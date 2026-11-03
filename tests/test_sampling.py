"""Tensor-level sampling regressions; no checkpoint or GPU is required."""

import types
import sys
import unittest
from unittest.mock import Mock, patch

import torch

from cybermoss_sampler.noise import contract_noise
from cybermoss_sampler.presets import PRESETS
from cybermoss_sampler.schedules import build_schedule
from support import engine_modules


def latent(*shape):
    return torch.zeros(*shape) if shape else torch.zeros(1, 128, 2, 2)


def moss_model(layers=42, class_name="MOSSImage"):
    base_class = type(class_name, (), {})
    base = base_class()
    base.diffusion_model = types.SimpleNamespace(layers=[object()] * layers)
    return types.SimpleNamespace(model=base)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.modules = self.enterContext(engine_modules())
        self.engine = self.modules.sampling

    def test_loop_obeys_small_step_budgets(self):
        for steps in (1, 2, 3, 4, 8, 12, 16):
            for plunge in (False, True):
                for frac in (0.0, .25, .60):
                    for sigma_r in (0.0, .65):
                        with self.subTest(steps=steps, plunge=plunge, frac=frac, sigma_r=sigma_r):
                            model = Mock(side_effect=lambda x, sigma: x * .5)
                            callback = Mock()
                            sigmas = build_schedule(steps, restart_frac=frac,
                                                    sigma_r=sigma_r, plunge=plunge)
                            result = self.engine.cybermoss_sampler_loop(
                                model, torch.ones(1, 128, 2, 2), sigmas,
                                callback=callback, eta0=1, order=2)
                            self.assertEqual(model.call_count, steps)
                            self.assertEqual(callback.call_count, steps)
                            self.assertTrue(torch.isfinite(result).all())

    def test_stochastic_streams_are_reproducible(self):
        sigmas = build_schedule(12, restart_frac=.25, sigma_r=.65, plunge=True)
        x = torch.ones(2, 128, 2, 2)
        def run(seed):
            return self.engine.cybermoss_sampler_loop(
                lambda value, sigma: value.tanh() * .7, x.clone(), sigmas,
                eta0=1, detail_amount=.6, restart_seed=seed)
        first = run(123)
        torch.randn(20)  # Global RNG use must not change the restart stream.
        self.assertTrue(torch.equal(first, run(123)))
        self.assertFalse(torch.equal(first, run(124)))

    def test_turbo_preset_is_effectively_euler(self):
        preset = PRESETS["turbo"]
        sigmas = build_schedule(preset["steps"], restart_frac=preset["restart_frac"],
                                sigma_r=preset["sigma_r"], plunge=preset["plunge"])
        results = [self.engine.cybermoss_sampler_loop(
            lambda x, sigma: x.tanh(), torch.ones(1, 128, 2, 2), sigmas,
            order=order, eta0=preset["eta0"], detail_amount=preset["detail"],
            restart_seed=42) for order in (1, 2)]
        self.assertTrue(torch.equal(*results))

    def test_ab2_accounts_for_unequal_step_widths(self):
        inputs = []
        def model(x, sigma):
            inputs.append(x.clone())
            return x - sigma.square().reshape(-1, 1, 1, 1)
        self.engine.cybermoss_sampler_loop(
            model, torch.zeros(1, 128, 1, 1), torch.tensor([1., .9, .7, 0.]),
            order=2, eta0=0)
        # d(sigma)=sigma: Euler startup contributes -.1, then integrating
        # the linear derivative from .9 to .7 contributes exactly -.16.
        torch.testing.assert_close(inputs[2], torch.full_like(inputs[2], -.26))

    def test_eta_gate_cuts_off_and_ramps(self):
        eta = self.engine._gated_eta
        for gate in (0, .1, .35, .5, 1):
            self.assertEqual(eta(gate, 1, gate), 0)
            self.assertEqual(eta(gate / 2, 1, gate), 0)
        self.assertAlmostEqual(eta(.225, 1, .1), .5)
        self.assertEqual(eta(.35, 1, .1), 1)
        self.assertEqual(eta(.6, 1, .5), 1)
        self.assertEqual(eta(.2, 1, .5), 0)
        self.assertEqual(eta(.6, 0, .5), 0)

    def test_high_gate_prevents_rng_draws_below_cutoff(self):
        sigmas = torch.tensor([.3, .2, .1, 0.])
        kwargs = dict(model=lambda x, sigma: x * .5,
                      x=torch.ones(1, 128, 2, 2), sigmas=sigmas, sigma_gate=.5)
        first = self.engine.cybermoss_sampler_loop(**kwargs, eta0=1, restart_seed=42)
        second = self.engine.cybermoss_sampler_loop(**kwargs, eta0=0, restart_seed=84)
        self.assertTrue(torch.equal(first, second))

    def test_contraction_only_affects_initial_noise(self):
        noise = torch.ones(1, 128, 2, 2)
        self.assertIs(contract_noise(noise, 1), noise)
        self.assertEqual(contract_noise(noise, 0).count_nonzero(), 0)
        torch.testing.assert_close(contract_noise(noise, .7), noise * .7)
        sigmas = build_schedule(4, restart_frac=.25, sigma_r=.65, plunge=True)
        result = self.engine.cybermoss_sampler_loop(
            lambda x, sigma: x * .5, contract_noise(noise, 0), sigmas, eta0=0)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(result.count_nonzero().item(), 0)
        for value in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                contract_noise(noise, value)

    def test_run_keeps_model_conditioning_mask_and_batch_indices(self):
        model = object()
        positive = [[torch.ones(1, 2, 3), {"pooled_output": torch.ones(1, 3)}]]
        negative = [[torch.zeros(1, 2, 3), {}]]
        mask = torch.ones(1, 2, 2)
        samples = {"samples": torch.zeros(2, 128, 2, 2), "noise_mask": mask,
                   "batch_index": [2, 5], "custom": "preserved",
                   "downscale_ratio_spacial": 8}
        sigmas = build_schedule(4, restart_frac=.25, plunge=True)
        for conditioning, cfg in ((None, 1.0), (negative, 1.0), (negative, 3.0)):
            out = self.engine.run_sampling(model, positive, conditioning, samples, sigmas,
                                           seed=123, cfg=cfg, skip_layers=(22,))
            guider = self.modules.guiders[-1]
            self.assertIsInstance(guider, self.modules.guidance.CyberMOSSGuider)
            self.assertIs(guider.model_patcher, model)
            self.assertEqual(guider.cfg, cfg)
            self.assertEqual(guider.skip_layers, (22,) if cfg != 1 else ())
            self.assertIs(guider.conds["positive"], positive)
            self.assertIs(guider.sample.call_args.kwargs["denoise_mask"], mask)
            self.assertEqual(guider.sample.call_args.kwargs["seed"], 123)
            if conditioning is None:
                self.assertEqual(guider.conds["negative"][0][0].count_nonzero(), 0)
                self.assertEqual(positive[0][0].count_nonzero(), 6)
            else:
                self.assertIs(guider.conds["negative"], negative)
            self.assertEqual(self.modules.comfy.sample.prepare_noise.call_args.args[2], [2, 5])
            self.assertIsNot(out, samples)
            self.assertIs(out["noise_mask"], mask)
            self.assertEqual(out["custom"], "preserved")
            self.assertNotIn("downscale_ratio_spacial", out)
            self.assertEqual(samples["downscale_ratio_spacial"], 8)
            self.assertEqual(samples["samples"].count_nonzero(), 0)

    def test_cfg_above_one_requires_negative(self):
        with self.assertRaisesRegex(ValueError, "needs negative conditioning"):
            self.engine.run_sampling(object(), [], None, {"samples": latent()},
                                     build_schedule(2), seed=0, cfg=3.0)

    def test_model_and_skip_layer_checks(self):
        negative = [[torch.zeros(1, 2, 3), {}]]
        self.engine.run_sampling(moss_model(), [], negative, {"samples": latent()},
                                 build_schedule(2), seed=0, cfg=3.0, skip_layers=(22, 41))
        with self.assertRaisesRegex(ValueError, "outside this model's layers 0-41"):
            self.engine.run_sampling(moss_model(), [], negative, {"samples": latent()},
                                     build_schedule(2), seed=0, cfg=3.0, skip_layers=(42,))
        # CFG 1 never runs the negative pass, so the skip list is irrelevant there.
        self.engine.run_sampling(moss_model(), [], None, {"samples": latent()},
                                 build_schedule(2), seed=0, cfg=1.0, skip_layers=(99,))
        with self.assertRaisesRegex(ValueError, "requires a MOSS-Image model; got Krea2"):
            self.engine.run_sampling(moss_model(class_name="Krea2"), [], None,
                                     {"samples": latent()}, build_schedule(2), seed=0)

    def test_preview_progress_excludes_restart_jump(self):
        model = object()
        self.engine.run_sampling(model, [], None, {"samples": latent()},
                                 build_schedule(4, restart_frac=.25, plunge=True), seed=0)
        self.modules.preview.prepare_callback.assert_called_once_with(model, 4)
        callback = self.modules.guiders[-1].sample.call_args.kwargs["callback"]
        for index in (0, 1, 2, 4):
            callback(index, "denoised", "latent", 5)
        self.assertEqual([call.args for call in self.modules.preview.prepare_callback.return_value.call_args_list],
                         [(index, "denoised", "latent", 4) for index in range(4)])

    def test_native_and_official_loader_models_coexist(self):
        base_model = type("BaseModel", (), {})
        native = type("MOSSImage", (base_model,), {"__module__": "comfy.model_base"})
        official = type("MOSSImage", (base_model,), {"__module__": "moss-image.loaders"})
        model_base = types.ModuleType("comfy.model_base")
        model_base.MOSSImage = native
        models = (native, official, type("NativeWrapper", (native,), {}),
                  type("OfficialWrapper", (official,), {}))
        with patch.dict(sys.modules, {"comfy.model_base": model_base}), \
             patch.object(self.modules.comfy, "model_base", model_base, create=True):
            for cls in models:
                with self.subTest(model=cls):
                    base = cls()
                    base.diffusion_model = types.SimpleNamespace(layers=[None] * 42)
                    model = types.SimpleNamespace(model=base)
                    self.assertEqual(self.engine._moss_layer_count(model), 42)
                    self.engine.run_sampling(model, [], [], {"samples": latent()},
                                             build_schedule(2), seed=0, cfg=3, skip_layers=(22,))
                    with self.assertRaisesRegex(ValueError, "outside this model's layers"):
                        self.engine.run_sampling(model, [], [], {"samples": latent()},
                                                 build_schedule(2), seed=0, cfg=3, skip_layers=(42,))
            with self.assertRaisesRegex(ValueError, "got Krea2"):
                self.engine._moss_layer_count(moss_model(class_name="Krea2"))

    def test_wrong_latent_shape_has_actionable_error(self):
        for shape in ((1, 16, 2, 2), (1, 16, 1, 2, 2), (1, 128, 1, 2, 2), (1, 4, 2, 2)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "MOSS-Image / Flux2"):
                self.engine.run_sampling(object(), [], None, {"samples": torch.zeros(shape)},
                                         build_schedule(1), seed=0)

    def test_comfy_converts_empty_image_latent_before_shape_validation(self):
        samples = {"samples": torch.zeros(1, 4, 4, 4), "downscale_ratio_spacial": 8}
        fix = self.modules.comfy.sample.fix_empty_latent_channels
        fix.side_effect = lambda model, latent, ratio=None: torch.zeros(1, 128, 2, 2)
        result = self.engine.run_sampling(object(), [], None, samples, build_schedule(1), seed=0)
        self.assertEqual(fix.call_args.args[2], 8)
        self.assertEqual(tuple(result["samples"].shape), (1, 128, 2, 2))
        self.assertEqual(tuple(samples["samples"].shape), (1, 4, 4, 4))
        fix.side_effect = lambda model, latent, *args: latent
        self.engine.run_sampling(object(), [], None, {"samples": latent()}, build_schedule(1), seed=0)
        self.assertEqual(len(fix.call_args.args), 2)


if __name__ == "__main__":
    unittest.main()
