"""Preview settings, first-image decode and node-to-engine wiring."""

from enum import Enum
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from support import engine_modules


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.modules = self.enterContext(engine_modules())
        self.nodes = self.modules.nodes

    def test_default_preview_preserves_global_setting(self):
        args = types.SimpleNamespace(preview_method="disabled")
        cli = types.ModuleType("comfy.cli_args")
        cli.args = args
        with patch.dict(sys.modules, {"comfy.cli_args": cli}):
            with self.nodes._live_preview("default"):
                self.assertEqual(args.preview_method, "disabled")
            self.assertEqual(args.preview_method, "disabled")

    def test_explicit_preview_override_is_restored_even_on_failure(self):
        methods = Enum("LatentPreviewMethod", "Auto Latent2RGB TAESD NoPreviews")
        self.modules.preview.LatentPreviewMethod = methods
        cli = types.ModuleType("comfy.cli_args")
        cli.args = types.SimpleNamespace(preview_method=methods.NoPreviews)
        with patch.dict(sys.modules, {"comfy.cli_args": cli}):
            for name, expected in (("auto", methods.Auto), ("latent2rgb", methods.Latent2RGB),
                                   ("taesd", methods.TAESD), ("none", methods.NoPreviews)):
                with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                    with self.nodes._live_preview(name):
                        self.assertIs(cli.args.preview_method, expected)
                        raise RuntimeError("sampling failed")
                self.assertIs(cli.args.preview_method, methods.NoPreviews)
        with self.assertRaises(ValueError):
            with self.nodes._live_preview("unknown"):
                pass

    def test_vae_decodes_only_first_image_and_keeps_full_latent(self):
        samples = torch.arange(3 * 128 * 4, dtype=torch.float32).reshape(3, 128, 2, 2)
        output = {"samples": samples}
        for decoded_shape in ((1, 32, 32, 3), (1, 1, 32, 32, 3)):
            vae = types.SimpleNamespace(decode=Mock(return_value=torch.zeros(decoded_shape)))
            save_images = Mock(return_value={"ui": {"images": ["thumbnail"]}})
            comfy_nodes = types.ModuleType("nodes")
            comfy_nodes.PreviewImage = Mock(return_value=types.SimpleNamespace(save_images=save_images))
            with patch.dict(sys.modules, {"nodes": comfy_nodes}):
                result = self.nodes._result_with_preview(output, vae)
            torch.testing.assert_close(vae.decode.call_args.args[0], samples[:1])
            self.assertEqual(tuple(save_images.call_args.args[0].shape), (1, 32, 32, 3))
            self.assertEqual(save_images.call_args.kwargs["filename_prefix"], "CyberMOSS")
            self.assertIs(result["result"][0], output)
            self.assertEqual(result["ui"]["images"], ["thumbnail"])
            self.assertEqual(output["samples"].shape[0], 3)
        self.assertEqual(self.nodes._result_with_preview(output, None), (output,))

    def test_node_uses_preset_defaults_and_explicit_overrides(self):
        model, positive, negative = object(), object(), object()
        latent = {"samples": torch.zeros(1, 128, 2, 2)}
        run = Mock(return_value=latent)
        with patch.object(self.nodes, "run_sampling", run):
            result = self.nodes.CyberMOSSSampler().sample(
                model, positive, latent, 42, "turbo", steps=2, sampler="euler_2m",
                contraction=0, negative=negative)
        self.assertIs(result[0], latent)
        args = run.call_args
        self.assertIs(args.args[0], model)
        self.assertIs(args.args[1], positive)
        self.assertIs(args.args[2], negative)
        self.assertIs(args.args[3], latent)
        self.assertEqual(args.args[4].tolist(), [1, .75, 0])
        self.assertEqual(args.kwargs["order"], 2)
        self.assertEqual(args.kwargs["contraction"], 0)
        self.assertEqual(args.kwargs["detail_amount"], .5)
        self.assertEqual(args.kwargs["cfg"], 1.0)
        self.assertEqual(args.kwargs["skip_layers"], ())

    def test_full_reference_uses_official_settings(self):
        latent = {"samples": torch.zeros(1, 128, 2, 2)}
        negative = object()
        run = Mock(return_value=latent)
        with patch.object(self.nodes, "run_sampling", run):
            self.nodes.CyberMOSSSampler().sample(
                object(), object(), latent, 7, "full_reference", negative=negative)
        args = run.call_args
        sigmas = args.args[4]
        self.assertEqual(len(sigmas), 51)
        self.assertEqual(sigmas[0], 1)
        self.assertEqual(sigmas[-1], 0)
        self.assertAlmostEqual(sigmas[-2].item(), .024, places=6)
        self.assertTrue((sigmas[1:] < sigmas[:-1]).all())
        self.assertIs(args.args[2], negative)
        self.assertEqual(args.kwargs["cfg"], 3.0)
        self.assertEqual(args.kwargs["skip_layers"], (22,))
        self.assertEqual(args.kwargs["eta0"], 0)
        self.assertEqual(args.kwargs["detail_amount"], 0)
        self.assertEqual(args.kwargs["contraction"], 1)
        self.assertEqual(args.kwargs["order"], 1)

    def test_turbo_reference_preserves_simple_schedule(self):
        latent = {"samples": torch.zeros(1, 128, 2, 2)}
        run = Mock(return_value=latent)
        with patch.object(self.nodes, "run_sampling", run):
            self.nodes.CyberMOSSSampler().sample(object(), object(), latent, 7, "turbo_reference")
        sigmas = run.call_args.args[4]
        self.assertEqual(len(sigmas), 9)
        self.assertAlmostEqual(sigmas[-2].item(), 5 / 12, places=6)
        self.assertEqual(run.call_args.kwargs["cfg"], 1)

    def test_manual_skip_layers_and_cfg_override_preset(self):
        latent = {"samples": torch.zeros(1, 128, 2, 2)}
        run = Mock(return_value=latent)
        with patch.object(self.nodes, "run_sampling", run):
            self.nodes.CyberMOSSSampler().sample(
                object(), object(), latent, 7, "full", cfg=4.5, skip_layers="22, 24, 26",
                negative=object())
        self.assertEqual(run.call_args.kwargs["cfg"], 4.5)
        self.assertEqual(run.call_args.kwargs["skip_layers"], (22, 24, 26))
        with patch.object(self.nodes, "run_sampling", run):
            self.nodes.CyberMOSSSampler().sample(
                object(), object(), latent, 7, "full", skip_layers="", negative=object())
        self.assertEqual(run.call_args.kwargs["skip_layers"], ())

    def test_reference_refinement_keeps_requested_steps(self):
        latent = {"samples": torch.ones(1, 128, 2, 2), "batch_index": [7]}
        model, positive = object(), object()
        run = Mock(return_value=latent)
        with patch.object(self.nodes, "run_sampling", run):
            self.nodes.CyberMOSSSampler().sample(
                model, positive, latent, 42, "turbo_reference", steps=3, denoise=.3)
        args = run.call_args
        self.assertIs(args.args[0], model)
        self.assertIs(args.args[1], positive)
        self.assertIs(args.args[3], latent)
        self.assertIsNone(args.args[2])
        sigmas = args.args[4]
        self.assertEqual(len(sigmas), 4)
        self.assertLess(sigmas[0], 1)
        self.assertEqual(sigmas[-1], 0)
        self.assertEqual(args.kwargs["contraction"], 1)
        self.assertEqual(args.kwargs["eta0"], 0)
        self.assertEqual(args.kwargs["detail_amount"], 0)

    def test_zero_denoise_preserves_latent_and_skips_sampling(self):
        latent = {"samples": torch.ones(1, 128, 2, 2), "noise_mask": object()}
        with patch.object(self.nodes, "run_sampling") as run:
            result = self.nodes.CyberMOSSSampler().sample(
                object(), object(), latent, 42, "turbo", denoise=0)
        run.assert_not_called()
        self.assertIs(result[0], latent)

    def test_widget_order(self):
        definition = self.nodes.CyberMOSSSampler.INPUT_TYPES()
        sockets = {"MODEL", "CONDITIONING", "LATENT", "VAE"}
        widgets = [name for name, spec in
                   {**definition["required"], **definition["optional"]}.items()
                   if not isinstance(spec[0], str) or spec[0] not in sockets]
        self.assertEqual(widgets, ["seed", "preset", "steps", "sampler", "cfg", "skip_layers",
                                  "restart_frac", "sigma_r", "plunge", "detail", "eta0",
                                  "sigma_gate", "contraction", "schedule_mode", "denoise",
                                  "preview_method"])


if __name__ == "__main__":
    unittest.main()
