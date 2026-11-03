"""Constant CFG, negative-branch layer skipping and option forwarding."""

import unittest

import torch

from support import engine_modules


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.modules = self.enterContext(engine_modules())
        self.guidance = self.modules.guidance

    def test_parse_skip_layers(self):
        parse = self.guidance.parse_skip_layers
        self.assertEqual(parse(""), ())
        self.assertEqual(parse(None), ())
        self.assertEqual(parse("22"), (22,))
        self.assertEqual(parse(" 22, 24 ;26 "), (22, 24, 26))
        self.assertEqual(parse("22, 22"), (22,))
        self.assertEqual(parse([22, 24]), (22, 24))
        for value in ("22a", "-1", "layer 22", [-1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value)

    def test_guider_forwards_model_patches_seed_and_conditioning(self):
        model = object()
        positive, negative = object(), object()
        for cfg in (1.0, 3.0):
            guider = self.guidance.CyberMOSSGuider(model, cfg)
            guider.set_conds(positive, negative)
            options = {"transformer_options": {"patches": object()}}
            guider.predict_noise(torch.ones(1), torch.tensor([.5]), model_options=options, seed=42)
            args = self.modules.comfy.samplers.sampling_function.call_args
            self.assertIs(args.args[0], model)
            self.assertIs(args.args[3], negative)
            self.assertIs(args.args[4], positive)
            self.assertEqual(args.args[5], cfg)
            self.assertIs(args.kwargs["model_options"], options)
            self.assertEqual(args.kwargs["seed"], 42)

    def test_skip_layers_only_change_the_uncond_pass(self):
        model = object()
        guider = self.guidance.CyberMOSSGuider(model, 3.0, skip_layers=(22, 24))
        positive, negative = object(), object()
        guider.set_conds(positive, negative)
        options = {"transformer_options": {"patches_replace": {"dit": {("single_block", 5): "user"}}}}
        guider.predict_noise(torch.ones(1), torch.tensor([.5]), model_options=options, seed=1)
        passed = self.modules.comfy.samplers.sampling_function.call_args.kwargs["model_options"]
        self.assertIsNot(passed, options)
        self.assertNotIn("sampler_calc_cond_batch_function", options)
        calc = passed["sampler_calc_cond_batch_function"]

        x, sigma = torch.ones(1), torch.tensor([.5])
        cond_out, uncond_out = calc({"conds": ["c", "u"], "input": x, "sigma": sigma,
                                     "model": model, "model_options": options})
        self.assertEqual(cond_out[:2], ("out", "c"))
        self.assertIs(cond_out[2], options)
        self.assertEqual(uncond_out[:2], ("out", "u"))
        replaced = uncond_out[2]["transformer_options"]["patches_replace"]["dit"]
        self.assertEqual(set(replaced), {("single_block", 5), ("single_block", 22), ("single_block", 24)})
        self.assertEqual(replaced[("single_block", 5)], "user")
        skip = replaced[("single_block", 22)]
        block_args = {"img": x}
        self.assertIs(skip(block_args, {"original_block": None}), block_args)
        # The caller's options are never mutated.
        self.assertEqual(set(options["transformer_options"]["patches_replace"]["dit"]),
                         {("single_block", 5)})

        # CFG 1 optimization: ComfyUI passes no uncond, so only the cond pass runs.
        calls = self.modules.comfy.samplers.calc_cond_batch.call_count
        out = calc({"conds": ["c", None], "input": x, "sigma": sigma,
                    "model": model, "model_options": options})
        self.assertIsNone(out[1])
        self.assertEqual(self.modules.comfy.samplers.calc_cond_batch.call_count, calls + 1)

    def test_skip_is_not_installed_when_it_cannot_apply(self):
        model, positive, negative = object(), object(), object()
        existing = {"transformer_options": {}, "sampler_calc_cond_batch_function": object()}
        cases = [
            (1.0, (22,), negative, {"transformer_options": {}}),
            (3.0, (), negative, {"transformer_options": {}}),
            (3.0, (22,), None, {"transformer_options": {}}),
            (3.0, (22,), negative, existing),
        ]
        for cfg, layers, neg, options in cases:
            with self.subTest(cfg=cfg, layers=layers, negative=neg is not None):
                guider = self.guidance.CyberMOSSGuider(model, cfg, skip_layers=layers)
                guider.set_conds(positive, neg)
                guider.predict_noise(torch.ones(1), torch.tensor([.5]), model_options=options)
                passed = self.modules.comfy.samplers.sampling_function.call_args.kwargs["model_options"]
                self.assertIs(passed, options)


if __name__ == "__main__":
    unittest.main()
