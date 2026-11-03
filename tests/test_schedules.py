"""Schedule invariants across the full supported step range."""

import math
import json
from pathlib import Path
import unittest

import torch

from cybermoss_sampler.presets import PRESETS, SHIFT
from cybermoss_sampler.schedules import (
    build_schedule, build_sampling_schedule, sigma_from_t, t_from_sigma,
)


class ScheduleTests(unittest.TestCase):
    def test_budget_endpoints_and_restart_structure(self):
        for steps in range(1, 101):
            for plunge in (False, True):
                for frac in (0.0, 0.01, 0.25, 0.60):
                    for sigma_r in (0.0, 0.01, 0.65, 1.0):
                        with self.subTest(steps=steps, plunge=plunge, frac=frac, sigma_r=sigma_r):
                            sigmas = build_schedule(steps, restart_frac=frac,
                                                    sigma_r=sigma_r, plunge=plunge)
                            self.assertEqual(sigmas[0].item(), 1.0)
                            self.assertEqual(sigmas[-1].item(), 0.0)
                            self.assertTrue(torch.isfinite(sigmas).all())
                            jumps = (sigmas[1:] > sigmas[:-1]).nonzero().flatten()
                            expected_restart = frac > 0 and sigma_r > 0 and steps > (2 if plunge else 1)
                            self.assertEqual(len(jumps), int(expected_restart))
                            evaluations = ((sigmas[:-1] > 0) & (sigmas[1:] <= sigmas[:-1])).sum()
                            self.assertEqual(evaluations.item(), steps)
                            if expected_restart:
                                self.assertEqual(sigmas[jumps[0]].item(), 0.0)

    def test_one_step_is_always_full_descent(self):
        for plunge in (False, True):
            self.assertEqual(build_schedule(1, restart_frac=.25, sigma_r=.65,
                                            plunge=plunge).tolist(), [1.0, 0.0])

    def test_turbo_preset_structure(self):
        preset = PRESETS["turbo"]
        sigmas = build_schedule(preset["steps"], restart_frac=preset["restart_frac"],
                                sigma_r=preset["sigma_r"], plunge=preset["plunge"])
        zero_index = (sigmas == 0).nonzero()[0].item()
        self.assertEqual(zero_index, preset["steps"] * 3 // 4)
        self.assertAlmostEqual(sigmas[zero_index - 1].item(), .75)
        self.assertAlmostEqual(sigmas[zero_index + 1].item(), .65)

    def test_full_preset_is_one_plain_descent(self):
        preset = PRESETS["full"]
        sigmas = build_sampling_schedule(preset["steps"], restart_frac=preset["restart_frac"],
                                         sigma_r=preset["sigma_r"], plunge=preset["plunge"])
        self.assertEqual(len(sigmas), preset["steps"] + 1)
        self.assertTrue((sigmas[1:] < sigmas[:-1]).all())

    def test_shift_inverse(self):
        for shift in (1.0, 3.158, SHIFT, 10.0):
            for t in (0.0, .1, .5, .9, 1.0):
                self.assertAlmostEqual(t_from_sigma(sigma_from_t(t, shift), shift), t)

    def test_full_cybermoss_schedules_match_build_schedule(self):
        for name in ("turbo", "full"):
            p = PRESETS[name]
            kwargs = {key: p[key] for key in ("restart_frac", "sigma_r", "plunge")}
            self.assertTrue(torch.equal(build_schedule(p["steps"], **kwargs),
                                        build_sampling_schedule(p["steps"], **kwargs)))

    def test_simple_matches_shift_five_uniform_grid_and_tail(self):
        self.assertEqual(SHIFT, 5.0)
        for steps in (1, 3, 8, 16, 50, 100):
            for denoise in (1.0, .3, .65, .01):
                total = int(steps / denoise)
                t = torch.linspace(1, 0, total + 1, dtype=torch.float64)
                expected = SHIFT * t / (1 + (SHIFT - 1) * t)
                actual = build_sampling_schedule(steps, schedule_mode="simple", denoise=denoise,
                                                 restart_frac=.25, plunge=True)
                torch.testing.assert_close(actual, expected[-(steps+1):].float(), rtol=0, atol=1e-6)
                self.assertEqual(len(actual), steps + 1)
                self.assertTrue((actual[1:] < actual[:-1]).all())

    def test_reference_matches_official_moss_scheduler_fixtures(self):
        fixture = json.loads(Path(__file__).with_name("moss_reference_sigmas.json").read_text())
        for steps, values in fixture["sigmas"].items():
            with self.subTest(steps=steps):
                actual = build_sampling_schedule(int(steps), schedule_mode="reference",
                                                 restart_frac=.25, plunge=True)
                torch.testing.assert_close(actual, torch.tensor(values), rtol=0, atol=1e-7)
        sigmas = build_sampling_schedule(50, schedule_mode="reference")
        self.assertAlmostEqual(sigmas[-2].item(), .024, places=6)

    def test_reference_preserves_supported_budgets_and_training_grid(self):
        for steps in range(1, 101):
            sigmas = build_sampling_schedule(steps, schedule_mode="reference")
            self.assertEqual(len(sigmas), steps + 1)
            self.assertEqual(sigmas[0].item(), 1)
            self.assertEqual(sigmas[-1].item(), 0)
            self.assertTrue((sigmas[1:] < sigmas[:-1]).all())
            torch.testing.assert_close(sigmas[:-1] * 1000,
                                       (sigmas[:-1] * 1000).round(), rtol=0, atol=1e-4)

    def test_partial_reference_keeps_continuous_tail_and_requested_budget(self):
        # Snapping long expanded schedules would merge steps at low denoise.
        for steps in (3, 8, 50, 100):
            for denoise in (.3, .65, .01, 1e-8):
                reference = build_sampling_schedule(steps, schedule_mode="reference", denoise=denoise)
                simple = build_sampling_schedule(steps, schedule_mode="simple", denoise=denoise)
                self.assertEqual(len(reference), steps + 1)
                torch.testing.assert_close(reference, simple, rtol=0, atol=0)
                self.assertTrue((reference[1:] < reference[:-1]).all())

    def test_partial_uses_plain_schedule_tail_without_restart(self):
        for steps in (3, 8):
            for denoise in (.3, .65):
                expected = build_schedule(int(steps / denoise))[-(steps+1):]
                actual = build_sampling_schedule(steps, denoise=denoise, plunge=True, restart_frac=.25)
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)

    def test_denoise_limits_and_small_values(self):
        for mode in ("cybermoss", "reference", "simple"):
            self.assertEqual(build_sampling_schedule(8, schedule_mode=mode, denoise=0).numel(), 0)
            tiny = build_sampling_schedule(8, schedule_mode=mode, denoise=1e-8)
            self.assertEqual(len(tiny), 9)
            self.assertTrue(torch.isfinite(tiny).all())
        for kwargs in ({"denoise": -1}, {"denoise": 1.01}, {"denoise": float("nan")},
                       {"schedule_mode": "unknown"}, {"shift": 0}):
            with self.assertRaises(ValueError):
                build_sampling_schedule(8, **kwargs)

    def test_invalid_schedule_inputs(self):
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                build_schedule(steps)
        for kwargs in ({"shift": 0}, {"shift": float("inf")},
                       {"sigma_r": -1}, {"restart_frac": float("nan")},
                       {"restart_frac": 1.1}):
            with self.assertRaises(ValueError):
                build_schedule(8, **kwargs)


if __name__ == "__main__":
    unittest.main()
