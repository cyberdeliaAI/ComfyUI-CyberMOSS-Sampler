"""Preset defaults, manual overrides, validation and frontend metadata."""

import unittest

from cybermoss_sampler.presets import DEFAULT_PRESET, PRESETS
from support import engine_modules


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.modules = self.enterContext(engine_modules())
        self.nodes = self.modules.nodes

    def test_preset_names_and_guidance(self):
        self.assertEqual(list(PRESETS), ["turbo", "turbo_reference", "full", "full_reference"])
        self.assertEqual(DEFAULT_PRESET, "turbo")
        for name in ("turbo", "turbo_reference"):
            self.assertEqual((PRESETS[name]["cfg"], PRESETS[name]["skip_layers"]), (1.0, ""))
        for name in ("full", "full_reference"):
            self.assertEqual((PRESETS[name]["cfg"], PRESETS[name]["skip_layers"]), (3.0, "22"))
            self.assertFalse(PRESETS[name]["plunge"])
            self.assertEqual(PRESETS[name]["restart_frac"], 0)
        self.assertEqual(PRESETS["full_reference"]["steps"], 50)
        self.assertEqual(PRESETS["full_reference"]["schedule_mode"], "reference")
        self.assertEqual(PRESETS["turbo_reference"]["schedule_mode"], "simple")

    def test_all_presets_resolve_without_frontend(self):
        for name, defaults in PRESETS.items():
            with self.subTest(preset=name):
                result = self.nodes.resolve_settings(name)
                self.assertEqual({k: result[k] for k in defaults}, defaults)
                self.assertEqual(result["order"], 1)
                self.assertEqual(result["shift"], 5.0)

    def test_explicit_controls_override_preset_including_zero_and_false(self):
        overrides = dict(steps=10, sampler="euler_2m", cfg=4.0, skip_layers="24",
                         restart_frac=0.0, sigma_r=0.55, plunge=False, detail=0.42,
                         eta0=0.0, sigma_gate=0.20, contraction=0.0)
        for name in PRESETS:
            result = self.nodes.resolve_settings(name, **overrides)
            self.assertEqual({k: result[k] for k in overrides}, overrides)
            self.assertEqual(result["order"], 2)
            self.assertEqual(result["skip_layer_list"], (24,))

    def test_settings_do_not_modify_preset(self):
        result = self.nodes.resolve_settings("turbo")
        result["steps"] = 1
        self.assertEqual(PRESETS["turbo"]["steps"], 8)

    def test_invalid_inputs_are_rejected(self):
        cases = [("steps", 0), ("steps", 101), ("steps", 1.5), ("steps", True),
                 ("sampler", "unknown"), ("plunge", "false"), ("restart_frac", .61),
                 ("sigma_r", -1), ("detail", 2), ("eta0", 3),
                 ("sigma_gate", float("nan")), ("contraction", float("inf")),
                 ("contraction", "invalid"), ("cfg", -1), ("cfg", 21), ("cfg", float("nan")),
                 ("skip_layers", "22a"), ("skip_layers", "-1")]
        cases += [("denoise", -0.1), ("denoise", 1.1), ("denoise", float("nan")),
                  ("schedule_mode", "unknown"), ("schedule_mode", "cyberkrea")]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.nodes.resolve_settings("turbo", **{key: value})
        with self.assertRaisesRegex(ValueError, "Unknown CyberMOSS preset"):
            self.nodes.resolve_settings("balanced")

    def test_ui_defaults_and_metadata_use_python_presets(self):
        definition = self.nodes.CyberMOSSSampler.INPUT_TYPES()
        inputs = {**definition["required"], **definition["optional"]}
        self.assertEqual(inputs["preset"][1]["cybermoss_presets"], PRESETS)
        for key, value in PRESETS[DEFAULT_PRESET].items():
            self.assertEqual(inputs[key][1]["default"], value)
        self.assertEqual(inputs["preview_method"][1]["default"], "default")
        self.assertIn("taef2_decoder", inputs["preview_method"][1]["tooltip"])


if __name__ == "__main__":
    unittest.main()
