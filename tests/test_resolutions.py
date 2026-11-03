"""Resolution tiers, stale widget recovery and real latent allocation."""

import math
import unittest

from cybermoss_sampler import resolutions


class ResolutionTests(unittest.TestCase):
    def test_tiers_dimensions_and_metadata(self):
        self.assertEqual(list(resolutions.RESOLUTION_OPTIONS),
                         ["S (~1.0 MP)", "M (~1.5 MP)", "L (~2.0 MP)", "XL (~3.0 MP)"])
        self.assertEqual(resolutions.DEFAULT_RESOLUTION, "1024x1536 (2:3)")
        self.assertEqual(resolutions.CyberMOSSEmptyLatent.INPUT_TYPES()["required"]["size"][1]
                         ["cybermoss_resolutions"], resolutions.RESOLUTION_OPTIONS)
        targets = {"S (~1.0 MP)": 1.0, "M (~1.5 MP)": 1.5, "L (~2.0 MP)": 2.0, "XL (~3.0 MP)": 3.0}
        for tier, options in resolutions.RESOLUTION_OPTIONS.items():
            self.assertEqual(len(options), 11)
            for option in options:
                width, height = resolutions.resolve_dimensions(tier, option)
                self.assertEqual(width % 16, 0)
                self.assertEqual(height % 16, 0)
                self.assertTrue(option.startswith(f"{width}x{height} "))
                megapixels = width * height / (1024 * 1024)
                self.assertLess(abs(math.log(megapixels / targets[tier])), 0.08, option)
                a, b = map(int, option.rpartition("(")[2].removesuffix(")").split(":"))
                self.assertLess(abs(math.log((width / height) / (a / b))), 0.01, option)

    def test_stale_resolution_preserves_each_aspect_in_each_tier(self):
        for source in resolutions.ALL_RESOLUTIONS:
            aspect = source.rpartition("(")[2].removesuffix(")")
            for tier in resolutions.RESOLUTION_OPTIONS:
                self.assertEqual(resolutions.resolve_dimensions(tier, source),
                                 resolutions.RESOLUTION_BUCKETS[tier][aspect])

    def test_unknown_size_or_resolution_is_rejected(self):
        for tier, label in [("unknown", resolutions.DEFAULT_RESOLUTION),
                            ("XL (~3.0 MP)", "invalid (3:2)")]:
            with self.assertRaises(ValueError):
                resolutions.resolve_dimensions(tier, label)

    def test_generate_allocates_flux2_latents(self):
        latent, width, height = resolutions.CyberMOSSEmptyLatent().generate(
            "S (~1.0 MP)", "768x1360 (9:16)", 2)
        self.assertEqual((width, height), (768, 1360))
        self.assertEqual(tuple(latent["samples"].shape), (2, 128, 85, 48))
        self.assertEqual(latent["samples"].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
