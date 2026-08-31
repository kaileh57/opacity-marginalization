import unittest

from sculptor.config import get_default_track_config
from sculptor.forward_model.atmosphere import build_temperature_profile, is_physical_profile


class AtmosphereTests(unittest.TestCase):
    def test_profile_builds_on_full_grid(self) -> None:
        config = get_default_track_config("g395h")
        profile = build_temperature_profile([600, 800, 1000, 1200, 1400], config)
        self.assertEqual(len(profile.pressures_bar), config.n_layers)
        self.assertEqual(len(profile.temperatures_K), config.n_layers)
        self.assertTrue(is_physical_profile(profile))


if __name__ == "__main__":
    unittest.main()
