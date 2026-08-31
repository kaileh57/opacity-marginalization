import unittest

import numpy as np

from sculptor.config import get_default_track_config
from sculptor.forward_model.atmosphere import build_temperature_profile
from sculptor.forward_model.chemistry import ChemistryModel, co_quench_timescale_seconds, nh3_quench_timescale_seconds


class ChemistryTests(unittest.TestCase):
    def test_quench_timescales_are_positive(self) -> None:
        temperatures = np.array([900.0, 1100.0, 1300.0])
        pressures = np.array([1.0e-3, 1.0e-1, 10.0])
        self.assertTrue(np.all(co_quench_timescale_seconds(temperatures, pressures, metallicity=0.0) > 0.0))
        self.assertTrue(np.all(nh3_quench_timescale_seconds(temperatures, pressures) > 0.0))

    def test_chemistry_state_shapes(self) -> None:
        config = get_default_track_config("g395h")
        profile = build_temperature_profile([700, 850, 1000, 1200, 1400], config)
        state = ChemistryModel(config).compute_state(profile, metallicity=0.0, carbon_to_oxygen=0.55, log_gravity=5.0, log_kzz=8.0)
        self.assertEqual(state.vmr.shape[0], config.n_layers)
        self.assertEqual(state.vmr.shape[1], len(state.species))
        self.assertTrue(np.all(np.isfinite(state.mean_molecular_weight)))
        np.testing.assert_allclose(state.vmr_for("H2-H2"), state.vmr_for("H2") ** 2)
        np.testing.assert_allclose(state.vmr_for("H2-He"), state.vmr_for("H2") * state.vmr_for("He"))


if __name__ == "__main__":
    unittest.main()
