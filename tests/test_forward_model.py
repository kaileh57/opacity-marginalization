import unittest

import numpy as np

from sculptor.forward_model.instrument import InstrumentResult
from sculptor.forward_model.radiative_transfer import RadiativeTransferResult
from sculptor.forward_model.spectrum import SpectrumModel


class _FakeRadiativeTransfer:
    def compute_flux(self, chemistry_state, opacity_result, gravity_cgs, radius_rjup, distance_pc,
                     cloud_parameters=None):
        nu_grid = np.asarray(opacity_result.nu_grid_cm1, dtype=np.float64)
        flux_nu = 1.0 - 0.25 * np.exp(-0.5 * ((nu_grid - 3030.0) / 30.0) ** 2)
        return RadiativeTransferResult(
            flux_nu=flux_nu,
            total_optical_depth=np.zeros_like(nu_grid),
            layer_optical_depth=np.zeros((len(chemistry_state.pressures_bar), len(nu_grid))),
            metadata={"backend": "test"},
        )


class _IdentityInstrument:
    def __init__(self, nu_grid_cm1: np.ndarray) -> None:
        self.nu_grid_cm1 = nu_grid_cm1

    def convolve(self, flux_nu):
        return InstrumentResult(nu_grid_cm1=self.nu_grid_cm1, flux_nu=np.asarray(flux_nu))


class ForwardModelTests(unittest.TestCase):
    def test_end_to_end_spectrum_has_expected_bands(self) -> None:
        base_model = SpectrumModel(radiative_transfer=_FakeRadiativeTransfer())
        model = SpectrumModel(
            track_config=base_model.track_config,
            chemistry_model=base_model.chemistry_model,
            opacity_backend=base_model.opacity_backend,
            radiative_transfer=_FakeRadiativeTransfer(),
            instrument_model=_IdentityInstrument(base_model.track_config.nu_grid_cm1),
        )
        spectrum = model.generate(
            {
                "T1": 700.0,
                "T2": 850.0,
                "T3": 1000.0,
                "T4": 1200.0,
                "T5": 1400.0,
                "metallicity": 0.0,
                "carbon_to_oxygen": 0.55,
                "log_kzz": 8.0,
                "log_gravity": 5.0,
                "radius_rjup": 1.0,
                "snr": 50.0,
            }
        )
        self.assertEqual(spectrum.flux_nu.shape, spectrum.nu_grid_cm1.shape)
        self.assertTrue(np.all(np.isfinite(spectrum.flux_nu)))
        ch4_index = int(np.argmin(np.abs(spectrum.nu_grid_cm1 - 3030.0)))
        continuum_index = int(np.argmin(np.abs(spectrum.nu_grid_cm1 - 2500.0)))
        self.assertLess(spectrum.flux_nu[ch4_index], spectrum.flux_nu[continuum_index])


if __name__ == "__main__":
    unittest.main()
