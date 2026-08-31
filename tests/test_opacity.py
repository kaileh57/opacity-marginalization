import os
import unittest
from unittest import mock

import numpy as np

from sculptor.config import get_default_track_config
from sculptor.forward_model.atmosphere import build_temperature_profile
from sculptor.forward_model.chemistry import ChemistryModel
from sculptor.forward_model.opacity import (
    SyntheticOpacityBackend,
    apply_broadening_perturbation,
    broadening_perturbation_from_env,
)


class _StubPremodit:
    """Minimal stand-in for a loaded OpaPremodit (grids + bookkeeping arrays)."""

    def __init__(self) -> None:
        from exojax.opacity.premodit.info import PreMODITInfo

        self.multi_index_uniqgrid = np.array([[0, 0], [1, 1], [2, 0]], dtype=int)
        self.pre_modit_info = PreMODITInfo(
            multi_index_uniqgrid=self.multi_index_uniqgrid,
            elower_grid=np.array([0.0, 1000.0]),
            ngamma_ref_grid=np.array([0.01, 0.05, 0.2]),
            n_Texp_grid=np.array([0.3, 0.6]),
            R=1_000_000.0,
            pmarray=np.array([1.0, -1.0]),
        )
        self.opainfo = self.pre_modit_info.as_tuple()
        self.gamma_ref = np.array([0.01, 0.05, 0.2])
        self.n_Texp = np.array([0.3, 0.6, 0.3])


class BroadeningPerturbationTests(unittest.TestCase):
    def test_env_lookup_prefers_species_over_global(self) -> None:
        env = {
            "SCULPTOR_BROADENING_GAMMA_SCALE": "1.1",
            "SCULPTOR_BROADENING_GAMMA_SCALE_CO2": "1.4",
            "SCULPTOR_BROADENING_TEXP_SHIFT": "-0.1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(broadening_perturbation_from_env("CO2"), (1.4, -0.1))
            self.assertEqual(broadening_perturbation_from_env("H2O"), (1.1, -0.1))

    def test_unset_environment_is_a_no_op(self) -> None:
        operator = _StubPremodit()
        before = np.array(operator.pre_modit_info.ngamma_ref_grid, copy=True)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(apply_broadening_perturbation(operator, "H2O"))
        np.testing.assert_array_equal(operator.pre_modit_info.ngamma_ref_grid, before)

    def test_perturbation_scales_the_evaluated_line_width_exactly(self) -> None:
        """The grid edit must reproduce ExoJAX's own width evaluation exactly.

        ``unbiased_ngamma_grid`` is the function PreMODIT calls inside
        ``xsvector``/``xsmatrix``, so pinning it pins the cross section.
        """
        from exojax.opacity.premodit.premodit import unbiased_ngamma_grid

        scale, shift = 1.3, -0.15
        temperature, pressure, tref = 1200.0, 3.0, 296.0

        reference = _StubPremodit()
        base_widths = np.asarray(
            unbiased_ngamma_grid(
                temperature,
                pressure,
                reference.pre_modit_info.ngamma_ref_grid,
                reference.pre_modit_info.n_Texp_grid,
                reference.multi_index_uniqgrid,
                tref,
            )
        )

        operator = _StubPremodit()
        env = {
            "SCULPTOR_BROADENING_GAMMA_SCALE": str(scale),
            "SCULPTOR_BROADENING_TEXP_SHIFT": str(shift),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertTrue(apply_broadening_perturbation(operator, "H2O"))

        got = np.asarray(
            unbiased_ngamma_grid(
                temperature,
                pressure,
                operator.pre_modit_info.ngamma_ref_grid,
                operator.pre_modit_info.n_Texp_grid,
                operator.multi_index_uniqgrid,
                tref,
            )
        )
        n_per_line = reference.pre_modit_info.n_Texp_grid[reference.multi_index_uniqgrid[:, 1]]
        expected = base_widths * scale * (temperature / tref) ** (-shift)
        np.testing.assert_allclose(got, expected, rtol=1e-6)
        # sanity: the shift really does act as a temperature-exponent change
        np.testing.assert_allclose(
            got,
            scale
            * reference.pre_modit_info.ngamma_ref_grid[reference.multi_index_uniqgrid[:, 0]]
            * (temperature / tref) ** (-(n_per_line + shift))
            * pressure,
            rtol=1e-6,
        )
        # opainfo mirror stays in sync with pre_modit_info
        np.testing.assert_array_equal(operator.opainfo[2], operator.pre_modit_info.ngamma_ref_grid)
        np.testing.assert_array_equal(operator.opainfo[3], operator.pre_modit_info.n_Texp_grid)

    def test_non_positive_gamma_scale_is_rejected(self) -> None:
        operator = _StubPremodit()
        with mock.patch.dict(os.environ, {"SCULPTOR_BROADENING_GAMMA_SCALE": "0"}, clear=False):
            with self.assertRaises(ValueError):
                apply_broadening_perturbation(operator, "H2O")


class OpacityTests(unittest.TestCase):
    def test_co2_feature_is_present_near_4p3_microns(self) -> None:
        config = get_default_track_config("g395h")
        profile = build_temperature_profile([700, 850, 1000, 1200, 1400], config)
        chemistry = ChemistryModel(config).compute_state(profile, metallicity=0.0, carbon_to_oxygen=0.55, log_gravity=5.0, log_kzz=8.0)
        opacity = SyntheticOpacityBackend(config).compute_cross_sections(chemistry)
        total = opacity.species_cross_sections["CO2"].sum(axis=0)
        center_index = int(np.argmin(np.abs(config.nu_grid_cm1 - 2325.0)))
        edge_index = int(np.argmin(np.abs(config.nu_grid_cm1 - 2800.0)))
        self.assertGreater(total[center_index], total[edge_index])


if __name__ == "__main__":
    unittest.main()
