import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sculptor.data_generation import opacity_perturbation as op


def _settings(species=("H2O", "CO2"), n_modes=4):
    sigma_const = {s: 0.05 if s != "CO2" else 0.10 for s in species}
    sigma_modes = {
        s: tuple(0.04 * (0.8**m) for m in range(n_modes)) for s in species
    }
    return op.OpacityPerturbationSettings(
        enabled=True,
        n_modes=n_modes,
        species=tuple(species),
        sigma_const=sigma_const,
        sigma_modes=sigma_modes,
        priors_source="test",
    )


class CoefficientSamplingTests(unittest.TestCase):
    def test_deterministic_given_seed_and_salt(self) -> None:
        settings = _settings()
        a = op.sample_coefficients(settings, 32, seed=7, salt=3)
        b = op.sample_coefficients(settings, 32, seed=7, salt=3)
        np.testing.assert_array_equal(a, b)

    def test_salt_and_seed_change_the_stream(self) -> None:
        settings = _settings()
        base = op.sample_coefficients(settings, 32, seed=7, salt=3)
        self.assertFalse(np.array_equal(base, op.sample_coefficients(settings, 32, seed=8, salt=3)))
        self.assertFalse(np.array_equal(base, op.sample_coefficients(settings, 32, seed=7, salt=4)))

    def test_sigma_scaling(self) -> None:
        settings = _settings()
        draws = op.sample_coefficients(settings, 20000, seed=0, salt=0)
        sigmas = settings.coefficient_sigmas()
        std = np.std(draws, axis=0)
        np.testing.assert_allclose(std, sigmas, rtol=0.05)

    def test_disabled_settings_produce_no_columns(self) -> None:
        disabled = op.disabled_settings()
        self.assertEqual(op.sample_coefficients(disabled, 8, seed=0).shape, (8, 0))
        self.assertEqual(disabled.coefficient_names(), ())


class FourierFieldTests(unittest.TestCase):
    def test_basis_shape_and_constant_row(self) -> None:
        basis = op.fourier_basis(256, 4)
        self.assertEqual(basis.shape, (9, 256))
        np.testing.assert_array_equal(basis[0], np.ones(256, dtype=np.float32))

    def test_zero_coefficients_are_identity_on_synthetic_xs(self) -> None:
        # flag-off identity at the math level: exp(0 @ basis) == 1 exactly,
        # so multiplying a tiny synthetic xs matrix leaves it bit-identical.
        basis = op.fourier_basis(64, 3)
        coeffs = np.zeros((1 + 2 * 3,))
        field = np.exp(op.evaluate_log_field(coeffs, basis))
        xs = np.random.default_rng(0).uniform(1e-30, 1e-20, size=(5, 64))
        np.testing.assert_array_equal(xs * field[None, :], xs)

    def test_field_matches_manual_formula(self) -> None:
        n_modes, n_points = 2, 128
        basis = op.fourier_basis(n_points, n_modes)
        coeffs = np.array([0.02, 0.05, -0.03, 0.01, 0.04])
        log_field = op.evaluate_log_field(coeffs, basis)
        u = np.linspace(0.0, 1.0, n_points)
        expected = (
            coeffs[0]
            + coeffs[1] * np.cos(2 * np.pi * u)
            + coeffs[2] * np.sin(2 * np.pi * u)
            + coeffs[3] * np.cos(4 * np.pi * u)
            + coeffs[4] * np.sin(4 * np.pi * u)
        )
        np.testing.assert_allclose(log_field, expected, atol=1e-6)


class TableColumnTests(unittest.TestCase):
    def test_append_and_extract_roundtrip(self) -> None:
        settings = _settings(species=("H2O", "CO", "CO2"), n_modes=4)
        n_rows = 16
        physical = np.random.default_rng(1).uniform(size=(n_rows, 6))
        physical_names = tuple(f"p{i}" for i in range(6))
        coefficients = op.sample_coefficients(settings, n_rows, seed=5, salt=0)
        combo = np.zeros(n_rows)
        table, names = op.append_perturbation_columns(
            physical, physical_names, settings, coefficients, combo
        )
        self.assertEqual(table.shape, (n_rows, 6 + coefficients.shape[1] + 1))
        self.assertEqual(names[-1], op.COMBO_COLUMN)
        extracted = op.extract_coefficient_matrix(table, names, settings)
        self.assertEqual(extracted.shape, (n_rows, 3, settings.n_coeff_per_species))
        np.testing.assert_allclose(
            extracted.reshape(n_rows, -1), coefficients.astype(np.float32), rtol=1e-6
        )

    def test_extract_respects_species_subset_order(self) -> None:
        settings = _settings(species=("H2O", "CO", "CO2"), n_modes=2)
        n_rows = 4
        coefficients = op.sample_coefficients(settings, n_rows, seed=2, salt=0)
        table, names = op.append_perturbation_columns(
            np.zeros((n_rows, 1)), ("x",), settings, coefficients, np.zeros(n_rows)
        )
        subset = op.extract_coefficient_matrix(table, names, settings, species_subset=("CO2", "H2O"))
        full = op.extract_coefficient_matrix(table, names, settings)
        np.testing.assert_array_equal(subset[:, 0], full[:, 2])  # CO2
        np.testing.assert_array_equal(subset[:, 1], full[:, 0])  # H2O

    def test_missing_column_raises(self) -> None:
        settings = _settings(species=("H2O",), n_modes=2)
        with self.assertRaises(KeyError):
            op.extract_coefficient_matrix(np.zeros((2, 3)), ("a", "b", "c"), settings)

    def test_duplicate_columns_rejected(self) -> None:
        settings = _settings(species=("H2O",), n_modes=1)
        coefficients = op.sample_coefficients(settings, 2, seed=0)
        table, names = op.append_perturbation_columns(
            np.zeros((2, 1)), ("x",), settings, coefficients, np.zeros(2)
        )
        with self.assertRaises(ValueError):
            op.append_perturbation_columns(table, names, settings, coefficients, np.zeros(2))


class LinelistComboTests(unittest.TestCase):
    def test_combo_bit_mapping(self) -> None:
        alt = ("CH4", "H2O")  # sorted order defines the bits
        self.assertEqual(op.n_linelist_combos(alt), 4)
        self.assertEqual(op.combo_selection(0, alt), {"CH4": 0, "H2O": 0})
        self.assertEqual(op.combo_selection(1, alt), {"CH4": 1, "H2O": 0})
        self.assertEqual(op.combo_selection(2, alt), {"CH4": 0, "H2O": 1})
        self.assertEqual(op.combo_selection(3, alt), {"CH4": 1, "H2O": 1})
        with self.assertRaises(ValueError):
            op.combo_selection(4, alt)

    def test_shard_combo_column_matches_shard_combo(self) -> None:
        alt = ("CH4", "H2O")
        shard_size, n_rows = 8, 50
        column = op.shard_combo_column(n_rows, shard_size, alt)
        for row in range(n_rows):
            shard_index = row // shard_size
            self.assertEqual(int(column[row]), op.shard_combo(shard_index, alt))

    def test_no_alt_species_is_single_combo(self) -> None:
        self.assertEqual(op.n_linelist_combos(()), 1)
        self.assertTrue(np.all(op.shard_combo_column(20, 4, ()) == 0.0))


class ForceComboTests(unittest.TestCase):
    """SCULPTOR_FORCE_COMBO: the SMOOTH-ONLY half of the discrete/smooth ablation."""

    ALT = ("CH4", "H2O")

    def test_unset_is_production_behaviour(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(op.FORCE_COMBO_ENV, None)
            self.assertIsNone(op.forced_combo_from_env(self.ALT))
            self.assertEqual([op.shard_combo(s, self.ALT) for s in range(9)],
                             [0, 1, 2, 3, 0, 1, 2, 3, 0])
            np.testing.assert_array_equal(
                op.shard_combo_column(12, 2, self.ALT),
                np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1], dtype=np.float64),
            )

    def test_forced_combo_pins_every_shard(self) -> None:
        for forced in (0, 1, 2, 3):
            with mock.patch.dict(os.environ, {op.FORCE_COMBO_ENV: str(forced)}, clear=False):
                self.assertEqual(op.forced_combo_from_env(self.ALT), forced)
                self.assertEqual({op.shard_combo(s, self.ALT) for s in range(16)}, {forced})
                column = op.shard_combo_column(12, 2, self.ALT)
                np.testing.assert_array_equal(column, np.full(12, float(forced)))

    def test_forced_combo_matches_column_and_selection(self) -> None:
        with mock.patch.dict(os.environ, {op.FORCE_COMBO_ENV: "0"}, clear=False):
            column = op.shard_combo_column(20, 4, self.ALT)
            for row in range(20):
                combo = op.shard_combo(row // 4, self.ALT)
                self.assertEqual(int(column[row]), combo)
                self.assertEqual(op.combo_selection(combo, self.ALT), {"CH4": 0, "H2O": 0})

    def test_out_of_range_and_malformed_are_hard_errors(self) -> None:
        with mock.patch.dict(os.environ, {op.FORCE_COMBO_ENV: "4"}, clear=False):
            with self.assertRaises(ValueError):
                op.shard_combo(0, self.ALT)
        with mock.patch.dict(os.environ, {op.FORCE_COMBO_ENV: "-1"}, clear=False):
            with self.assertRaises(ValueError):
                op.shard_combo(0, self.ALT)
        with mock.patch.dict(os.environ, {op.FORCE_COMBO_ENV: "base"}, clear=False):
            with self.assertRaises(ValueError):
                op.shard_combo(0, self.ALT)
        # a combo valid for 4 variants is invalid when only one species has an alt
        with mock.patch.dict(os.environ, {op.FORCE_COMBO_ENV: "3"}, clear=False):
            with self.assertRaises(ValueError):
                op.shard_combo(0, ("CH4",))

    def test_blank_value_is_treated_as_unset(self) -> None:
        with mock.patch.dict(os.environ, {op.FORCE_COMBO_ENV: "  "}, clear=False):
            self.assertIsNone(op.forced_combo_from_env(self.ALT))
            self.assertEqual(op.shard_combo(5, self.ALT), 1)


class ZeroAmplitudeTests(unittest.TestCase):
    """Zero-sigma priors: the DISCRETE-ONLY half of the ablation.

    The continuous field must switch off EXACTLY (not merely be small), so the
    discrete-only dataset differs from an unperturbed forward model in nothing
    but the line-list operators it uses.
    """

    def _zero_settings(self, species=("H2O", "CO2"), n_modes=4):
        return op.OpacityPerturbationSettings(
            enabled=True,
            n_modes=n_modes,
            species=tuple(species),
            sigma_const={s: 0.0 for s in species},
            sigma_modes={s: tuple([0.0] * n_modes) for s in species},
            priors_source="zero",
        )

    def test_zero_priors_json_yields_exactly_zero_sigmas(self) -> None:
        priors = {
            "suggested_priors": {
                "H2O": {"sigma_const": 0.0, "sigma_modes": [0.0, 0.0, 0.0, 0.0]},
                "CO2": {"sigma_const": 0.0, "sigma_modes": [0.0, 0.0, 0.0, 0.0]},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zero.json"
            path.write_text(json.dumps(priors))
            env = {
                "SCULPTOR_OPACITY_PERTURB": "1",
                "SCULPTOR_OPACITY_PERTURB_MODES": "4",
                "SCULPTOR_OPACITY_PERTURB_PRIORS": str(path),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                settings = op.settings_from_env(("H2O", "CO2"))
        self.assertTrue(settings.enabled)
        np.testing.assert_array_equal(settings.coefficient_sigmas(),
                                      np.zeros(2 * (1 + 2 * 4)))

    def test_zero_sigma_draws_are_exactly_zero(self) -> None:
        settings = self._zero_settings()
        draws = op.sample_coefficients(settings, 1024, seed=413612, salt=0)
        # sigma == 0 makes every draw a signed zero (normal * 0.0 keeps the sign);
        # +0.0 and -0.0 both compare equal to zero and both exponentiate to
        # EXACTLY 1.0, which is what the identity below actually requires.
        self.assertTrue(np.all(draws == 0.0))
        self.assertTrue(np.all(np.exp(draws) == 1.0))
        self.assertEqual(float(np.abs(draws).max()), 0.0)

    def test_zero_field_is_bit_identity_on_cross_sections(self) -> None:
        settings = self._zero_settings(species=("H2O",), n_modes=4)
        basis = op.fourier_basis(512, settings.n_modes)
        coefficients = op.sample_coefficients(settings, 4, seed=1, salt=0)
        field = np.exp(op.evaluate_log_field(coefficients, basis))
        np.testing.assert_array_equal(field, np.ones_like(field))
        xs = np.random.default_rng(0).uniform(1e-30, 1e-18, size=(12, 512)).astype(np.float32)
        for row in range(len(coefficients)):
            perturbed = xs * field[row][None, :].astype(np.float32)
            np.testing.assert_array_equal(perturbed, xs)
            self.assertEqual(perturbed.tobytes(), xs.tobytes())

    def test_zero_field_is_bit_identity_under_jax_float32(self) -> None:
        """Same identity through the jnp.exp / matmul the generator actually runs."""
        try:
            import jax
            import jax.numpy as jnp
        except Exception:  # pragma: no cover - jax optional locally
            self.skipTest("jax not available")
        jax.config.update("jax_enable_x64", False)
        settings = self._zero_settings(species=("H2O", "CO2"), n_modes=4)
        basis = jnp.asarray(op.fourier_basis(256, settings.n_modes), dtype=jnp.float32)
        coefficients = jnp.asarray(
            op.sample_coefficients(settings, 3, seed=5, salt=0).reshape(3, 2, -1),
            dtype=jnp.float32,
        )
        xs = jnp.asarray(
            np.random.default_rng(3).uniform(1e-30, 1e-18, size=(8, 256)).astype(np.float32)
        )
        for row in range(3):
            fields = jnp.exp(coefficients[row] @ basis)
            np.testing.assert_array_equal(np.asarray(fields), np.ones((2, 256), dtype=np.float32))
            for species_index in range(2):
                perturbed = xs * fields[species_index][None, :]
                np.testing.assert_array_equal(np.asarray(perturbed), np.asarray(xs))


class SettingsTests(unittest.TestCase):
    def test_disabled_without_env(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SCULPTOR_OPACITY_PERTURB", None)
            settings = op.settings_from_env(("H2O", "CO2"))
        self.assertFalse(settings.enabled)

    def test_enabled_with_defaults(self) -> None:
        env = {"SCULPTOR_OPACITY_PERTURB": "1", "SCULPTOR_OPACITY_PERTURB_MODES": "6"}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("SCULPTOR_OPACITY_PERTURB_PRIORS", None)
            settings = op.settings_from_env(("H2O", "CO2"))
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.n_modes, 6)
        self.assertEqual(len(settings.sigma_modes["H2O"]), 6)  # decay-extended from 4
        # CO2 prior is the widest (no alternative cache; dominant floor window)
        self.assertGreater(settings.sigma_const["CO2"], settings.sigma_const["H2O"])
        self.assertEqual(len(settings.coefficient_names()), 2 * (1 + 2 * 6))

    def test_priors_json_override(self) -> None:
        priors = {
            "suggested_priors": {
                "H2O": {"sigma_const": 0.011, "sigma_modes": [0.021, 0.012]},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "priors.json"
            path.write_text(json.dumps(priors))
            env = {
                "SCULPTOR_OPACITY_PERTURB": "1",
                "SCULPTOR_OPACITY_PERTURB_MODES": "2",
                "SCULPTOR_OPACITY_PERTURB_PRIORS": str(path),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                settings = op.settings_from_env(("H2O", "CO2"))
        self.assertAlmostEqual(settings.sigma_const["H2O"], 0.011)
        self.assertEqual(settings.sigma_modes["H2O"], (0.021, 0.012))
        # CO2 not in the JSON -> builtin default
        self.assertAlmostEqual(settings.sigma_const["CO2"], op.DEFAULT_PRIOR_TABLE["CO2"]["sigma_const"])

    def test_alt_species_from_env(self) -> None:
        env = {"SCULPTOR_OPA_ALT_CACHE_DIR_H2O": "/tmp/x", "SCULPTOR_OPA_ALT_CACHE_DIR_CH4": "/tmp/y"}
        with mock.patch.dict(os.environ, env, clear=False):
            alt = op.alt_species_from_env(("H2O", "CO", "CH4", "CO2"))
        self.assertEqual(alt, ("CH4", "H2O"))  # sorted


class GeneratorFlagOffTests(unittest.TestCase):
    def test_generator_default_is_disabled_and_scalar_path_unchanged(self) -> None:
        # Mirrors tests/test_data_generation.py: with the flag off, the
        # generator must construct with perturbation disabled and the eager
        # scalar fallback must behave exactly as before.
        from sculptor.config import get_default_track_config
        from sculptor.data_generation.chemistry_precompute import PrecomputedAtmosphere
        from sculptor.data_generation.tpu_generator import SpectrumBatchGenerator
        from sculptor.forward_model.chemistry import ChemistryState

        os.environ.pop("SCULPTOR_OPACITY_PERTURB", None)

        class _FakeSpectrumModel:
            def __init__(self) -> None:
                self.track_config = SimpleNamespace(
                    nu_grid_cm1=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    tpu_batch_size=1,
                )

            def generate_from_state(self, row, profile, chemistry_state):
                return SimpleNamespace(
                    flux_nu=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    chemistry_state=chemistry_state,
                )

        pressure_grid = np.array([1.0e-3, 1.0e-2], dtype=np.float32)
        temperatures = np.array([500.0, 600.0], dtype=np.float32)
        chemistry_state = ChemistryState(
            pressures_bar=pressure_grid,
            temperatures_K=temperatures,
            species=("H2O", "CO"),
            vmr=np.ones((2, 2), dtype=np.float32),
            mean_molecular_weight=np.array([2.3, 2.3], dtype=np.float32),
            metadata={},
        )
        batch = PrecomputedAtmosphere(
            parameters=np.ones((1, 11), dtype=np.float32),
            parameter_names=get_default_track_config().parameter_names,
            profiles=[SimpleNamespace()],
            chemistry_states=[chemistry_state],
            pressure_grid_bar=pressure_grid,
            temperature_profiles_K=temperatures[None, :],
            vmr_profiles=np.ones((1, 2, 2), dtype=np.float32),
            mean_molecular_weight_profiles=np.array([[2.3, 2.3]], dtype=np.float32),
            gravity_cgs=np.array([1.0e5], dtype=np.float32),
            radius_rjup=np.array([1.0], dtype=np.float32),
            species=chemistry_state.species,
        )
        generator = SpectrumBatchGenerator(_FakeSpectrumModel())
        self.assertFalse(generator.perturbation.enabled)
        generator._get_batch_function = lambda unused: (_ for _ in ()).throw(ImportError("force scalar"))
        result = generator.generate(batch)
        self.assertEqual(result.spectra_clean.shape, (1, 3))

    def test_generator_perturbation_blocks_silent_scalar_fallback(self) -> None:
        from sculptor.config import get_default_track_config
        from sculptor.data_generation.chemistry_precompute import PrecomputedAtmosphere
        from sculptor.data_generation.tpu_generator import SpectrumBatchGenerator
        from sculptor.forward_model.chemistry import ChemistryState

        settings = _settings(species=("H2O",), n_modes=2)

        class _FakeSpectrumModel:
            def __init__(self) -> None:
                self.track_config = SimpleNamespace(
                    nu_grid_cm1=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    tpu_batch_size=1,
                    molecules=("H2O",),
                )

        pressure_grid = np.array([1.0e-3], dtype=np.float32)
        chemistry_state = ChemistryState(
            pressures_bar=pressure_grid,
            temperatures_K=np.array([500.0], dtype=np.float32),
            species=("H2O",),
            vmr=np.ones((1, 1), dtype=np.float32),
            mean_molecular_weight=np.array([2.3], dtype=np.float32),
            metadata={},
        )
        batch = PrecomputedAtmosphere(
            parameters=np.ones((1, 11), dtype=np.float32),
            parameter_names=get_default_track_config().parameter_names[:11],
            profiles=[SimpleNamespace()],
            chemistry_states=[chemistry_state],
            pressure_grid_bar=pressure_grid,
            temperature_profiles_K=np.array([[500.0]], dtype=np.float32),
            vmr_profiles=np.ones((1, 1, 1), dtype=np.float32),
            mean_molecular_weight_profiles=np.array([[2.3]], dtype=np.float32),
            gravity_cgs=np.array([1.0e5], dtype=np.float32),
            radius_rjup=np.array([1.0], dtype=np.float32),
            species=chemistry_state.species,
        )
        generator = SpectrumBatchGenerator(_FakeSpectrumModel(), perturbation=settings)
        self.assertTrue(generator.perturbation.enabled)
        generator._get_batch_function = lambda unused: (_ for _ in ()).throw(ImportError("no jax"))
        with self.assertRaises(ImportError):
            generator.generate(batch)


if __name__ == "__main__":
    unittest.main()
