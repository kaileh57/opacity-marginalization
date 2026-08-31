import unittest
from types import SimpleNamespace

import numpy as np

from sculptor.config import get_default_track_config
from sculptor.data_generation.chemistry_precompute import PrecomputedAtmosphere
from sculptor.data_generation.tpu_generator import SpectrumBatchGenerator
from sculptor.forward_model.chemistry import ChemistryState


class _FakeSpectrumModel:
    def __init__(self) -> None:
        self.track_config = SimpleNamespace(
            nu_grid_cm1=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            tpu_batch_size=1,
        )

    def generate(self, row):  # pragma: no cover - this should not be called
        raise AssertionError("generate() should not be called when precomputed states are available.")

    def generate_from_state(self, row, profile, chemistry_state):
        return SimpleNamespace(
            flux_nu=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            chemistry_state=chemistry_state,
        )


class DataGenerationTests(unittest.TestCase):
    def test_batch_generator_uses_precomputed_states(self) -> None:
        pressure_grid = np.array([1.0e-3, 1.0e-2], dtype=np.float32)
        temperatures = np.array([500.0, 600.0], dtype=np.float32)
        vmr = np.ones((2, 2), dtype=np.float32)
        mean_molecular_weight = np.array([2.3, 2.3], dtype=np.float32)
        chemistry_state = ChemistryState(
            pressures_bar=pressure_grid,
            temperatures_K=temperatures,
            species=("H2O", "CO"),
            vmr=vmr,
            mean_molecular_weight=mean_molecular_weight,
            metadata={},
        )
        batch = PrecomputedAtmosphere(
            parameters=np.ones((1, 11), dtype=np.float32),
            parameter_names=get_default_track_config().parameter_names,
            profiles=[SimpleNamespace()],
            chemistry_states=[chemistry_state],
            pressure_grid_bar=pressure_grid,
            temperature_profiles_K=temperatures[None, :],
            vmr_profiles=vmr[None, :, :],
            mean_molecular_weight_profiles=mean_molecular_weight[None, :],
            gravity_cgs=np.array([1.0e5], dtype=np.float32),
            radius_rjup=np.array([1.0], dtype=np.float32),
            species=chemistry_state.species,
        )
        generator = SpectrumBatchGenerator(_FakeSpectrumModel())
        generator._get_batch_function = lambda unused_batch: (_ for _ in ()).throw(ImportError("force scalar path"))
        result = generator.generate(batch)
        self.assertEqual(result.spectra_clean.shape, (1, 3))
        np.testing.assert_array_equal(result.vmr_profiles[0], chemistry_state.vmr)
