"""CPU-side precomputation of chemistry states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sculptor.forward_model.atmosphere import TemperatureProfile, build_temperature_profile
from sculptor.forward_model.chemistry import ChemistryModel, ChemistryState


@dataclass(frozen=True)
class PrecomputedAtmosphere:
    parameters: np.ndarray
    parameter_names: tuple[str, ...]
    profiles: list[TemperatureProfile]
    chemistry_states: list[ChemistryState]
    pressure_grid_bar: np.ndarray
    temperature_profiles_K: np.ndarray
    vmr_profiles: np.ndarray
    mean_molecular_weight_profiles: np.ndarray
    gravity_cgs: np.ndarray
    radius_rjup: np.ndarray
    species: tuple[str, ...]


class ChemistryPrecomputer:
    """Prepare temperature profiles and chemistry states before spectral synthesis."""

    def __init__(self, chemistry_model: ChemistryModel) -> None:
        self.chemistry_model = chemistry_model
        self.track_config = chemistry_model.track_config

    def precompute(self, parameters: np.ndarray, parameter_names: tuple[str, ...]) -> PrecomputedAtmosphere:
        profiles: list[TemperatureProfile] = []
        states: list[ChemistryState] = []
        gravity_values: list[float] = []
        radius_values: list[float] = []
        for row in parameters:
            parameter_map = {name: float(row[index]) for index, name in enumerate(parameter_names)}
            profile = build_temperature_profile(
                [parameter_map[name] for name in parameter_names if name.startswith("T")],
                self.track_config,
            )
            state = self.chemistry_model.compute_state(
                profile=profile,
                metallicity=parameter_map["metallicity"],
                carbon_to_oxygen=parameter_map["carbon_to_oxygen"],
                log_gravity=parameter_map["log_gravity"],
                log_kzz=parameter_map["log_kzz"],
            )
            profiles.append(profile)
            states.append(state)
            gravity_values.append(10.0 ** parameter_map["log_gravity"])
            radius_values.append(parameter_map["radius_rjup"])

        if states:
            pressure_grid_bar = np.asarray(states[0].pressures_bar, dtype=np.float32)
            species = states[0].species
            temperature_profiles = np.asarray([profile.temperatures_K for profile in profiles], dtype=np.float32)
            vmr_profiles = np.asarray([state.vmr for state in states], dtype=np.float32)
            mean_molecular_weight_profiles = np.asarray(
                [state.mean_molecular_weight for state in states],
                dtype=np.float32,
            )
        else:
            pressure_grid_bar = self.track_config.pressure_grid_bar.astype(np.float32)
            species = tuple(self.track_config.molecules + self.track_config.continuum_species + ("H2", "He"))
            temperature_profiles = np.empty((0, self.track_config.n_layers), dtype=np.float32)
            vmr_profiles = np.empty((0, self.track_config.n_layers, len(species)), dtype=np.float32)
            mean_molecular_weight_profiles = np.empty((0, self.track_config.n_layers), dtype=np.float32)

        return PrecomputedAtmosphere(
            parameters=np.asarray(parameters, dtype=np.float64),
            parameter_names=parameter_names,
            profiles=profiles,
            chemistry_states=states,
            pressure_grid_bar=pressure_grid_bar,
            temperature_profiles_K=temperature_profiles,
            vmr_profiles=vmr_profiles,
            mean_molecular_weight_profiles=mean_molecular_weight_profiles,
            gravity_cgs=np.asarray(gravity_values, dtype=np.float32),
            radius_rjup=np.asarray(radius_values, dtype=np.float32),
            species=species,
        )
