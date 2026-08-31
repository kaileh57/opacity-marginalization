"""Chemistry utilities: FastChem interpolation and quench approximation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sculptor.config import TrackConfig
from sculptor.forward_model.atmosphere import TemperatureProfile

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None


BOLTZMANN_CGS = 1.380649e-16
PROTON_MASS_G = 1.67262192369e-24

MOLECULAR_WEIGHTS = {
    "H2": 2.01588,
    "He": 4.0026,
    "H2O": 18.01528,
    "CO": 28.0101,
    "CH4": 16.0425,
    "NH3": 17.03052,
    "CO2": 44.0095,
    "H2S": 34.0809,
}


@dataclass(frozen=True)
class FastChemGrid:
    """A structured chemistry grid stored on regular axes."""

    axes: dict[str, np.ndarray]
    species: tuple[str, ...]
    values: np.ndarray

    @staticmethod
    def from_hdf5(path: str | Path) -> "FastChemGrid":
        if h5py is None:
            raise ImportError("h5py is required to load FastChem grids from HDF5.")
        axis_aliases = {
            "temperature": ("temperature", "temperatures", "T"),
            "pressure": ("pressure", "pressures", "P"),
            "metallicity": ("metallicity", "mh", "[M/H]"),
            "carbon_to_oxygen": ("carbon_to_oxygen", "c_o", "co_ratio", "C/O"),
        }
        with h5py.File(path, "r") as handle:
            axes: dict[str, np.ndarray] = {}
            for target_key, aliases in axis_aliases.items():
                for alias in aliases:
                    if alias in handle:
                        axes[target_key] = np.asarray(handle[alias], dtype=np.float64)
                        break
                    if f"grid/{alias}" in handle:
                        axes[target_key] = np.asarray(handle[f"grid/{alias}"], dtype=np.float64)
                        break
                if target_key not in axes:
                    raise KeyError(f"Missing FastChem axis for {target_key}")

            if "vmr" in handle:
                if isinstance(handle["vmr"], h5py.Dataset):
                    species = tuple(
                        s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in handle["species"][...]
                    )
                    values = np.asarray(handle["vmr"], dtype=np.float64)
                elif isinstance(handle["vmr"], h5py.Group):
                    species = tuple(handle["vmr"].keys())
                    stacked = [np.asarray(handle["vmr"][name], dtype=np.float64) for name in species]
                    values = np.stack(stacked, axis=-1)
                else:
                    raise KeyError("Unsupported FastChem HDF5 layout for vmr.")
            elif "species" in handle and isinstance(handle["species"], h5py.Group):
                species = tuple(handle["species"].keys())
                stacked = [np.asarray(handle["species"][name], dtype=np.float64) for name in species]
                values = np.stack(stacked, axis=-1)
            else:
                raise KeyError("Unsupported FastChem HDF5 layout.")

        expected_shape = (
            len(axes["temperature"]),
            len(axes["pressure"]),
            len(axes["metallicity"]),
            len(axes["carbon_to_oxygen"]),
        )
        if values.shape[:4] != expected_shape:
            raise ValueError(
                "FastChem grid shape does not match axes order. "
                f"Expected {expected_shape} for (T, P, metallicity, C/O), got {values.shape[:4]}"
            )
        return FastChemGrid(axes=axes, species=species, values=values)


@dataclass(frozen=True)
class ChemistryState:
    """Atmospheric composition on the pressure grid.

    The `vmr` matrix stores true VMRs for molecular species plus absorber weights
    for CIA channels such as `H2-H2` and `H2-He`.
    """

    pressures_bar: np.ndarray
    temperatures_K: np.ndarray
    species: tuple[str, ...]
    vmr: np.ndarray
    mean_molecular_weight: np.ndarray
    metadata: dict[str, Any]

    def vmr_for(self, species_name: str) -> np.ndarray:
        if species_name not in self.species:
            return np.zeros_like(self.pressures_bar)
        return self.vmr[:, self.species.index(species_name)]


def _multilinear_interp(point: tuple[float, ...], axes: tuple[np.ndarray, ...], values: np.ndarray) -> np.ndarray:
    """N-dimensional multilinear interpolation returning the trailing vector axis."""

    indices: list[int] = []
    fractions: list[float] = []
    for coordinate, axis in zip(point, axes):
        if coordinate <= axis[0]:
            indices.append(0)
            fractions.append(0.0)
            continue
        if coordinate >= axis[-1]:
            indices.append(len(axis) - 2)
            fractions.append(1.0)
            continue
        upper = int(np.searchsorted(axis, coordinate, side="right"))
        lower = upper - 1
        frac = (coordinate - axis[lower]) / (axis[upper] - axis[lower])
        indices.append(lower)
        fractions.append(float(frac))

    corners = np.array(np.meshgrid(*[[0, 1]] * len(axes))).T.reshape(-1, len(axes))
    result = np.zeros(values.shape[-1], dtype=np.float64)
    for corner in corners:
        weight = 1.0
        slices = []
        for dim, bit in enumerate(corner):
            weight *= fractions[dim] if bit else 1.0 - fractions[dim]
            slices.append(indices[dim] + bit)
        result += weight * values[tuple(slices)]
    return result


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _heuristic_equilibrium_abundances(
    temperatures_K: np.ndarray,
    pressures_bar: np.ndarray,
    metallicity: float,
    carbon_to_oxygen: float,
    molecules: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Fallback equilibrium chemistry when a FastChem grid is unavailable."""

    t = np.asarray(temperatures_K, dtype=np.float64)
    p = np.asarray(pressures_bar, dtype=np.float64)
    metal_scale = 10.0 ** metallicity
    co_ratio = carbon_to_oxygen
    cool = _sigmoid((1300.0 - t) / 140.0)
    hot = 1.0 - cool
    very_cool = _sigmoid((950.0 - t) / 120.0)
    oxidizing = np.clip((1.0 / np.maximum(co_ratio, 1.0e-3)) ** 0.7, 0.3, 3.0)
    reducing = np.clip(co_ratio**0.8, 0.3, 3.0)
    pressure_boost = np.clip((p / 1.0e-2) ** 0.05, 0.7, 1.6)

    abundances: dict[str, np.ndarray] = {
        "H2O": 2.5e-4 * metal_scale * oxidizing * (0.4 + 0.6 * cool) * pressure_boost,
        "CO": 1.2e-4 * metal_scale * reducing * (0.25 + 0.75 * hot) * pressure_boost,
        "CH4": 1.0e-4 * metal_scale * reducing * (0.1 + 0.9 * cool) * pressure_boost,
        "NH3": 8.0e-5 * metal_scale * (0.1 + 0.9 * very_cool) * pressure_boost,
        "CO2": 2.5e-6 * metal_scale**1.2 * oxidizing * (0.3 + 0.7 * cool) * pressure_boost,
        "H2S": 6.0e-6 * metal_scale * (0.8 + 0.2 * cool) * pressure_boost,
        "H2": np.full_like(t, 0.84),
        "He": np.full_like(t, 0.15),
    }

    selected = {name: np.clip(abundances.get(name, np.full_like(t, 1.0e-12)), 1.0e-12, 0.95) for name in molecules}
    heavy_total = np.sum(np.stack([selected[name] for name in selected if name not in {"H2", "He"}], axis=0), axis=0)
    selected["H2"] = np.clip(0.84 - 0.5 * heavy_total, 0.2, 0.95)
    selected["He"] = np.clip(0.15 - 0.1 * heavy_total, 0.03, 0.3)
    return selected


def co_quench_timescale_seconds(
    temperatures_K: np.ndarray,
    pressures_bar: np.ndarray,
    metallicity: float,
) -> np.ndarray:
    """Zahnle & Marley (2014) Eqs. 12-14 approximation."""

    p = np.maximum(np.asarray(pressures_bar, dtype=np.float64), 1.0e-12)
    t = np.maximum(np.asarray(temperatures_K, dtype=np.float64), 1.0)
    metal_factor = np.maximum(10.0**metallicity, 1.0e-6)
    tq1 = 1.5e-6 * p**-1.0 * metal_factor**-0.7 * np.exp(42000.0 / t)
    tq2 = 40.0 * p**-2.0 * np.exp(25000.0 / t)
    return 1.0 / (1.0 / tq1 + 1.0 / tq2)


def nh3_quench_timescale_seconds(
    temperatures_K: np.ndarray,
    pressures_bar: np.ndarray,
) -> np.ndarray:
    """Zahnle & Marley (2014) Eq. 32 approximation."""

    p = np.maximum(np.asarray(pressures_bar, dtype=np.float64), 1.0e-12)
    t = np.maximum(np.asarray(temperatures_K, dtype=np.float64), 1.0)
    return 1.0e-7 * p**-1.0 * np.exp(52000.0 / t)


def _compute_scale_height_cm(
    temperatures_K: np.ndarray,
    mean_molecular_weight: np.ndarray,
    gravity_cgs: float,
) -> np.ndarray:
    return (BOLTZMANN_CGS * temperatures_K) / (np.maximum(mean_molecular_weight, 1.0) * PROTON_MASS_G * gravity_cgs)


def _mixing_timescale_seconds(
    temperatures_K: np.ndarray,
    mean_molecular_weight: np.ndarray,
    gravity_cgs: float,
    log_kzz: float,
) -> np.ndarray:
    scale_height_cm = _compute_scale_height_cm(temperatures_K, mean_molecular_weight, gravity_cgs)
    kzz = 10.0**log_kzz
    return scale_height_cm**2 / kzz


def _find_quench_index(t_chem: np.ndarray, t_mix: np.ndarray) -> int:
    ratio = np.maximum(t_chem / np.maximum(t_mix, 1.0e-30), 1.0e-30)
    return int(np.argmin(np.abs(np.log10(ratio))))


def _freeze_above(profile: np.ndarray, index: int) -> np.ndarray:
    frozen = np.array(profile, copy=True)
    frozen[:index] = profile[index]
    return frozen


def compute_mean_molecular_weight(
    abundances: dict[str, np.ndarray],
    track_config: TrackConfig,
) -> np.ndarray:
    n_layers = len(next(iter(abundances.values())))
    total = np.zeros(n_layers, dtype=np.float64)
    weight = np.zeros(n_layers, dtype=np.float64)
    for species, values in abundances.items():
        vmr = np.asarray(values, dtype=np.float64)
        total += vmr
        weight += vmr * MOLECULAR_WEIGHTS.get(species, 2.3)
    total = np.where(total <= 0.0, 1.0, total)
    return weight / total


class ChemistryModel:
    """Interpolates equilibrium chemistry and applies quench approximations."""

    def __init__(self, track_config: TrackConfig, fastchem_grid: FastChemGrid | None = None) -> None:
        self.track_config = track_config
        self.fastchem_grid = fastchem_grid

    @classmethod
    def from_fastchem_hdf5(cls, path: str | Path, track_config: TrackConfig) -> "ChemistryModel":
        return cls(track_config=track_config, fastchem_grid=FastChemGrid.from_hdf5(path))

    def equilibrium_abundances(
        self,
        profile: TemperatureProfile,
        metallicity: float,
        carbon_to_oxygen: float,
    ) -> dict[str, np.ndarray]:
        if self.fastchem_grid is None:
            return _heuristic_equilibrium_abundances(
                temperatures_K=profile.temperatures_K,
                pressures_bar=profile.pressures_bar,
                metallicity=metallicity,
                carbon_to_oxygen=carbon_to_oxygen,
                molecules=self.track_config.molecules + ("H2", "He"),
            )

        axes = (
            self.fastchem_grid.axes["temperature"],
            self.fastchem_grid.axes["pressure"],
            self.fastchem_grid.axes["metallicity"],
            self.fastchem_grid.axes["carbon_to_oxygen"],
        )
        interpolated = []
        for temperature, pressure in zip(profile.temperatures_K, profile.pressures_bar):
            interpolated.append(
                _multilinear_interp(
                    (temperature, pressure, metallicity, carbon_to_oxygen),
                    axes,
                    self.fastchem_grid.values,
                )
            )
        values = np.asarray(interpolated, dtype=np.float64)
        abundances = {
            species: np.clip(values[:, index], 1.0e-12, 0.95)
            for index, species in enumerate(self.fastchem_grid.species)
        }
        if "H2" not in abundances:
            abundances["H2"] = np.full_like(profile.pressures_bar, 0.84)
        if "He" not in abundances:
            abundances["He"] = np.full_like(profile.pressures_bar, 0.15)
        return abundances

    def apply_quench(
        self,
        abundances: dict[str, np.ndarray],
        profile: TemperatureProfile,
        log_gravity: float,
        log_kzz: float,
        metallicity: float,
    ) -> dict[str, np.ndarray]:
        mean_molecular_weight = compute_mean_molecular_weight(abundances, self.track_config)
        gravity_cgs = 10.0**log_gravity
        t_mix = _mixing_timescale_seconds(
            temperatures_K=profile.temperatures_K,
            mean_molecular_weight=mean_molecular_weight,
            gravity_cgs=gravity_cgs,
            log_kzz=log_kzz,
        )

        quenched = {name: np.array(values, copy=True) for name, values in abundances.items()}
        co_index = _find_quench_index(
            co_quench_timescale_seconds(profile.temperatures_K, profile.pressures_bar, metallicity),
            t_mix,
        )
        nh3_index = _find_quench_index(
            nh3_quench_timescale_seconds(profile.temperatures_K, profile.pressures_bar),
            t_mix,
        )

        for species in ("CO", "CH4", "CO2"):
            if species in quenched:
                quenched[species] = _freeze_above(quenched[species], co_index)
        if "NH3" in quenched:
            quenched["NH3"] = _freeze_above(quenched["NH3"], nh3_index)
        return quenched

    def compute_state(
        self,
        profile: TemperatureProfile,
        metallicity: float,
        carbon_to_oxygen: float,
        log_gravity: float,
        log_kzz: float,
    ) -> ChemistryState:
        equilibrium = self.equilibrium_abundances(profile, metallicity, carbon_to_oxygen)
        quenched = self.apply_quench(equilibrium, profile, log_gravity, log_kzz, metallicity)
        species = tuple(self.track_config.molecules + self.track_config.continuum_species + ("H2", "He"))

        h2_profile = np.asarray(quenched.get("H2", np.full_like(profile.pressures_bar, 1.0e-12)), dtype=np.float64)
        he_profile = np.asarray(quenched.get("He", np.full_like(profile.pressures_bar, 1.0e-12)), dtype=np.float64)

        def _profile_for(species_name: str) -> np.ndarray:
            if species_name == "H2-H2":
                return h2_profile * h2_profile
            if species_name == "H2-He":
                return h2_profile * he_profile
            return np.asarray(quenched.get(species_name, np.full_like(profile.pressures_bar, 1.0e-12)), dtype=np.float64)

        vmr = np.stack(
            [_profile_for(name) for name in species],
            axis=1,
        )
        mean_molecular_weight = compute_mean_molecular_weight(quenched, self.track_config)
        return ChemistryState(
            pressures_bar=profile.pressures_bar,
            temperatures_K=profile.temperatures_K,
            species=species,
            vmr=vmr,
            mean_molecular_weight=mean_molecular_weight,
            metadata={
                "metallicity": metallicity,
                "carbon_to_oxygen": carbon_to_oxygen,
                "log_gravity": log_gravity,
                "log_kzz": log_kzz,
            },
        )
