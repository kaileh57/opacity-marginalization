"""Configuration primitives for Sculptor."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ParameterPrior:
    """Uniform prior bounds for a scalar parameter."""

    name: str
    lower: float
    upper: float
    is_conditioning: bool = False

    def denormalize(self, value: float) -> float:
        return self.lower + (self.upper - self.lower) * float(value)

    def normalize(self, value: float) -> float:
        return (float(value) - self.lower) / (self.upper - self.lower)


@dataclass(frozen=True)
class OpacityDatabaseConfig:
    """Metadata for a single opacity database."""

    name: str
    database_class: str
    relative_path: str
    database_name: str
    crit: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class NoiseConfig:
    """Training noise and domain-randomization configuration."""

    snr_min: float
    snr_max: float
    flux_calibration_sigma_max: float = 0.05
    resolving_power_jitter_fraction: float = 0.05
    max_pixel_shift: float = 1.0
    baseline_tilt_max: float = 0.03


@dataclass(frozen=True)
class TrackConfig:
    """End-to-end configuration for a specific observing track."""

    name: str
    wavelength_label: str
    nu_min_cm1: float
    nu_max_cm1: float
    n_wavenumber: int
    n_layers: int
    pressure_top_bar: float
    pressure_bottom_bar: float
    knot_log10_pressures_bar: tuple[float, ...]
    instrument_resolving_power: float
    default_distance_pc: float
    molecules: tuple[str, ...]
    continuum_species: tuple[str, ...]
    opacity_databases: tuple[OpacityDatabaseConfig, ...]
    parameter_priors: tuple[ParameterPrior, ...]
    noise: NoiseConfig
    track_root: str
    fastchem_path: str | None = None
    tpu_batch_size: int = 16
    cloud_model: str = "off"  # "off" | "ackerman_marley"; gray backend on CPU, mie on pod
    cloud_condensates: tuple[str, ...] = ("MgSiO3", "Na2S")
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pressure_grid_bar(self) -> np.ndarray:
        return np.logspace(
            np.log10(self.pressure_top_bar),
            np.log10(self.pressure_bottom_bar),
            self.n_layers,
            dtype=np.float64,
        )

    @property
    def nu_grid_cm1(self) -> np.ndarray:
        log_min = np.log(self.nu_min_cm1)
        log_max = np.log(self.nu_max_cm1)
        return np.exp(np.linspace(log_min, log_max, self.n_wavenumber, dtype=np.float64))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(prior.name for prior in self.parameter_priors)

    @property
    def retrieval_parameter_names(self) -> tuple[str, ...]:
        return tuple(prior.name for prior in self.parameter_priors if not prior.is_conditioning)

    @property
    def conditioning_parameter_names(self) -> tuple[str, ...]:
        return tuple(prior.name for prior in self.parameter_priors if prior.is_conditioning)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def package_root() -> Path:
    return Path(__file__).resolve().parent


def config_root() -> Path:
    return project_root() / "configs"


def _g395h_h2o_database() -> "OpacityDatabaseConfig":
    """H2O line list for G395H, selectable via SCULPTOR_H2O_LINELIST.

    Default 'hitemp' = HITEMP 2010 (the v3/v4 production cache). 'pokazatel' = ExoMol
    1H2-16O POKAZATEL, the candidate fix for the 4.35-4.80 um NRS2 over-absorption
    (postfix_floor.json: model OVER-absorbs there; the lever is a line list with
    corrected hot-H2O line strengths). POKAZATEL files are staged at
    <EXOJAX_ROOT>/g395h/H2O_POKAZATEL/1H2-16O/POKAZATEL.
    """
    choice = os.environ.get("SCULPTOR_H2O_LINELIST", "hitemp").strip().lower()
    if choice == "pokazatel":
        return OpacityDatabaseConfig(
            "H2O", "MdbExomol", "g395h/H2O_POKAZATEL/1H2-16O/POKAZATEL",
            "ExoMol/POKAZATEL", crit=1.0e-30,
        )
    if choice not in ("hitemp", ""):
        raise ValueError(f"Unknown SCULPTOR_H2O_LINELIST={choice!r} (expected 'hitemp' or 'pokazatel')")
    return OpacityDatabaseConfig("H2O", "MdbHitemp", "g395h/H2O_HITEMP/H2O", "HITEMP 2010", crit=1.0e-30)


def _g395h_ch4_database() -> "OpacityDatabaseConfig":
    """CH4 line list for G395H, selectable via SCULPTOR_CH4_LINELIST.

    Default 'hitemp' = HITEMP 2020 (the v3-v6 production cache). 'exomol_mm' = ExoMol
    12C-1H4 MM (Yurchenko et al. 2023, refined PES + ab initio DMS, 50.4 Gline, valid
    to 2000 K), the R1-identified lever for the 4.35-4.80 um red-carbon over-absorption
    (R1: CH4 abundance is immovable while the floor stays large => a line-SHAPE/strength
    error a uniform abundance scale cannot fix). MM files staged at
    <EXOJAX_ROOT>/g395h/CH4_EXOMOL/12C-1H4/MM.
    """
    choice = os.environ.get("SCULPTOR_CH4_LINELIST", "hitemp").strip().lower()
    if choice in ("exomol_mm", "mm", "exomol"):
        return OpacityDatabaseConfig(
            "CH4", "MdbExomol", "g395h/CH4_EXOMOL/12C-1H4/MM", "ExoMol/MM", crit=1.0e-30,
        )
    if choice not in ("hitemp", ""):
        raise ValueError(f"Unknown SCULPTOR_CH4_LINELIST={choice!r} (expected 'hitemp' or 'exomol_mm')")
    return OpacityDatabaseConfig("CH4", "MdbHitemp", "g395h/CH4_HITEMP/CH4", "HITEMP 2020", crit=1.0e-30)


def _g395h_config() -> TrackConfig:
    return TrackConfig(
        name="g395h",
        wavelength_label="JWST NIRSpec G395H",
        nu_min_cm1=1850.0,
        nu_max_cm1=3550.0,
        n_wavenumber=10_000,
        n_layers=100,
        pressure_top_bar=1.0e-6,
        pressure_bottom_bar=1.0e3,
        knot_log10_pressures_bar=(-5.0, -3.0, -1.0, 1.0, 3.0),
        instrument_resolving_power=2700.0,
        default_distance_pc=10.0,
        molecules=("H2O", "CO", "CH4", "NH3", "CO2", "H2S"),
        continuum_species=("H2-H2", "H2-He"),
        opacity_databases=(
            OpacityDatabaseConfig("CO", "MdbExomol", "g395h/CO/12C-16O/Li2015", "ExoMol/Li2015"),
            OpacityDatabaseConfig("NH3", "MdbExomol", "g395h/NH3/14N-1H3/BYTe", "ExoMol/BYTe"),
            _g395h_ch4_database(),
            _g395h_h2o_database(),
            OpacityDatabaseConfig("CO2", "MdbHitemp", "g395h/CO2_HITEMP/CO2", "HITEMP 2024", crit=1.0e-30),
            OpacityDatabaseConfig("H2S", "MdbExomol", "g395h/H2S/1H2-32S/AYT2", "ExoMol/AYT2"),
            OpacityDatabaseConfig("H2-H2", "CdbCIA", "CIA/H2-H2", "HITRAN CIA"),
            OpacityDatabaseConfig("H2-He", "CdbCIA", "CIA/H2-He", "HITRAN CIA"),
        ),
        parameter_priors=(
            ParameterPrior("T1", 150.0, 1500.0),
            ParameterPrior("T2", 200.0, 2000.0),
            ParameterPrior("T3", 300.0, 2500.0),
            ParameterPrior("T4", 500.0, 3000.0),
            ParameterPrior("T5", 800.0, 3500.0),
            ParameterPrior("metallicity", -1.0, 1.5),
            ParameterPrior("carbon_to_oxygen", 0.1, 1.5),
            ParameterPrior("log_kzz", 6.0, 10.0),
            ParameterPrior("log_gravity", 3.0, 5.5),
            ParameterPrior("radius_rjup", 0.5, 2.0),
            ParameterPrior("f_sed", 0.5, 8.0),
            ParameterPrior("sigma_c", 1.2, 2.5),
            ParameterPrior("rv_km_s", -100.0, 100.0),
            ParameterPrior("vsini_km_s", 0.0, 30.0),
            ParameterPrior("sigma_cal", 0.0, 0.05),
            ParameterPrior("snr", 10.0, 200.0, is_conditioning=True),
        ),
        noise=NoiseConfig(snr_min=10.0, snr_max=200.0),
        track_root="g395h",
        fastchem_path="/tmp/exojax_db/fastchem_grid.h5",
        tpu_batch_size=16,
        cloud_model=os.environ.get("SCULPTOR_CLOUD_MODEL", "off").strip().lower(),
        metadata={
            "priority": "primary",
            "instrument_convolution_mode": "variable_resolution",
            "instrument_resolution_curve_path": "tmp_g395h_disp_reference.fits",
            "instrument_resolution_curve_hdu": 1,
            "instrument_resolution_wavelength_column": "WAVELENGTH",
            "instrument_resolution_value_column": "R",
        },
    )


def _hband_config() -> TrackConfig:
    return TrackConfig(
        name="hband",
        wavelength_label="SpeX Prism H-band",
        nu_min_cm1=5500.0,
        nu_max_cm1=7200.0,
        n_wavenumber=4096,
        n_layers=100,
        pressure_top_bar=1.0e-6,
        pressure_bottom_bar=1.0e3,
        knot_log10_pressures_bar=(-5.0, -3.0, -1.0, 1.0, 3.0),
        instrument_resolving_power=100.0,
        default_distance_pc=10.0,
        molecules=("H2O", "CH4", "NH3"),
        continuum_species=("H2-H2", "H2-He"),
        opacity_databases=(
            OpacityDatabaseConfig("H2O", "MdbHitemp", "hband/H2O_HITEMP/H2O", "HITEMP", crit=1.0e-30),
            OpacityDatabaseConfig("CH4", "MdbHitemp", "hband/CH4_HITEMP/CH4", "HITEMP", crit=1.0e-30),
            OpacityDatabaseConfig("NH3", "MdbExomol", "hband/NH3/14N-1H3/BYTe", "ExoMol/BYTe"),
            OpacityDatabaseConfig("H2-H2", "CdbCIA", "CIA/H2-H2", "HITRAN CIA"),
            OpacityDatabaseConfig("H2-He", "CdbCIA", "CIA/H2-He", "HITRAN CIA"),
        ),
        parameter_priors=(
            ParameterPrior("T1", 150.0, 1500.0),
            ParameterPrior("T2", 200.0, 2000.0),
            ParameterPrior("T3", 300.0, 2500.0),
            ParameterPrior("T4", 500.0, 3000.0),
            ParameterPrior("T5", 800.0, 3500.0),
            ParameterPrior("metallicity", -1.0, 1.5),
            ParameterPrior("carbon_to_oxygen", 0.1, 1.5),
            ParameterPrior("log_kzz", 6.0, 10.0),
            ParameterPrior("log_gravity", 3.0, 5.5),
            ParameterPrior("radius_rjup", 0.5, 2.0),
            ParameterPrior("snr", 30.0, 150.0, is_conditioning=True),
        ),
        noise=NoiseConfig(snr_min=30.0, snr_max=150.0),
        track_root="hband",
        fastchem_path=None,
        tpu_batch_size=16,
        metadata={"priority": "secondary"},
    )


def _g395h_miri_config() -> TrackConfig:
    """G395H (2.82-5.41 um) + MIRI MRS extension out to ~28 um as one contiguous
    ESLOG grid. Used to quantify how much longer-wavelength coverage breaks the
    G395H-only degeneracies (especially log_kzz / disequilibrium chemistry, whose
    NH3 10.5 um, CH4 7.7 um and CO2 15 um bands sit beyond G395H). Same 9-param
    priors as g395h so the maps are directly comparable. Variable-resolution
    convolution uses a combined R curve (G395H curve + MIRI MRS R) built into
    tmp_g395h_miri_disp_reference.fits before running."""
    return TrackConfig(
        name="g395h_miri",
        wavelength_label="JWST NIRSpec G395H + MIRI MRS",
        nu_min_cm1=800.0,
        nu_max_cm1=3550.0,
        n_wavenumber=10000,
        n_layers=100,
        pressure_top_bar=1.0e-6,
        pressure_bottom_bar=1.0e3,
        knot_log10_pressures_bar=(-5.0, -3.0, -1.0, 1.0, 3.0),
        instrument_resolving_power=2000.0,
        default_distance_pc=10.0,
        molecules=("H2O", "CO", "CH4", "NH3", "CO2", "H2S"),
        continuum_species=("H2-H2", "H2-He"),
        opacity_databases=(
            OpacityDatabaseConfig("CO", "MdbExomol", "g395h_miri/CO/12C-16O/Li2015", "ExoMol/Li2015"),
            OpacityDatabaseConfig("NH3", "MdbExomol", "g395h_miri/NH3/14N-1H3/BYTe", "ExoMol/BYTe"),
            OpacityDatabaseConfig("CH4", "MdbHitran", "g395h_miri/CH4_HITRAN/CH4", "HITRAN2020 (egress/login fallback)", crit=1.0e-30),
            OpacityDatabaseConfig("H2O", "MdbHitran", "g395h_miri/H2O_HITRAN/H2O", "HITRAN2020 (egress/login fallback)", crit=1.0e-30),
            OpacityDatabaseConfig("CO2", "MdbHitran", "g395h_miri/CO2_HITRAN/CO2", "HITRAN2020 (egress/login fallback)", crit=1.0e-30),
            OpacityDatabaseConfig("H2S", "MdbExomol", "g395h_miri/H2S/1H2-32S/AYT2", "ExoMol/AYT2"),
            OpacityDatabaseConfig("H2-H2", "CdbCIA", "CIA/H2-H2", "HITRAN CIA"),
            OpacityDatabaseConfig("H2-He", "CdbCIA", "CIA/H2-He", "HITRAN CIA"),
        ),
        parameter_priors=(
            ParameterPrior("T1", 150.0, 1500.0),
            ParameterPrior("T2", 200.0, 2000.0),
            ParameterPrior("T3", 300.0, 2500.0),
            ParameterPrior("T4", 500.0, 3000.0),
            ParameterPrior("T5", 800.0, 3500.0),
            ParameterPrior("metallicity", -1.0, 1.5),
            ParameterPrior("carbon_to_oxygen", 0.1, 1.5),
            ParameterPrior("log_kzz", 6.0, 10.0),
            ParameterPrior("log_gravity", 3.0, 5.5),
            ParameterPrior("radius_rjup", 0.5, 2.0),
            ParameterPrior("f_sed", 0.5, 8.0),
            ParameterPrior("sigma_c", 1.2, 2.5),
            ParameterPrior("rv_km_s", -100.0, 100.0),
            ParameterPrior("vsini_km_s", 0.0, 30.0),
            ParameterPrior("sigma_cal", 0.0, 0.05),
            ParameterPrior("snr", 10.0, 200.0, is_conditioning=True),
        ),
        noise=NoiseConfig(snr_min=10.0, snr_max=200.0),
        track_root="g395h_miri",
        fastchem_path="/tmp/exojax_db/fastchem_grid.h5",
        tpu_batch_size=16,
        metadata={
            "priority": "experimental",
            "instrument_convolution_mode": "variable_resolution",
            "instrument_resolution_curve_path": "tmp_g395h_miri_disp_reference.fits",
            "instrument_resolution_curve_hdu": 1,
            "instrument_resolution_wavelength_column": "WAVELENGTH",
            "instrument_resolution_value_column": "R",
        },
    )


DEFAULT_TRACK_CONFIGS = {
    "g395h": _g395h_config(),
    "hband": _hband_config(),
    "g395h_miri": _g395h_miri_config(),
}


def get_default_track_config(name: str = "g395h") -> TrackConfig:
    key = name.lower()
    if key not in DEFAULT_TRACK_CONFIGS:
        raise KeyError(f"Unknown track config: {name}")
    return DEFAULT_TRACK_CONFIGS[key]


def _dict_to_track_config(data: dict[str, Any]) -> TrackConfig:
    opacity_databases = tuple(
        OpacityDatabaseConfig(**entry) for entry in data["opacity_databases"]
    )
    # Honor the same H2O line-list selector as the Python g395h config, so YAML-driven steps
    # (exact-fit, generation) pick up SCULPTOR_H2O_LINELIST=pokazatel consistently.
    if os.environ.get("SCULPTOR_H2O_LINELIST", "hitemp").strip().lower() == "pokazatel":
        opacity_databases = tuple(
            (_g395h_h2o_database() if (d.name == "H2O" and d.relative_path == "g395h/H2O_HITEMP/H2O") else d)
            for d in opacity_databases
        )
    # Same selector for CH4 (R2: ExoMol MM lever), so YAML-driven exact-fit/generation honor it.
    if os.environ.get("SCULPTOR_CH4_LINELIST", "hitemp").strip().lower() in ("exomol_mm", "mm", "exomol"):
        opacity_databases = tuple(
            (_g395h_ch4_database() if (d.name == "CH4" and d.relative_path == "g395h/CH4_HITEMP/CH4") else d)
            for d in opacity_databases
        )
    parameter_priors = tuple(
        ParameterPrior(**entry) for entry in data["parameter_priors"]
    )
    noise = NoiseConfig(**data["noise"])
    return TrackConfig(
        name=data["name"],
        wavelength_label=data["wavelength_label"],
        nu_min_cm1=float(data["nu_min_cm1"]),
        nu_max_cm1=float(data["nu_max_cm1"]),
        n_wavenumber=int(data["n_wavenumber"]),
        n_layers=int(data["n_layers"]),
        pressure_top_bar=float(data["pressure_top_bar"]),
        pressure_bottom_bar=float(data["pressure_bottom_bar"]),
        knot_log10_pressures_bar=tuple(float(x) for x in data["knot_log10_pressures_bar"]),
        instrument_resolving_power=float(data["instrument_resolving_power"]),
        default_distance_pc=float(data.get("default_distance_pc", 10.0)),
        molecules=tuple(data["molecules"]),
        continuum_species=tuple(data["continuum_species"]),
        opacity_databases=opacity_databases,
        parameter_priors=parameter_priors,
        noise=noise,
        track_root=data["track_root"],
        fastchem_path=data.get("fastchem_path"),
        tpu_batch_size=int(data.get("tpu_batch_size", 16)),
        metadata=dict(data.get("metadata", {})),
    )


def load_track_config(path: str | Path | None = None, name: str = "g395h") -> TrackConfig:
    if path is None:
        default_path = config_root() / f"{name.lower()}.yaml"
        if not default_path.exists():
            return get_default_track_config(name)
        path = default_path

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return _dict_to_track_config(data)
