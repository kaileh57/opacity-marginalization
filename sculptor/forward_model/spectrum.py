"""Top-level spectrum generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sculptor.config import TrackConfig, get_default_track_config
from sculptor.forward_model.atmosphere import TemperatureProfile, build_temperature_profile, is_physical_profile
from sculptor.forward_model.chemistry import ChemistryModel, ChemistryState
from sculptor.forward_model.instrument import InstrumentModel
from sculptor.forward_model.opacity import BaseOpacityBackend, SyntheticOpacityBackend
from sculptor.forward_model.radiative_transfer import EmissionRadiativeTransfer, RadiativeTransferResult

C_LIGHT_KM_S = 299792.458


def apply_radial_velocity_shift(
    nu_grid_cm1: np.ndarray,
    flux_nu: np.ndarray,
    rv_km_s: float,
) -> np.ndarray:
    """Doppler-shift a spectrum onto the fixed observation grid.

    Positive radial velocity corresponds to a redward shift in wavelength,
    which maps to sampling the rest-frame spectrum at slightly higher
    wavenumber on a fixed wavenumber grid.
    """
    rv_km_s = float(rv_km_s)
    if abs(rv_km_s) < 1.0e-12:
        return np.asarray(flux_nu, dtype=np.float64)
    nu_grid_cm1 = np.asarray(nu_grid_cm1, dtype=np.float64)
    flux_nu = np.asarray(flux_nu, dtype=np.float64)
    shifted_nu = nu_grid_cm1 * (1.0 + rv_km_s / C_LIGHT_KM_S)
    return np.interp(shifted_nu, nu_grid_cm1, flux_nu, left=flux_nu[0], right=flux_nu[-1])


def apply_vsini_broadening(
    nu_grid_cm1: np.ndarray,
    flux_nu: np.ndarray,
    vsini_km_s: float,
    epsilon: float = 0.3,
) -> np.ndarray:
    """Rotational (v sin i) broadening with a linear limb-darkening kernel.

    Convolves on the log-wavenumber (constant-velocity) grid: G395H uses an ESLOG grid so each
    pixel is a fixed delta-v. Below ~0.1 km/s this is the identity. Real Gray (2008) rotation
    profile; the ExoJAX ``convolve_rigid_rotation`` kernel is used on the pod batch path.
    """
    vsini_km_s = float(vsini_km_s)
    if vsini_km_s < 0.1:
        return np.asarray(flux_nu, dtype=np.float64)
    nu = np.asarray(nu_grid_cm1, dtype=np.float64)
    flux = np.asarray(flux_nu, dtype=np.float64)
    # per-pixel velocity step (constant on an ESLOG grid)
    dv = C_LIGHT_KM_S * np.median(np.diff(np.log(nu)))
    dv = abs(dv)
    if dv <= 0:
        return flux
    half = int(np.ceil(vsini_km_s / dv))
    if half < 1:
        return flux
    v = np.arange(-half, half + 1) * dv
    x = np.clip(v / vsini_km_s, -1.0, 1.0)
    mu = np.sqrt(1.0 - x ** 2)
    kernel = (2.0 * (1.0 - epsilon) * mu + 0.5 * np.pi * epsilon * mu ** 2)
    kernel = np.where(np.abs(x) <= 1.0, kernel, 0.0)
    ksum = kernel.sum()
    if ksum <= 0:
        return flux
    kernel /= ksum
    return np.convolve(flux, kernel, mode="same")


@dataclass(frozen=True)
class GeneratedSpectrum:
    """Full output bundle from a single forward-model evaluation."""

    nu_grid_cm1: np.ndarray
    flux_nu: np.ndarray
    flux_nu_clean: np.ndarray
    temperature_profile: TemperatureProfile
    chemistry_state: ChemistryState
    radiative_transfer: RadiativeTransferResult
    parameters: dict[str, float]
    metadata: dict[str, Any]


class SpectrumModel:
    """Compose atmosphere, chemistry, opacity, radiative transfer, and instrument steps."""

    def __init__(
        self,
        track_config: TrackConfig | None = None,
        chemistry_model: ChemistryModel | None = None,
        opacity_backend: BaseOpacityBackend | None = None,
        radiative_transfer: EmissionRadiativeTransfer | None = None,
        instrument_model: InstrumentModel | None = None,
    ) -> None:
        self.track_config = track_config or get_default_track_config("g395h")
        self.chemistry_model = chemistry_model or ChemistryModel(self.track_config)
        self.opacity_backend = opacity_backend or SyntheticOpacityBackend(self.track_config)
        self.radiative_transfer = radiative_transfer or EmissionRadiativeTransfer()
        self.instrument_model = instrument_model or InstrumentModel(self.track_config)

    def parse_parameters(self, parameters: dict[str, float] | np.ndarray) -> dict[str, float]:
        if isinstance(parameters, dict):
            parsed = {name: float(value) for name, value in parameters.items()}
        else:
            array = np.asarray(parameters, dtype=np.float64).ravel()
            names = self.track_config.parameter_names
            if len(array) != len(names):
                raise ValueError(f"Expected {len(names)} parameters, got {len(array)}")
            parsed = {name: float(array[index]) for index, name in enumerate(names)}
        return parsed

    def _render_spectrum(
        self,
        parsed: dict[str, float],
        temperature_profile: TemperatureProfile,
        chemistry_state: ChemistryState,
        distance_pc: float | None = None,
    ) -> GeneratedSpectrum:
        opacity_result = self.opacity_backend.compute_cross_sections(chemistry_state)
        cloud_parameters = None
        if getattr(self.track_config, "cloud_model", "off") != "off":
            cloud_parameters = {
                "cloud_model": self.track_config.cloud_model,
                "condensates": self.track_config.cloud_condensates,
                "cloud_backend": self.track_config.metadata.get("cloud_backend", "gray"),
                "metallicity": parsed["metallicity"],
                "log_kzz": parsed["log_kzz"],
                "f_sed": parsed.get("f_sed", 8.0),
                "sigma_c": parsed.get("sigma_c", 2.0),
            }
        rt_result = self.radiative_transfer.compute_flux(
            chemistry_state=chemistry_state,
            opacity_result=opacity_result,
            gravity_cgs=10.0 ** parsed["log_gravity"],
            radius_rjup=parsed["radius_rjup"],
            distance_pc=distance_pc or self.track_config.default_distance_pc,
            cloud_parameters=cloud_parameters,
        )
        clean_flux = rt_result.flux_nu
        # processing order: rotational broadening -> radial velocity -> instrument LSF
        clean_flux = apply_vsini_broadening(
            self.track_config.nu_grid_cm1,
            clean_flux,
            parsed.get("vsini_km_s", 0.0),
        )
        clean_flux = apply_radial_velocity_shift(
            self.track_config.nu_grid_cm1,
            clean_flux,
            parsed.get("rv_km_s", 0.0),
        )
        instrument_result = self.instrument_model.convolve(clean_flux)
        return GeneratedSpectrum(
            nu_grid_cm1=instrument_result.nu_grid_cm1,
            flux_nu=instrument_result.flux_nu,
            flux_nu_clean=clean_flux,
            temperature_profile=temperature_profile,
            chemistry_state=chemistry_state,
            radiative_transfer=rt_result,
            parameters=parsed,
            metadata={
                "track": self.track_config.name,
                "opacity_backend": opacity_result.metadata.get("backend", "unknown"),
            },
        )

    def generate_from_state(
        self,
        parameters: dict[str, float] | np.ndarray,
        temperature_profile: TemperatureProfile,
        chemistry_state: ChemistryState,
        distance_pc: float | None = None,
    ) -> GeneratedSpectrum:
        parsed = self.parse_parameters(parameters)
        return self._render_spectrum(parsed, temperature_profile, chemistry_state, distance_pc=distance_pc)

    def generate(self, parameters: dict[str, float] | np.ndarray, distance_pc: float | None = None) -> GeneratedSpectrum:
        parsed = self.parse_parameters(parameters)
        temperature_profile = build_temperature_profile(
            knot_temperatures_K=[parsed[name] for name in self.track_config.parameter_names if name.startswith("T")],
            track_config=self.track_config,
        )
        if not is_physical_profile(temperature_profile):
            raise ValueError("Rejected unphysical temperature-pressure profile.")

        chemistry_state = self.chemistry_model.compute_state(
            profile=temperature_profile,
            metallicity=parsed["metallicity"],
            carbon_to_oxygen=parsed["carbon_to_oxygen"],
            log_gravity=parsed["log_gravity"],
            log_kzz=parsed["log_kzz"],
        )
        return self._render_spectrum(parsed, temperature_profile, chemistry_state, distance_pc=distance_pc)
