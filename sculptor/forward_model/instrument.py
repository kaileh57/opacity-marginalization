"""Instrument response and line-spread-function utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sculptor.config import TrackConfig, project_root


@dataclass(frozen=True)
class InstrumentResult:
    nu_grid_cm1: np.ndarray
    flux_nu: np.ndarray


def _gaussian_kernel(sigma_pixels: float, radius_sigma: float = 4.0) -> np.ndarray:
    half_width = max(int(np.ceil(radius_sigma * sigma_pixels)), 1)
    x = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(sigma_pixels, 1.0e-6)) ** 2)
    return kernel / np.sum(kernel)


class InstrumentModel:
    """Convolve a spectrum on an ESLOG-like grid with a fixed resolving power."""

    def __init__(self, track_config: TrackConfig) -> None:
        self.track_config = track_config
        self._resolution_curve_cache: tuple[np.ndarray, np.ndarray] | None = None

    def convolution_kernel(self, resolving_power: float | None = None) -> np.ndarray:
        nu_grid = self.track_config.nu_grid_cm1
        resolving_power = resolving_power or self.track_config.instrument_resolving_power
        delta_log_nu = float(np.mean(np.diff(np.log(nu_grid))))
        sigma_log = 1.0 / (resolving_power * 2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_pixels = sigma_log / max(delta_log_nu, 1.0e-12)
        return _gaussian_kernel(sigma_pixels)

    def variable_sigma_pixels(
        self,
        resolving_power_curve_wavelength_um: np.ndarray,
        resolving_power_curve_values: np.ndarray,
    ) -> np.ndarray:
        nu_grid = np.asarray(self.track_config.nu_grid_cm1, dtype=np.float64)
        wavelength_um = 1.0e4 / nu_grid
        order = np.argsort(resolving_power_curve_wavelength_um)
        curve_wavelength = np.asarray(resolving_power_curve_wavelength_um, dtype=np.float64)[order]
        curve_resolving_power = np.asarray(resolving_power_curve_values, dtype=np.float64)[order]
        sampled_resolving_power = np.interp(
            wavelength_um,
            curve_wavelength,
            curve_resolving_power,
            left=curve_resolving_power[0],
            right=curve_resolving_power[-1],
        )
        delta_log_nu = float(np.mean(np.diff(np.log(nu_grid))))
        sigma_log = 1.0 / (sampled_resolving_power * 2.0 * np.sqrt(2.0 * np.log(2.0)))
        return sigma_log / max(delta_log_nu, 1.0e-12)

    def uses_variable_resolution(self) -> bool:
        return str(self.track_config.metadata.get("instrument_convolution_mode", "")).lower() == "variable_resolution"

    def load_resolution_curve(self) -> tuple[np.ndarray, np.ndarray]:
        if self._resolution_curve_cache is not None:
            return self._resolution_curve_cache

        curve_path_value = self.track_config.metadata.get("instrument_resolution_curve_path")
        if not curve_path_value:
            raise ValueError("Missing instrument_resolution_curve_path in track config metadata.")
        curve_path = Path(str(curve_path_value))
        if not curve_path.is_absolute():
            curve_path = project_root() / curve_path
        if not curve_path.exists():
            raise FileNotFoundError(f"Instrument resolution curve not found: {curve_path}")

        from astropy.io import fits

        hdu_index = int(self.track_config.metadata.get("instrument_resolution_curve_hdu", 1))
        wavelength_column = str(self.track_config.metadata.get("instrument_resolution_wavelength_column", "WAVELENGTH"))
        resolving_power_column = str(self.track_config.metadata.get("instrument_resolution_value_column", "R"))
        with fits.open(curve_path) as hdul:
            data = hdul[hdu_index].data
            wavelength_um = np.asarray(data[wavelength_column], dtype=np.float64)
            resolving_power = np.asarray(data[resolving_power_column], dtype=np.float64)
        self._resolution_curve_cache = (wavelength_um, resolving_power)
        return self._resolution_curve_cache

    def variable_convolution_plan(
        self,
        resolving_power_curve_wavelength_um: np.ndarray | None = None,
        resolving_power_curve_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if resolving_power_curve_wavelength_um is None or resolving_power_curve_values is None:
            resolving_power_curve_wavelength_um, resolving_power_curve_values = self.load_resolution_curve()

        sigma_pixels = self.variable_sigma_pixels(
            resolving_power_curve_wavelength_um=resolving_power_curve_wavelength_um,
            resolving_power_curve_values=resolving_power_curve_values,
        )
        half_width = max(int(np.ceil(4.0 * float(np.max(sigma_pixels)))), 1)
        offsets = np.arange(-half_width, half_width + 1, dtype=np.int32)
        base_indices = np.arange(len(self.track_config.nu_grid_cm1), dtype=np.int32)[:, None]
        gather_indices = np.clip(base_indices + offsets[None, :], 0, len(self.track_config.nu_grid_cm1) - 1)
        kernels = np.exp(-0.5 * (offsets[None, :] / np.maximum(sigma_pixels[:, None], 1.0e-6)) ** 2)
        kernels /= np.sum(kernels, axis=1, keepdims=True)
        return gather_indices, kernels.astype(np.float64)

    def convolve_variable_resolution(
        self,
        flux_nu: np.ndarray,
        resolving_power_curve_wavelength_um: np.ndarray | None = None,
        resolving_power_curve_values: np.ndarray | None = None,
    ) -> InstrumentResult:
        flux = np.asarray(flux_nu, dtype=np.float64)
        gather_indices, kernels = self.variable_convolution_plan(
            resolving_power_curve_wavelength_um=resolving_power_curve_wavelength_um,
            resolving_power_curve_values=resolving_power_curve_values,
        )
        convolved = np.sum(flux[gather_indices] * kernels, axis=1)
        return InstrumentResult(nu_grid_cm1=self.track_config.nu_grid_cm1, flux_nu=convolved)

    def convolve(self, flux_nu: np.ndarray, resolving_power: float | None = None) -> InstrumentResult:
        if resolving_power is None and self.uses_variable_resolution():
            return self.convolve_variable_resolution(flux_nu)
        nu_grid = self.track_config.nu_grid_cm1
        kernel = self.convolution_kernel(resolving_power=resolving_power)
        convolved = np.convolve(np.asarray(flux_nu, dtype=np.float64), kernel, mode="same")
        return InstrumentResult(nu_grid_cm1=nu_grid, flux_nu=convolved)

    def rebin(self, flux_nu: np.ndarray, source_nu_cm1: np.ndarray, target_nu_cm1: np.ndarray) -> np.ndarray:
        return np.interp(target_nu_cm1, source_nu_cm1, flux_nu, left=flux_nu[0], right=flux_nu[-1])
