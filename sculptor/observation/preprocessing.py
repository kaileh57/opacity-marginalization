"""Observed-spectrum preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from astropy.io import fits
except Exception:  # pragma: no cover - optional dependency
    fits = None


@dataclass(frozen=True)
class ObservedSpectrum:
    wavelength_um: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    dq: np.ndarray | None = None

    @property
    def nu_cm1(self) -> np.ndarray:
        return 1.0e4 / np.asarray(self.wavelength_um, dtype=np.float64)


def load_ascii_spectrum(path: str | Path) -> ObservedSpectrum:
    data = np.loadtxt(path)
    if data.shape[1] < 2:
        raise ValueError("Expected at least wavelength and flux columns.")
    wavelength = data[:, 0]
    flux = data[:, 1]
    error = data[:, 2] if data.shape[1] > 2 else np.full_like(flux, np.std(flux) * 0.05 + 1.0e-18)
    return ObservedSpectrum(wavelength_um=wavelength, flux=flux, error=error)


def continuum_normalize(flux: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    reference = np.percentile(flux, percentile)
    return np.asarray(flux, dtype=np.float64) / max(reference, 1.0e-18)


def load_x1d_spectrum(
    path: str | Path,
    wavelength_min_um: float = 2.85,
    wavelength_max_um: float = 5.3,
) -> ObservedSpectrum:
    """Load a JWST x1d spectrum and keep the native DQ flags."""

    if fits is None:
        raise ImportError("astropy is required to load JWST FITS spectra.")

    with fits.open(path) as hdul:
        table_hdu = next(
            (
                hdu
                for hdu in hdul
                if hasattr(hdu, "columns") and hdu.columns is not None and "WAVELENGTH" in hdu.columns.names
            ),
            None,
        )
        if table_hdu is None:
            raise ValueError(f"Could not find a spectrum table in {path}")
        data = table_hdu.data
        wavelength = np.asarray(data["WAVELENGTH"], dtype=np.float64).ravel()
        if np.nanmedian(wavelength) > 1000:
            wavelength = wavelength / 1.0e4
        flux = np.asarray(data["FLUX"], dtype=np.float64).ravel()
        err_name = "FLUX_ERROR" if "FLUX_ERROR" in table_hdu.columns.names else "ERROR"
        error = np.asarray(data[err_name], dtype=np.float64).ravel()
        dq = np.asarray(data["DQ"], dtype=np.int64).ravel() if "DQ" in table_hdu.columns.names else np.zeros_like(flux, dtype=np.int64)

    mask = (
        np.isfinite(wavelength)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (wavelength >= wavelength_min_um)
        & (wavelength <= wavelength_max_um)
        & (flux > 0)
    )
    return ObservedSpectrum(
        wavelength_um=wavelength[mask],
        flux=flux[mask],
        error=error[mask],
        dq=dq[mask],
    )


def dq_valid_mask(dq: np.ndarray | None) -> np.ndarray:
    if dq is None:
        return np.ones(0, dtype=bool)
    return np.asarray(dq, dtype=np.int64) == 0


def _nearest_resample(values: np.ndarray, source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    source_x = np.asarray(source_x, dtype=np.float64)
    target_x = np.asarray(target_x, dtype=np.float64)
    if source_x.size == 0:
        return np.full_like(target_x, np.nan, dtype=np.float64)
    insert = np.searchsorted(source_x, target_x, side="left")
    right = np.clip(insert, 0, source_x.size - 1)
    left = np.clip(insert - 1, 0, source_x.size - 1)
    choose_right = np.abs(source_x[right] - target_x) <= np.abs(target_x - source_x[left])
    nearest = np.where(choose_right, right, left)
    return values[nearest]


def _large_interpolation_gap_mask(
    source_x: np.ndarray,
    target_x: np.ndarray,
    max_gap_factor: float,
) -> np.ndarray:
    """Return targets whose bracketing source pixels are separated by a large gap."""
    source_x = np.asarray(source_x, dtype=np.float64)
    target_x = np.asarray(target_x, dtype=np.float64)
    if source_x.size < 3:
        return np.ones_like(target_x, dtype=bool)

    source_steps = np.diff(source_x)
    positive_steps = source_steps[np.isfinite(source_steps) & (source_steps > 0)]
    if positive_steps.size == 0:
        return np.ones_like(target_x, dtype=bool)
    max_allowed_gap = float(max_gap_factor) * float(np.nanmedian(positive_steps))

    insert = np.searchsorted(source_x, target_x, side="left")
    interior = (insert > 0) & (insert < source_x.size)
    gap_mask = np.ones_like(target_x, dtype=bool)
    gap_mask[interior] = (source_x[insert[interior]] - source_x[insert[interior] - 1]) > max_allowed_gap
    return gap_mask


def align_to_grid(
    nu_cm1: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    target_nu_cm1: np.ndarray,
    dq: np.ndarray | None = None,
    interpolate_masked_segments: bool = False,
    mask_large_gaps: bool = True,
    max_source_gap_factor: float = 5.0,
) -> ObservedSpectrum:
    source_nu = np.asarray(nu_cm1, dtype=np.float64)
    source_flux = np.asarray(flux, dtype=np.float64)
    source_error = np.asarray(error, dtype=np.float64)
    source_dq = np.asarray(dq, dtype=np.int64) if dq is not None else None

    sort_index = np.argsort(source_nu)
    sorted_nu = source_nu[sort_index]
    sorted_flux = source_flux[sort_index]
    sorted_error = source_error[sort_index]
    sorted_dq = source_dq[sort_index] if source_dq is not None else None

    if sorted_dq is None or interpolate_masked_segments:
        aligned_flux = np.interp(target_nu_cm1, sorted_nu, sorted_flux)
        aligned_error = np.interp(target_nu_cm1, sorted_nu, sorted_error)
        aligned_dq = (
            np.zeros_like(target_nu_cm1, dtype=np.int64)
            if sorted_dq is None
            else _nearest_resample((sorted_dq != 0).astype(np.int64), sorted_nu, target_nu_cm1).astype(np.int64)
        )
        interpolation_support_nu = sorted_nu
    else:
        aligned_flux = np.full_like(target_nu_cm1, np.nan, dtype=np.float64)
        aligned_error = np.full_like(target_nu_cm1, np.nan, dtype=np.float64)
        aligned_dq = np.ones_like(target_nu_cm1, dtype=np.int64)
        valid = sorted_dq == 0
        interpolation_support_nu = sorted_nu[valid]
        if valid.sum() >= 2:
            valid_flux = sorted_flux[valid]
            valid_error = sorted_error[valid]
            aligned_flux = np.interp(target_nu_cm1, interpolation_support_nu, valid_flux, left=np.nan, right=np.nan)
            aligned_error = np.interp(target_nu_cm1, interpolation_support_nu, valid_error, left=np.nan, right=np.nan)
        aligned_dq = _nearest_resample((sorted_dq != 0).astype(np.int64), sorted_nu, target_nu_cm1).astype(np.int64)

    if mask_large_gaps:
        large_gap = _large_interpolation_gap_mask(
            interpolation_support_nu,
            target_nu_cm1,
            max_gap_factor=max_source_gap_factor,
        )
        aligned_flux[large_gap] = np.nan
        aligned_error[large_gap] = np.nan
        aligned_dq[large_gap] = 1

    wavelength_um = 1.0e4 / target_nu_cm1
    return ObservedSpectrum(wavelength_um=wavelength_um, flux=aligned_flux, error=aligned_error, dq=aligned_dq)
