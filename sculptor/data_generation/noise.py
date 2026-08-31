"""Noise augmentation and domain randomization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sculptor.config import NoiseConfig, TrackConfig


@dataclass(frozen=True)
class NoiseBatch:
    spectra_noisy: np.ndarray
    error_bars: np.ndarray
    metadata: dict[str, np.ndarray]


class NoiseAugmentor:
    """Apply wavelength-dependent noise and calibration perturbations."""

    def __init__(self, track_config: TrackConfig, random_seed: int = 0) -> None:
        self.track_config = track_config
        self.config: NoiseConfig = track_config.noise
        self.random = np.random.default_rng(random_seed)

    def _apply_shift(self, flux: np.ndarray, shift_pixels: float) -> np.ndarray:
        grid = np.arange(len(flux), dtype=np.float64)
        shifted_grid = np.clip(grid - shift_pixels, grid[0], grid[-1])
        return np.interp(grid, shifted_grid, flux)

    def _correlated_component(self, sigma: np.ndarray, frac: float) -> np.ndarray:
        """Smooth wavelength-correlated noise (real JWST NIRSpec has 1/f-like correlated noise, not
        pure white). Build a low-frequency smoothed field scaled to ``frac`` of sigma."""
        n = len(sigma)
        white = self.random.normal(0.0, 1.0, size=n)
        w = max(3, n // 40)
        smooth = np.convolve(white, np.ones(w) / w, mode="same")
        smooth /= np.std(smooth) + 1e-12
        return frac * sigma * smooth

    def augment(self, clean_flux: np.ndarray, snr: float | None = None,
                sigma_cal: float | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        flux = np.asarray(clean_flux, dtype=np.float64)
        snr_value = float(snr) if snr is not None else float(self.random.uniform(self.config.snr_min, self.config.snr_max))
        wavelength_coordinate = np.linspace(-1.0, 1.0, len(flux))
        profile = 1.0 + 0.15 * wavelength_coordinate
        sigma = np.maximum(np.abs(flux) / (snr_value / profile), 1.0e-18)

        # sigma_cal: use the sampled value when provided (B4 param), else the random draw
        if sigma_cal is not None:
            calibration_amp = float(sigma_cal)
        else:
            calibration_amp = float(self.random.uniform(0.0, self.config.flux_calibration_sigma_max))
        calibration_curve = 1.0 + calibration_amp * (0.5 * wavelength_coordinate + 0.5 * wavelength_coordinate**2)
        tilt = float(self.random.uniform(-self.config.baseline_tilt_max, self.config.baseline_tilt_max))
        shift_pixels = float(self.random.uniform(-self.config.max_pixel_shift, self.config.max_pixel_shift))

        perturbed = flux * calibration_curve * (1.0 + tilt * wavelength_coordinate)
        perturbed = self._apply_shift(perturbed, shift_pixels)
        # noise = white + wavelength-correlated (1/f-like) in quadrature so total stays ~ sigma
        corr_frac = float(self.random.uniform(0.0, 0.5))  # domain-randomize the correlated fraction
        white = self.random.normal(0.0, np.sqrt(max(1.0 - corr_frac ** 2, 0.0)) * sigma, size=flux.shape)
        noisy = perturbed + white + self._correlated_component(sigma, corr_frac)
        return noisy.astype(np.float32), sigma.astype(np.float32), {
            "snr": snr_value,
            "calibration_amp": calibration_amp,
            "tilt": tilt,
            "shift_pixels": shift_pixels,
            "corr_frac": corr_frac,
        }

    def augment_batch(self, clean_flux_batch: np.ndarray, snr_values: np.ndarray | None = None) -> NoiseBatch:
        noisy_list = []
        sigma_list = []
        metadata = {"snr": [], "calibration_amp": [], "tilt": [], "shift_pixels": []}
        for index, flux in enumerate(clean_flux_batch):
            snr = None if snr_values is None else float(snr_values[index])
            noisy, sigma, info = self.augment(flux, snr=snr)
            noisy_list.append(noisy)
            sigma_list.append(sigma)
            for key in metadata:
                metadata[key].append(info[key])
        return NoiseBatch(
            spectra_noisy=np.asarray(noisy_list, dtype=np.float32),
            error_bars=np.asarray(sigma_list, dtype=np.float32),
            metadata={key: np.asarray(values, dtype=np.float32) for key, values in metadata.items()},
        )
