"""Temperature-pressure profile construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sculptor.config import TrackConfig


@dataclass(frozen=True)
class TemperatureProfile:
    """A temperature-pressure profile sampled on the model layers."""

    pressures_bar: np.ndarray
    temperatures_K: np.ndarray
    knot_log10_pressures_bar: np.ndarray
    knot_temperatures_K: np.ndarray


def _natural_cubic_spline(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Evaluate a natural cubic spline without relying on SciPy."""

    n = len(x)
    if n < 3:
        return np.interp(x_new, x, y)

    h = np.diff(x)
    alpha = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1])

    l = np.ones(n, dtype=np.float64)
    mu = np.zeros(n, dtype=np.float64)
    z = np.zeros(n, dtype=np.float64)

    for i in range(1, n - 1):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    b = np.zeros(n - 1, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    d = np.zeros(n - 1, dtype=np.float64)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    result = np.empty_like(x_new, dtype=np.float64)
    interval_indices = np.searchsorted(x, x_new, side="right") - 1
    interval_indices = np.clip(interval_indices, 0, n - 2)
    dx = x_new - x[interval_indices]
    result[:] = y[interval_indices] + b[interval_indices] * dx + c[interval_indices] * dx**2 + d[interval_indices] * dx**3
    return result


def build_temperature_profile(
    knot_temperatures_K: Iterable[float],
    track_config: TrackConfig,
    pressure_grid_bar: np.ndarray | None = None,
) -> TemperatureProfile:
    """Build a spline-interpolated temperature profile."""

    knot_temperatures = np.asarray(tuple(knot_temperatures_K), dtype=np.float64)
    knot_log_pressures = np.asarray(track_config.knot_log10_pressures_bar, dtype=np.float64)
    pressures = (
        np.asarray(pressure_grid_bar, dtype=np.float64)
        if pressure_grid_bar is not None
        else track_config.pressure_grid_bar.astype(np.float64)
    )
    log_pressures = np.log10(pressures)
    temperatures = _natural_cubic_spline(knot_log_pressures, knot_temperatures, log_pressures)
    temperatures = np.clip(temperatures, 50.0, 6000.0)
    return TemperatureProfile(
        pressures_bar=pressures,
        temperatures_K=temperatures,
        knot_log10_pressures_bar=knot_log_pressures,
        knot_temperatures_K=knot_temperatures,
    )


def is_physical_profile(profile: TemperatureProfile, min_temperature_K: float = 100.0) -> bool:
    """Screen clearly unphysical profiles before running the spectrum model."""

    if np.any(~np.isfinite(profile.temperatures_K)):
        return False
    if np.any(profile.temperatures_K < min_temperature_K):
        return False
    knot_jumps = np.abs(np.diff(profile.knot_temperatures_K))
    if np.any(knot_jumps > 1500.0):
        return False
    if np.max(np.abs(np.diff(profile.temperatures_K))) > 250.0:
        return False
    return True
