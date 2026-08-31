"""Coverage and calibration diagnostics."""

from __future__ import annotations

import numpy as np


def empirical_coverage(
    posterior_samples: np.ndarray,
    truths: np.ndarray,
    credible_levels: tuple[float, ...] = (0.5, 0.68, 0.9, 0.95),
) -> dict[float, float]:
    """Estimate marginal empirical coverage using central intervals.

    The returned value at each credible level is the fraction of object-parameter
    pairs whose truth falls within the corresponding marginal interval.
    """

    samples = np.asarray(posterior_samples, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError("posterior_samples must have shape [n_objects, n_samples, n_params]")
    if truths.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("truths must have shape [n_objects, n_params]")

    coverage: dict[float, float] = {}
    for level in credible_levels:
        alpha = 1.0 - level
        lower = np.quantile(samples, alpha / 2.0, axis=1)
        upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=1)
        in_interval = (truths >= lower) & (truths <= upper)
        coverage[level] = float(np.mean(in_interval))
    return coverage


def mean_absolute_rank_error(posterior_samples: np.ndarray, truths: np.ndarray) -> float:
    samples = np.asarray(posterior_samples, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    ranks = np.mean(samples < truths[:, None, :], axis=1)
    return float(np.mean(np.abs(ranks - 0.5)))
