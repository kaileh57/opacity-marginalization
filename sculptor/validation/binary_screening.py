"""Binary anomaly screening utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryScreeningResult:
    flagged: bool
    radius_z_score: float
    predictive_residual: float


def screen_for_binary(
    posterior_samples: np.ndarray,
    parameter_names: tuple[str, ...],
    observed_flux: np.ndarray,
    predictive_flux: np.ndarray,
    expected_radius_rjup: float = 1.0,
    radius_sigma_rjup: float = 0.15,
    residual_threshold: float = 3.0,
) -> BinaryScreeningResult:
    radius_index = parameter_names.index("radius_rjup")
    inferred_radius = float(np.median(posterior_samples[:, radius_index]))
    radius_z = (inferred_radius - expected_radius_rjup) / max(radius_sigma_rjup, 1.0e-6)
    residual = float(np.sqrt(np.mean((observed_flux - predictive_flux) ** 2)) / (np.std(observed_flux) + 1.0e-12))
    flagged = radius_z > 2.0 or residual > residual_threshold
    return BinaryScreeningResult(flagged=flagged, radius_z_score=radius_z, predictive_residual=residual)
