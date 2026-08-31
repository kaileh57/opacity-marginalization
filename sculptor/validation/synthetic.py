"""Tier-1 synthetic validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sculptor.npe.calibration import empirical_coverage, mean_absolute_rank_error
from sculptor.npe.inference import PosteriorInferencer


@dataclass(frozen=True)
class SyntheticValidationResult:
    coverage: dict[float, float]
    rank_error: float
    num_objects: int
    metadata: dict[str, Any]


def run_synthetic_validation(
    inferencer: PosteriorInferencer,
    spectra: np.ndarray,
    truths: np.ndarray,
    conditioning: np.ndarray | None = None,
    num_samples: int = 1000,
) -> SyntheticValidationResult:
    posterior_samples = inferencer.sample(spectra, conditioning=conditioning, num_samples=num_samples)
    coverage = empirical_coverage(posterior_samples, truths)
    rank_error = mean_absolute_rank_error(posterior_samples, truths)
    return SyntheticValidationResult(
        coverage=coverage,
        rank_error=rank_error,
        num_objects=int(len(spectra)),
        metadata={"num_samples": num_samples},
    )
