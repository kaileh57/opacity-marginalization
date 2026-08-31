"""Post-hoc quantile recalibration for amortized posteriors.

Implements per-parameter quantile recalibration (Kuleshov, Fenner & Ermon 2018,
"Accurate Uncertainties for Deep Learning Using Calibrated Regression") using the
probability-integral-transform (PIT) / rank values of a held-out set.

For each parameter the PIT value of an object is r = P(sample < truth), estimated
as the fraction of posterior samples below the truth. If the posterior is
calibrated these are Uniform(0,1). We fit the empirical CDF R(r) of the held-out
PIT values; the calibrated predictive quantile at level p is then the posterior
sample-quantile at level R^{-1}(p). This corrects both dispersion (over/under
confidence) and, because R need not be symmetric, location bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pit_values(samples: np.ndarray, truths: np.ndarray) -> np.ndarray:
    """PIT/rank per object-parameter: fraction of samples below the truth.

    samples: [n_obj, n_samples, n_params]; truths: [n_obj, n_params].
    Returns [n_obj, n_params] in [0, 1].
    """
    samples = np.asarray(samples, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError("samples must be [n_obj, n_samples, n_params]")
    return np.mean(samples < truths[:, None, :], axis=1)


@dataclass
class QuantileRecalibrator:
    """Per-parameter monotone PIT-CDF recalibration map and its inverse."""

    param_names: tuple[str, ...]
    # sorted PIT grid and CDF value per parameter (for R and R^{-1})
    _pit_sorted: list  # list[np.ndarray]
    _cdf: list         # list[np.ndarray]

    @classmethod
    def fit(cls, samples: np.ndarray, truths: np.ndarray,
            param_names: tuple[str, ...]) -> "QuantileRecalibrator":
        pit = pit_values(samples, truths)  # [n_obj, n_params]
        n_obj, n_params = pit.shape
        if n_params != len(param_names):
            raise ValueError("param_names length mismatch")
        pit_sorted, cdf = [], []
        for j in range(n_params):
            r = np.sort(pit[:, j])
            # empirical CDF with endpoints pinned so R(0)=0, R(1)=1
            r = np.concatenate([[0.0], r, [1.0]])
            c = np.concatenate([[0.0], (np.arange(1, n_obj + 1)) / (n_obj + 1), [1.0]])
            pit_sorted.append(r)
            cdf.append(c)
        return cls(tuple(param_names), pit_sorted, cdf)

    def _R(self, j: int, r: np.ndarray) -> np.ndarray:
        """Empirical PIT CDF: observed coverage of nominal level r."""
        return np.interp(r, self._pit_sorted[j], self._cdf[j])

    def _R_inv(self, j: int, p: np.ndarray) -> np.ndarray:
        """Inverse map: nominal sample-level achieving calibrated level p."""
        return np.interp(p, self._cdf[j], self._pit_sorted[j])

    def calibrated_quantile_levels(self, j: int, levels: np.ndarray) -> np.ndarray:
        """Sample-quantile levels to use to obtain calibrated `levels`."""
        return self._R_inv(j, np.asarray(levels, dtype=np.float64))

    def calibrated_interval(self, samples_j: np.ndarray, j: int, cred: float):
        """Calibrated central credible interval for one parameter.

        samples_j: [n_obj, n_samples] posterior samples of parameter j.
        Returns (lo, hi) arrays of shape [n_obj].
        """
        p_lo, p_hi = (1.0 - cred) / 2.0, (1.0 + cred) / 2.0
        q_lo = float(self._R_inv(j, np.array([p_lo]))[0])
        q_hi = float(self._R_inv(j, np.array([p_hi]))[0])
        lo = np.quantile(samples_j, np.clip(q_lo, 0.0, 1.0), axis=1)
        hi = np.quantile(samples_j, np.clip(q_hi, 0.0, 1.0), axis=1)
        return lo, hi

    def coverage_after(self, samples: np.ndarray, truths: np.ndarray,
                       levels=(0.5, 0.68, 0.9, 0.95)) -> dict[float, float]:
        """Marginal coverage (over all object-parameter pairs) after recalibration."""
        samples = np.asarray(samples, dtype=np.float64)
        truths = np.asarray(truths, dtype=np.float64)
        out: dict[float, float] = {}
        for cred in levels:
            inside = []
            for j in range(samples.shape[2]):
                lo, hi = self.calibrated_interval(samples[:, :, j], j, cred)
                inside.append((truths[:, j] >= lo) & (truths[:, j] <= hi))
            out[cred] = float(np.mean(np.stack(inside, axis=1)))
        return out

    def to_dict(self) -> dict:
        return {
            "param_names": list(self.param_names),
            "pit_sorted": [a.tolist() for a in self._pit_sorted],
            "cdf": [a.tolist() for a in self._cdf],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuantileRecalibrator":
        return cls(tuple(d["param_names"]),
                   [np.asarray(a) for a in d["pit_sorted"]],
                   [np.asarray(a) for a in d["cdf"]])
