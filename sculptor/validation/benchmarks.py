"""Tier-2 benchmark comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BenchmarkTarget:
    name: str
    expected_metallicity: float | None = None
    expected_mass_mjup: float | None = None
    metadata: dict[str, Any] | None = None


BENCHMARKS = {
    "epsilon_indi_ba": BenchmarkTarget(name="epsilon_indi_ba", expected_mass_mjup=66.92),
    "epsilon_indi_bb": BenchmarkTarget(name="epsilon_indi_bb", expected_mass_mjup=53.25),
    "gl_570d": BenchmarkTarget(name="gl_570d", expected_metallicity=0.0),
    "hd_3651b": BenchmarkTarget(name="hd_3651b", expected_metallicity=0.1),
}


def summarize_posterior(samples: np.ndarray, parameter_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for index, name in enumerate(parameter_names):
        column = samples[:, index]
        summary[name] = {
            "median": float(np.median(column)),
            "p16": float(np.quantile(column, 0.16)),
            "p84": float(np.quantile(column, 0.84)),
        }
    return summary


def compare_to_benchmark(target_name: str, posterior_samples: np.ndarray, parameter_names: tuple[str, ...]) -> dict[str, Any]:
    target = BENCHMARKS[target_name]
    summary = summarize_posterior(posterior_samples, parameter_names)
    result: dict[str, Any] = {"target": target.name, "summary": summary}
    if target.expected_metallicity is not None and "metallicity" in summary:
        result["metallicity_offset"] = summary["metallicity"]["median"] - target.expected_metallicity
    if target.expected_mass_mjup is not None and "log_gravity" in summary and "radius_rjup" in summary:
        gravity = 10.0 ** summary["log_gravity"]["median"]
        radius_cm = summary["radius_rjup"]["median"] * 7.1492e9
        mass_g = gravity * radius_cm**2 / 6.67430e-8
        result["mass_mjup"] = mass_g / 1.898e30
        result["mass_offset"] = result["mass_mjup"] - target.expected_mass_mjup
    return result
