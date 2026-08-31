"""Prior sampling for synthetic training data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sculptor.config import TrackConfig
from sculptor.forward_model.atmosphere import build_temperature_profile, is_physical_profile


def _first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        is_prime = True
        for prime in primes:
            if candidate % prime == 0:
                is_prime = False
                break
            if prime * prime > candidate:
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def _van_der_corput(index: int, base: int) -> float:
    result = 0.0
    denominator = 1.0
    n = index
    while n > 0:
        n, remainder = divmod(n, base)
        denominator *= base
        result += remainder / denominator
    return result


def halton_sequence(start_index: int, count: int, dimension: int) -> np.ndarray:
    primes = _first_primes(dimension)
    samples = np.zeros((count, dimension), dtype=np.float64)
    for row in range(count):
        index = start_index + row + 1
        for column, base in enumerate(primes):
            samples[row, column] = _van_der_corput(index, base)
    return samples


@dataclass(frozen=True)
class ParameterBatch:
    parameters: np.ndarray
    parameter_names: tuple[str, ...]
    start_index: int
    next_index: int

    def as_dicts(self) -> list[dict[str, float]]:
        result = []
        for row in self.parameters:
            result.append({name: float(row[index]) for index, name in enumerate(self.parameter_names)})
        return result


class ParameterSampler:
    """Deterministic low-discrepancy sampling over the prior cube."""

    def __init__(self, track_config: TrackConfig, seed: int = 0) -> None:
        self.track_config = track_config
        self.seed = seed

    def _map_unit_cube(self, unit_samples: np.ndarray) -> np.ndarray:
        mapped = np.zeros_like(unit_samples)
        for index, prior in enumerate(self.track_config.parameter_priors):
            mapped[:, index] = prior.lower + unit_samples[:, index] * (prior.upper - prior.lower)
        return mapped

    def _is_valid(self, parameter_row: np.ndarray) -> bool:
        parameters = {
            name: float(parameter_row[index]) for index, name in enumerate(self.track_config.parameter_names)
        }
        profile = build_temperature_profile(
            [parameters[name] for name in self.track_config.parameter_names if name.startswith("T")],
            self.track_config,
        )
        return is_physical_profile(profile)

    def sample(self, count: int, start_index: int = 0) -> ParameterBatch:
        target_dim = len(self.track_config.parameter_priors)
        accepted: list[np.ndarray] = []
        base_offset = self.seed * 100_000
        cursor = start_index + base_offset
        while len(accepted) < count:
            candidates = self._map_unit_cube(halton_sequence(cursor, max(count, 16), target_dim))
            cursor += max(count, 16)
            for row in candidates:
                if self._is_valid(row):
                    accepted.append(row)
                if len(accepted) >= count:
                    break
        return ParameterBatch(
            parameters=np.asarray(accepted, dtype=np.float64),
            parameter_names=self.track_config.parameter_names,
            start_index=start_index,
            next_index=cursor - base_offset,
        )
