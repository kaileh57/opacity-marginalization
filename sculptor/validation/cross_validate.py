"""Tier-3 posterior cross-validation against a simple MCMC baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MCMCResult:
    samples: np.ndarray
    acceptance_rate: float


def metropolis_hastings(
    log_posterior: Callable[[np.ndarray], float],
    initial: np.ndarray,
    step_scale: np.ndarray,
    steps: int = 5000,
    burn_in: int = 1000,
    random_seed: int = 0,
) -> MCMCResult:
    rng = np.random.default_rng(random_seed)
    current = np.asarray(initial, dtype=np.float64)
    current_logp = float(log_posterior(current))
    accepted = 0
    chain = []
    for step in range(steps):
        proposal = current + rng.normal(0.0, step_scale, size=current.shape)
        proposal_logp = float(log_posterior(proposal))
        if np.log(rng.uniform()) < proposal_logp - current_logp:
            current = proposal
            current_logp = proposal_logp
            accepted += 1
        if step >= burn_in:
            chain.append(current.copy())
    return MCMCResult(samples=np.asarray(chain, dtype=np.float64), acceptance_rate=accepted / max(steps, 1))
