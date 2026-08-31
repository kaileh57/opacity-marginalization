"""Proposal distributions for sequential neural posterior estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sculptor.config import ParameterPrior, TrackConfig
from sculptor.forward_model.atmosphere import build_temperature_profile, is_physical_profile
from sculptor.npe.inference import PosteriorInferencer


ArrayLike = np.ndarray


@dataclass(frozen=True)
class ProposalSample:
    """Samples drawn from a sequential proposal."""

    parameters: np.ndarray
    parameter_names: tuple[str, ...]
    target_parameters: np.ndarray
    target_parameter_names: tuple[str, ...]
    source: np.ndarray


class ProposalDistribution:
    """Mixture of real-data NPE posteriors and the original prior.

    The round-2 proposal is

    ``p(theta) = alpha * mixture(q_phi(theta | x_i)) + (1 - alpha) * p_prior(theta)``.

    The posterior mixture is represented empirically by cached samples or by
    samples generated from a checkpoint. Non-target generation parameters, such
    as ``snr`` or ``radius_rjup`` when they are not part of the NPE target, are
    drawn from their original priors.
    """

    def __init__(
        self,
        *,
        track_config: TrackConfig,
        target_parameter_names: Sequence[str],
        posterior_samples: Mapping[str, ArrayLike],
        alpha: float = 0.5,
        posterior_weights: Mapping[str, float] | None = None,
        posterior_jitter_fraction: float = 0.02,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if posterior_jitter_fraction < 0.0:
            raise ValueError("posterior_jitter_fraction must be non-negative.")
        if not posterior_samples and alpha > 0.0:
            raise ValueError("posterior_samples are required when alpha > 0.")

        self.track_config = track_config
        self.alpha = float(alpha)
        self.target_parameter_names = tuple(target_parameter_names)
        self.parameter_names = track_config.parameter_names
        self.posterior_jitter_fraction = float(posterior_jitter_fraction)
        self.priors = {prior.name: prior for prior in track_config.parameter_priors}

        missing = [name for name in self.target_parameter_names if name not in self.priors]
        if missing:
            raise ValueError(f"Target parameters are not present in track priors: {missing}")

        cleaned: dict[str, np.ndarray] = {}
        for target_key, samples in posterior_samples.items():
            values = np.asarray(samples, dtype=np.float32)
            if values.ndim != 2:
                raise ValueError(f"Posterior samples for {target_key} must be 2D.")
            if values.shape[1] < len(self.target_parameter_names):
                raise ValueError(
                    f"Posterior samples for {target_key} have {values.shape[1]} columns, "
                    f"need at least {len(self.target_parameter_names)}."
                )
            values = values[:, : len(self.target_parameter_names)]
            mask = np.all(np.isfinite(values), axis=1) & self._target_in_bounds(values)
            if not np.any(mask):
                raise ValueError(f"No finite in-bounds posterior samples for {target_key}.")
            cleaned[target_key] = values[mask]
        self.posterior_samples = cleaned

        if cleaned:
            keys = tuple(cleaned)
            if posterior_weights is None:
                weights = np.ones(len(keys), dtype=np.float64)
            else:
                weights = np.array([float(posterior_weights.get(key, 0.0)) for key in keys], dtype=np.float64)
            if np.any(weights < 0.0) or np.sum(weights) <= 0.0:
                raise ValueError("posterior_weights must be non-negative and sum to a positive value.")
            self.posterior_keys = keys
            self.posterior_weights = weights / np.sum(weights)
        else:
            self.posterior_keys = ()
            self.posterior_weights = np.zeros(0, dtype=np.float64)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_dir: str | Path,
        observed_spectra: Mapping[str, ArrayLike],
        track_config: TrackConfig,
        num_samples_per_target: int = 10_000,
        alpha: float = 0.5,
        conditioning: Mapping[str, ArrayLike] | None = None,
        posterior_jitter_fraction: float = 0.02,
    ) -> "ProposalDistribution":
        """Build a proposal by running a trained NPE on observed spectra."""

        inferencer = PosteriorInferencer(checkpoint_dir)
        samples: dict[str, np.ndarray] = {}
        for target_key, spectrum in observed_spectra.items():
            spectrum_array = np.asarray(spectrum, dtype=np.float32)
            if spectrum_array.ndim == 1:
                spectrum_array = spectrum_array[None, :]
            cond = None if conditioning is None else conditioning.get(target_key)
            drawn = inferencer.sample(spectrum_array, conditioning=cond, num_samples=num_samples_per_target)
            samples[target_key] = drawn.reshape(-1, drawn.shape[-1])
        return cls(
            track_config=track_config,
            target_parameter_names=inferencer.target_parameter_names,
            posterior_samples=samples,
            alpha=alpha,
            posterior_jitter_fraction=posterior_jitter_fraction,
        )

    @classmethod
    def from_sample_files(
        cls,
        *,
        sample_files: Sequence[str | Path],
        track_config: TrackConfig,
        target_parameter_names: Sequence[str],
        alpha: float = 0.5,
        posterior_jitter_fraction: float = 0.02,
    ) -> "ProposalDistribution":
        samples: dict[str, np.ndarray] = {}
        for path in sample_files:
            path = Path(path)
            samples[path.stem.removeprefix("samples_")] = np.load(path)
        return cls(
            track_config=track_config,
            target_parameter_names=target_parameter_names,
            posterior_samples=samples,
            alpha=alpha,
            posterior_jitter_fraction=posterior_jitter_fraction,
        )

    def sample(
        self,
        count: int,
        seed: int | None = None,
        physical_only: bool = False,
        max_draw_multiplier: int = 20,
    ) -> ProposalSample:
        rng = np.random.default_rng(seed)
        count = int(count)
        if count < 1:
            raise ValueError("count must be positive.")
        if not physical_only:
            return self._sample_unfiltered(count=count, rng=rng)

        accepted: list[ProposalSample] = []
        accepted_count = 0
        draw_count = 0
        batch_size = max(1024, min(count, 16_384))
        max_draws = max(count, int(count * max_draw_multiplier))
        while accepted_count < count and draw_count < max_draws:
            batch_count = min(batch_size, max_draws - draw_count)
            sample = self._sample_unfiltered(count=batch_count, rng=rng)
            draw_count += batch_count
            mask = self._physical_mask(sample.parameters)
            if not np.any(mask):
                continue
            accepted.append(
                ProposalSample(
                    parameters=sample.parameters[mask],
                    parameter_names=sample.parameter_names,
                    target_parameters=sample.target_parameters[mask],
                    target_parameter_names=sample.target_parameter_names,
                    source=sample.source[mask],
                )
            )
            accepted_count += int(np.sum(mask))
        if accepted_count < count:
            raise RuntimeError(
                f"Could only draw {accepted_count} physical samples after {draw_count} proposal draws; "
                f"increase max_draw_multiplier or broaden the proposal."
            )
        parameters = np.concatenate([sample.parameters for sample in accepted], axis=0)[:count]
        target_parameters = np.concatenate([sample.target_parameters for sample in accepted], axis=0)[:count]
        source = np.concatenate([sample.source for sample in accepted], axis=0)[:count]
        return ProposalSample(
            parameters=parameters,
            parameter_names=self.parameter_names,
            target_parameters=target_parameters,
            target_parameter_names=self.target_parameter_names,
            source=source,
        )

    def _sample_unfiltered(self, count: int, rng: np.random.Generator) -> ProposalSample:

        use_posterior = rng.random(count) < self.alpha
        if not self.posterior_keys:
            use_posterior[:] = False

        full = np.empty((count, len(self.parameter_names)), dtype=np.float32)
        source = np.empty(count, dtype=object)
        target = np.empty((count, len(self.target_parameter_names)), dtype=np.float32)

        posterior_rows = np.flatnonzero(use_posterior)
        prior_rows = np.flatnonzero(~use_posterior)

        if posterior_rows.size:
            target[posterior_rows] = self._sample_posterior_targets(rng, posterior_rows.size)
            source[posterior_rows] = "posterior"
        if prior_rows.size:
            target[prior_rows] = self._sample_target_prior(rng, prior_rows.size)
            source[prior_rows] = "prior"

        for column, name in enumerate(self.parameter_names):
            prior = self.priors[name]
            if name in self.target_parameter_names:
                target_column = self.target_parameter_names.index(name)
                full[:, column] = target[:, target_column]
            else:
                full[:, column] = self._sample_prior_values(rng, prior, count)

        return ProposalSample(
            parameters=full,
            parameter_names=self.parameter_names,
            target_parameters=target,
            target_parameter_names=self.target_parameter_names,
            source=source.astype(str),
        )

    def _physical_mask(self, parameters: np.ndarray) -> np.ndarray:
        temp_indices = [
            self.parameter_names.index(name)
            for name in self.parameter_names
            if name.startswith("T")
        ]
        mask = np.zeros(parameters.shape[0], dtype=bool)
        for row_index, row in enumerate(parameters):
            profile = build_temperature_profile(row[temp_indices], self.track_config)
            mask[row_index] = is_physical_profile(profile)
        return mask

    def summary(self) -> dict:
        posterior_counts = {key: int(value.shape[0]) for key, value in self.posterior_samples.items()}
        return {
            "alpha": self.alpha,
            "posterior_jitter_fraction": self.posterior_jitter_fraction,
            "parameter_names": list(self.parameter_names),
            "target_parameter_names": list(self.target_parameter_names),
            "posterior_target_count": len(self.posterior_samples),
            "posterior_sample_counts": posterior_counts,
            "posterior_weights": {
                key: float(weight) for key, weight in zip(self.posterior_keys, self.posterior_weights)
            },
        }

    def _sample_posterior_targets(self, rng: np.random.Generator, count: int) -> np.ndarray:
        target_choices = rng.choice(len(self.posterior_keys), size=count, p=self.posterior_weights)
        output = np.empty((count, len(self.target_parameter_names)), dtype=np.float32)
        for target_index in range(len(self.posterior_keys)):
            rows = np.flatnonzero(target_choices == target_index)
            if not rows.size:
                continue
            samples = self.posterior_samples[self.posterior_keys[target_index]]
            chosen = samples[rng.integers(0, len(samples), size=rows.size)]
            output[rows] = chosen
        if self.posterior_jitter_fraction:
            output = output + rng.normal(0.0, self._target_widths() * self.posterior_jitter_fraction, output.shape)
            output = self._clip_targets(output)
        return output.astype(np.float32)

    def _sample_target_prior(self, rng: np.random.Generator, count: int) -> np.ndarray:
        output = np.empty((count, len(self.target_parameter_names)), dtype=np.float32)
        for column, name in enumerate(self.target_parameter_names):
            output[:, column] = self._sample_prior_values(rng, self.priors[name], count)
        return output

    @staticmethod
    def _sample_prior_values(rng: np.random.Generator, prior: ParameterPrior, count: int) -> np.ndarray:
        return rng.uniform(float(prior.lower), float(prior.upper), size=count).astype(np.float32)

    def _target_widths(self) -> np.ndarray:
        return np.array(
            [self.priors[name].upper - self.priors[name].lower for name in self.target_parameter_names],
            dtype=np.float32,
        )

    def _target_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([self.priors[name].lower for name in self.target_parameter_names], dtype=np.float32)
        upper = np.array([self.priors[name].upper for name in self.target_parameter_names], dtype=np.float32)
        return lower, upper

    def _target_in_bounds(self, values: np.ndarray) -> np.ndarray:
        lower, upper = self._target_bounds()
        return np.all((values >= lower) & (values <= upper), axis=1)

    def _clip_targets(self, values: np.ndarray) -> np.ndarray:
        lower, upper = self._target_bounds()
        return np.clip(values, lower, upper)
