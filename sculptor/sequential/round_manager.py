"""Round bookkeeping for sequential NPE."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from sculptor.config import TrackConfig
from sculptor.sequential.proposal import ProposalDistribution, ProposalSample


@dataclass(frozen=True)
class SequentialRoundPaths:
    output_dir: Path
    metadata_path: Path
    proposal_parameters_path: Path
    proposal_target_parameters_path: Path
    proposal_stats_path: Path


class SequentialNPERound:
    """Manage metadata and proposal sampling for one sequential NPE round."""

    def __init__(
        self,
        *,
        round_number: int,
        output_dir: str | Path,
        track_config: TrackConfig,
        previous_checkpoint: str | Path | None,
    ) -> None:
        if int(round_number) < 1:
            raise ValueError("round_number must be >= 1.")
        self.round_number = int(round_number)
        self.output_dir = Path(output_dir)
        self.track_config = track_config
        self.previous_checkpoint = None if previous_checkpoint is None else Path(previous_checkpoint)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def paths(self) -> SequentialRoundPaths:
        return SequentialRoundPaths(
            output_dir=self.output_dir,
            metadata_path=self.output_dir / "round_metadata.json",
            proposal_parameters_path=self.output_dir / "proposal_parameters.npy",
            proposal_target_parameters_path=self.output_dir / "proposal_target_parameters.npy",
            proposal_stats_path=self.output_dir / "proposal_stats.json",
        )

    def proposal_from_sample_files(
        self,
        *,
        sample_files: Sequence[str | Path],
        target_parameter_names: Sequence[str],
        alpha: float,
        posterior_jitter_fraction: float = 0.02,
    ) -> ProposalDistribution:
        return ProposalDistribution.from_sample_files(
            sample_files=sample_files,
            track_config=self.track_config,
            target_parameter_names=target_parameter_names,
            alpha=alpha,
            posterior_jitter_fraction=posterior_jitter_fraction,
        )

    def save_baseline_metadata(
        self,
        *,
        dataset_uri: str,
        checkpoint_path: str | Path,
        checkpoint_epoch: int | None,
        notes: str,
        extra: dict | None = None,
    ) -> dict:
        metadata = {
            "round_number": self.round_number,
            "status": "complete_baseline",
            "track": self.track_config.name,
            "dataset_uri": dataset_uri,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "notes": notes,
            "parameter_names": list(self.track_config.parameter_names),
            "retrieval_parameter_names": list(self.track_config.retrieval_parameter_names),
            "extra": extra or {},
        }
        self._write_json(self.paths.metadata_path, metadata)
        return metadata

    def sample_and_save(
        self,
        proposal: ProposalDistribution,
        *,
        count: int,
        seed: int,
        physical_only: bool = False,
    ) -> tuple[ProposalSample, dict]:
        sample = proposal.sample(count=count, seed=seed, physical_only=physical_only)
        np.save(self.paths.proposal_parameters_path, sample.parameters)
        np.save(self.paths.proposal_target_parameters_path, sample.target_parameters)
        stats = self._sampling_stats(sample, proposal)
        stats["physical_only"] = bool(physical_only)
        self._write_json(self.paths.proposal_stats_path, stats)
        return sample, stats

    def save_round_metadata(
        self,
        *,
        proposal: ProposalDistribution,
        sample_count: int,
        seed: int,
        real_data_sources: Sequence[str | Path],
        status: str = "proposal_prepared",
        physical_only: bool = False,
        extra: dict | None = None,
    ) -> dict:
        metadata = {
            "round_number": self.round_number,
            "status": status,
            "track": self.track_config.name,
            "previous_checkpoint": None if self.previous_checkpoint is None else str(self.previous_checkpoint),
            "sample_count": int(sample_count),
            "seed": int(seed),
            "physical_only": bool(physical_only),
            "proposal": proposal.summary(),
            "real_data_sources": [str(path) for path in real_data_sources],
            "outputs": {
                "proposal_parameters": str(self.paths.proposal_parameters_path),
                "proposal_target_parameters": str(self.paths.proposal_target_parameters_path),
                "proposal_stats": str(self.paths.proposal_stats_path),
            },
            "extra": extra or {},
        }
        self._write_json(self.paths.metadata_path, metadata)
        return metadata

    def _sampling_stats(self, sample: ProposalSample, proposal: ProposalDistribution) -> dict:
        stats = {
            "count": int(sample.parameters.shape[0]),
            "source_counts": {
                name: int(np.sum(sample.source == name))
                for name in sorted(set(sample.source.tolist()))
            },
            "proposal": proposal.summary(),
            "parameters": {},
        }
        for column, name in enumerate(sample.parameter_names):
            values = sample.parameters[:, column]
            stats["parameters"][name] = {
                "min": float(np.min(values)),
                "q16": float(np.percentile(values, 16)),
                "median": float(np.percentile(values, 50)),
                "q84": float(np.percentile(values, 84)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        return stats

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
