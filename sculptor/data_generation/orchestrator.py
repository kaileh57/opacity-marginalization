"""End-to-end orchestration for synthetic dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from sculptor.config import get_default_track_config, load_track_config
from sculptor.data_generation.chemistry_precompute import ChemistryPrecomputer
from sculptor.data_generation.noise import NoiseAugmentor
from sculptor.data_generation.opacity_perturbation import (
    COMBO_COLUMN,
    alt_species_from_env,
    append_perturbation_columns,
    combo_selection,
    sample_coefficients,
    settings_from_env,
    shard_combo,
)
from sculptor.data_generation.sampler import ParameterSampler
from sculptor.data_generation.tpu_generator import SpectrumBatchGenerator
from sculptor.data_generation.writer import ShardPayload, ShardWriter
from sculptor.forward_model.chemistry import ChemistryModel
from sculptor.forward_model.opacity import ExoJAXOpacityBackend
from sculptor.forward_model.spectrum import SpectrumModel
from sculptor.utils.gcs import exists, is_gcs_path, read_text, write_text
from sculptor.utils.logging import ThroughputTracker, get_logger


@dataclass(frozen=True)
class GenerationConfig:
    track: str = "g395h"
    config_path: str | None = None
    samples: int = 1024
    shard_size: int = 256
    output: str = "./artifacts/generation"
    manifest_path: str | None = None
    seed: int = 0
    fastchem_path: str | None = None
    device_batch_size: int | None = None
    max_rejection_attempts_per_shard: int = 32
    parameter_file: str | None = None


class GenerationOrchestrator:
    """Tie together sampling, chemistry, spectrum generation, noise, and writing."""

    def __init__(self, config: GenerationConfig, prebuilt_opacity_backend=None) -> None:
        self.config = config
        self.track_config = (
            load_track_config(config.config_path)
            if config.config_path is not None
            else get_default_track_config(config.track)
        )
        fastchem_path = config.fastchem_path or self.track_config.fastchem_path
        if not fastchem_path:
            raise ValueError(
                "FastChem grid required for physical-model generation, "
                "use --fastchem-path or set fastchem_path in config."
            )
        fastchem_path_obj = Path(fastchem_path)
        if not fastchem_path_obj.exists():
            raise FileNotFoundError(
                "FastChem grid required for physical-model generation, "
                f"but {fastchem_path_obj} does not exist. "
                "Use --fastchem-path or set fastchem_path in config."
            )
        self.config = replace(config, fastchem_path=str(fastchem_path_obj))
        chemistry_model = ChemistryModel.from_fastchem_hdf5(fastchem_path_obj, self.track_config)
        if prebuilt_opacity_backend is not None:
            opacity_backend = prebuilt_opacity_backend
        elif "EXOJAX_ROOT" in os.environ:
            opacity_backend = ExoJAXOpacityBackend(
                track_config=self.track_config,
                exojax_root=os.environ["EXOJAX_ROOT"],
            )
        else:
            opacity_backend = None
        self.spectrum_model = SpectrumModel(
            track_config=self.track_config,
            chemistry_model=chemistry_model,
            opacity_backend=opacity_backend,
        )
        self.sampler = ParameterSampler(self.track_config, seed=config.seed)
        self.precomputer = ChemistryPrecomputer(chemistry_model)
        # Opacity-marginalization (SCULPTOR_OPACITY_PERTURB=1): per-draw
        # perturbed opacities. Disabled (default) leaves this path bit-identical
        # to the unperturbed pipeline.
        self.perturbation = settings_from_env(self.track_config.molecules)
        self.alt_species = (
            alt_species_from_env(self.track_config.molecules) if self.perturbation.enabled else ()
        )
        self.generator = SpectrumBatchGenerator(
            self.spectrum_model,
            device_batch_size=config.device_batch_size,
            perturbation=self.perturbation,
        )
        self.noise = NoiseAugmentor(self.track_config, random_seed=config.seed)
        self.writer = ShardWriter()
        self.logger = get_logger(__name__)

    def _set_shard_linelist_combo(self, shard_index: int) -> dict[str, Any] | None:
        """Activate the line-list combo for a shard; return shard metadata (or None)."""
        if not self.perturbation.enabled:
            return None
        combo = shard_combo(shard_index, self.alt_species)
        selection = combo_selection(combo, self.alt_species)
        if self.alt_species:
            self.spectrum_model.opacity_backend.set_linelist_selection(selection)
        return {
            **self.perturbation.as_metadata(),
            "linelist_combo": combo,
            "linelist_selection": selection,
            "alt_species": list(self.alt_species),
        }

    def _extend_batch_with_perturbations(
        self,
        parameters: np.ndarray,
        parameter_names: tuple[str, ...],
        salt: int,
        combo: int,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Append nuisance columns for a sampled batch (salt = Halton cursor)."""
        coefficients = sample_coefficients(
            self.perturbation, len(parameters), seed=self.config.seed, salt=salt
        )
        combo_column = np.full(len(parameters), float(combo), dtype=np.float64)
        return append_perturbation_columns(
            parameters, parameter_names, self.perturbation, coefficients, combo_column
        )

    @property
    def manifest_path(self) -> str:
        if self.config.manifest_path is not None:
            return self.config.manifest_path
        return self._join_output_path("manifest.json")

    def _join_output_path(self, filename: str) -> str:
        if is_gcs_path(self.config.output):
            return f"{self.config.output.rstrip('/')}/{filename}"
        return str(Path(self.config.output) / filename)

    def _load_manifest(self) -> dict[str, Any]:
        if exists(self.manifest_path):
            return json.loads(read_text(self.manifest_path))
        return {"completed_shards": []}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        write_text(self.manifest_path, json.dumps(manifest, indent=2))

    def _log_rejected_rows(
        self,
        reason: str,
        parameters: np.ndarray,
        parameter_names: tuple[str, ...],
        spectra: np.ndarray,
        invalid_mask: np.ndarray,
    ) -> None:
        invalid_indices = np.flatnonzero(invalid_mask)
        for row_index in invalid_indices:
            parameter_vector = {
                name: float(parameters[row_index, column_index])
                for column_index, name in enumerate(parameter_names)
            }
            flux = np.asarray(spectra[row_index], dtype=np.float64)
            finite_flux = flux[np.isfinite(flux)]
            self.logger.warning(
                "Rejected invalid spectrum sample",
                extra={
                    "extra_fields": {
                        "reason": reason,
                        "parameters": parameter_vector,
                        "nan_count": int(np.isnan(flux).sum()),
                        "inf_count": int(np.isinf(flux).sum()),
                        "negative_count": int((flux < 0.0).sum()),
                        "flux_min": float(np.min(finite_flux)) if finite_flux.size else 0.0,
                        "flux_max": float(np.max(finite_flux)) if finite_flux.size else 0.0,
                    }
                },
            )

    def _valid_clean_mask(self, spectra_clean: np.ndarray) -> np.ndarray:
        finite = np.all(np.isfinite(spectra_clean), axis=1)
        non_negative = np.all(spectra_clean >= 0.0, axis=1)
        return finite & non_negative

    def _valid_noise_mask(self, spectra_noisy: np.ndarray, error_bars: np.ndarray) -> np.ndarray:
        # Noisy flux may scatter below zero, but it must remain finite.
        return np.all(np.isfinite(spectra_noisy), axis=1) & np.all(np.isfinite(error_bars), axis=1)

    def run(self) -> list[str]:
        if self.config.parameter_file is not None:
            return self._run_parameter_file()

        tracker = ThroughputTracker()
        manifest = self._load_manifest()
        self.logger.info(
            "Starting generation run",
            extra={
                "extra_fields": {
                    "samples": self.config.samples,
                    "shard_size": self.config.shard_size,
                    "output": self.config.output,
                    "manifest_path": self.manifest_path,
                    "opacity_backend": type(self.spectrum_model.opacity_backend).__name__,
                }
            },
        )
        completed = set(manifest.get("completed_shards", []))
        if "next_halton_index" in manifest:
            halton_cursor = int(manifest["next_halton_index"])
        elif completed:
            self.logger.warning(
                "Manifest missing next_halton_index; falling back to shard_size-based cursor, "
                "which can overlap when rejection filtering is enabled."
            )
            halton_cursor = (max(completed) + 1) * self.config.shard_size
        else:
            halton_cursor = 0
        destinations: list[str] = []
        total_shards = int(np.ceil(self.config.samples / self.config.shard_size))

        for shard_index in range(total_shards):
            if shard_index in completed:
                continue
            start = shard_index * self.config.shard_size
            count = min(self.config.shard_size, self.config.samples - start)
            attempts = 0
            rejected_total = 0
            accepted_clean: list[np.ndarray] = []
            accepted_noisy: list[np.ndarray] = []
            accepted_error_bars: list[np.ndarray] = []
            accepted_parameters: list[np.ndarray] = []
            accepted_vmr_profiles: list[np.ndarray] = []
            accepted_noise_metadata = {"snr": [], "calibration_amp": [], "tilt": [], "shift_pixels": []}
            parameter_names = self.track_config.parameter_names
            shard_perturbation_meta = self._set_shard_linelist_combo(shard_index)

            while sum(chunk.shape[0] for chunk in accepted_clean) < count:
                if attempts >= self.config.max_rejection_attempts_per_shard:
                    raise RuntimeError(
                        f"Exceeded rejection limit while filling shard {shard_index}; "
                        f"accepted {sum(chunk.shape[0] for chunk in accepted_clean)}/{count} spectra."
                    )
                attempts += 1
                remaining = count - sum(chunk.shape[0] for chunk in accepted_clean)
                batch_salt = halton_cursor
                batch = self.sampler.sample(count=remaining, start_index=halton_cursor)
                halton_cursor = batch.next_index
                batch_parameters = batch.parameters
                parameter_names = batch.parameter_names
                if shard_perturbation_meta is not None:
                    batch_parameters, parameter_names = self._extend_batch_with_perturbations(
                        batch_parameters,
                        parameter_names,
                        salt=batch_salt,
                        combo=int(shard_perturbation_meta["linelist_combo"]),
                    )
                precomputed = self.precomputer.precompute(batch_parameters, parameter_names)
                generated = self.generator.generate(precomputed)

                invalid_clean_mask = ~self._valid_clean_mask(generated.spectra_clean)
                if np.any(invalid_clean_mask):
                    rejected_total += int(np.sum(invalid_clean_mask))
                    self._log_rejected_rows(
                        reason="invalid_clean_flux",
                        parameters=generated.parameters,
                        parameter_names=generated.parameter_names,
                        spectra=generated.spectra_clean,
                        invalid_mask=invalid_clean_mask,
                    )
                valid_generated = generated.select_rows(~invalid_clean_mask)
                if len(valid_generated.parameters) == 0:
                    continue

                snr_index = valid_generated.parameter_names.index("snr")
                noise = self.noise.augment_batch(
                    valid_generated.spectra_clean,
                    valid_generated.parameters[:, snr_index],
                )
                invalid_noise_mask = ~self._valid_noise_mask(noise.spectra_noisy, noise.error_bars)
                if np.any(invalid_noise_mask):
                    rejected_total += int(np.sum(invalid_noise_mask))
                    self._log_rejected_rows(
                        reason="invalid_noisy_flux",
                        parameters=valid_generated.parameters,
                        parameter_names=valid_generated.parameter_names,
                        spectra=noise.spectra_noisy,
                        invalid_mask=invalid_noise_mask,
                    )
                valid_noise_mask = ~invalid_noise_mask
                if not np.any(valid_noise_mask):
                    continue

                accepted_clean.append(valid_generated.spectra_clean[valid_noise_mask])
                accepted_noisy.append(noise.spectra_noisy[valid_noise_mask])
                accepted_error_bars.append(noise.error_bars[valid_noise_mask])
                accepted_parameters.append(valid_generated.parameters[valid_noise_mask])
                accepted_vmr_profiles.append(valid_generated.vmr_profiles[valid_noise_mask])
                for key in accepted_noise_metadata:
                    accepted_noise_metadata[key].append(noise.metadata[key][valid_noise_mask])

            spectra_clean = np.concatenate(accepted_clean, axis=0)[:count]
            spectra_noisy = np.concatenate(accepted_noisy, axis=0)[:count]
            error_bars = np.concatenate(accepted_error_bars, axis=0)[:count]
            parameters = np.concatenate(accepted_parameters, axis=0)[:count]
            vmr_profiles = np.concatenate(accepted_vmr_profiles, axis=0)[:count]
            noise_metadata = {
                key: np.concatenate(values, axis=0)[:count]
                for key, values in accepted_noise_metadata.items()
            }

            destination = self._join_output_path(f"shard_{shard_index:04d}.h5")
            payload = ShardPayload(
                spectra=spectra_noisy,
                spectra_clean=spectra_clean,
                params=parameters.astype(np.float32),
                vmr_profiles=vmr_profiles.astype(np.float32),
                nu_grid=self.spectrum_model.track_config.nu_grid_cm1.astype(np.float32),
                param_names=parameter_names,
                metadata={
                    "track": self.track_config.name,
                    "seed": self.config.seed,
                    "shard_index": shard_index,
                    "noise": {key: value.tolist() for key, value in noise_metadata.items()},
                    "rejected_samples": rejected_total,
                    "device_batch_size": self.generator.device_batch_size,
                    # Only present on perturbed runs: flag-off shards stay
                    # byte-identical to the pre-perturbation pipeline.
                    **(
                        {"opacity_perturbation": shard_perturbation_meta}
                        if shard_perturbation_meta is not None
                        else {}
                    ),
                },
                error_bars=error_bars,
            )
            written = self.writer.write(payload, destination)
            destinations.append(written)
            completed.add(shard_index)
            manifest["completed_shards"] = sorted(completed)
            manifest["config"] = {
                **asdict(self.config),
                "device_batch_size": self.generator.device_batch_size,
            }
            manifest["next_halton_index"] = halton_cursor
            self._save_manifest(manifest)

            rate = tracker.update(count)
            self.logger.info(
                "Generated shard",
                extra={
                    "extra_fields": {
                        "shard_index": shard_index,
                        "samples": count,
                        "throughput": rate,
                        "rejected_samples": rejected_total,
                    }
                },
            )
        return destinations

    def _run_parameter_file(self) -> list[str]:
        parameter_file = Path(self.config.parameter_file or "")
        if not parameter_file.exists():
            raise FileNotFoundError(f"Parameter file does not exist: {parameter_file}")
        explicit_parameters = np.load(parameter_file).astype(np.float64)
        parameter_names = self.track_config.parameter_names
        if explicit_parameters.ndim != 2:
            raise ValueError(f"Parameter file must be a 2D array, got shape {explicit_parameters.shape}.")
        if self.perturbation.enabled:
            # Perturbed explicit-parameter runs must supply the nuisance columns
            # (field coefficients + line-list combo) alongside the physical
            # parameters, in the canonical order.
            extended_names = tuple(parameter_names) + self.perturbation.column_names()
            if explicit_parameters.shape[1] != len(extended_names):
                raise ValueError(
                    f"SCULPTOR_OPACITY_PERTURB=1: parameter file has "
                    f"{explicit_parameters.shape[1]} columns but {len(extended_names)} are required "
                    f"({len(parameter_names)} physical + {len(self.perturbation.column_names())} "
                    f"oppert_* nuisance columns): {extended_names}"
                )
            parameter_names = extended_names
        elif explicit_parameters.shape[1] != len(parameter_names):
            raise ValueError(
                f"Parameter file has {explicit_parameters.shape[1]} columns, "
                f"but config expects {len(parameter_names)} parameters: {parameter_names}"
            )
        requested = int(self.config.samples)
        if explicit_parameters.shape[0] < requested:
            raise ValueError(
                f"Parameter file has {explicit_parameters.shape[0]} rows, "
                f"but {requested} output spectra were requested."
            )

        tracker = ThroughputTracker()
        manifest = self._load_manifest()
        completed = set(manifest.get("completed_shards", []))
        if "explicit_candidate_cursor" in manifest:
            candidate_cursor = int(manifest["explicit_candidate_cursor"])
        elif "next_halton_index" in manifest:
            candidate_cursor = int(manifest["next_halton_index"])
        elif completed:
            self.logger.warning(
                "Manifest missing explicit candidate cursor; falling back to shard_size-based cursor, "
                "which can overlap when rejection filtering is enabled."
            )
            candidate_cursor = (max(completed) + 1) * self.config.shard_size
        else:
            candidate_cursor = 0
        destinations: list[str] = []
        total_shards = int(np.ceil(requested / self.config.shard_size))
        self.logger.info(
            "Starting explicit-parameter generation run",
            extra={
                "extra_fields": {
                    "samples": requested,
                    "shard_size": self.config.shard_size,
                    "output": self.config.output,
                    "parameter_file": str(parameter_file),
                    "manifest_path": self.manifest_path,
                }
            },
        )

        for shard_index in range(total_shards):
            if shard_index in completed:
                continue
            start = shard_index * self.config.shard_size
            count = min(self.config.shard_size, requested - start)
            rejected_total = 0
            accepted_clean: list[np.ndarray] = []
            accepted_noisy: list[np.ndarray] = []
            accepted_error_bars: list[np.ndarray] = []
            accepted_parameters: list[np.ndarray] = []
            accepted_vmr_profiles: list[np.ndarray] = []
            accepted_noise_metadata = {"snr": [], "calibration_amp": [], "tilt": [], "shift_pixels": []}

            while sum(chunk.shape[0] for chunk in accepted_clean) < count:
                if candidate_cursor >= explicit_parameters.shape[0]:
                    raise RuntimeError(
                        f"Explicit parameter file exhausted at row {candidate_cursor}; "
                        f"accepted {sum(chunk.shape[0] for chunk in accepted_clean)}/{count} "
                        f"for shard {shard_index}. Provide a larger candidate pool."
                    )
                remaining = count - sum(chunk.shape[0] for chunk in accepted_clean)
                draw_count = min(
                    max(remaining, self.generator.device_batch_size),
                    explicit_parameters.shape[0] - candidate_cursor,
                )
                parameters = explicit_parameters[candidate_cursor : candidate_cursor + draw_count]
                candidate_cursor += draw_count
                if self.perturbation.enabled:
                    # The compiled spectrum function embeds one line-list combo as a
                    # trace-time constant, so each draw chunk must be combo-uniform.
                    combo_values = np.unique(
                        np.rint(parameters[:, parameter_names.index(COMBO_COLUMN)]).astype(int)
                    )
                    if combo_values.size != 1:
                        raise ValueError(
                            f"Explicit parameter rows {candidate_cursor - draw_count}:{candidate_cursor} "
                            f"mix line-list combos {combo_values.tolist()}; perturbed explicit files "
                            f"must be combo-uniform within each draw chunk (sort rows by "
                            f"{COMBO_COLUMN!r})."
                        )
                    if self.alt_species:
                        self.spectrum_model.opacity_backend.set_linelist_selection(
                            combo_selection(int(combo_values[0]), self.alt_species)
                        )
                precomputed = self.precomputer.precompute(parameters, parameter_names)
                generated = self.generator.generate(precomputed)

                invalid_clean_mask = ~self._valid_clean_mask(generated.spectra_clean)
                if np.any(invalid_clean_mask):
                    rejected_total += int(np.sum(invalid_clean_mask))
                    self._log_rejected_rows(
                        reason="invalid_clean_flux_explicit_parameter_file",
                        parameters=generated.parameters,
                        parameter_names=generated.parameter_names,
                        spectra=generated.spectra_clean,
                        invalid_mask=invalid_clean_mask,
                    )
                valid_generated = generated.select_rows(~invalid_clean_mask)
                if len(valid_generated.parameters) == 0:
                    continue

                snr_index = valid_generated.parameter_names.index("snr")
                noise = self.noise.augment_batch(
                    valid_generated.spectra_clean,
                    valid_generated.parameters[:, snr_index],
                )
                invalid_noise_mask = ~self._valid_noise_mask(noise.spectra_noisy, noise.error_bars)
                if np.any(invalid_noise_mask):
                    rejected_total += int(np.sum(invalid_noise_mask))
                    self._log_rejected_rows(
                        reason="invalid_noisy_flux_explicit_parameter_file",
                        parameters=valid_generated.parameters,
                        parameter_names=valid_generated.parameter_names,
                        spectra=noise.spectra_noisy,
                        invalid_mask=invalid_noise_mask,
                    )
                valid_noise_mask = ~invalid_noise_mask
                if not np.any(valid_noise_mask):
                    continue

                accepted_clean.append(valid_generated.spectra_clean[valid_noise_mask])
                accepted_noisy.append(noise.spectra_noisy[valid_noise_mask])
                accepted_error_bars.append(noise.error_bars[valid_noise_mask])
                accepted_parameters.append(valid_generated.parameters[valid_noise_mask])
                accepted_vmr_profiles.append(valid_generated.vmr_profiles[valid_noise_mask])
                for key in accepted_noise_metadata:
                    accepted_noise_metadata[key].append(noise.metadata[key][valid_noise_mask])

            spectra_clean = np.concatenate(accepted_clean, axis=0)[:count]
            spectra_noisy = np.concatenate(accepted_noisy, axis=0)[:count]
            error_bars = np.concatenate(accepted_error_bars, axis=0)[:count]
            parameters = np.concatenate(accepted_parameters, axis=0)[:count]
            vmr_profiles = np.concatenate(accepted_vmr_profiles, axis=0)[:count]
            noise_metadata = {
                key: np.concatenate(values, axis=0)[:count]
                for key, values in accepted_noise_metadata.items()
            }

            destination = self._join_output_path(f"shard_{shard_index:04d}.h5")
            payload = ShardPayload(
                spectra=spectra_noisy,
                spectra_clean=spectra_clean,
                params=parameters.astype(np.float32),
                vmr_profiles=vmr_profiles.astype(np.float32),
                nu_grid=self.spectrum_model.track_config.nu_grid_cm1.astype(np.float32),
                param_names=parameter_names,
                metadata={
                    "track": self.track_config.name,
                    "seed": self.config.seed,
                    "shard_index": shard_index,
                    "parameter_source": "explicit_parameter_file",
                    "parameter_file": str(parameter_file),
                    "parameter_rows": [int(start), int(start + count)],
                    "explicit_candidate_cursor": int(candidate_cursor),
                    **(
                        {
                            "opacity_perturbation": {
                                **self.perturbation.as_metadata(),
                                "alt_species": list(self.alt_species),
                                "linelist_source": "parameter_column",
                            }
                        }
                        if self.perturbation.enabled
                        else {}
                    ),
                    "noise": {key: value.tolist() for key, value in noise_metadata.items()},
                    "rejected_samples": rejected_total,
                    "device_batch_size": self.generator.device_batch_size,
                },
                error_bars=error_bars,
            )
            written = self.writer.write(payload, destination)
            destinations.append(written)
            completed.add(shard_index)
            manifest["completed_shards"] = sorted(completed)
            manifest["config"] = {
                **asdict(self.config),
                "samples": requested,
                "device_batch_size": self.generator.device_batch_size,
                "parameter_source": "explicit_parameter_file",
            }
            manifest["parameter_count"] = requested
            manifest["next_halton_index"] = candidate_cursor
            manifest["explicit_candidate_cursor"] = candidate_cursor
            self._save_manifest(manifest)
            rate = tracker.update(count)
            self.logger.info(
                "Generated explicit-parameter shard",
                extra={
                    "extra_fields": {
                        "shard_index": shard_index,
                        "samples": count,
                        "throughput": rate,
                        "rejected_samples": rejected_total,
                    }
                },
            )
        return destinations
