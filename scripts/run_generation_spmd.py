#!/usr/bin/env python3
"""8-way SPMD data generation for a v4-64 TPU pod (8 hosts x 4 chips).

ALL workers generate concurrently on their own 4 local TPU chips. The shard
space is split by ``shard_idx % process_count == process_index`` so every host
owns a disjoint subset; the union of all hosts equals a single-writer run.

Why SPMD (not the single-writer run_generation_distributed wrapper): the
single-writer design wastes 7/8 of the pod (workers 1-7 only poll). Here each
host saturates its local TPU. Critically (Infra lesson, SWAP_LEDGER):

  (i)  ALL workers exit SYMMETRICALLY. There is NO worker-0-only in-process
       gather/poll loop after the work -- an asymmetric exit makes the late
       worker reach the jax shutdown barrier seconds after the others and the
       pod dies with "Shutdown barrier has failed" / DEADLINE_EXCEEDED. Each
       worker writes its own GCS done-marker and returns. Manifest assembly is
       done OUT-OF-BAND from the laptop via scripts/check_generation_spmd.py.

  (ii) Generation uses the jitted/vmapped path (device_batch_size > 1). The
       scalar eager path (device_batch_size == 1) OOMs (>400 GB) -- never use it.

DETERMINISM ACROSS WORKERS
--------------------------
The orchestrator's parameter sampling is ORDER-DEPENDENT: ParameterSampler.sample
draws Halton blocks of ``max(count, 16)`` and chains a cursor (``next_index``)
shard-by-shard, so shard N's parameters depend on shards 0..N-1. We cannot simply
sample one big table (different block size -> different cursor arithmetic -> a
different table; verified locally). Instead every worker REPLAYS the orchestrator's
exact per-shard cursor chain in pure numpy (cheap) to build the identical full
parameter table, then generates ONLY its modulo-assigned shards from that table.
Same seed -> same Halton draws -> same is_physical_profile rejection -> identical
table on every host -> byte-compatible with single-writer v3 shards.

Resume: before generating a shard, check GCS for the shard h5; skip if present.

Opacity: env-driven staged cache (SCULPTOR_OPA_CACHE_DIR + SCULPTOR_READ_OPA_CACHE=1)
is the primary path; the live OpaPremodit build is the automatic fallback (the
backend reads those env vars itself inside prepare()).
"""
from __future__ import annotations

import os
import sys as _sys

# EXOJAX_ROOT mirrors run_generation_distributed so the live-build fallback and
# the FastChem/opacity loaders resolve their data directory identically.
_default_exojax_root = os.path.expanduser("~/exojax_db")
if not os.path.isdir(_default_exojax_root) and os.path.isdir("/tmp/exojax_db"):
    _default_exojax_root = "/tmp/exojax_db"
os.environ.setdefault("EXOJAX_ROOT", _default_exojax_root)

import argparse
import json
import socket
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from sculptor.config import get_default_track_config, load_track_config  # noqa: E402
from sculptor.data_generation.opacity_perturbation import (  # noqa: E402
    alt_species_from_env,
    append_perturbation_columns,
    combo_selection,
    forced_combo_from_env,
    sample_coefficients,
    settings_from_env,
    shard_combo,
    shard_combo_column,
)
from sculptor.data_generation.orchestrator import GenerationConfig  # noqa: E402
from sculptor.data_generation.sampler import ParameterSampler  # noqa: E402
from sculptor.utils.gcs import exists, is_gcs_path, write_text  # noqa: E402


def _log(**fields) -> None:
    print(json.dumps(fields), flush=True)


def _join(output: str, filename: str) -> str:
    if is_gcs_path(output):
        return f"{output.rstrip('/')}/{filename}"
    return str(Path(output) / filename)


def build_parameter_table(
    sampler: ParameterSampler, num_spectra: int, shard_size: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Replay the orchestrator's exact per-shard cursor chain.

    Returns the full (num_spectra, n_params) table plus parameter names. This
    is pure deterministic numpy: every worker produces an identical table from
    the same seed, and the table is byte-compatible with a single-writer
    GenerationOrchestrator.run() over the same config.
    """
    total_shards = int(np.ceil(num_spectra / shard_size))
    cursor = 0
    rows: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for shard_index in range(total_shards):
        start = shard_index * shard_size
        count = min(shard_size, num_spectra - start)
        batch = sampler.sample(count=count, start_index=cursor)
        cursor = batch.next_index
        rows.append(batch.parameters)
        names = batch.parameter_names
    table = np.concatenate(rows, axis=0) if rows else np.empty((0, 0), dtype=np.float64)
    assert names is not None, "empty parameter table"
    return table, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="g395h")
    parser.add_argument("--config-path", "--config", dest="config_path")
    parser.add_argument("--num-spectra", "--samples", dest="num_spectra", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--output", required=True, help="GCS prefix (or local dir) for shard_*.h5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fastchem-path")
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument(
        "--process-index",
        type=int,
        help=(
            "Manual SPMD rank. Requires --process-count and SKIPS "
            "jax.distributed.initialize(). Use ONLY when the pod cannot form an "
            "8-host TPU mesh (dead/occupied workers) and generation therefore runs "
            "on host CPU (JAX_PLATFORMS=cpu) over a worker SUBSET. The shard split, "
            "parameter-table replay and GCS rendezvous are unchanged, so the union "
            "of the subset's shards is byte-compatible with a full SPMD run."
        ),
    )
    parser.add_argument("--process-count", type=int, help="Manual SPMD world size (see --process-index).")
    args = parser.parse_args()

    if args.device_batch_size < 2:
        raise ValueError(
            "device-batch-size must be >= 2: the scalar eager path (==1) OOMs (>400 GB). "
            "Use the jitted/vmapped path."
        )
    manual_rank = args.process_index is not None or args.process_count is not None
    if manual_rank and (args.process_index is None or args.process_count is None):
        raise ValueError("--process-index and --process-count must be given together.")

    output = args.output.rstrip("/")

    import jax

    if manual_rank:
        if not (0 <= args.process_index < args.process_count):
            raise ValueError(
                f"--process-index {args.process_index} out of range for "
                f"--process-count {args.process_count}"
            )
        process_index = int(args.process_index)
        process_count = int(args.process_count)
        _log(
            msg="skipping_distributed_initialize",
            reason="manual_process_rank",
            hostname=socket.gethostname(),
            jax_platforms=os.environ.get("JAX_PLATFORMS", ""),
        )
    else:
        _log(msg="calling_distributed_initialize", hostname=socket.gethostname())
        # Init window must cover the pod-wide launch stagger + per-worker staging
        # variance (workers are launched by sequential single SSH connections).
        jax.distributed.initialize(
            initialization_timeout=int(os.environ.get("SPMD_INIT_TIMEOUT_S", "600"))
        )
        process_index = int(jax.process_index())
        process_count = int(jax.process_count())
    _log(
        msg="jax_ready",
        process_index=process_index,
        process_count=process_count,
        device_count=int(jax.device_count()),
        local_device_count=int(jax.local_device_count()),
        hostname=socket.gethostname(),
    )

    # ---- Track config + deterministic parameter table (identical on every host) ----
    track_config = (
        load_track_config(args.config_path)
        if args.config_path
        else get_default_track_config(args.track)
    )
    fastchem_path = args.fastchem_path or track_config.fastchem_path
    if not fastchem_path or not Path(fastchem_path).exists():
        raise FileNotFoundError(
            "FastChem grid required for physical-model generation; "
            f"--fastchem-path or config value missing/not found: {fastchem_path!r}"
        )

    sampler = ParameterSampler(track_config, seed=args.seed)
    parameters, parameter_names = build_parameter_table(sampler, args.num_spectra, args.shard_size)
    total_shards = int(np.ceil(args.num_spectra / args.shard_size))

    # ---- Opacity-marginalization (SCULPTOR_OPACITY_PERTURB=1) ----
    # Nuisance columns are appended AFTER the Halton replay so the physical
    # parameter draws are untouched; the coefficient block comes from ONE rng
    # over the full table, so every worker reproduces the identical extension.
    # Line-list combos map to shards as combo = shard % 2**n_alt; with 8 pod
    # workers and shard % 8 == worker, each worker compiles exactly one combo.
    perturbation = settings_from_env(track_config.molecules)
    alt_species = alt_species_from_env(track_config.molecules) if perturbation.enabled else ()
    # SCULPTOR_FORCE_COMBO (ablation): pin the discrete line-list combo for every
    # shard instead of the shard % n_combos cycle. None = production behaviour.
    forced_combo = forced_combo_from_env(alt_species) if perturbation.enabled else None
    if perturbation.enabled:
        coefficients = sample_coefficients(perturbation, len(parameters), seed=args.seed, salt=0)
        combo_column = shard_combo_column(len(parameters), args.shard_size, alt_species)
        parameters, parameter_names = append_perturbation_columns(
            parameters, parameter_names, perturbation, coefficients, combo_column
        )
        _log(
            msg="opacity_perturbation_enabled",
            process_index=process_index,
            n_modes=perturbation.n_modes,
            priors_source=perturbation.priors_source,
            alt_species=list(alt_species),
            n_combos=2 ** len(alt_species),
            forced_combo=forced_combo,
            n_nuisance_columns=len(perturbation.column_names()),
        )

    snr_index = parameter_names.index("snr")
    my_shards = [s for s in range(total_shards) if s % process_count == process_index]
    _log(
        msg="parameter_table_ready",
        process_index=process_index,
        n_params=int(parameters.shape[1]),
        total_shards=total_shards,
        n_my_shards=len(my_shards),
        my_shards=my_shards[:8] + (["..."] if len(my_shards) > 8 else []),
    )

    # ---- Forward-model objects (this host's local TPU only; no collectives) ----
    from sculptor.data_generation.chemistry_precompute import ChemistryPrecomputer
    from sculptor.data_generation.noise import NoiseAugmentor
    from sculptor.data_generation.tpu_generator import SpectrumBatchGenerator
    from sculptor.data_generation.writer import ShardPayload, ShardWriter
    from sculptor.forward_model.chemistry import ChemistryModel
    from sculptor.forward_model.opacity import ExoJAXOpacityBackend
    from sculptor.forward_model.spectrum import SpectrumModel

    chemistry_model = ChemistryModel.from_fastchem_hdf5(Path(fastchem_path), track_config)
    opacity_backend = None
    exojax_root = os.environ.get("EXOJAX_ROOT")
    if exojax_root and os.path.isdir(exojax_root):
        opacity_backend = ExoJAXOpacityBackend(track_config=track_config, exojax_root=exojax_root)
        _log(msg="opacity_prepare_start", process_index=process_index,
             opa_cache_dir=os.environ.get("SCULPTOR_OPA_CACHE_DIR"),
             read_opa_cache=os.environ.get("SCULPTOR_READ_OPA_CACHE", "0"))
        opacity_backend.prepare()
        _log(msg="opacity_prepare_done", process_index=process_index)
    else:
        raise RuntimeError(f"EXOJAX_ROOT not set or missing: {exojax_root!r}")

    spectrum_model = SpectrumModel(
        track_config=track_config,
        chemistry_model=chemistry_model,
        opacity_backend=opacity_backend,
    )
    precomputer = ChemistryPrecomputer(chemistry_model)
    generator = SpectrumBatchGenerator(
        spectrum_model, device_batch_size=args.device_batch_size, perturbation=perturbation
    )
    writer = ShardWriter()

    # Per-shard write tempfile must be unique per host: ShardWriter uses a fixed
    # "_sculptor_tmp_shard.h5" in cwd. One process per host on a pod, but make the
    # cwd unique anyway so concurrent local testing never collides.
    work_dir = Path(f"_spmd_gen_w{process_index}")
    work_dir.mkdir(parents=True, exist_ok=True)
    prev_cwd = Path.cwd()
    os.chdir(work_dir)

    generation_config = GenerationConfig(
        track=args.track,
        config_path=args.config_path,
        samples=args.num_spectra,
        shard_size=args.shard_size,
        output=output,
        seed=args.seed,
        fastchem_path=str(fastchem_path),
        device_batch_size=args.device_batch_size,
    )
    config_meta = {**asdict(generation_config), "device_batch_size": generator.device_batch_size}

    written: list[str] = []
    skipped: list[int] = []
    t_start = time.time()
    try:
        for shard_index in my_shards:
            destination = _join(output, f"shard_{shard_index:04d}.h5")
            if exists(destination):
                skipped.append(shard_index)
                _log(msg="shard_skip_exists", process_index=process_index, shard_index=shard_index)
                continue

            start = shard_index * args.shard_size
            count = min(args.shard_size, args.num_spectra - start)
            shard_parameters = parameters[start : start + count]

            shard_perturbation_meta = None
            if perturbation.enabled:
                combo = shard_combo(shard_index, alt_species)
                selection = combo_selection(combo, alt_species)
                if alt_species:
                    opacity_backend.set_linelist_selection(selection)
                shard_perturbation_meta = {
                    **perturbation.as_metadata(),
                    "linelist_combo": combo,
                    "linelist_selection": selection,
                    "alt_species": list(alt_species),
                    "forced_combo": forced_combo,
                }

            precomputed = precomputer.precompute(shard_parameters, parameter_names)
            generated = generator.generate(precomputed)
            spectra_clean = np.asarray(generated.spectra_clean, dtype=np.float32)
            vmr_profiles = np.asarray(generated.vmr_profiles, dtype=np.float32)

            # Per-shard noise RNG (seed + shard_index): deterministic and
            # independent of host execution order. DEVIATION from the orchestrator,
            # which uses one RNG advanced across all shards in order -- impossible
            # to reproduce when hosts generate disjoint shard subsets concurrently.
            noise = NoiseAugmentor(track_config, random_seed=args.seed + shard_index)
            noise_batch = noise.augment_batch(spectra_clean, shard_parameters[:, snr_index])
            spectra_noisy = np.asarray(noise_batch.spectra_noisy, dtype=np.float32)
            error_bars = np.asarray(noise_batch.error_bars, dtype=np.float32)
            noise_metadata = {k: np.asarray(v) for k, v in noise_batch.metadata.items()}

            payload = ShardPayload(
                spectra=spectra_noisy,
                spectra_clean=spectra_clean,
                params=shard_parameters.astype(np.float32),
                vmr_profiles=vmr_profiles,
                nu_grid=spectrum_model.track_config.nu_grid_cm1.astype(np.float32),
                param_names=parameter_names,
                metadata={
                    "track": track_config.name,
                    "seed": args.seed,
                    "shard_index": shard_index,
                    "noise": {k: v.tolist() for k, v in noise_metadata.items()},
                    "noise_seed": args.seed + shard_index,
                    "rejected_samples": 0,
                    "device_batch_size": generator.device_batch_size,
                    "spmd_process_index": process_index,
                    "spmd_process_count": process_count,
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
            written_path = writer.write(payload, destination)
            written.append(written_path)
            _log(
                msg="shard_written",
                process_index=process_index,
                shard_index=shard_index,
                count=int(count),
                destination=written_path,
                elapsed_s=round(time.time() - t_start, 1),
            )
    except Exception as exc:
        os.chdir(prev_cwd)
        marker = f"{output}/done_spmd/worker_{process_index}.json"
        write_text(
            marker,
            json.dumps(
                {
                    "status": "failure",
                    "process_index": process_index,
                    "process_count": process_count,
                    "hostname": socket.gethostname(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "written": written,
                    "skipped": skipped,
                    "completed_at": time.time(),
                },
                indent=2,
            ),
        )
        _log(msg="worker_failed", process_index=process_index, error=str(exc))
        raise

    os.chdir(prev_cwd)

    # Symmetric exit: every worker writes its own marker and returns. No gather.
    marker = f"{output}/done_spmd/worker_{process_index}.json"
    write_text(
        marker,
        json.dumps(
            {
                "status": "success",
                "process_index": process_index,
                "process_count": process_count,
                "hostname": socket.gethostname(),
                "assigned_shards": my_shards,
                "written": written,
                "skipped": skipped,
                "num_spectra": args.num_spectra,
                "shard_size": args.shard_size,
                "total_shards": total_shards,
                "param_names": list(parameter_names),
                "config": config_meta,
                **(
                    {
                        "opacity_perturbation": {
                            **perturbation.as_metadata(),
                            "alt_species": list(alt_species),
                            "forced_combo": forced_combo,
                        }
                    }
                    if perturbation.enabled
                    else {}
                ),
                "completed_at": time.time(),
            },
            indent=2,
        ),
    )
    _log(
        msg="worker_done",
        process_index=process_index,
        n_written=len(written),
        n_skipped=len(skipped),
        elapsed_s=round(time.time() - t_start, 1),
    )

    # EXIT RENDEZVOUS (learned 2026-06-11, 100K run): symmetric exit CODE is not enough on a
    # pod — exits must be symmetric in TIME. When the fastest workers finished ~30 min before
    # the slowest, their jax shutdown started the coordination-service barrier, which timed out
    # (DEADLINE_EXCEEDED -> coredump cascade) and killed the still-working workers at 741/782
    # shards. Fix: every worker, after writing its own marker, polls GCS until ALL workers'
    # done markers exist, so all 8 reach the interpreter exit (and the jax shutdown barrier)
    # within one poll interval of each other. On timeout we exit anyway — shards are already
    # uploaded per-shard, so a barrier abort at that point loses nothing.
    rendezvous_deadline = time.time() + float(os.environ.get("SPMD_RENDEZVOUS_TIMEOUT_S", "10800"))
    poll_s = 20.0
    while time.time() < rendezvous_deadline:
        missing = [
            i for i in range(process_count)
            if not exists(f"{output}/done_spmd/worker_{i}.json")
        ]
        if not missing:
            _log(msg="rendezvous_complete", process_index=process_index)
            break
        time.sleep(poll_s)
    else:
        _log(msg="rendezvous_timeout", process_index=process_index)


if __name__ == "__main__":
    main()
