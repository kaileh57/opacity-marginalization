"""CLI entry point for NPE training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sculptor.config import get_default_track_config, load_track_config
from sculptor.data_generation.reader import load_generated_dataset
from sculptor.npe.training import NPETrainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Directory containing generated shards.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--track")
    parser.add_argument("--config-path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed torch + numpy before building the model (weight init, DataLoader shuffle). "
        "Set distinct seeds for the seed-stability comparison; leave unset for legacy behavior.",
    )
    parser.add_argument(
        "--unbounded-flow",
        action="store_true",
        help="Train an unbounded flow and handle prior support at inference time.",
    )
    parser.add_argument("--embedding-arch", default="cnn",
                        choices=["cnn", "transformer", "resnet", "tcn", "gru", "mlp"],
                        help="Spectrum embedding architecture.")
    parser.add_argument("--discrepancy-template", default=None,
                        help="Path to a forward-model floor template (.npy, self-norm residual) for "
                             "misspecification-robust training. Off if unset.")
    parser.add_argument("--discrepancy-amplitude", type=float, default=1.0)
    parser.add_argument("--discrepancy-prob", type=float, default=0.6)
    parser.add_argument(
        "--mask-augment-prob", type=float, default=0.0,
        help="Journal item 9 (mask-robust training): with this probability per training "
        "spectrum, apply the DEPLOYMENT masked-pixel median-fill corruption (sampled real "
        "masks) to RAW flux BEFORE self-normalization, so the network learns to stay "
        "calibrated under apply_opmarg_real's preprocessing. 0 = off (default; no effect on "
        "any existing training).",
    )
    parser.add_argument(
        "--checkpoint-path", default=None,
        help="Preemption resilience: write a resumable checkpoint (model+optimizer+epoch+best-val) "
        "here every --checkpoint-every epochs, and resume from it on restart. Essential on "
        "spot/preemptible hosts; off if unset.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore an existing --checkpoint-path and train from epoch 0.",
    )
    parser.add_argument(
        "--mask-inputs", default=None,
        help="Directory of real *_mask.npy (+ optional *_observed.npy) files used to sample "
        "mask patterns when --mask-augment-prob > 0.",
    )
    args = parser.parse_args()

    import numpy as _np
    if args.seed is not None:
        # Seed the streams that make one training run differ from another: weight
        # initialization and DataLoader shuffle both draw from torch's global RNG;
        # numpy is seeded for defensiveness. The train/val split is deterministic
        # (first (1-val_split) rows) so the seeds isolate optimization stochasticity,
        # which is exactly the seed-stability axis the opacity-marginalization paper reports.
        _np.random.seed(int(args.seed))
        from sculptor.npe.training import torch as _torch
        if _torch is not None:
            _torch.manual_seed(int(args.seed))
        print(f"[run_training] seeded torch+numpy with seed={args.seed}", flush=True)

    dataset = load_generated_dataset(Path(args.train))
    from sculptor.npe.dataset import self_normalize as _selfnorm
    from sculptor.data_generation.opacity_perturbation import NUISANCE_PREFIX as _NUISANCE_PREFIX

    # Opacity-marginalized datasets (v7_opmarg) carry oppert_* NUISANCE columns
    # (per-draw opacity-field coefficients + line-list combo). They are recorded
    # for reproducibility only and must NEVER become flow targets (they also have
    # no prior bounds, which would KeyError the bounded flow). Drop them here so
    # the trainer marginalizes over the perturbations implicitly.
    _keep_cols = [i for i, n in enumerate(dataset.param_names) if not str(n).startswith(_NUISANCE_PREFIX)]
    if len(_keep_cols) != len(dataset.param_names):
        print(
            f"[run_training] dropping {len(dataset.param_names) - len(_keep_cols)} opacity-nuisance "
            f"columns ({_NUISANCE_PREFIX}*): NPE marginalizes over the opacity perturbations",
            flush=True,
        )
        dataset = dataset.__class__(
            spectra=dataset.spectra,
            params=dataset.params[:, _keep_cols],
            param_names=tuple(dataset.param_names[i] for i in _keep_cols),
            metadata=dataset.metadata,
        )
    _sp = dataset.spectra
    # STRICT filter: drop any row with a non-finite or non-positive pixel (SPMD drops ~10% such rows;
    # NPETrainer feeds spectra to the CNN without sanitizing, so partial-NaN rows NaN-out the loss).
    _keep = _np.all(_np.isfinite(_sp) & (_sp > 0.0), axis=1)
    if not bool(_keep.all()):
        print(f"[run_training] strict finite filter: keep {int(_keep.sum())}/{len(_keep)} "
              f"({100.0 * _keep.mean():.1f}%)", flush=True)
    _spk = _sp[_keep]
    # MASK AUGMENTATION (journal item 9, off by default). Deployment (apply_opmarg_real)
    # median-fills masked pixels (bad pixels + the NRS1/NRS2 gap) of the RAW flux before
    # self-normalization; a network trained only on clean spectra loses C/O calibration
    # under this preprocessing (item-6 stress: C/O cov90 0.90 -> 0.48). Here we corrupt a
    # high-probability subset of the RAW training spectra with the SAME real-mask median-fill
    # (identical logic to opmarg_deploy_preproc_stress.build_variant's real_mask_fill), so the
    # flow is trained UNDER the deployment preprocessing. Applied to raw flux BEFORE
    # self_normalize; each corrupted spectrum draws its own real mask -> dataset-level diversity.
    if args.mask_augment_prob and args.mask_augment_prob > 0.0:
        import glob as _glob
        _L = _spk.shape[1]
        _masks = []
        for _mf in sorted(_glob.glob(str(Path(args.mask_inputs) / "*_mask.npy"))):
            _m = _np.load(_mf).astype(bool)
            if _m.shape[0] != _L:
                continue
            _of = Path(_mf).with_name(Path(_mf).name.replace("_mask.npy", "_observed.npy"))
            if _of.exists():
                _obs = _np.load(_of).astype(float)
                _m = _m & _np.isfinite(_obs) & (_obs > 0)
            if _m.sum() >= 1000:
                _masks.append(_m)
        if not _masks:
            raise RuntimeError(
                f"--mask-augment-prob={args.mask_augment_prob} set but no usable *_mask.npy "
                f"(>=1000 good px, len {_L}) found in {args.mask_inputs}")
        _augrng = _np.random.default_rng(int(args.seed) if args.seed is not None else 0)
        _spk = _spk.astype(_np.float64).copy()
        _naug = 0
        for _i in range(len(_spk)):
            if _augrng.random() >= args.mask_augment_prob:
                continue
            _flux = _spk[_i]
            _good = _masks[int(_augrng.integers(0, len(_masks)))].copy()
            _good &= _np.isfinite(_flux) & (_flux > 0)
            if _good.sum() < 1000:
                _good = _np.isfinite(_flux) & (_flux > 0)
            _fill = _np.nanmedian(_flux[_good])
            _spk[_i] = _np.where(_good, _flux, _fill)
            _naug += 1
        print(
            f"[run_training] mask-augment: corrupted {_naug}/{len(_spk)} raw spectra "
            f"(p={args.mask_augment_prob}, {len(_masks)} real masks, median good-frac "
            f"{_np.median([m.mean() for m in _masks]):.3f}) before self-normalization",
            flush=True,
        )
    # SELF-NORMALIZE (the project's invariance; matches the diseq/production recipe). Raw ~1e-15 flux
    # into the CNN is ill-scaled; self_normalize (log -> zero-mean/unit-std) is what trains stably.
    _spn = _np.asarray([_selfnorm(s) for s in _spk], dtype=_np.float32)
    print(f"[run_training] self-normalized {len(_spn)} spectra; finite={bool(_np.all(_np.isfinite(_spn)))}", flush=True)
    dataset = dataset.__class__(
        spectra=_spn, params=dataset.params[_keep],
        param_names=dataset.param_names, metadata=dataset.metadata,
    )
    track_name = args.track or str(dataset.metadata.get("track", "g395h"))
    track_config = load_track_config(args.config_path) if args.config_path else get_default_track_config(track_name)
    trainer = NPETrainer(
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            embedding_arch=args.embedding_arch,
            discrepancy_template_path=args.discrepancy_template,
            discrepancy_amplitude=args.discrepancy_amplitude,
            discrepancy_prob=args.discrepancy_prob,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            resume=not args.no_resume,
        )
    )
    result = trainer.train(
        dataset.spectra,
        dataset.params,
        args.output,
        parameter_names=dataset.param_names,
        conditioning_names=track_config.conditioning_parameter_names,
        parameter_bounds=None
        if args.unbounded_flow
        else {prior.name: (prior.lower, prior.upper) for prior in track_config.parameter_priors},
    )
    print(result["checkpoint"])


if __name__ == "__main__":
    main()
