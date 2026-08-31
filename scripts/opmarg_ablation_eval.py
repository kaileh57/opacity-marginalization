#!/usr/bin/env python3
"""Coverage eval for the discrete-vs-smooth opacity-perturbation ablation.

Evaluates an arbitrary set of NPE models -- single models and/or pooled-sample
seed ensembles -- on one or more held-out shard directories, and reports the two
numbers the ablation turns on: C/O coverage @90% and overall (mean-over-target)
coverage @90%.

Two evaluation modes:

  RAW (``--n-obj N``)          -- one pass over N objects, raw posterior quantile
                                  intervals. Used for the stage-1 512-object
                                  discrete-only / smooth-only test sets.
  SPLIT (``--n-cal C --n-test T``) -- the STANDARD factorial protocol: load C+T
                                  objects, split them with the same deterministic
                                  permutation seed as opmarg_factorial_eval.py /
                                  opmarg_recal_benchmark.py, fit the quantile
                                  recalibrator on the cal half and report BOTH raw
                                  and recalibrated coverage on the test half.
                                  Used for stage 2 on the 2048-object perturbed
                                  factorial split.

Each distinct model directory is sampled ONCE per set and reused by every
ensemble that contains it, so an N-seed ensemble costs no extra forward passes.
Ensembles pool draws along the sample axis (identical to opmarg_factorial_eval).

Spectra are self-normalized here exactly as the trainer does (models record
spectrum_normalization="none" and are fed self-normalized inputs).

Usage:
  python scripts/opmarg_ablation_eval.py \
      --set discrete_only=/data/discrete --set smooth_only=/data/smooth \
      --model marg_seed0=/models/npe_v7_opmarg_seed0 \
      --model control=/models/npe_v6_cloud \
      --ensemble marg_ensemble3=/models/npe_v7_opmarg_seed0,/models/npe_v7_opmarg_seed1,/models/npe_v7_opmarg_seed2 \
      --n-obj 512 --samples 400 --out stage1_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sculptor.npe.dataset import self_normalize  # noqa: E402
from sculptor.npe.inference import PosteriorInferencer  # noqa: E402
from sculptor.npe.recalibration import QuantileRecalibrator  # noqa: E402

LEVELS = (0.5, 0.68, 0.9, 0.95)
# Same deterministic cal/test split seed as opmarg_recal_benchmark.py and
# opmarg_factorial_eval.py, so stage-2 numbers sit on the STANDARD split.
SPLIT_SEED = 20260716
CO_NAME = "carbon_to_oxygen"


def parse_named(value: str, what: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--{what} expects NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"--{what} expects NAME=PATH, got {value!r}")
    return name, path


def load_heldout(shard_dir: Path, target_names, cond_names, input_length: int, n_obj: int):
    """Spectra (self-normalized), physical truths and conditioning from shard_*.h5.

    Nuisance (oppert_*) columns are ignored: the truths are the physical
    parameters that generated the perturbed spectrum. Strict finite filter
    matches the trainer's.
    """
    paths = sorted(Path(shard_dir).glob("shard_*.h5"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.h5 in {shard_dir}")
    spx, trx, cnx, n = [], [], [], 0
    for path in paths:
        if n >= n_obj:
            break
        with h5py.File(path, "r") as handle:
            sp = np.asarray(handle["spectra"][...], np.float32)
            pr = np.asarray(handle["params"][...], np.float32)
            names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in np.asarray(handle["param_names"][...])]
        idx = {nm: i for i, nm in enumerate(names)}
        missing = [c for c in list(target_names) + list(cond_names) if c not in idx]
        if missing:
            raise KeyError(f"{path.name} missing columns {missing}")
        if sp.shape[-1] != input_length:
            raise ValueError(f"{path.name}: spectra length {sp.shape[-1]} != model input {input_length}")
        keep = np.all(np.isfinite(sp) & (sp > 0.0), axis=1)
        sp, pr = sp[keep], pr[keep]
        spx.append(np.stack([self_normalize(s) for s in sp]).astype(np.float32))
        trx.append(pr[:, [idx[t] for t in target_names]].astype(np.float64))
        cnx.append(pr[:, [idx[c] for c in cond_names]].astype(np.float32) if cond_names
                   else np.zeros((len(pr), 0), np.float32))
        n += len(sp)
    return (np.concatenate(spx)[:n_obj], np.concatenate(trx)[:n_obj],
            np.concatenate(cnx)[:n_obj])


def sample_model(inf: PosteriorInferencer, spectra, cond, n_samples: int, chunk: int, label: str):
    out = np.empty((len(spectra), n_samples, len(inf.target_parameter_names)), np.float64)
    cond_all = cond if inf.conditioning_parameter_names else None
    for start in range(0, len(spectra), chunk):
        sl = slice(start, start + chunk)
        out[sl] = inf.sample(
            spectra[sl],
            conditioning=(cond_all[sl] if cond_all is not None else None),
            num_samples=n_samples,
        )
        if (start // chunk) % 8 == 0:
            print(f"    [{label}] {min(start + chunk, len(spectra))}/{len(spectra)}", flush=True)
    return out


def raw_interval(samples_j, level: float):
    alpha = (1.0 - level) / 2.0
    return np.quantile(samples_j, alpha, axis=1), np.quantile(samples_j, 1.0 - alpha, axis=1)


def eval_posterior(post, truths, names, recal):
    """Per-parameter coverage/width plus overall (mean over targets) coverage."""
    stats = {}
    for j, name in enumerate(names):
        sj, tj = post[:, :, j], truths[:, j]
        cov, width90 = {}, None
        for level in LEVELS:
            if recal is None:
                lo, hi = raw_interval(sj, level)
            else:
                lo, hi = recal.calibrated_interval(sj, j, level)
            cov[str(level)] = float(np.mean((tj >= lo) & (tj <= hi)))
            if level == 0.9:
                width90 = hi - lo
        stats[name] = {"coverage": cov, "median_width90": float(np.median(width90))}
    overall = {str(level): float(np.mean([stats[n]["coverage"][str(level)] for n in names]))
               for level in LEVELS}
    return stats, overall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="sets", action="append", required=True,
                    help="NAME=DIR of held-out shard_*.h5 (repeatable)")
    ap.add_argument("--model", dest="models", action="append", default=[],
                    help="NAME=MODEL_DIR single model (repeatable)")
    ap.add_argument("--ensemble", dest="ensembles", action="append", default=[],
                    help="NAME=DIR1,DIR2,... pooled-sample ensemble (repeatable)")
    ap.add_argument("--n-obj", type=int, default=0, help="RAW mode: objects per set")
    ap.add_argument("--n-cal", type=int, default=0, help="SPLIT mode: recalibration-fit objects")
    ap.add_argument("--n-test", type=int, default=0, help="SPLIT mode: evaluation objects")
    ap.add_argument("--samples", type=int, default=400, help="draws per model per object")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split_mode = args.n_cal > 0 or args.n_test > 0
    if split_mode:
        if args.n_cal <= 0 or args.n_test <= 0:
            ap.error("SPLIT mode needs both --n-cal and --n-test > 0")
        if args.n_obj:
            ap.error("--n-obj is RAW mode only; drop it when using --n-cal/--n-test")
        n_total = args.n_cal + args.n_test
    else:
        if args.n_obj <= 0:
            ap.error("give either --n-obj (raw) or --n-cal/--n-test (split)")
        n_total = args.n_obj

    sets = dict(parse_named(s, "set") for s in args.sets)
    singles = dict(parse_named(m, "model") for m in args.models)
    ensembles = {name: [p for p in paths.split(",") if p]
                 for name, paths in (parse_named(e, "ensemble") for e in args.ensembles)}
    if not singles and not ensembles:
        ap.error("give at least one --model or --ensemble")

    # every distinct model directory referenced anywhere, sampled once per set
    model_paths: list[str] = []
    for path in list(singles.values()) + [p for paths in ensembles.values() for p in paths]:
        if path not in model_paths:
            model_paths.append(path)

    inferencers = {path: PosteriorInferencer(path) for path in model_paths}
    reference = inferencers[model_paths[0]]
    target_names = list(reference.target_parameter_names)
    cond_names = list(reference.conditioning_parameter_names)
    for path, inf in inferencers.items():
        if tuple(inf.target_parameter_names) != tuple(target_names):
            raise ValueError(f"target-parameter mismatch for {path}: {inf.target_parameter_names}")
        if tuple(inf.conditioning_parameter_names) != tuple(cond_names):
            raise ValueError(f"conditioning mismatch for {path}: {inf.conditioning_parameter_names}")
    input_length = int(reference.input_length)
    print(f"targets={target_names}\nconditioning={cond_names}", flush=True)

    result = {
        "mode": "split" if split_mode else "raw",
        "sets": {name: str(path) for name, path in sets.items()},
        "models": singles,
        "ensembles": ensembles,
        "samples_per_model": int(args.samples),
        "target_names": target_names,
        "split_seed": SPLIT_SEED if split_mode else None,
        "n_cal": int(args.n_cal) if split_mode else None,
        "n_test": int(args.n_test) if split_mode else None,
        "results": {},
        "summary": [],
    }

    for set_name, shard_dir in sets.items():
        print(f"\n=== SET {set_name} ({shard_dir}) ===", flush=True)
        spectra, truths, cond = load_heldout(
            Path(shard_dir), target_names, cond_names, input_length, n_total
        )
        print(f"loaded {len(spectra)} objects (requested {n_total})", flush=True)
        if split_mode:
            n_cal = args.n_cal if len(spectra) >= n_total else len(spectra) // 2
            if n_cal != args.n_cal:
                print(f"WARNING: only {len(spectra)} objects; cal={n_cal}", flush=True)
            perm = np.random.default_rng(SPLIT_SEED).permutation(len(spectra))
            cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
        else:
            cal_idx, test_idx = None, np.arange(len(spectra))

        posts = {
            path: sample_model(inf, spectra, cond, args.samples, args.chunk, f"{set_name}/{Path(path).name}")
            for path, inf in inferencers.items()
        }
        bases: dict[str, np.ndarray] = {name: posts[path] for name, path in singles.items()}
        for name, paths in ensembles.items():
            bases[name] = np.concatenate([posts[p] for p in paths], axis=1)

        set_result: dict[str, dict] = {"n_obj": int(len(spectra)),
                                       "n_test": int(len(test_idx)),
                                       "shard_dir": str(shard_dir),
                                       "models": {}}
        for model_name, post in bases.items():
            variants = {"raw": None}
            if split_mode:
                variants["recal"] = QuantileRecalibrator.fit(
                    post[cal_idx], truths[cal_idx], tuple(target_names)
                )
            entry = {}
            for variant, recal in variants.items():
                stats, overall = eval_posterior(post[test_idx], truths[test_idx], target_names, recal)
                co = stats[CO_NAME]
                entry[variant] = {
                    "co_cov90": co["coverage"]["0.9"],
                    "co_cov": co["coverage"],
                    "co_median_width90": co["median_width90"],
                    "overall_cov90": overall["0.9"],
                    "overall_cov": overall,
                    "per_param": stats,
                }
                result["summary"].append({
                    "set": set_name, "model": model_name, "variant": variant,
                    "co_cov90": entry[variant]["co_cov90"],
                    "overall_cov90": entry[variant]["overall_cov90"],
                    "co_median_width90": entry[variant]["co_median_width90"],
                    "n_test": int(len(test_idx)),
                })
                print(f"  {set_name:14} {model_name:18} {variant:5}: "
                      f"C/O cov90={entry[variant]['co_cov90']:.3f} "
                      f"w90={entry[variant]['co_median_width90']:.3f} "
                      f"overall90={entry[variant]['overall_cov90']:.3f}", flush=True)
            set_result["models"][model_name] = entry
        result["results"][set_name] = set_result
        del posts, bases

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== SUMMARY (target cov90 = 0.90) ===")
    print(f"{'set':16} {'model':18} {'variant':7} {'C/O cov90':>10} {'overall cov90':>14}")
    for row in result["summary"]:
        print(f"{row['set']:16} {row['model']:18} {row['variant']:7} "
              f"{row['co_cov90']:10.3f} {row['overall_cov90']:14.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
