#!/usr/bin/env python3
"""SBC / coverage comparison for the opacity-marginalization flagship (direction A).

Evaluates TWO amortized posteriors on the SAME held-out, OPACITY-PERTURBED shards:

  * MARGINALIZED NPE  -- trained on v7_opmarg (per-draw opacity-field perturbations);
                         it is supposed to have folded the line-list uncertainty into
                         its posterior width and to STAY CALIBRATED under perturbation.
  * CONTROL NPE       -- trained on the unperturbed v6_cloud dataset; it has never
                         seen a perturbed opacity and is expected to be OVERCONFIDENT
                         (coverage below nominal) when the test opacity is perturbed.

The honest opacity error bar is the *width inflation* of the marginalized posterior
relative to the control: per parameter we report cov50/cov90 for both models plus the
median 90%-interval width ratio (marginalized / control). A calibrated-yet-wider
marginalized posterior is the paper's core claim.

Both models were trained on SELF-NORMALIZED spectra (run_training.py self-normalizes
before the trainer and records spectrum_normalization="none"), so we self-normalize the
held-out spectra here before sampling -- identical to apply_bestval_npe_real.py.

Reuses the project SBC machinery: sculptor.npe.calibration.empirical_coverage,
sculptor.npe.recalibration.pit_values / QuantileRecalibrator.

Usage:
  python scripts/opmarg_sbc_compare.py \
      --marg-model outputs/npe_v7_opmarg_seed0 \
      --control-model outputs/npe_v6_cloud \
      --heldout <dir of held-out PERTURBED shard_*.h5> \
      [--heldout-clean <dir of held-out UNPERTURBED shards, optional reference>] \
      --n-obj 512 --samples 400 --out outputs/opmarg/sbc_compare.json
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
from sculptor.npe.recalibration import QuantileRecalibrator, pit_values  # noqa: E402

try:
    from sculptor.data_generation.opacity_perturbation import NUISANCE_PREFIX
except Exception:  # pragma: no cover - defensive
    NUISANCE_PREFIX = "oppert_"

LEVELS = (0.5, 0.68, 0.9, 0.95)


def _shard_param_names(handle) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(handle["param_names"][...])]


def load_heldout(shard_dir: Path, target_names, cond_names, input_length, n_obj):
    """Load raw spectra + physical truths + conditioning from held-out shards.

    Nuisance (oppert_*) columns are ignored; the truths are the physical parameters
    that generated the (perturbed) spectrum. Applies the same STRICT finite filter and
    self-normalization the trainer uses, so the model sees identically-preprocessed input.
    """
    shard_paths = sorted(shard_dir.glob("shard_*.h5"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.h5 in {shard_dir}")
    spectra_l, truth_l, cond_l, n = [], [], [], 0
    for path in shard_paths:
        if n >= n_obj:
            break
        with h5py.File(path, "r") as handle:
            sp = np.asarray(handle["spectra"][...], dtype=np.float32)
            pr = np.asarray(handle["params"][...], dtype=np.float32)
            names = _shard_param_names(handle)
        idx = {name: i for i, name in enumerate(names)}
        missing = [t for t in list(target_names) + list(cond_names) if t not in idx]
        if missing:
            raise KeyError(f"held-out shard {path.name} missing columns {missing}")
        if sp.shape[-1] != input_length:
            raise ValueError(f"{path.name}: spectra length {sp.shape[-1]} != model input {input_length}")
        keep = np.all(np.isfinite(sp) & (sp > 0.0), axis=1)
        sp, pr = sp[keep], pr[keep]
        spectra_l.append(np.stack([self_normalize(s) for s in sp], axis=0).astype(np.float32))
        truth_l.append(pr[:, [idx[t] for t in target_names]].astype(np.float64))
        cond_l.append(pr[:, [idx[c] for c in cond_names]].astype(np.float32) if cond_names else
                      np.zeros((len(pr), 0), dtype=np.float32))
        n += len(sp)
    spectra = np.concatenate(spectra_l)[:n_obj]
    truths = np.concatenate(truth_l)[:n_obj]
    cond = np.concatenate(cond_l)[:n_obj]
    return spectra, truths, cond


def sample_model(inf: PosteriorInferencer, spectra, cond, samples, chunk):
    out = np.empty((len(spectra), samples, len(inf.target_parameter_names)), dtype=np.float64)
    cond_arg = cond if inf.conditioning_parameter_names else None
    for s in range(0, len(spectra), chunk):
        sl = slice(s, s + chunk)
        c = cond_arg[sl] if cond_arg is not None else None
        out[sl] = inf.sample(spectra[sl], conditioning=c, num_samples=samples)
        print(f"    sampled {min(s + chunk, len(spectra))}/{len(spectra)}", flush=True)
    return out


def per_param_stats(samples, truths, target_names, prior_bounds):
    """cov@levels, rank_mean, and median 90%-interval width per parameter."""
    stats = {}
    for j, name in enumerate(target_names):
        sj, tj = samples[:, :, j], truths[:, j]
        cov = {}
        for lv in LEVELS:
            a = 1.0 - lv
            lo = np.quantile(sj, a / 2.0, axis=1)
            hi = np.quantile(sj, 1.0 - a / 2.0, axis=1)
            cov[str(lv)] = float(np.mean((tj >= lo) & (tj <= hi)))
        w90 = np.quantile(sj, 0.95, axis=1) - np.quantile(sj, 0.05, axis=1)
        prior_w = None
        if prior_bounds and name in prior_bounds:
            lo_p, hi_p = prior_bounds[name]
            prior_w = float(hi_p - lo_p)
        ranks = np.mean(sj < tj[:, None], axis=1)
        stats[name] = {
            "coverage": cov,
            "rank_mean": float(ranks.mean()),
            "median_width90": float(np.median(w90)),
            "prior_width": prior_w,
            "median_width90_over_prior": float(np.median(w90) / prior_w) if prior_w else None,
        }
    return stats


def evaluate(inf, spectra, truths, cond, samples, chunk, label):
    print(f"  [{label}] sampling {len(spectra)} objects x {samples} draws", flush=True)
    post = sample_model(inf, spectra, cond, samples, chunk)
    stats = per_param_stats(post, truths, inf.target_parameter_names, inf.target_parameter_bounds)
    overall = {
        str(lv): float(np.mean([stats[n]["coverage"][str(lv)] for n in inf.target_parameter_names]))
        for lv in LEVELS
    }
    return post, stats, overall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marg-model", required=True, help="marginalized NPE dir (v7_opmarg)")
    ap.add_argument("--control-model", required=True, help="control NPE dir (v6_cloud, unperturbed)")
    ap.add_argument("--heldout", required=True, help="dir of held-out PERTURBED shard_*.h5")
    ap.add_argument("--heldout-clean", default=None,
                    help="optional dir of held-out UNPERTURBED shards (control sanity reference)")
    ap.add_argument("--n-obj", type=int, default=512)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    marg = PosteriorInferencer(args.marg_model)
    control = PosteriorInferencer(args.control_model)
    # Align on the marginalized model's target ordering (both share the 16-param cloudy layout).
    if tuple(marg.target_parameter_names) != tuple(control.target_parameter_names):
        raise ValueError(
            "marginalized and control models have different target parameters; "
            f"marg={marg.target_parameter_names} control={control.target_parameter_names}"
        )
    target_names = list(marg.target_parameter_names)
    cond_names = list(marg.conditioning_parameter_names)
    print(f"targets={target_names}\nconditioning={cond_names}", flush=True)

    spectra, truths, cond = load_heldout(
        Path(args.heldout), target_names, cond_names, marg.input_length, args.n_obj
    )
    print(f"held-out perturbed: {len(spectra)} objects", flush=True)

    marg_post, marg_stats, marg_overall = evaluate(
        marg, spectra, truths, cond, args.samples, args.chunk, "marginalized/perturbed"
    )
    ctrl_post, ctrl_stats, ctrl_overall = evaluate(
        control, spectra, truths, cond, args.samples, args.chunk, "control/perturbed"
    )

    # width-inflation = the honest opacity error bar (marginalized / control), per param
    width_ratio = {}
    for name in target_names:
        mw = marg_stats[name]["median_width90"]
        cw = ctrl_stats[name]["median_width90"]
        width_ratio[name] = float(mw / cw) if cw > 0 else None

    # post-hoc recalibration sanity (fit on half, evaluate the other half) for each model
    def recal_holdout(post):
        half = len(post) // 2
        recal = QuantileRecalibrator.fit(post[:half], truths[:half], tuple(target_names))
        return {str(k): float(v) for k, v in
                recal.coverage_after(post[half:], truths[half:], levels=LEVELS).items()}

    result = {
        "marg_model": str(args.marg_model),
        "control_model": str(args.control_model),
        "heldout": str(args.heldout),
        "n_obj": int(len(spectra)),
        "samples": int(args.samples),
        "target_names": target_names,
        "perturbed": {
            "marginalized": {"overall_coverage": marg_overall, "per_param": marg_stats,
                             "recal_holdout": recal_holdout(marg_post)},
            "control": {"overall_coverage": ctrl_overall, "per_param": ctrl_stats,
                        "recal_holdout": recal_holdout(ctrl_post)},
        },
        "width_ratio_marg_over_control": width_ratio,
    }

    # optional clean reference: control should be ~calibrated on unperturbed held-out
    if args.heldout_clean:
        cs, ct, cc = load_heldout(Path(args.heldout_clean), target_names, cond_names,
                                  marg.input_length, args.n_obj)
        _, cctrl_stats, cctrl_overall = evaluate(
            control, cs, ct, cc, args.samples, args.chunk, "control/clean")
        _, cmarg_stats, cmarg_overall = evaluate(
            marg, cs, ct, cc, args.samples, args.chunk, "marginalized/clean")
        result["clean"] = {
            "control": {"overall_coverage": cctrl_overall, "per_param": cctrl_stats},
            "marginalized": {"overall_coverage": cmarg_overall, "per_param": cmarg_stats},
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # console summary
    print("\n=== coverage on held-out PERTURBED shards (target=nominal) ===")
    print(f"{'level':>6} | {'marginalized':>12} | {'control':>8}")
    for lv in LEVELS:
        print(f"{lv:6.2f} | {marg_overall[str(lv)]:12.3f} | {ctrl_overall[str(lv)]:8.3f}")
    print("\n=== per-param 90% width inflation (marginalized / control) ===")
    for name in target_names:
        wr = width_ratio[name]
        print(f"  {name:18} cov90 marg={marg_stats[name]['coverage']['0.9']:.2f} "
              f"ctrl={ctrl_stats[name]['coverage']['0.9']:.2f}  width_ratio="
              f"{wr:.2f}" if wr is not None else f"  {name:18} width_ratio=n/a")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
