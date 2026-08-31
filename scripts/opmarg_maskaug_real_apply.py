#!/usr/bin/env python3
"""Apply the MASK-AUGMENTED marginalized NPE to the 105 real JWST/G395H spectra.

Referee finding (v3.1 panel): the paper's real-data results use the un-augmented
marginalized seeds, i.e. the very models the preceding subsection shows collapse
under real bad-pixel masking. This script closes that gap for the single
mask-augmented seed that exists (npe_v7_opmarg_maskaug_seed0) by re-running the
identical real-data preprocessing and posterior-summary path used by
scripts/opmarg_factorial_eval.py (median-fill of bad pixels, self-normalization,
median-SNR conditioning, 667 posterior draws), and reporting the T8 benchmark
interval plus the population width summary.

Raw intervals only: no leakage-free calibration half is available locally for
this checkpoint, so no recalibrated row is produced.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sculptor.npe.dataset import self_normalize  # noqa: E402
from sculptor.npe.inference import PosteriorInferencer  # noqa: E402

# Without zuko the flow silently falls back to a diagonal Gaussian and every
# number below is meaningless. Fail loudly instead.
import zuko  # noqa: E402,F401

PRIOR_WIDTH_CO = 1.4  # C/O prior [0.1, 1.5]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, nargs="+",
                    help="one or more checkpoint dirs; >1 pools their posterior "
                         "samples by concatenation (same convention as "
                         "opmarg_factorial_eval.py)")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--samples", type=int, default=667)
    ap.add_argument("--t8-key", default="jw02124-o059")
    ap.add_argument("--hood-co", type=float, default=0.53)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    infs = [PosteriorInferencer(m) for m in args.model]
    for m, inf in zip(args.model, infs):
        backend = inf.model.flow.backend_name
        if backend != "zuko_nsf":
            raise RuntimeError(
                f"{m}: flow backend is {backend!r}, not 'zuko_nsf' -- the "
                "diagonal-Gaussian fallback is active and results would be junk.")
        print(f"model loaded: {m} (backend={backend})", flush=True)
    inf = infs[0]
    names = list(inf.target_parameter_names)
    jco = names.index("carbon_to_oxygen")
    L = int(inf.input_length)
    for other in infs[1:]:
        if list(other.target_parameter_names) != names:
            raise RuntimeError("target parameter names differ across seeds")
        if int(other.input_length) != L:
            raise RuntimeError("input_length differs across seeds")
    print(f"{len(infs)} model(s): {len(names)} targets, input_length={L}, "
          f"{args.samples} draws each -> {len(infs) * args.samples} pooled",
          flush=True)

    rows = []
    for of in sorted(glob.glob(str(Path(args.inputs) / "*_observed.npy"))):
        of = Path(of)
        key = of.name[: -len("_observed.npy")]
        ef = of.with_name(f"{key}_error.npy")
        mf = of.with_name(f"{key}_mask.npy")
        if not ef.exists():
            continue
        obs = np.load(of).astype(float)
        err = np.load(ef).astype(float)
        if obs.shape[0] != L:
            continue
        mask = np.load(mf).astype(bool) if mf.exists() else np.ones_like(obs, bool)
        good = np.isfinite(obs) & (obs > 0) & mask & np.isfinite(err) & (err > 0)
        if int(good.sum()) < 1000:
            continue
        snr = float(np.median(obs[good] / err[good]))
        filled = np.where(good, obs, np.nanmedian(obs[good])).astype(np.float32)
        x = self_normalize(filled)[None, :].astype(np.float32)
        cond = np.array([[snr]], np.float32)
        per_seed = [
            f.sample(x, cond if f.conditioning_parameter_names else None,
                     num_samples=args.samples)[0]
            for f in infs
        ]
        s = np.concatenate(per_seed, axis=0)  # pool along the sample axis
        sj = s[:, jco]
        row = {
            "key": key, "snr": snr,
            "good_fraction": float(good.mean()),
            "median": float(np.median(sj)),
            "lo68": float(np.percentile(sj, 16)),
            "hi68": float(np.percentile(sj, 84)),
            "lo90": float(np.percentile(sj, 5)),
            "hi90": float(np.percentile(sj, 95)),
        }
        row["width90"] = row["hi90"] - row["lo90"]
        row["width90_over_prior"] = row["width90"] / PRIOR_WIDTH_CO
        if len(infs) > 1:
            row["per_seed_median"] = [float(np.median(ps[:, jco]))
                                      for ps in per_seed]
        rows.append(row)
        print(f"  {key} SNR={snr:6.2f} C/O={row['median']:.3f} "
              f"90%=[{row['lo90']:.3f},{row['hi90']:.3f}]", flush=True)

    t8_rows = [r for r in rows if args.t8_key in r["key"]]
    t8 = None
    if t8_rows:
        r = max(t8_rows, key=lambda z: z["snr"])
        t8 = dict(r)
        t8["covers_hood_68"] = bool(r["lo68"] <= args.hood_co <= r["hi68"])
        t8["covers_hood_90"] = bool(r["lo90"] <= args.hood_co <= r["hi90"])
        t8["hood_co"] = args.hood_co

    insup = [r for r in rows if r["snr"] >= 10.0]
    result = {
        "model": args.model if len(args.model) > 1 else args.model[0],
        "models": args.model, "n_models": len(infs),
        "pooling": "concatenate posterior samples across seeds",
        "inputs": args.inputs,
        "samples_per_model": args.samples,
        "samples": args.samples * len(infs), "n_spectra": len(rows),
        "n_in_support_snr_ge_10": len(insup),
        "median_width90_all": float(np.median([r["width90"] for r in rows])),
        "median_width90_in_support": (
            float(np.median([r["width90"] for r in insup])) if insup else None),
        "median_width_over_prior_in_support": (
            float(np.median([r["width90"] for r in insup]) / PRIOR_WIDTH_CO)
            if insup else None),
        "median_width_over_prior_all": float(
            np.median([r["width90"] for r in rows]) / PRIOR_WIDTH_CO),
        "t8_benchmark": t8, "objects": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "objects"}, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
