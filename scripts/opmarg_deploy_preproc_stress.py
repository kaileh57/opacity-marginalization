#!/usr/bin/env python3
"""Item 6 (journal round): deployment-preprocessing stress test (journal-blocking).

At deployment the real spectra pass through preprocessing that the training set
never saw (apply_opmarg_real): masked pixels (bad pixels + the NRS1/NRS2 detector
gap) are median-filled, and the object is conditioned on a single SCALAR median
SNR. If that preprocessing degrades calibration, every real-data number in the
paper is suspect.

We take PERTURBED held-out SIM spectra (whose truths we know) and push them
through the *actual* deployment preprocessing, then re-measure coverage:

  baseline        : self-normalize the sim spectrum as-is (training convention).
  real_mask_fill  : overlay a REAL observed mask (median-fill masked pixels,
                    exactly as apply_opmarg_real), then self-normalize. This is
                    the dominant deployment deviation (the contiguous NRS gap).
  scalar_renoise  : re-noise the sim spectrum with a FLAT scalar-SNR Gaussian
                    (sigma = median|flux|/snr, no wavelength profile / correlated
                    component), isolating the scalar-noise-model mismatch.
  full_deploy     : real_mask_fill + scalar_renoise together (the real pipeline).

Real masks are sampled from the actual pop_exact_fit_inputs masks, so the masked
fraction and the gap geometry match deployment. Evaluated for control_seed0 and
marg_ensemble3. A coverage collapse here is a SCIENCE FINDING, flagged in stdout.
Pure CPU inference, no forward model.
"""
from __future__ import annotations

import argparse
import glob
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

LEVELS = (0.5, 0.68, 0.9, 0.95)


def load_heldout_raw(shard_dir, target_names, cond_names, input_length, n_obj):
    """Return RAW (un-normalized) spectra + truths + conditioning; we normalize
    per-variant after applying deployment preprocessing."""
    paths = sorted(Path(shard_dir).glob("shard_*.h5"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.h5 in {shard_dir}")
    spx, trx, cnx, n = [], [], [], 0
    for p in paths:
        if n >= n_obj:
            break
        with h5py.File(p, "r") as h:
            sp = np.asarray(h["spectra"][...], np.float32)
            pr = np.asarray(h["params"][...], np.float32)
            names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in np.asarray(h["param_names"][...])]
        idx = {nm: i for i, nm in enumerate(names)}
        keep = np.all(np.isfinite(sp) & (sp > 0.0), axis=1)
        sp, pr = sp[keep], pr[keep]
        spx.append(sp.astype(np.float64))
        trx.append(pr[:, [idx[t] for t in target_names]].astype(np.float64))
        cnx.append(pr[:, [idx[c] for c in cond_names]].astype(np.float32) if cond_names
                   else np.zeros((len(pr), 0), np.float32))
        n += len(sp)
    return (np.concatenate(spx)[:n_obj], np.concatenate(trx)[:n_obj],
            np.concatenate(cnx)[:n_obj])


def load_real_masks(inputs_dir, input_length):
    """Boolean 'good pixel' masks from real observations, length input_length."""
    masks = []
    for mf in sorted(glob.glob(str(Path(inputs_dir) / "*_mask.npy"))):
        m = np.load(mf).astype(bool)
        if m.shape[0] != input_length:
            continue
        # combine with the observed finite/>0 condition the real pipeline uses
        of = Path(mf).with_name(Path(mf).name.replace("_mask.npy", "_observed.npy"))
        if of.exists():
            obs = np.load(of).astype(float)
            m = m & np.isfinite(obs) & (obs > 0)
        if m.sum() >= 1000:
            masks.append(m)
    return masks


def coverage_stats(post, truths, names):
    per, jco = {}, names.index("carbon_to_oxygen")
    for j, name in enumerate(names):
        sj, tj = post[:, :, j], truths[:, j]
        cov, w90 = {}, None
        for lv in LEVELS:
            a = 1.0 - lv
            lo = np.quantile(sj, a / 2.0, axis=1)
            hi = np.quantile(sj, 1.0 - a / 2.0, axis=1)
            cov[str(lv)] = float(np.mean((tj >= lo) & (tj <= hi)))
            if lv == 0.9:
                w90 = hi - lo
        per[name] = {"coverage": cov, "median_width90": float(np.median(w90))}
    overall = {str(lv): float(np.mean([per[n]["coverage"][str(lv)] for n in names]))
               for lv in LEVELS}
    return {"overall_coverage": overall, "per_param": per,
            "co_cov90": per["carbon_to_oxygen"]["coverage"]["0.9"],
            "co_median_width90": per["carbon_to_oxygen"]["median_width90"]}


def build_variant(raw_sp, snr, masks, rng, variant):
    """Apply a deployment-preprocessing variant, return self-normalized spectra."""
    n, L = raw_sp.shape
    out = np.empty((n, L), np.float32)
    for i in range(n):
        flux = raw_sp[i].astype(np.float64).copy()
        if variant in ("scalar_renoise", "full_deploy"):
            # scalar (flat) Gaussian noise at the object SNR, on the flux scale
            sigma = np.median(np.abs(flux)) / max(snr[i], 1e-3)
            flux = flux + rng.normal(0.0, sigma, size=L)
        if variant in ("real_mask_fill", "full_deploy"):
            good = masks[rng.integers(0, len(masks))].copy()
            good &= np.isfinite(flux) & (flux > 0)
            if good.sum() < 1000:
                good = np.isfinite(flux) & (flux > 0)
            fill = np.nanmedian(flux[good])
            flux = np.where(good, flux, fill)
        # bad/non-positive pixels -> NaN so self_normalize median-fills them (the
        # deployment convention). NEVER clamp to a fixed floor: physical model flux
        # is << 1e-8, so a fixed floor would flatten the whole spectrum to a
        # constant and self_normalize would return zeros (near-prior posteriors).
        flux[~np.isfinite(flux) | (flux <= 0.0)] = np.nan
        out[i] = self_normalize(flux.astype(np.float32))
    return out


def sample_ensemble(models, spectra, cond, k_per, chunk=64, label=""):
    posts = []
    for mi, inf in enumerate(models):
        out = np.empty((len(spectra), k_per, len(inf.target_parameter_names)), np.float64)
        ca = cond if inf.conditioning_parameter_names else None
        for s in range(0, len(spectra), chunk):
            sl = slice(s, s + chunk)
            out[sl] = inf.sample(spectra[sl],
                                 conditioning=(ca[sl] if ca is not None else None),
                                 num_samples=k_per)
        posts.append(out)
        print(f"    [{label}] model {mi} done", flush=True)
    return np.concatenate(posts, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marg-models", nargs=3, required=True)
    ap.add_argument("--control-model", required=True)
    ap.add_argument("--heldout", required=True, help="perturbed held-out dir")
    ap.add_argument("--inputs", required=True, help="real inputs dir (for masks)")
    ap.add_argument("--n-obj", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=400, help="control draws; marg = /3 per seed")
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    margs = [PosteriorInferencer(m) for m in args.marg_models]
    control = PosteriorInferencer(args.control_model)
    names = list(margs[0].target_parameter_names)
    cond_names = list(margs[0].conditioning_parameter_names)
    L = int(margs[0].input_length)

    raw_sp, tr, cn = load_heldout_raw(Path(args.heldout), names, cond_names, L, args.n_obj)
    snr = cn[:, 0].astype(np.float64) if cn.shape[1] else np.full(len(raw_sp), 30.0)
    masks = load_real_masks(args.inputs, L)
    if not masks:
        raise RuntimeError("no usable real masks found for the stress test")
    print(f"{len(raw_sp)} sim objects, {len(masks)} real masks "
          f"(median good-frac {np.median([m.mean() for m in masks]):.3f})", flush=True)

    rng = np.random.default_rng(args.seed)
    k_marg = max(1, args.samples // 3)
    result = {
        "marg_models": args.marg_models, "control_model": args.control_model,
        "heldout": args.heldout, "n_obj": int(len(raw_sp)),
        "n_real_masks": len(masks), "target_names": names,
        "median_real_good_fraction": float(np.median([m.mean() for m in masks])),
        "variants": {},
    }
    variants = ["baseline", "real_mask_fill", "scalar_renoise", "full_deploy"]
    baseline_cov = {}
    for variant in variants:
        vr = np.random.default_rng(args.seed + hash(variant) % 10000)
        spec = build_variant(raw_sp, snr, masks, vr, variant)
        cond = cn if cond_names else None
        ens = sample_ensemble(margs, spec, cond, k_marg, label=f"{variant}/marg")
        cpost = sample_ensemble([control], spec, cond, args.samples, label=f"{variant}/ctrl")
        cov_m = coverage_stats(ens, tr, names)
        cov_c = coverage_stats(cpost, tr, names)
        result["variants"][variant] = {"marg_ensemble3": cov_m, "control_seed0": cov_c}
        if variant == "baseline":
            baseline_cov = {"marg": cov_m, "control": cov_c}
            # HARD self-check: the no-preprocessing baseline MUST reproduce the
            # canonical marg-ensemble coverage (factorial: C/O cov90 0.894,
            # overall 0.902). If it does not, the loader/input path is broken and
            # NO stressed variant is interpretable.
            co90 = cov_m["co_cov90"]; ov90 = cov_m["overall_coverage"]["0.9"]
            ok = (0.84 <= co90 <= 0.94) and (0.86 <= ov90 <= 0.94)
            result["baseline_selfcheck"] = {
                "marg_co_cov90": co90, "marg_overall_cov90": ov90,
                "expected_co_cov90_approx": 0.894, "expected_overall_cov90_approx": 0.902,
                "passes": bool(ok)}
            print(f"BASELINE SELF-CHECK: marg C/O cov90={co90:.3f} (exp~0.894) "
                  f"overall90={ov90:.3f} (exp~0.902) -> {'PASS' if ok else 'FAIL'}",
                  flush=True)
            if not ok:
                print("*** BASELINE SELF-CHECK FAILED: input/loader path is broken; "
                      "stressed-variant deltas are NOT trustworthy ***", flush=True)
        dm = cov_m["overall_coverage"]["0.9"] - baseline_cov["marg"]["overall_coverage"]["0.9"]
        dc = cov_c["overall_coverage"]["0.9"] - baseline_cov["control"]["overall_coverage"]["0.9"]
        result["variants"][variant]["delta_overall_cov90_vs_baseline"] = {
            "marg_ensemble3": float(dm), "control_seed0": float(dc)}
        flag = "  <<< COVERAGE SHIFT" if (abs(dm) > 0.05 or abs(dc) > 0.05) else ""
        print(f"{variant:16} marg overall90={cov_m['overall_coverage']['0.9']:.3f} "
              f"(d={dm:+.3f}) co90={cov_m['co_cov90']:.3f} | "
              f"ctrl overall90={cov_c['overall_coverage']['0.9']:.3f} (d={dc:+.3f}){flag}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
