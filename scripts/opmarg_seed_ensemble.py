#!/usr/bin/env python3
"""TRACK C -- seed-ensemble estimator for the opacity-marginalization flagship.

The two marginalized NPE seeds (npe_v7_opmarg_seed0/seed1) give different C/O on
the T8 benchmark (0.66 vs 0.91 at v1; a seed-instability the referee will flag).
The ensemble estimator pools an equal number of posterior draws from BOTH seeds
per object -> a single mixture posterior that averages over the seed (a poor
person's deep ensemble / BMA over the amortization-network init). This script:

  * REAL sample: for every object in --inputs, draw N samples from each seed,
    pool them, and report ensemble C/O median + 68/90% intervals + 90% width,
    alongside the per-seed medians. The T8 benchmark row is reported vs Hood C/O.
  * (optional) HELD-OUT coverage: if --heldout is given, compute the ensemble's
    empirical coverage (cov50/68/90/95, overall + C/O) on the perturbed held-out
    set and compare to each seed's -- does pooling restore calibration?

Self-normalizes spectra identically to apply_opmarg_real.py / opmarg_sbc_compare.py.
Outputs JSON to --out. Pure inference (torch/zuko on CPU); no forward model.
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

LEVELS = (0.5, 0.68, 0.9, 0.95)
DEFAULT_T8_KEY = "jw02124-o059"
DEFAULT_HOOD_CO = 0.53
DEFAULT_HOOD_CO_ERR = 0.01


def summarize(samples: np.ndarray, names, prior_bounds, survive_frac):
    out = {}
    for j, name in enumerate(names):
        s = samples[:, j]
        med = float(np.median(s))
        lo68, hi68 = float(np.percentile(s, 16)), float(np.percentile(s, 84))
        lo90, hi90 = float(np.percentile(s, 5)), float(np.percentile(s, 95))
        w90 = hi90 - lo90
        ratio = constrained = prior_w = None
        if name in prior_bounds:
            lo_p, hi_p = prior_bounds[name]
            prior_w = float(hi_p - lo_p)
            if prior_w > 0:
                ratio = w90 / prior_w
                constrained = bool(ratio < survive_frac)
        out[name] = {"median": med, "lo68": lo68, "hi68": hi68,
                     "lo90": lo90, "hi90": hi90, "width90": w90,
                     "width90_over_prior": ratio, "constrained": constrained}
    return out


def coverage_of(post, truths, names):
    """post: (n_obj, n_samp, n_targ); truths: (n_obj, n_targ)."""
    per = {}
    for j, name in enumerate(names):
        sj, tj = post[:, :, j], truths[:, j]
        cov = {}
        for lv in LEVELS:
            a = 1.0 - lv
            lo = np.quantile(sj, a / 2.0, axis=1)
            hi = np.quantile(sj, 1.0 - a / 2.0, axis=1)
            cov[str(lv)] = float(np.mean((tj >= lo) & (tj <= hi)))
        per[name] = cov
    overall = {str(lv): float(np.mean([per[n][str(lv)] for n in names])) for lv in LEVELS}
    return overall, per


def load_heldout(shard_dir, target_names, cond_names, input_length, n_obj):
    import h5py
    paths = sorted(Path(shard_dir).glob("shard_*.h5"))
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
        spx.append(np.stack([self_normalize(s) for s in sp]).astype(np.float32))
        trx.append(pr[:, [idx[t] for t in target_names]].astype(np.float64))
        cnx.append(pr[:, [idx[c] for c in cond_names]].astype(np.float32) if cond_names
                   else np.zeros((len(pr), 0), np.float32))
        n += len(sp)
    return (np.concatenate(spx)[:n_obj], np.concatenate(trx)[:n_obj],
            np.concatenate(cnx)[:n_obj])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed0-model", required=True)
    ap.add_argument("--seed1-model", required=True)
    ap.add_argument("--inputs", default=str(PROJECT_ROOT / "outputs/pop_exact_fit_inputs_local"))
    ap.add_argument("--exact-fit-dir", default=None)
    ap.add_argument("--heldout", default=None, help="perturbed held-out dir for ensemble coverage")
    ap.add_argument("--n-per-seed", type=int, default=1000, help="draws per seed (pooled -> 2x)")
    ap.add_argument("--n-obj-heldout", type=int, default=512)
    ap.add_argument("--survive-frac", type=float, default=0.5)
    ap.add_argument("--min-good-pixels", type=int, default=1000)
    ap.add_argument("--t8-key", default=DEFAULT_T8_KEY)
    ap.add_argument("--hood-co", type=float, default=DEFAULT_HOOD_CO)
    ap.add_argument("--hood-co-err", type=float, default=DEFAULT_HOOD_CO_ERR)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m0 = PosteriorInferencer(args.seed0_model)
    m1 = PosteriorInferencer(args.seed1_model)
    if tuple(m0.target_parameter_names) != tuple(m1.target_parameter_names):
        raise ValueError("seed models have different targets")
    names = list(m0.target_parameter_names)
    prior_bounds = dict(m0.target_parameter_bounds)
    L = int(m0.input_length)
    print(f"targets={names}", flush=True)

    # ---- real sample ----
    obs_files = [Path(p) for p in sorted(glob.glob(str(Path(args.inputs) / "*_observed.npy")))]
    rows = []
    for of in obs_files:
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
        if int(good.sum()) < args.min_good_pixels:
            continue
        snr = float(np.median(obs[good] / err[good]))
        filled = np.where(good, obs, np.nanmedian(obs[good])).astype(np.float32)
        x = self_normalize(filled)[None, :].astype(np.float32)
        cond0 = np.array([[snr]], np.float32) if m0.conditioning_parameter_names else None
        cond1 = np.array([[snr]], np.float32) if m1.conditioning_parameter_names else None
        p0 = m0.sample(x, cond0, num_samples=args.n_per_seed)[0]
        p1 = m1.sample(x, cond1, num_samples=args.n_per_seed)[0]
        ens = np.concatenate([p0, p1], axis=0)
        e_sum = summarize(ens, names, prior_bounds, args.survive_frac)
        rows.append({
            "key": key, "snr": snr, "n_good": int(good.sum()),
            "is_t8_benchmark": bool(args.t8_key and args.t8_key in key),
            "ensemble": e_sum,
            "seed0_co": float(np.median(p0[:, names.index("carbon_to_oxygen")])),
            "seed1_co": float(np.median(p1[:, names.index("carbon_to_oxygen")])),
        })
        co = e_sum["carbon_to_oxygen"]
        print(f"{key[:44]:44} snr={snr:6.1f} ens C/O={co['median']:.3f}"
              f"[{co['lo68']:.2f},{co['hi68']:.2f}] s0={rows[-1]['seed0_co']:.3f} "
              f"s1={rows[-1]['seed1_co']:.3f}", flush=True)

    def pooled(field):
        tab = {}
        for nm in names:
            r = [x["ensemble"][nm]["width90_over_prior"] for x in rows
                 if x["ensemble"][nm]["width90_over_prior"] is not None]
            f = [x["ensemble"][nm]["constrained"] for x in rows
                 if x["ensemble"][nm]["constrained"] is not None]
            tab[nm] = {"median_width90_over_prior": float(np.median(r)) if r else None,
                       "frac_constrained": float(np.mean(f)) if f else None, "n": len(f)}
        return tab

    t8 = None
    t8_rows = [r for r in rows if r["is_t8_benchmark"]]
    if t8_rows:
        r = max(t8_rows, key=lambda z: z["snr"])
        co = r["ensemble"]["carbon_to_oxygen"]
        t8 = {"key": r["key"], "snr": r["snr"], "hood_co": args.hood_co,
              "hood_co_err": args.hood_co_err, "ensemble_co": co,
              "seed0_co": r["seed0_co"], "seed1_co": r["seed1_co"]}

    result = {
        "seed0_model": str(args.seed0_model), "seed1_model": str(args.seed1_model),
        "n_objects": len(rows), "n_per_seed": args.n_per_seed,
        "target_names": names, "population_survival": pooled("ensemble"),
        "t8_benchmark": t8, "objects": rows,
    }

    # ---- optional held-out ensemble coverage ----
    if args.heldout:
        cond_names = list(m0.conditioning_parameter_names)
        sp, tr, cn = load_heldout(args.heldout, names, cond_names, L, args.n_obj_heldout)
        print(f"held-out ensemble coverage on {len(sp)} objects", flush=True)

        def sample_all(inf, spectra, cond, k):
            ca = cond if inf.conditioning_parameter_names else None
            out = np.empty((len(spectra), k, len(names)))
            for s in range(0, len(spectra), 64):
                sl = slice(s, s + 64)
                c = ca[sl] if ca is not None else None
                out[sl] = inf.sample(spectra[sl], conditioning=c, num_samples=k)
            return out

        e0 = sample_all(m0, sp, cn, args.n_per_seed)
        e1 = sample_all(m1, sp, cn, args.n_per_seed)
        ens = np.concatenate([e0, e1], axis=1)
        ov_e, pp_e = coverage_of(ens, tr, names)
        ov_0, _ = coverage_of(e0, tr, names)
        ov_1, _ = coverage_of(e1, tr, names)
        result["heldout_coverage"] = {
            "n_obj": int(len(sp)),
            "ensemble_overall": ov_e, "seed0_overall": ov_0, "seed1_overall": ov_1,
            "ensemble_co": pp_e["carbon_to_oxygen"],
        }
        print(f"held-out overall cov90: ens={ov_e['0.9']:.3f} s0={ov_0['0.9']:.3f} "
              f"s1={ov_1['0.9']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    if t8:
        print(f"\nT8 {t8['key'][:40]}: Hood {t8['hood_co']}+/-{t8['hood_co_err']} | "
              f"ensemble C/O={t8['ensemble_co']['median']:.3f}"
              f"[{t8['ensemble_co']['lo68']:.2f},{t8['ensemble_co']['hi68']:.2f}] "
              f"(s0={t8['seed0_co']:.3f} s1={t8['seed1_co']:.3f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
