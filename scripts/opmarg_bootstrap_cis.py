#!/usr/bin/env python3
"""Item 7 (journal round): paired bootstrap CIs + target-level repeatability.

Two deliverables:

(A) PAIRED bootstrap 95% CIs on the HEADLINE differences. The paper claims the
    marginalized ensemble is better-calibrated than the control ensemble; a point
    estimate is not enough for a journal. We resample OBJECTS with replacement
    (paired: the same resampled index set is applied to every method, so the CI
    is on the difference, not two independent noisy numbers) on the perturbed
    test half and report 95% percentile CIs on:
      * overall cov90 delta   (marg_ensemble3 - control_ensemble3), raw & recal
      * C/O cov90 delta       (same)
      * C/O median width90 delta (the honest opacity error bar inflation)
    A CI that excludes 0 is a defensible claim.

(B) TARGET-LEVEL repeatability. The 105 real spectra are a small number of unique
    late-T targets observed multiple times (different visits / sub-apertures /
    detectors). A trustworthy amortized posterior should give CONSISTENT C/O for
    repeat spectra of the SAME target. We group spectra by their matched
    exact-fit target_key (popcold) and report within-target vs across-target C/O
    median scatter (a repeatability ratio << 1 means the spread across the whole
    sample is real target-to-target variation, not per-spectrum posterior noise).

Recal maps are fit leakage-free on a disjoint cal half. Pure CPU inference.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
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
SPLIT_SEED = 20260716
N_BOOT = 2000


def load_heldout(shard_dir, target_names, cond_names, input_length, n_obj):
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
        spx.append(np.stack([self_normalize(s) for s in sp]).astype(np.float32))
        trx.append(pr[:, [idx[t] for t in target_names]].astype(np.float64))
        cnx.append(pr[:, [idx[c] for c in cond_names]].astype(np.float32) if cond_names
                   else np.zeros((len(pr), 0), np.float32))
        n += len(sp)
    return (np.concatenate(spx)[:n_obj], np.concatenate(trx)[:n_obj],
            np.concatenate(cnx)[:n_obj])


def sample_model(inf, spectra, cond, k, chunk=64, label=""):
    out = np.empty((len(spectra), k, len(inf.target_parameter_names)), np.float64)
    ca = cond if inf.conditioning_parameter_names else None
    for s in range(0, len(spectra), chunk):
        sl = slice(s, s + chunk)
        out[sl] = inf.sample(spectra[sl],
                             conditioning=(ca[sl] if ca is not None else None),
                             num_samples=k)
        if (s // chunk) % 8 == 0:
            print(f"    [{label}] {min(s + chunk, len(spectra))}/{len(spectra)}",
                  flush=True)
    return out


def per_object_inside_and_width(post, truths, j, cred, recal):
    """Return (inside[n_obj] bool, width[n_obj]) for parameter j at level cred."""
    sj = post[:, :, j]
    if recal is None:
        # central interval at level `cred`: tail mass a on each side.
        # BUGFIX (2026-07-31): the tail mass was halved a second time here, so
        # `cred=0.90` produced 0.025/0.975 quantiles -- i.e. 95% intervals
        # labelled 90%. Every raw-branch number written before this fix is a 95%
        # result and must not be quoted at 90%. The `recal` branch below never
        # used this code path and is unaffected.
        a = (1.0 - cred) / 2.0
        lo, hi = np.quantile(sj, a, axis=1), np.quantile(sj, 1.0 - a, axis=1)
    else:
        lo, hi = recal.calibrated_interval(sj, j, cred)
    return (truths[:, j] >= lo) & (truths[:, j] <= hi), hi - lo


def all_param_inside(post, truths, names, cred, recal):
    """Per-object mean 'inside' across all params (for overall coverage bootstrap)."""
    ins = np.zeros((post.shape[0], len(names)))
    for j in range(len(names)):
        inside, _ = per_object_inside_and_width(post, truths, j, cred, recal)
        ins[:, j] = inside
    return ins  # [n_obj, n_params]


def bootstrap_deltas(marg_post, ctrl_post, truths, names, recal_m, recal_c, seed):
    jco = names.index("carbon_to_oxygen")
    n = marg_post.shape[0]
    # precompute per-object statistics so the bootstrap is a cheap resample of indices
    stat = {}
    for tag, post, recal in (("marg", marg_post, recal_m), ("ctrl", ctrl_post, recal_c)):
        stat[tag] = {
            "co_inside90": per_object_inside_and_width(post, truths, jco, 0.9, recal)[0].astype(float),
            "co_width90": per_object_inside_and_width(post, truths, jco, 0.9, recal)[1],
            "all_inside90": all_param_inside(post, truths, names, 0.9, recal),
        }
    rng = np.random.default_rng(seed)
    d_co_cov, d_all_cov, d_co_w = [], [], []
    for _ in range(N_BOOT):
        bi = rng.integers(0, n, n)
        d_co_cov.append(stat["marg"]["co_inside90"][bi].mean()
                        - stat["ctrl"]["co_inside90"][bi].mean())
        d_all_cov.append(stat["marg"]["all_inside90"][bi].mean()
                         - stat["ctrl"]["all_inside90"][bi].mean())
        d_co_w.append(np.median(stat["marg"]["co_width90"][bi])
                      - np.median(stat["ctrl"]["co_width90"][bi]))

    def summarize(arr):
        arr = np.asarray(arr)
        return {"point": float(arr.mean()),
                "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
                "excludes_zero": bool(np.percentile(arr, 2.5) > 0 or np.percentile(arr, 97.5) < 0)}

    # point estimates on the full sample (not bootstrap mean)
    pt_co = float(stat["marg"]["co_inside90"].mean() - stat["ctrl"]["co_inside90"].mean())
    pt_all = float(stat["marg"]["all_inside90"].mean() - stat["ctrl"]["all_inside90"].mean())
    pt_w = float(np.median(stat["marg"]["co_width90"]) - np.median(stat["ctrl"]["co_width90"]))
    out = {"co_cov90_delta": summarize(d_co_cov), "overall_cov90_delta": summarize(d_all_cov),
           "co_width90_delta": summarize(d_co_w)}
    out["co_cov90_delta"]["point"] = pt_co
    out["overall_cov90_delta"]["point"] = pt_all
    out["co_width90_delta"]["point"] = pt_w
    out["marg_co_cov90"] = float(stat["marg"]["co_inside90"].mean())
    out["ctrl_co_cov90"] = float(stat["ctrl"]["co_inside90"].mean())
    out["marg_overall_cov90"] = float(stat["marg"]["all_inside90"].mean())
    out["ctrl_overall_cov90"] = float(stat["ctrl"]["all_inside90"].mean())
    return out


# --------------------------------------------------------- target repeatability
def parse_source_id(key: str) -> str:
    """Map a spectrum key to its PHYSICAL source id, merging the two JWST naming
    schemes so repeat exposures / detectors / sub-apertures of one target group:
      jw01189-o004_t002_...            -> jw01189_o004   (program 01189, object 004)
      jw01189004001_03106_00001_nrs1   -> jw01189_o004   (program 01189, obs 004)
      jw02124-o001_t001-s...           -> jw02124_o001
    """
    m = re.match(r"jw(\d{5})-o(\d+)", key)
    if m:
        return f"jw{m.group(1)}_o{int(m.group(2)):03d}"
    m = re.match(r"jw(\d{5})(\d{3})\d+_", key)
    if m:
        return f"jw{m.group(1)}_o{int(m.group(2)):03d}"
    return key


def target_repeatability(inputs_dir, margs, control, names, min_good=1000):
    jco = names.index("carbon_to_oxygen")
    L = int(margs[0].input_length)
    per_obj = []
    for of in sorted(glob.glob(str(Path(inputs_dir) / "*_observed.npy"))):
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
        if int(good.sum()) < min_good:
            continue
        snr = float(np.median(obs[good] / err[good]))
        filled = np.where(good, obs, np.nanmedian(obs[good])).astype(np.float32)
        x = self_normalize(filled)[None, :].astype(np.float32)
        cond = np.array([[snr]], np.float32)
        ens = np.concatenate([m.sample(x, cond if m.conditioning_parameter_names else None,
                                       num_samples=667)[0] for m in margs], axis=0)
        cs = control.sample(x, cond if control.conditioning_parameter_names else None,
                            num_samples=2000)[0]
        tgt = parse_source_id(key)
        per_obj.append({
            "key": key, "target": tgt, "snr": snr,
            "marg_co_median": float(np.median(ens[:, jco])),
            "marg_co_width90": float(np.percentile(ens[:, jco], 95) - np.percentile(ens[:, jco], 5)),
            "control_co_median": float(np.median(cs[:, jco])),
        })
    # group by target
    groups = {}
    for r in per_obj:
        groups.setdefault(r["target"], []).append(r)
    within = []  # within-target std of C/O medians (for targets with >=2 spectra)
    for t, rs in groups.items():
        if len(rs) >= 2:
            within.append(np.std([r["marg_co_median"] for r in rs], ddof=1))
    marg_medians = [r["marg_co_median"] for r in per_obj]
    across = float(np.std(marg_medians, ddof=1)) if len(marg_medians) > 1 else None
    within_mean = float(np.mean(within)) if within else None
    # median within-target posterior width for reference
    med_width = float(np.median([r["marg_co_width90"] for r in per_obj])) if per_obj else None
    return {
        "n_spectra": len(per_obj), "n_targets": len(groups),
        "n_targets_with_repeats": int(sum(1 for rs in groups.values() if len(rs) >= 2)),
        "within_target_co_scatter_mean": within_mean,
        "across_target_co_scatter": across,
        "repeatability_ratio_within_over_across": (
            float(within_mean / across) if (within_mean is not None and across) else None),
        "median_marg_co_posterior_width90": med_width,
        "per_target": {t: {"n": len(rs), "keys": [r["key"] for r in rs],
                           "co_medians": [r["marg_co_median"] for r in rs],
                           "co_std": float(np.std([r["marg_co_median"] for r in rs], ddof=1))
                                     if len(rs) >= 2 else None}
                       for t, rs in groups.items()},
        "objects": per_obj,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marg-models", nargs=3, required=True)
    ap.add_argument("--control-models", nargs=3, required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--n-cal", type=int, default=2048)
    ap.add_argument("--n-test", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=400, help="draws per seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    margs = [PosteriorInferencer(m) for m in args.marg_models]
    controls = [PosteriorInferencer(m) for m in args.control_models]
    names = list(margs[0].target_parameter_names)
    cond_names = list(margs[0].conditioning_parameter_names)
    L = int(margs[0].input_length)

    n_total = args.n_cal + args.n_test
    sp, tr, cn = load_heldout(Path(args.heldout), names, cond_names, L, n_total)
    if len(sp) < n_total:
        args.n_cal = len(sp) // 2
        print(f"WARNING: only {len(sp)} objects; cal={args.n_cal}", flush=True)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(sp))
    cal_idx, test_idx = perm[: args.n_cal], perm[args.n_cal:]

    m_posts = [sample_model(m, sp, cn, args.samples, label=f"marg{i}")
               for i, m in enumerate(margs)]
    c_posts = [sample_model(c, sp, cn, args.samples, label=f"ctrl{i}")
               for i, c in enumerate(controls)]
    marg_ens = np.concatenate(m_posts, axis=1)
    ctrl_ens = np.concatenate(c_posts, axis=1)

    recal_m = QuantileRecalibrator.fit(marg_ens[cal_idx], tr[cal_idx], tuple(names))
    recal_c = QuantileRecalibrator.fit(ctrl_ens[cal_idx], tr[cal_idx], tuple(names))
    mt, ct, tt = marg_ens[test_idx], ctrl_ens[test_idx], tr[test_idx]

    result = {
        "marg_models": args.marg_models, "control_models": args.control_models,
        "heldout": args.heldout, "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)), "samples_per_seed": args.samples,
        "n_boot": N_BOOT, "split_seed": SPLIT_SEED, "target_names": names,
        "paired_bootstrap": {
            "raw": bootstrap_deltas(mt, ct, tt, names, None, None, seed=1),
            "recal": bootstrap_deltas(mt, ct, tt, names, recal_m, recal_c, seed=2),
        },
    }
    for variant in ("raw", "recal"):
        b = result["paired_bootstrap"][variant]
        for metric in ("co_cov90_delta", "overall_cov90_delta", "co_width90_delta"):
            d = b[metric]
            print(f"[{variant}] {metric:20} point={d['point']:+.4f} "
                  f"CI95=[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
                  f"excl0={d['excludes_zero']}", flush=True)

    if args.inputs:
        print("target-level repeatability on the real sample", flush=True)
        rep = target_repeatability(args.inputs, margs, controls[0], names)
        result["target_repeatability"] = rep
        print(f"  {rep['n_spectra']} spectra / {rep['n_targets']} targets "
              f"({rep['n_targets_with_repeats']} with repeats); "
              f"within={rep['within_target_co_scatter_mean']} "
              f"across={rep['across_target_co_scatter']} "
              f"ratio={rep['repeatability_ratio_within_over_across']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
