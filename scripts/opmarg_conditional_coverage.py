#!/usr/bin/env python3
"""Item 4 (journal round): CONDITIONAL coverage + full per-parameter table (J2).

Marginal coverage can hide conditional miscalibration. The reviews specifically
flag the recalibrated marginalized posterior as weakest at SNR 10-49, so we bin
the PERTURBED held-out by SNR and by a Teff proxy and report, per bin:

  * coverage at 0.5/0.68/0.9/0.95 (all-parameter aggregate and C/O specifically),
  * median 90% interval width (all-param aggregate + C/O).

We report this for four methods (leakage-free recal, fit on a disjoint cal half):
  control_seed0 raw, marg_ensemble3 raw, control_seed0 recal, marg_ensemble3 recal.

Also produced:
  * the FULL per-parameter (all 15) marginal coverage + width table for every
    method (not just C/O), and
  * the REAL 105-spectrum 90% width distribution binned by SNR (C/O + all-param
    median), using the exact apply_opmarg_real preprocessing.

Teff proxy: the deepest-well-constrained knot T4 (photospheric for late-T dwarfs
in this 5-knot parameterization); reported as a proxy and clearly labelled.
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
from sculptor.npe.recalibration import QuantileRecalibrator  # noqa: E402

LEVELS = (0.5, 0.68, 0.9, 0.95)
SPLIT_SEED = 20260716
N_BINS = 5
TEFF_PROXY = "T4"


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


def interval(sj, j, cred, recal):
    if recal is None:
        a = (1.0 - cred) / 2.0
        return np.quantile(sj, a, axis=1), np.quantile(sj, 1.0 - a, axis=1)
    return recal.calibrated_interval(sj, j, cred)


def per_param_table(post, truths, names, recal):
    """Marginal coverage@levels + median width90 for every parameter + aggregate."""
    table = {}
    for j, name in enumerate(names):
        sj, tj = post[:, :, j], truths[:, j]
        cov, w90 = {}, None
        for lv in LEVELS:
            lo, hi = interval(sj, j, lv, recal)
            cov[str(lv)] = float(np.mean((tj >= lo) & (tj <= hi)))
            if lv == 0.9:
                w90 = hi - lo
        table[name] = {"coverage": cov, "median_width90": float(np.median(w90))}
    overall = {str(lv): float(np.mean([table[n]["coverage"][str(lv)] for n in names]))
               for lv in LEVELS}
    return table, overall


def conditional_bins(post, truths, names, values, recal):
    """Coverage@levels + median width90 (all-param aggregate + C/O) in equal-count
    bins of `values`."""
    jco = names.index("carbon_to_oxygen")
    qs = np.quantile(values, np.linspace(0, 1, N_BINS + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    # precompute inside/width arrays per (param, level)
    rows = []
    for b in range(N_BINS):
        m = (values >= qs[b]) & (values < qs[b + 1])
        if not m.any():
            continue
        p_b, t_b = post[m], truths[m]
        cov_all, cov_co = {}, {}
        w_all, w_co = None, None
        for lv in LEVELS:
            ins_params, w_params = [], []
            for j in range(len(names)):
                lo, hi = interval(p_b[:, :, j], j, lv, recal)
                inside = (t_b[:, j] >= lo) & (t_b[:, j] <= hi)
                ins_params.append(inside)
                if lv == 0.9:
                    w_params.append(np.median(hi - lo))  # median width for param j
                if j == jco:
                    cov_co[str(lv)] = float(np.mean(inside))
                    if lv == 0.9:
                        w_co = float(np.median(hi - lo))
            cov_all[str(lv)] = float(np.mean(np.stack(ins_params, axis=1)))
            if lv == 0.9:
                w_all = float(np.median(w_params))  # median over params of median width
        rows.append({
            "bin_index": b, "n": int(m.sum()),
            "value_lo": float(values[m].min()), "value_hi": float(values[m].max()),
            "value_median": float(np.median(values[m])),
            "coverage_all": cov_all, "coverage_co": cov_co,
            "median_width90_all": w_all, "median_width90_co": w_co,
        })
    return rows


def real_width_by_snr(inputs_dir, margs, control, names, min_good=1000):
    """Real 105-spectrum 90% width distribution binned by median SNR."""
    jco = names.index("carbon_to_oxygen")
    L = int(margs[0].input_length)
    rows = []
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
        w90 = lambda s: float(np.percentile(s, 95) - np.percentile(s, 5))
        rows.append({
            "key": key, "snr": snr,
            "marg_ens_co_width90": w90(ens[:, jco]),
            "control_co_width90": w90(cs[:, jco]),
            "marg_ens_median_width90_allparam": float(np.median(
                [w90(ens[:, j]) for j in range(len(names))])),
        })
    # bin by SNR (fixed deployment-relevant edges + quantile bins)
    snrs = np.array([r["snr"] for r in rows], float)
    fixed_edges = [0, 10, 20, 50, 100, np.inf]
    binned = []
    for b in range(len(fixed_edges) - 1):
        m = (snrs >= fixed_edges[b]) & (snrs < fixed_edges[b + 1])
        if not m.any():
            continue
        sub = [rows[i] for i in np.where(m)[0]]
        binned.append({
            "snr_lo": fixed_edges[b], "snr_hi": fixed_edges[b + 1], "n": int(m.sum()),
            "marg_ens_co_width90_median": float(np.median([r["marg_ens_co_width90"] for r in sub])),
            "control_co_width90_median": float(np.median([r["control_co_width90"] for r in sub])),
            "marg_ens_allparam_width90_median": float(np.median(
                [r["marg_ens_median_width90_allparam"] for r in sub])),
        })
    return {"n_objects": len(rows), "snr_bins_fixed": binned, "objects": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marg-models", nargs=3, required=True)
    ap.add_argument("--control-model", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--n-cal", type=int, default=2048)
    ap.add_argument("--n-test", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=400, help="draws per seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    margs = [PosteriorInferencer(m) for m in args.marg_models]
    control = PosteriorInferencer(args.control_model)
    names = list(margs[0].target_parameter_names)
    cond_names = list(margs[0].conditioning_parameter_names)
    L = int(margs[0].input_length)
    jteff = names.index(TEFF_PROXY)

    n_total = args.n_cal + args.n_test
    sp, tr, cn = load_heldout(Path(args.heldout), names, cond_names, L, n_total)
    if len(sp) < n_total:
        args.n_cal = len(sp) // 2
        print(f"WARNING: only {len(sp)} objects; cal={args.n_cal}", flush=True)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(sp))
    cal_idx, test_idx = perm[: args.n_cal], perm[args.n_cal:]
    print(f"cal={len(cal_idx)} test={len(test_idx)}", flush=True)

    m_posts = [sample_model(m, sp, cn, args.samples, label=f"marg{i}")
               for i, m in enumerate(margs)]
    c_post = sample_model(control, sp, cn, args.samples, label="ctrl0")
    ens_post = np.concatenate(m_posts, axis=1)

    bases = {"control_seed0": c_post, "marg_ensemble3": ens_post}
    recals = {b: QuantileRecalibrator.fit(p[cal_idx], tr[cal_idx], tuple(names))
              for b, p in bases.items()}

    snr_test = cn[test_idx, 0].astype(np.float64) if cn.shape[1] else None
    teff_test = tr[test_idx, jteff].astype(np.float64)
    tr_test = tr[test_idx]

    result = {
        "marg_models": args.marg_models, "control_model": args.control_model,
        "heldout": args.heldout, "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)), "samples_per_seed": args.samples,
        "split_seed": SPLIT_SEED, "teff_proxy": TEFF_PROXY, "target_names": names,
        "methods": {},
    }
    method_specs = [
        ("control_seed0_raw", "control_seed0", None),
        ("marg_ensemble3_raw", "marg_ensemble3", None),
        ("control_seed0_recal", "control_seed0", "control_seed0"),
        ("marg_ensemble3_recal", "marg_ensemble3", "marg_ensemble3"),
    ]
    for mname, base, rk in method_specs:
        post_test = bases[base][test_idx]
        recal = recals[rk] if rk else None
        table, overall = per_param_table(post_test, tr_test, names, recal)
        entry = {
            "overall_coverage": overall,
            "per_param_coverage_width": table,
            "conditional_by_teff_proxy": conditional_bins(
                post_test, tr_test, names, teff_test, recal),
        }
        if snr_test is not None:
            entry["conditional_by_snr"] = conditional_bins(
                post_test, tr_test, names, snr_test, recal)
        result["methods"][mname] = entry
        co = table["carbon_to_oxygen"]
        print(f"{mname:22} overall90={overall['0.9']:.3f} "
              f"C/O cov90={co['coverage']['0.9']:.3f} w90={co['median_width90']:.3f}",
              flush=True)
        if snr_test is not None:
            for r in entry["conditional_by_snr"]:
                print(f"    SNR[{r['value_lo']:.0f},{r['value_hi']:.0f}] n={r['n']:4d} "
                      f"all90={r['coverage_all']['0.9']:.3f} co90={r['coverage_co']['0.9']:.3f}",
                      flush=True)

    if args.inputs:
        print("real 105-spectrum width-by-SNR", flush=True)
        result["real_width_by_snr"] = real_width_by_snr(args.inputs, margs, control, names)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
