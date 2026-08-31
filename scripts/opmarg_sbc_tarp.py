#!/usr/bin/env python3
"""Item 3 (journal round): FULL 15-parameter SBC + TARP expected-coverage.

The reviewer panel (J1/C9) wants proper simulation-based calibration for *every*
target parameter, not just the C/O central interval:

  * per-parameter PIT/rank histograms with quantitative uniformity tests
    (chi-square goodness-of-fit and Kolmogorov-Smirnov, each with a p-value),
  * the TARP expected-coverage-probability diagnostic (Lemos et al. 2023,
    arXiv:2302.03026), which is sensitive to joint (multivariate) miscalibration
    that per-parameter marginals can miss.

Evaluated for three posteriors on the 2048-object PERTURBED held-out set:
  control_seed0  (npe_v6_cloud)                  -- naive amortized posterior
  marg_seed0     (npe_v7_opmarg_seed0)           -- single marginalized seed
  marg_ensemble3 (npe_v7_opmarg_seed0+1+2)       -- 3-seed marginalized mixture

Pure CPU inference, no forward model. Reuses the project held-out loader and
sculptor.npe.recalibration.pit_values (identical PIT convention as the rest of
the paper). p-values use scipy when available and fall back to analytic
implementations (series Kolmogorov CDF, Wilson-Hilferty chi-square) otherwise,
so the script never hard-depends on scipy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sculptor.npe.dataset import self_normalize  # noqa: E402
from sculptor.npe.inference import PosteriorInferencer  # noqa: E402
from sculptor.npe.recalibration import pit_values  # noqa: E402

try:
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    _scipy_stats = None

PIT_BINS = 20
TARP_ALPHA = np.linspace(0.0, 1.0, 41)


# ---------------------------------------------------------------- data loading
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


# ------------------------------------------------------------- p-value helpers
def _kolmogorov_cdf(t: float) -> float:
    """P(K <= t) for the asymptotic Kolmogorov distribution (series form)."""
    if t <= 0:
        return 0.0
    s = 0.0
    for k in range(1, 101):
        s += (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * t * t)
    return max(0.0, min(1.0, 1.0 - 2.0 * s))


def ks_uniform(x: np.ndarray):
    """KS statistic + p-value of x against Uniform(0,1)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0:
        return None, None
    cdf = (np.arange(1, n + 1)) / n
    d_plus = np.max(cdf - x)
    d_minus = np.max(x - (np.arange(0, n)) / n)
    d = float(max(d_plus, d_minus))
    if _scipy_stats is not None:
        p = float(_scipy_stats.kstest(x, "uniform").pvalue)
    else:
        p = float(1.0 - _kolmogorov_cdf((math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d))
    return d, p


def _chi2_sf(x: float, dof: int) -> float:
    """Survival function of chi-square (Wilson-Hilferty normal approximation)."""
    if x <= 0:
        return 1.0
    z = ((x / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(2.0 / (9.0 * dof))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def chi2_uniform(x: np.ndarray, bins: int = PIT_BINS):
    """Chi-square goodness-of-fit of x to Uniform(0,1) over `bins` equal bins."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return None, None, None
    counts, _ = np.histogram(x, bins=np.linspace(0.0, 1.0, bins + 1))
    expected = n / bins
    stat = float(np.sum((counts - expected) ** 2 / expected))
    dof = bins - 1
    if _scipy_stats is not None:
        p = float(_scipy_stats.chi2.sf(stat, dof))
    else:
        p = _chi2_sf(stat, dof)
    return stat, dof, p


# --------------------------------------------------------------------- SBC/PIT
def sbc_per_param(post, truths, names):
    pit = pit_values(post, truths)  # [n_obj, n_params]
    edges = np.linspace(0.0, 1.0, PIT_BINS + 1)
    out = {}
    all_pit = []
    for j, name in enumerate(names):
        r = pit[:, j]
        all_pit.append(r)
        counts, _ = np.histogram(r, bins=edges)
        ks_d, ks_p = ks_uniform(r)
        c2, c2_dof, c2_p = chi2_uniform(r)
        out[name] = {
            "pit_hist_counts": counts.astype(int).tolist(),
            "pit_bin_edges": edges.tolist(),
            "rank_mean": float(r.mean()),
            "ks_stat": ks_d, "ks_pvalue": ks_p,
            "chi2_stat": c2, "chi2_dof": c2_dof, "chi2_pvalue": c2_p,
        }
    # pooled over all object-parameter pairs
    pooled = np.concatenate(all_pit)
    ks_d, ks_p = ks_uniform(pooled)
    c2, c2_dof, c2_p = chi2_uniform(pooled)
    out["_pooled"] = {"ks_stat": ks_d, "ks_pvalue": ks_p,
                      "chi2_stat": c2, "chi2_dof": c2_dof, "chi2_pvalue": c2_p,
                      "n_pairs": int(len(pooled))}
    return out


# ------------------------------------------------------------------------ TARP
def tarp_ecp(post, truths, bounds, names, n_ref=50, seed=20260717):
    """TARP expected coverage probability (Lemos et al. 2023).

    Parameters are whitened to the unit hypercube using the prior bounds so the
    Euclidean distance is scale-free. For each of `n_ref` repetitions a reference
    point is drawn uniformly in the unit cube per object; the object's credibility
    is the fraction of posterior draws closer to the reference than the truth.
    ECP(alpha) = mean over all (object, repetition) of 1[credibility < alpha];
    a perfectly calibrated posterior gives ECP(alpha) = alpha.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([bounds[n][0] for n in names], np.float64)
    hi = np.array([bounds[n][1] for n in names], np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)
    p = (post - lo) / span                      # [n_obj, n_samp, D]
    t = (truths - lo) / span                    # [n_obj, D]
    n_obj, n_samp, D = p.shape
    creds = np.empty((n_ref, n_obj), np.float64)
    for r in range(n_ref):
        ref = rng.random((n_obj, D))            # uniform reference per object
        d_truth = np.linalg.norm(t - ref, axis=1)            # [n_obj]
        d_samp = np.linalg.norm(p - ref[:, None, :], axis=2)  # [n_obj, n_samp]
        creds[r] = np.mean(d_samp < d_truth[:, None], axis=1)
    ecp = np.array([np.mean(creds < a) for a in TARP_ALPHA])
    # bootstrap over objects for a CI band on the max deviation from the diagonal
    max_dev = float(np.max(np.abs(ecp - TARP_ALPHA)))
    boot_devs = []
    flat = creds.reshape(-1)
    for _ in range(200):
        bi = rng.integers(0, n_obj, n_obj)
        c = creds[:, bi].reshape(-1)
        e = np.array([np.mean(c < a) for a in TARP_ALPHA])
        boot_devs.append(np.max(np.abs(e - TARP_ALPHA)))
    return {
        "alpha_grid": TARP_ALPHA.tolist(),
        "ecp": ecp.tolist(),
        "max_deviation": max_dev,
        "max_deviation_boot95": [float(np.percentile(boot_devs, 2.5)),
                                 float(np.percentile(boot_devs, 97.5))],
        "mean_abs_deviation": float(np.mean(np.abs(ecp - TARP_ALPHA))),
        "n_ref": n_ref, "n_obj": int(n_obj),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marg-models", nargs=3, required=True)
    ap.add_argument("--control-model", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n-obj", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=400, help="draws per seed")
    ap.add_argument("--tarp-nref", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    margs = [PosteriorInferencer(m) for m in args.marg_models]
    control = PosteriorInferencer(args.control_model)
    names = list(margs[0].target_parameter_names)
    for inf in margs + [control]:
        if tuple(inf.target_parameter_names) != tuple(names):
            raise ValueError("model target mismatch")
    cond_names = list(margs[0].conditioning_parameter_names)
    bounds = dict(margs[0].target_parameter_bounds)
    L = int(margs[0].input_length)

    sp, tr, cn = load_heldout(Path(args.heldout), names, cond_names, L, args.n_obj)
    print(f"held-out perturbed: {len(sp)} objects, {len(names)} params", flush=True)

    m_posts = [sample_model(m, sp, cn, args.samples, label=f"marg{i}")
               for i, m in enumerate(margs)]
    c_post = sample_model(control, sp, cn, args.samples, label="ctrl0")

    posteriors = {
        "control_seed0": c_post,
        "marg_seed0": m_posts[0],
        "marg_ensemble3": np.concatenate(m_posts, axis=1),
    }

    result = {
        "marg_models": args.marg_models, "control_model": args.control_model,
        "heldout": args.heldout, "n_obj": int(len(sp)),
        "samples_per_seed": args.samples, "target_names": names,
        "scipy_available": _scipy_stats is not None, "models": {},
    }
    for label, post in posteriors.items():
        print(f"=== {label}: SBC + TARP ===", flush=True)
        sbc = sbc_per_param(post, tr, names)
        tarp = tarp_ecp(post, tr, bounds, names, n_ref=args.tarp_nref)
        # count params failing uniformity at 0.05 (Bonferroni-flag both raw + corrected)
        n_params = len(names)
        ks_fail = [n for n in names if (sbc[n]["ks_pvalue"] or 1.0) < 0.05]
        ks_fail_bonf = [n for n in names if (sbc[n]["ks_pvalue"] or 1.0) < 0.05 / n_params]
        result["models"][label] = {
            "sbc_per_param": sbc,
            "tarp": tarp,
            "n_params_ks_fail_p05": len(ks_fail),
            "params_ks_fail_p05": ks_fail,
            "n_params_ks_fail_bonferroni": len(ks_fail_bonf),
            "params_ks_fail_bonferroni": ks_fail_bonf,
        }
        print(f"  pooled KS p={sbc['_pooled']['ks_pvalue']:.3g} "
              f"chi2 p={sbc['_pooled']['chi2_pvalue']:.3g}  "
              f"TARP maxdev={tarp['max_deviation']:.3f} "
              f"CI{tarp['max_deviation_boot95']}  "
              f"KS-fail@.05: {len(ks_fail)}/{n_params}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
