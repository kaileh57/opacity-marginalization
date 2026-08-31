#!/usr/bin/env python3
"""Item 5b (journal round): DE-CONFOUNDED posterior-predictive checks on real G395H.

The v1 PPC (opmarg_ppc_real.py) computed a standardized residual whose observed
spectrum was self-normalized over the FULL 10000-pixel array AFTER median-filling
the ~53%-masked pixels, while the model was normalized clean. The filled constant
region contaminates the observed mean/std, so the reported chi2/dof (T8=43,
worst=1e4-5e4) is inflated by a NORMALIZATION MISMATCH, not necessarily by true
forward-model error. This script removes that confound and reports the numbers a
brown-dwarf retrieval referee actually cares about.

Everything is computed in LOG-FLUX on GOOD PIXELS ONLY (masked pixels never enter
any mean/statistic). Per object:

  * chi2_goodpix_absolute   : sum(((log_obs - log_model)/sigma)^2)/dof, sigma =
                              err/obs. Still dominated by the absolute flux scale
                              (radius/distance/calibration) -> reported for context.
  * chi2_goodpix_offset     : after removing a SINGLE global DC offset
                              a = mean(log_obs - log_model) (the flux-scale
                              nuisance every published BD G395H retrieval fits).
                              This is the de-confounded headline. NOTE: dividing
                              log-flux by any constant std cancels in chi2, so this
                              number is normalization-invariant by construction.
  * f_inflation             = sqrt(chi2_goodpix_offset) = the flux-jitter / error-
                              inflation (10^b term of Line+2015 / Hood+2024) needed
                              to bring chi2/dof -> 1. f ~ 2-4x is mundane/expected;
                              f >> ~10x on good pixels after offset removal = a real
                              forward-model gap.
  * chi2_goodpix_perarm     : after removing SEPARATE DC offsets for NRS1 and NRS2.
                              If this drops near ~1, the misfit is an inter-arm
                              multiplicative calibration offset (mundane). If it
                              stays high, it is genuine spectral-shape incompatibility.
  * nrs_flux_ratio          = exp(a_nrs1 - a_nrs2) : multiplicative flux ratio
                              between the detectors (1.0 = consistent).
  * within_arm_rms_dex      : RMS of the per-arm-offset-removed log residual within
                              each arm (small -> offset/mundane; large -> shape/real).

Verdict per object (esp. the T8 Hood benchmark, --t8-key jw02124-o059): MUNDANE
(gap resolves on good pixels with modest inflation / an inter-arm offset) vs
GENUINE (large f surviving good-pixel + global + per-arm offset removal).

Reuses the forward-model env already staged on worker 4 (jax/exojax, opa cache,
CPU-forced). Pure inference + one median-parameter forward per object.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sculptor.config import get_default_track_config  # noqa: E402
from sculptor.forward_model.chemistry import ChemistryModel  # noqa: E402
from sculptor.forward_model.opacity import ExoJAXOpacityBackend  # noqa: E402
from sculptor.forward_model.spectrum import SpectrumModel  # noqa: E402
from sculptor.npe.dataset import self_normalize  # noqa: E402
from sculptor.npe.inference import PosteriorInferencer  # noqa: E402

BANDS = [(2.87, 3.30, "H2O_blue"), (3.30, 3.60, "CH4_nu3"), (3.60, 4.00, "gap"),
         (4.00, 4.35, "CO2"), (4.35, 4.80, "CO_PH3_CH4"), (4.80, 5.14, "CO_fund")]
NRS_GAP_UM = (3.72, 3.82)


def forward_flux(sm, param_vec, names):
    """Raw (linear) model flux + nu grid for one parameter vector, or (None, None)."""
    pdict = {n: float(v) for n, v in zip(names, param_vec)}
    try:
        gen = sm.generate(pdict)
    except Exception:
        return None, None
    flux = np.asarray(gen.flux_nu, dtype=np.float64)
    if not np.any(np.isfinite(flux) & (flux > 0)):
        return None, None
    return flux, np.asarray(gen.nu_grid_cm1, np.float64)


def deconfounded_metrics(obs, err, model, good, wl):
    """All metrics in log-flux on GOOD pixels only."""
    g = good & np.isfinite(model) & (model > 0)
    n = int(g.sum())
    if n < 200:
        return None
    lo = np.log(obs[g]); lm = np.log(model[g])
    sig = err[g] / obs[g]                      # measurement error in log-flux
    sig = np.where(sig > 0, sig, np.nan)
    d = lo - lm                                 # raw log residual (abs scale in it)
    ndof = max(n - 1, 1)

    def chi2(resid):
        v = np.isfinite(resid) & np.isfinite(sig)
        return float(np.sum((resid[v] / sig[v]) ** 2) / max(int(v.sum()) - 1, 1))

    chi2_abs = chi2(d)
    a_glob = float(np.mean(d))
    chi2_off = chi2(d - a_glob)
    f_infl = float(np.sqrt(chi2_off))

    wlg = wl[g]
    # partition EVERY good pixel to one detector at the gap midpoint, so no pixel
    # is left without an offset removed (a pixel inside the 3.72-3.82 gap would
    # otherwise keep the full absolute flux offset and blow up the per-arm chi2).
    gap_mid = 0.5 * (NRS_GAP_UM[0] + NRS_GAP_UM[1])
    nrs1 = wlg <= gap_mid
    nrs2 = wlg > gap_mid
    a1 = float(np.mean(d[nrs1])) if nrs1.any() else a_glob
    a2 = float(np.mean(d[nrs2])) if nrs2.any() else a_glob
    d_perarm = d.copy()
    if nrs1.any():
        d_perarm[nrs1] -= a1
    if nrs2.any():
        d_perarm[nrs2] -= a2
    chi2_perarm = chi2(d_perarm)

    def arm_rms(sel):
        return float(np.sqrt(np.mean(d_perarm[sel] ** 2))) if sel.any() else None

    return {
        "n_good": n,
        "chi2_goodpix_absolute": chi2_abs,
        "global_log_offset": a_glob,
        "chi2_goodpix_offset": chi2_off,
        "f_inflation": f_infl,
        "chi2_goodpix_perarm": chi2_perarm,
        "nrs_flux_ratio_nrs1_over_nrs2": float(np.exp(a1 - a2)),
        "within_arm_rms_dex_nrs1": arm_rms(nrs1),
        "within_arm_rms_dex_nrs2": arm_rms(nrs2),
        "n_nrs1": int(nrs1.sum()), "n_nrs2": int(nrs2.sum()),
        # offset-removed standardized residual (for band stats on the detailed set)
        "_resid_perarm_over_sig": (d_perarm / sig),
        "_good_idx": np.where(g)[0],
    }


def band_stats_offset(resid_std, wl_good, good_idx):
    out = {}
    wl_g = wl_good[good_idx] if False else None  # placeholder; wl passed already good-indexed
    return out


def confounded_chi2(obs, err, model, good):
    """Reproduce the v1 (confounded) chi2 for cross-reference."""
    g = good & np.isfinite(model) & (model > 0)
    if int(g.sum()) < 200:
        return None
    ofn = obs.copy(); ofn[~good] = np.nanmedian(obs[good])
    on = self_normalize(ofn).astype(np.float64)
    fn = model.copy(); finm = np.isfinite(fn) & (fn > 0)
    fn[~finm] = np.nanmedian(fn[finm])
    mn = self_normalize(fn).astype(np.float64)
    log_std = float(np.nanstd(np.log(np.maximum(ofn[good], 1e-30))))
    en = err[g] / (obs[g] * max(log_std, 1e-6))
    v = np.isfinite(on[g]) & np.isfinite(mn[g]) & np.isfinite(en) & (en > 0)
    if not v.any():
        return None
    return float(np.mean(((on[g][v] - mn[g][v]) / en[v]) ** 2))


def load_real(inputs_dir, L, min_good=1000):
    objs = []
    for of in sorted(glob.glob(str(Path(inputs_dir) / "*_observed.npy"))):
        of = Path(of)
        key = of.name[: -len("_observed.npy")]
        ef = of.with_name(f"{key}_error.npy")
        mf = of.with_name(f"{key}_mask.npy")
        if not ef.exists():
            continue
        obs = np.load(of).astype(float); err = np.load(ef).astype(float)
        if obs.shape[0] != L:
            continue
        mask = np.load(mf).astype(bool) if mf.exists() else np.ones_like(obs, bool)
        good = np.isfinite(obs) & (obs > 0) & mask & np.isfinite(err) & (err > 0)
        if int(good.sum()) < min_good:
            continue
        snr = float(np.median(obs[good] / err[good]))
        objs.append({"key": key, "obs": obs, "err": err, "good": good, "snr": snr})
    return objs


def classify(m):
    """MUNDANE vs GENUINE verdict from the de-confounded metrics."""
    if m is None:
        return "no_data"
    f = m["f_inflation"]
    perarm = m["chi2_goodpix_perarm"]
    if f <= 4.0:
        return "MUNDANE_modest_inflation"
    if perarm <= 4.0 and f > 4.0:
        return "MUNDANE_interarm_offset"
    if f > 10.0 and perarm > 10.0:
        return "GENUINE_large_gap"
    return "INTERMEDIATE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marg-models", nargs=3, required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--t8-key", default="jw02124-o059")
    ap.add_argument("--post-samples", type=int, default=400)
    ap.add_argument("--n-detail", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = get_default_track_config("g395h")
    backend = ExoJAXOpacityBackend(track_config=cfg, exojax_root=os.environ.get("EXOJAX_ROOT"))
    backend.prepare()
    chem = ChemistryModel.from_fastchem_hdf5(
        Path(os.environ.get("SCULPTOR_FASTCHEM_PATH", "/tmp/exojax_db/fastchem_grid.h5")), cfg)
    sm = SpectrumModel(track_config=cfg, chemistry_model=chem, opacity_backend=backend)
    print("forward model ready", flush=True)

    margs = [PosteriorInferencer(m) for m in args.marg_models]
    names = list(margs[0].target_parameter_names)
    L = int(margs[0].input_length)
    objs = load_real(args.inputs, L)
    print(f"{len(objs)} real objects", flush=True)

    def post_median(o, k):
        good = o["good"]
        x = self_normalize(np.where(good, o["obs"], np.nanmedian(o["obs"][good])).astype(np.float32))[None, :]
        cond = np.array([[o["snr"]]], np.float32)
        post = np.concatenate([m.sample(x, cond if m.conditioning_parameter_names else None,
                                        num_samples=k)[0] for m in margs], axis=0)
        return np.median(post, axis=0)

    nu_grid = None
    rows = []
    for o in objs:
        pmed = post_median(o, args.post_samples // 3)
        model, nu = forward_flux(sm, pmed, names)
        if nu is not None:
            nu_grid = nu
        if model is None:
            rows.append({"key": o["key"], "snr": o["snr"], "error": "forward_failed"})
            continue
        wl = 1e4 / nu
        m = deconfounded_metrics(o["obs"], o["err"], model, o["good"], wl)
        conf = confounded_chi2(o["obs"], o["err"], model, o["good"])
        if m is None:
            rows.append({"key": o["key"], "snr": o["snr"], "error": "too_few_good"})
            continue
        resid_perarm = m.pop("_resid_perarm_over_sig")
        good_idx = m.pop("_good_idx")
        row = {"key": o["key"], "snr": o["snr"],
               "is_t8_benchmark": bool(args.t8_key in o["key"]),
               "confounded_chi2_v1": conf, **m, "verdict": classify(m)}
        # band stats on the offset-removed standardized residual (good pixels)
        wl_good = wl[good_idx]
        bands = {}
        for lo, hi, nm in BANDS:
            sel = (wl_good >= lo) & (wl_good < hi) & np.isfinite(resid_perarm)
            if sel.any():
                bands[nm] = {"n": int(sel.sum()),
                             "chi2_per_dof": float(np.mean(resid_perarm[sel] ** 2))}
        row["band_chi2_perarm_offremoved"] = bands
        rows.append(row)
        print(f"  {o['key'][:42]:42} snr={o['snr']:6.1f} "
              f"conf={conf if conf is None else round(conf,1)} "
              f"abs={m['chi2_goodpix_absolute']:.1f} off={m['chi2_goodpix_offset']:.2f} "
              f"f={m['f_inflation']:.2f} perarm={m['chi2_goodpix_perarm']:.2f} "
              f"nrsR={m['nrs_flux_ratio_nrs1_over_nrs2']:.3f} -> {row['verdict']}", flush=True)

    scored = [r for r in rows if "f_inflation" in r]
    # summary population stats
    def med(key):
        vals = [r[key] for r in scored if r.get(key) is not None]
        return float(np.median(vals)) if vals else None
    t8 = next((r for r in scored if r["is_t8_benchmark"]), None)
    verd_counts = {}
    for r in scored:
        verd_counts[r["verdict"]] = verd_counts.get(r["verdict"], 0) + 1

    result = {
        "marg_models": args.marg_models, "inputs": args.inputs, "t8_key": args.t8_key,
        "target_names": names, "n_objects": len(scored),
        "population_median": {
            "confounded_chi2_v1": med("confounded_chi2_v1"),
            "chi2_goodpix_absolute": med("chi2_goodpix_absolute"),
            "chi2_goodpix_offset": med("chi2_goodpix_offset"),
            "f_inflation": med("f_inflation"),
            "chi2_goodpix_perarm": med("chi2_goodpix_perarm"),
        },
        "verdict_counts": verd_counts,
        "t8_benchmark": t8,
        "objects": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("\n=== DE-CONFOUNDED SUMMARY ===", flush=True)
    print(f"population median: confounded_v1={result['population_median']['confounded_chi2_v1']} "
          f"-> goodpix_offset={result['population_median']['chi2_goodpix_offset']} "
          f"f={result['population_median']['f_inflation']} "
          f"perarm={result['population_median']['chi2_goodpix_perarm']}", flush=True)
    if t8:
        print(f"T8 {t8['key'][:36]}: confounded={t8['confounded_chi2_v1']:.1f} "
              f"goodpix_offset={t8['chi2_goodpix_offset']:.2f} f={t8['f_inflation']:.2f} "
              f"perarm={t8['chi2_goodpix_perarm']:.2f} nrsR={t8['nrs_flux_ratio_nrs1_over_nrs2']:.3f} "
              f"-> {t8['verdict']}", flush=True)
    print(f"verdict counts: {verd_counts}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
