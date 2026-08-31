#!/usr/bin/env python3
"""Apply the marginalized and control NPEs to the REAL JWST G395H sample.

For every real object we self-normalize the (mask-filled) spectrum, condition on its
median SNR, and draw the amortized posterior from BOTH models:

  * MARGINALIZED  -- v7_opmarg NPE (opacity uncertainty folded into the posterior).
  * CONTROL       -- v6_cloud NPE (unperturbed; the naive amortized posterior).

The paper question is *which published constraints survive marginalization*: a parameter
is "constrained" for an object when its 90% posterior width is materially narrower than
the prior width (ratio < --survive-frac). We report this per object and pooled over the
population, for both models -- the constraints that survive under marginalization are the
honest, opacity-robust ones. The benchmark T8 dwarf is reported explicitly against the
Hood et al. C/O = 0.53 +/- 0.01 line.

Real spectra live locally in outputs/pop_exact_fit_inputs_local/ (<key>_observed.npy,
<key>_error.npy, optional <key>_mask.npy) and on GCS at
gs://sculptor-research-c2/pop_exact_fit_inputs/ (masks GCS-only). Exact-fit reference
values (Powell g395h_7knot fits) are read from outputs/popcold/<target_key>.json.

Usage:
  python scripts/apply_opmarg_real.py \
      --marg-model outputs/npe_v7_opmarg_seed0 \
      --control-model outputs/npe_v6_cloud \
      --inputs outputs/pop_exact_fit_inputs_local \
      --exact-fit-dir outputs/popcold \
      --out outputs/opmarg/real_apply.json
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

from sculptor.npe.dataset import self_normalize  # noqa: E402
from sculptor.npe.inference import PosteriorInferencer  # noqa: E402

# Default benchmark T8 dwarf key + Hood et al. reference C/O. The key is matched as a
# substring against the object keys; override with --t8-key if the sample naming differs.
DEFAULT_T8_KEY = "jw02124"
DEFAULT_HOOD_CO = 0.53
DEFAULT_HOOD_CO_ERR = 0.01
# Exact-fit parameters shared between the g395h_7knot Powell fit and the NPE targets.
EXACT_FIT_SHARED = ("carbon_to_oxygen", "metallicity", "log_kzz", "log_gravity")


def load_exact_fit(exact_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not exact_dir or not exact_dir.exists():
        return out
    for path in glob.glob(str(exact_dir / "*.json")):
        try:
            doc = json.loads(Path(path).read_text())
        except Exception:
            continue
        key = doc.get("target_key")
        bp = doc.get("best_parameters")
        if key and isinstance(bp, dict):
            out[key] = {"chi2_per_dof": doc.get("chi2_per_dof"),
                        "best_parameters": bp}
    return out


def match_exact(key: str, exact: dict[str, dict]) -> dict | None:
    if key in exact:
        return exact[key]
    # tolerate sub-aperture / suffix mismatches via prefix match
    for ek, ev in exact.items():
        if key.startswith(ek) or ek.startswith(key):
            return ev
    return None


def posterior_summary(post, target_names, prior_bounds, survive_frac):
    """Per-parameter median, 68% interval, 90% width, and survival flag."""
    out = {}
    for j, name in enumerate(target_names):
        s = post[:, j]
        med = float(np.median(s))
        lo68, hi68 = float(np.percentile(s, 16)), float(np.percentile(s, 84))
        w90 = float(np.percentile(s, 95) - np.percentile(s, 5))
        prior_w = None
        ratio = None
        constrained = None
        if name in prior_bounds:
            lo_p, hi_p = prior_bounds[name]
            prior_w = float(hi_p - lo_p)
            if prior_w > 0:
                ratio = w90 / prior_w
                constrained = bool(ratio < survive_frac)
        out[name] = {"median": med, "lo68": lo68, "hi68": hi68, "width90": w90,
                     "prior_width": prior_w, "width90_over_prior": ratio,
                     "constrained": constrained}
    return out


def apply_model(inf, x, snr, nsamp):
    cond = np.array([[snr]], dtype=np.float32) if inf.conditioning_parameter_names else None
    return inf.sample(x, cond, num_samples=nsamp)[0]  # (nsamp, n_targets)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marg-model", required=True)
    ap.add_argument("--control-model", required=True)
    ap.add_argument("--inputs", default=str(PROJECT_ROOT / "outputs/pop_exact_fit_inputs_local"))
    ap.add_argument("--objects-json", default=None,
                    help="optional JSON listing target_key entries to restrict/order the sample")
    ap.add_argument("--exact-fit-dir", default=str(PROJECT_ROOT / "outputs/popcold"))
    ap.add_argument("--nsamp", type=int, default=2000)
    ap.add_argument("--survive-frac", type=float, default=0.5,
                    help="a param is 'constrained' when 90%% posterior width < survive_frac * prior width")
    ap.add_argument("--min-good-pixels", type=int, default=1000)
    ap.add_argument("--t8-key", default=DEFAULT_T8_KEY,
                    help="substring identifying the benchmark T8 object (Hood et al.)")
    ap.add_argument("--hood-co", type=float, default=DEFAULT_HOOD_CO)
    ap.add_argument("--hood-co-err", type=float, default=DEFAULT_HOOD_CO_ERR)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    marg = PosteriorInferencer(args.marg_model)
    control = PosteriorInferencer(args.control_model)
    if tuple(marg.target_parameter_names) != tuple(control.target_parameter_names):
        raise ValueError("marginalized and control target parameters differ")
    target_names = list(marg.target_parameter_names)
    prior_bounds = dict(marg.target_parameter_bounds)
    L = int(marg.input_length)
    exact = load_exact_fit(Path(args.exact_fit_dir))
    print(f"targets={target_names}", flush=True)
    print(f"loaded {len(exact)} exact-fit reference objects", flush=True)

    # object list: optional objects.json ordering, else every *_observed.npy in --inputs
    inputs_dir = Path(args.inputs)
    if args.objects_json:
        keys = [d["target_key"] for d in json.loads(Path(args.objects_json).read_text())]
        obs_files = [inputs_dir / f"{k}_observed.npy" for k in keys]
    else:
        obs_files = [Path(p) for p in sorted(glob.glob(str(inputs_dir / "*_observed.npy")))]

    rows = []
    for of in obs_files:
        if not of.exists():
            continue
        key = of.name[: -len("_observed.npy")]
        ef = of.with_name(f"{key}_error.npy")
        mf = of.with_name(f"{key}_mask.npy")
        if not ef.exists():
            continue
        obs = np.load(of).astype(float)
        err = np.load(ef).astype(float)
        if obs.shape[0] != L:
            continue  # grid mismatch
        mask = np.load(mf).astype(bool) if mf.exists() else np.ones_like(obs, dtype=bool)
        good = np.isfinite(obs) & (obs > 0) & mask & np.isfinite(err) & (err > 0)
        if int(good.sum()) < args.min_good_pixels:
            continue
        snr = float(np.median(obs[good] / err[good]))
        filled = np.where(good, obs, np.nanmedian(obs[good])).astype(np.float32)
        x = self_normalize(filled)[None, :].astype(np.float32)

        marg_post = apply_model(marg, x, snr, args.nsamp)
        ctrl_post = apply_model(control, x, snr, args.nsamp)
        marg_sum = posterior_summary(marg_post, target_names, prior_bounds, args.survive_frac)
        ctrl_sum = posterior_summary(ctrl_post, target_names, prior_bounds, args.survive_frac)

        ex = match_exact(key, exact)
        row = {
            "key": key, "snr": snr, "n_good": int(good.sum()),
            "is_t8_benchmark": bool(args.t8_key and args.t8_key in key),
            "marginalized": marg_sum, "control": ctrl_sum,
            "exact_fit": {k: ex["best_parameters"].get(k) for k in EXACT_FIT_SHARED} if ex else None,
            "exact_fit_chi2_per_dof": ex.get("chi2_per_dof") if ex else None,
        }
        rows.append(row)
        co_m = marg_sum.get("carbon_to_oxygen", {})
        print(f"{key[:46]:46} snr={snr:6.1f}  C/O marg={co_m.get('median', float('nan')):.3f}"
              f"[w90={co_m.get('width90', float('nan')):.3f}]  "
              f"ctrl={ctrl_sum.get('carbon_to_oxygen', {}).get('median', float('nan')):.3f}", flush=True)

    # population-level survival: fraction of objects constrained + median width ratio per param
    def pooled(model_key):
        table = {}
        for name in target_names:
            ratios = [r[model_key][name]["width90_over_prior"] for r in rows
                      if r[model_key][name]["width90_over_prior"] is not None]
            flags = [r[model_key][name]["constrained"] for r in rows
                     if r[model_key][name]["constrained"] is not None]
            table[name] = {
                "median_width90_over_prior": float(np.median(ratios)) if ratios else None,
                "frac_constrained": float(np.mean(flags)) if flags else None,
                "n": len(flags),
            }
        return table

    population = {"marginalized": pooled("marginalized"), "control": pooled("control")}

    # benchmark T8 comparison line
    t8 = None
    t8_rows = [r for r in rows if r["is_t8_benchmark"]]
    if t8_rows:
        # if multiple sub-apertures match, take the highest-SNR one
        r = max(t8_rows, key=lambda z: z["snr"])
        t8 = {
            "key": r["key"], "snr": r["snr"],
            "hood_co": args.hood_co, "hood_co_err": args.hood_co_err,
            "marg_co": r["marginalized"].get("carbon_to_oxygen"),
            "control_co": r["control"].get("carbon_to_oxygen"),
            "exact_fit_co": (r["exact_fit"] or {}).get("carbon_to_oxygen") if r["exact_fit"] else None,
        }

    result = {
        "marg_model": str(args.marg_model), "control_model": str(args.control_model),
        "inputs": str(inputs_dir), "n_objects": len(rows),
        "survive_frac": args.survive_frac, "target_names": target_names,
        "prior_bounds": {k: [float(v[0]), float(v[1])] for k, v in prior_bounds.items()},
        "population_survival": population, "t8_benchmark": t8, "objects": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== population survival (fraction of objects constrained: width90 < "
          f"{args.survive_frac} x prior) ===")
    print(f"{'param':18} | {'marg frac':>9} {'marg wr':>8} | {'ctrl frac':>9} {'ctrl wr':>8}")
    for name in target_names:
        m = population["marginalized"][name]
        c = population["control"][name]
        def fmt(x, p=".2f"):
            return format(x, p) if x is not None else "  n/a"
        print(f"{name:18} | {fmt(m['frac_constrained']):>9} {fmt(m['median_width90_over_prior']):>8} | "
              f"{fmt(c['frac_constrained']):>9} {fmt(c['median_width90_over_prior']):>8}")
    if t8:
        print(f"\nT8 benchmark {t8['key'][:40]}: Hood C/O={t8['hood_co']}+/-{t8['hood_co_err']}"
              f"  marg={t8['marg_co']['median']:.3f}[{t8['marg_co']['lo68']:.2f},{t8['marg_co']['hi68']:.2f}]"
              f"  control={t8['control_co']['median']:.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
