#!/usr/bin/env python3
"""Rescale an opacity-perturbation priors JSON by a global amplitude factor.

The generation pipeline reads its continuous-field amplitudes from the
``suggested_priors`` block of the measure_interlist_ratios.py document
(SCULPTOR_OPACITY_PERTURB_PRIORS). Every sigma in that block is a standard
deviation in natural-log units, and the field enters as ``exp(f)`` with f linear
in the coefficients, so multiplying ALL sigmas by one factor is exactly a global
rescaling of the perturbation amplitude.

Two production uses:

  * amplitude sweep (Track B): scale 0.4286 / 1.0 / 1.4286 / 2.1429 around the
    0.07-equivalent training amplitude.
  * DISCRETE-ONLY ablation: ``--scale 0`` gives sigma == 0 for every species and
    mode, so ``sample_coefficients`` returns EXACT zeros, the log field is
    exactly 0.0, and ``exp(0) == 1.0`` multiplies the cross-sections bit-exactly
    (IEEE-754: x * 1.0 == x). The continuous axis is then OFF while the discrete
    line-list-swap axis (combo = shard % 2**n_alt) still varies shard by shard.

The whole source document is preserved (measurements, provenance, timestamps);
only the priors block is rescaled, and a ``prior_scale`` record is added.

Usage:
  python scripts/scale_opmarg_priors.py --in interlist_ratio_priors_v1.json \
      --scale 0.0 --out interlist_ratio_priors_amp000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIOR_KEYS = ("suggested_priors", "priors")


def scale_priors(document: dict, scale: float) -> tuple[dict, str]:
    """Return a copy of ``document`` with every prior sigma multiplied by ``scale``."""
    if scale < 0.0:
        raise ValueError(f"--scale must be >= 0 (got {scale})")
    key = next((k for k in PRIOR_KEYS if isinstance(document.get(k), dict)), None)
    if key is None:
        raise ValueError(f"priors JSON has none of {PRIOR_KEYS} as a dict")
    scaled = json.loads(json.dumps(document))  # deep copy, JSON-safe
    block = scaled[key]
    if not block:
        raise ValueError(f"priors JSON block {key!r} is empty")
    for species, entry in block.items():
        if "sigma_const" not in entry or "sigma_modes" not in entry:
            raise ValueError(f"entry {species!r} must define sigma_const and sigma_modes")
        entry["sigma_const"] = float(entry["sigma_const"]) * scale
        entry["sigma_modes"] = [float(x) * scale for x in entry["sigma_modes"]]
    scaled["prior_scale"] = float(scale)
    return scaled, key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, help="source priors JSON")
    ap.add_argument("--scale", type=float, required=True, help="global multiplier on every sigma")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default=None, help="free-text provenance note stored as scale_note")
    args = ap.parse_args()

    source = json.loads(Path(args.in_path).read_text())
    scaled, key = scale_priors(source, args.scale)
    scaled["scaled_from"] = str(args.in_path)
    if args.note:
        scaled["scale_note"] = args.note

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scaled, indent=2), encoding="utf-8")

    print(f"scaled {key} by {args.scale} -> {out_path}")
    for species, entry in scaled[key].items():
        print(f"  {species:5} sigma_const={entry['sigma_const']:.6f} "
              f"sigma_modes={[round(x, 6) for x in entry['sigma_modes']]}")


if __name__ == "__main__":
    main()
