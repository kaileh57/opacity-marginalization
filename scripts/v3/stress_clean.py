"""Lean rerun of the deployment stress test with clean masks.

Uses the functions in scripts/opmarg_deploy_preproc_stress.py; every model sees
the same preprocessed spectra per variant, so masks are matched across models.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
torch.set_num_threads(2)

import opmarg_deploy_preproc_stress as S  # noqa: E402
from sculptor.npe.inference import PosteriorInferencer  # noqa: E402

H, MASKDIR, OUT = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
unaug = [PosteriorInferencer(str(H / f"models/npe_v7_opmarg_seed{s}")) for s in range(3)]
aug = [PosteriorInferencer(str(H / f"models/journal_round/npe_v7_opmarg_maskaug_seed{s}"))
       for s in range(3)]
ctrl = [PosteriorInferencer(str(H / "models/npe_v6_cloud_seed0"))]
for m in unaug + aug + ctrl:
    assert m.model.flow.backend_name == "zuko_nsf"

inf = unaug[0]
names = list(inf.target_parameter_names)
cnames = list(inf.conditioning_parameter_names)
L = int(inf.input_length)
raw, truths, cond = S.load_heldout_raw(H / "heldout/perturbed", names, cnames, L, 1024)
snr = cond[:, cnames.index("snr")]
masks = S.load_real_masks(MASKDIR, L)
res = {"n_obj": len(raw), "n_real_masks": len(masks),
       "median_real_good_fraction": float(np.median([m.mean() for m in masks])),
       "mask_dir": str(MASKDIR), "variants": {}}
print(len(raw), "objects,", len(masks), "masks, median good fraction",
      res["median_real_good_fraction"], flush=True)

VARIANTS = sys.argv[5].split(",") if len(sys.argv) > 5 else ["baseline", "real_mask_fill", "full_deploy"]
TAGS = sys.argv[6].split(",") if len(sys.argv) > 6 else None
ALL = ["baseline", "real_mask_fill", "full_deploy"]
for variant in VARIANTS:
    vi = ALL.index(variant)
    rng = np.random.default_rng(20260718 + vi)
    spec = S.build_variant(raw, snr, masks, rng, variant)
    out = {}
    for tag, models, k in [("marg_ensemble3", unaug, 133),
                           ("maskaug_ensemble3", aug, 133),
                           ("control_seed0", ctrl, 400)]:
        if TAGS and tag not in TAGS:
            continue
        t = time.time()
        post = S.sample_ensemble(models, spec, cond, k, chunk=4, label=f"{variant}/{tag}")
        out[tag] = S.coverage_stats(post, truths, names)
        print(f"{variant:15s} {tag:18s} C/O cov90 {out[tag]['co_cov90']:.3f} "
              f"overall {out[tag]['overall_coverage']['0.9']:.3f} "
              f"w90/prior {out[tag]['co_median_width90'] / 1.4:.3f} "
              f"({time.time() - t:.0f}s)", flush=True)
    res["variants"][variant] = out
    OUT.write_text(json.dumps(res, indent=1))
print("done", flush=True)
