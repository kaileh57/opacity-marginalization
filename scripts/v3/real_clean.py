"""Real-data C/O posteriors for the clean 14-object JWST set, every model.

Preprocessing matches scripts/opmarg_maskaug_real_apply.py: good-pixel mask,
median fill, self-normalization, median-SNR conditioning.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
torch.set_num_threads(1)
torch.manual_seed(20261029)

from sculptor.npe.dataset import self_normalize  # noqa: E402
from sculptor.npe.inference import PosteriorInferencer  # noqa: E402

H, INP, OUT = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
N = 2001
models = {f"control_s{s}": [str(H / f"models/npe_v6_cloud_seed{s}")] for s in range(3)}
models.update({f"marg_s{s}": [str(H / f"models/npe_v7_opmarg_seed{s}")] for s in range(3)})
models.update({f"maskaug_s{s}": [str(H / f"models/journal_round/npe_v7_opmarg_maskaug_seed{s}")]
               for s in range(3)})
models["control_ens3"] = [m for k in ("control_s0", "control_s1", "control_s2") for m in models[k]]
models["marg_ens3"] = [m for k in ("marg_s0", "marg_s1", "marg_s2") for m in models[k]]
models["maskaug_ens3"] = [m for k in ("maskaug_s0", "maskaug_s1", "maskaug_s2") for m in models[k]]
cache = {}
for paths in models.values():
    for p in paths:
        if p not in cache:
            cache[p] = PosteriorInferencer(p)
            assert cache[p].model.flow.backend_name == "zuko_nsf"

out = {"draws_per_model_total": N, "objects": {}}
for of in sorted(glob.glob(str(INP / "*_observed.npy"))):
    key = Path(of).name[: -len("_observed.npy")]
    obs = np.load(of).astype(float)
    err = np.load(INP / f"{key}_error.npy").astype(float)
    mask = np.load(INP / f"{key}_mask.npy").astype(bool)
    good = np.isfinite(obs) & (obs > 0) & mask & np.isfinite(err) & (err > 0)
    snr = float(np.median(obs[good] / err[good]))
    x = self_normalize(np.where(good, obs, np.nanmedian(obs[good])).astype(np.float32))
    x = x[None, :].astype(np.float32)
    row = {"snr": snr, "good_fraction": float(good.mean())}
    for tag, paths in models.items():
        per = N // len(paths)
        s = np.concatenate([
            cache[p].sample(x, np.array([[snr]], np.float32), num_samples=per)[0]
            for p in paths])
        j = list(cache[paths[0]].target_parameter_names).index("carbon_to_oxygen")
        c = s[:, j]
        row[tag] = {q: float(v) for q, v in zip(
            ("median", "lo68", "hi68", "lo90", "hi90"),
            (np.median(c), *np.percentile(c, [16, 84, 5, 95])))}
    out["objects"][key] = row
    print(key[:40], f"snr={snr:.1f}", f"maskaug_ens3={row['maskaug_ens3']['median']:.3f}",
          flush=True)
    OUT.write_text(json.dumps(out, indent=1))
print("done", flush=True)
