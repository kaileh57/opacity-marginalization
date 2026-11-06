"""Figure 1 for v3: deployment stress with the 14 clean masks.

Both series are three-seed ensembles scored on the same masked spectra.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_ENS, C_SEED0, C_REF, GRID = "#001B44", "#56B4E9", "#444444", "#DDDDDD"
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

res = {}
for f in sys.argv[1:-1]:
    for v, d in json.loads(Path(f).read_text())["variants"].items():
        res.setdefault(v, {}).update(d)
out = Path(sys.argv[-1])
variants = ["baseline", "real_mask_fill", "full_deploy"]
labels = ["unmasked", "real-mask\nmedian fill", "full path\n(mask + re-noise)"]
orig = [res[v]["marg_ensemble3"]["co_cov90"] for v in variants]
aug = [res[v]["maskaug_ensemble3"]["co_cov90"] for v in variants]
print("unaug", orig, "aug", aug)

x = np.arange(len(variants))
bw = 0.34
fig, ax = plt.subplots(figsize=(3.55, 2.05))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(direction="out", length=3)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.axhline(0.90, ls="--", lw=1.0, color=C_REF, zorder=1)
ax.text(x[-1] + bw + 0.13, 0.912, "nominal 0.90", ha="right", va="bottom", fontsize=7,
        color=C_REF)
b1 = ax.bar(x - bw / 2, orig, width=bw, color=C_SEED0, edgecolor="white", lw=0.8,
            zorder=2, label="no mask augmentation")
b2 = ax.bar(x + bw / 2, aug, width=bw, color=C_ENS, edgecolor="white", lw=0.8,
            zorder=2, label="mask-augmented")
for bars, vals in ((b1, orig), (b2, aug)):
    for rect, v in zip(bars, vals):
        inside = v > 0.80
        ax.text(rect.get_x() + rect.get_width() / 2, v - 0.035 if inside else v + 0.014,
                f"{v:.3f}", ha="center", va="top" if inside else "bottom", fontsize=7,
                color="white" if inside else "black")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("C/O coverage at nominal 0.90")
ax.set_ylim(0.0, 1.02)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.spines["left"].set_bounds(0.0, 1.0)
fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=8,
           bbox_to_anchor=(0.55, 1.03), handlelength=1.4, columnspacing=1.8,
           handletextpad=0.5)
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"))
print("wrote", out)
