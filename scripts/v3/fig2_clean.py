"""Figure 2 for v3: C/O for the 14 clean JWST spectra against the T8 benchmark.

Style copied from scripts/plot_ml4ps_final_figs.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_CTRL, C_ENS, C_REF, GRID = "#E69F00", "#001B44", "#444444", "#DDDDDD"
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 6.5, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333", "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

NAMES = {
    "jw01189-o004": "W0458", "jw01189-o011": "W1828", "jw02124-o003": "J0535",
    "jw02124-o004": "J0825", "jw02124-o005": "J1541", "jw02124-o007": "J1711",
    "jw02124-o008": "J0751", "jw02124-o010": "J2220", "jw02124-o011": "J1405",
    "jw02124-o012": "J2354", "jw02124-o051": "J1935", "jw02124-o052": "J0503",
    "jw02124-o056": "J0146", "jw02124-o059": "J0415",
}

d = json.loads(Path(sys.argv[1]).read_text())["objects"]
out = Path(sys.argv[2])
keys = sorted(d, key=lambda k: d[k]["snr"])
xs = np.arange(len(keys))
ens = [d[k]["maskaug_ens3"] for k in keys]
ctrl = [d[k]["control_s0"]["median"] for k in keys]
snr = np.array([d[k]["snr"] for k in keys])
labels = [f"{NAMES[k[:12]]} ({d[k]['snr']:.0f})" for k in keys]
hood_co, hood_err = 0.53, 0.01

fig, ax = plt.subplots(figsize=(5.5, 2.25))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="out", length=3)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.axhspan(hood_co - hood_err, hood_co + hood_err, color="#F0C808", alpha=0.55, zorder=0)
ax.axhline(hood_co, color="#C8A200", lw=1.0, zorder=1)
ax.text(len(keys) - 0.35, hood_co, "Hood et al.\nT8 C/O$=$0.53", ha="left",
        va="center", fontsize=7, color="#8A7200")

n_out = int((snr < 10).sum())
for i in range(len(keys)):
    a = 0.35 if snr[i] < 10 else 1.0
    e = ens[i]
    ax.errorbar(xs[i], e["median"], yerr=[[e["median"] - e["lo68"]], [e["hi68"] - e["median"]]],
                fmt="o", color=C_ENS, ms=4.2, mec="white", mew=0.4, ecolor=C_ENS,
                elinewidth=0.9, capsize=1.8, alpha=a, zorder=3)
    ax.plot(xs[i] + 0.18, ctrl[i], "s", color=C_CTRL, mec="white", mew=0.4, ms=4.0,
            alpha=a, zorder=2)
ax.axvline(n_out - 0.5, color=C_REF, ls="--", lw=0.9, zorder=1)
ax.text(n_out - 0.6, 1.13, "SNR $<$ 10", ha="right", va="top", fontsize=7, color=C_REF)
ax.text(n_out - 0.3, 1.13, "mask-aug. ensemble, median and 68% interval", ha="left",
        va="top", fontsize=7.5, color=C_ENS)
ax.text(n_out - 0.3, 1.05, "fixed-opacity control (seed 0), median", ha="left",
        va="top", fontsize=7.5, color="#B87D00")
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=40, ha="right", rotation_mode="anchor")
ax.set_xlabel("Object and median SNR (one combined G395H spectrum each)")
ax.set_ylabel("Retrieved C/O")
ax.set_xlim(-0.6, len(keys) + 1.9)
ax.set_ylim(0.1, 1.15)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.spines["left"].set_bounds(0.25, 1.0)
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"))
print("wrote", out)
