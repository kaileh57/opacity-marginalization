#!/usr/bin/env python
"""Generate the three publication figures for the ML4PS opacity-marginalized
retrieval paper.

Final figure set (v3.0):

  fig1_deployment_maskaug -- deployment-preprocessing collapse and the
                             mask-augmentation fix (item6 / item9)
  fig2_real_sample        -- 105 real G395H spectra, C/O vs the T8 benchmark
  fig3_sbc_tarp           -- appendix: C/O PIT histograms + TARP ECP deviation
                             (n=2048, item3)

Figures dropped in v3.0 (superseded by tables in the text, or orphaned):
the C/O reliability curves and the four-method calibration benchmark (both
fully contained in Table 1), the amplitude sweep and the OOD bar chart (a
handful of numbers each, given in the text).

Every plotted value is read directly from the committed result JSONs under
``outputs/paper_opmarg/`` -- nothing is hand-typed. Run from anywhere; paths
are resolved relative to this file:

    python scripts/plot_ml4ps_final_figs.py

Outputs 300-dpi PNG plus matching vector PDF into
``outputs/paper_opmarg/ml4ps_draft/figures/``.

Style: no in-image titles beyond panel labels (captions live in LaTeX),
Okabe-Ito colorblind-safe palette, half-frame axes, faint y-grid only. Each
figure is drawn at the size it is printed at (fig1 3.55 in, fig2/fig3 at the
5.5 in NeurIPS text width) so LaTeX includes it at scale 1.0 and the 7-9 pt
type stays 7-9 pt on the page.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "outputs" / "paper_opmarg" / "final"
V2 = REPO / "outputs" / "paper_opmarg" / "v2"
JOURNAL = REPO / "outputs" / "paper_opmarg" / "journal"
OUT = REPO / "outputs" / "paper_opmarg" / "ml4ps_draft" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Shared style
# --------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette. Blue family = marginalized models,
# orange = un-marginalized control, gray = chrome/reference.
C_CTRL = "#E69F00"     # orange     -> control (clean-trained)
C_ENS = "#001B44"      # dark navy  -> 3-seed marginalized ensemble / mask-aug
C_SEED0 = "#56B4E9"    # light blue -> single marginalized seed
C_REF = "#444444"      # nominal / reference lines
GRID = "#DDDDDD"

NOMINAL = 0.90

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COL_W = 5.5  # single-column NeurIPS text width (inches)


def _tufte(ax):
    """Strip chartjunk: hide top/right spines, ticks out, data above grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)
    ax.set_axisbelow(True)


def _save(fig, name):
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png.name} + {pdf.name}")


# --------------------------------------------------------------------------
# Figure 1 -- deployment preprocessing collapse and the mask-augmentation fix
# --------------------------------------------------------------------------
def fig1_deployment():
    """Grouped bars, C/O coverage at nominal 0.90, on one held-out set.

    Both series come from item9_maskaug_stress.json so that the comparison is
    same-eval and same-seed: its ``control_seed0`` slot holds the ORIGINAL
    marginalized seed 0 and its ``marg_ensemble3`` slot holds the
    mask-augmented seed 0 (listed three times, so it is a single network, not
    an ensemble). The three-seed ensemble's collapse under the same stress
    (item6) is quoted in the caption, not plotted, because it was measured on
    a different held-out set.
    """
    d9 = json.loads((JOURNAL / "item9_maskaug_stress.json").read_text())
    d6 = json.loads((JOURNAL / "item6_deploy_preproc_stress.json").read_text())

    variants = ["baseline", "real_mask_fill", "full_deploy"]
    labels = ["clean\n(no masking)", "real-mask\nmedian fill", "full deploy\n(mask + renoise)"]
    orig = [d9["variants"][v]["control_seed0"]["co_cov90"] for v in variants]
    aug = [d9["variants"][v]["marg_ensemble3"]["co_cov90"] for v in variants]
    ens6 = [d6["variants"][v]["marg_ensemble3"]["co_cov90"] for v in variants]

    print("fig1 (variant -> original marg seed0 / mask-aug seed0 | item6 ens3):")
    for lab, o, a, e in zip(variants, orig, aug, ens6):
        print(f"   {lab:16s} {o:.4f} / {a:.4f}  | {e:.4f}")

    x = np.arange(len(variants))
    bw = 0.34
    fig, ax = plt.subplots(figsize=(3.55, 2.05))
    _tufte(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.6)

    ax.axhline(NOMINAL, ls="--", lw=1.0, color=C_REF, zorder=1)
    ax.text(x[-1] + bw + 0.13, NOMINAL + 0.012, "nominal 0.90", ha="right",
            va="bottom", fontsize=7, color=C_REF)

    b1 = ax.bar(x - bw / 2, orig, width=bw, color=C_SEED0, edgecolor="white",
                lw=0.8, zorder=2, label="no mask augmentation")
    b2 = ax.bar(x + bw / 2, aug, width=bw, color=C_ENS, edgecolor="white",
                lw=0.8, zorder=2, label="mask-augmented")
    for bars, vals in [(b1, orig), (b2, aug)]:
        for rect, v in zip(bars, vals):
            # tall bars would push their label into the nominal-0.90 rule, so
            # those labels sit inside the bar instead
            inside = v > 0.80
            ax.text(rect.get_x() + rect.get_width() / 2,
                    v - 0.035 if inside else v + 0.014, f"{v:.3f}",
                    ha="center", va="top" if inside else "bottom", fontsize=7,
                    color="white" if inside else "black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("C/O coverage at nominal 0.90")
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_bounds(0.0, 1.0)
    # two series, three groups: direct labels would collide with the bars, so
    # the series key goes in the top margin instead
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.55, 1.03), handlelength=1.4,
               columnspacing=1.8, handletextpad=0.5)
    _save(fig, "fig1_deployment_maskaug")


# --------------------------------------------------------------------------
# Figure 2 -- real-sample C/O against the T8 benchmark
# --------------------------------------------------------------------------
def fig2_real_sample():
    ens = json.loads((DATA / "ensemble3.json").read_text())
    real = json.loads((V2 / "real_apply_seed0.json").read_text())
    ctrl_by_key = {o["key"]: o for o in real["objects"]}
    hood_co = ens["t8_benchmark"]["hood_co"]
    hood_err = ens["t8_benchmark"]["hood_co_err"]
    t8_key = ens["t8_benchmark"]["key"]

    objs = sorted(ens["objects"], key=lambda o: o["snr"])
    xs = np.arange(len(objs))
    med, lo, hi, ctrl_med, t8_idx = [], [], [], [], []
    for i, o in enumerate(objs):
        co = o["ensemble"]["carbon_to_oxygen"]
        med.append(co["median"])
        lo.append(co["median"] - co["lo68"])
        hi.append(co["hi68"] - co["median"])
        c = ctrl_by_key.get(o["key"], {}).get("control", {}).get("carbon_to_oxygen", {})
        ctrl_med.append(c.get("median", np.nan))
        if o.get("is_t8_benchmark") or o["key"] == t8_key:
            t8_idx.append(i)

    print(f"fig2: {len(objs)} spectra, Hood C/O={hood_co}+-{hood_err}")

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    _tufte(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.6)

    ax.axhspan(hood_co - hood_err, hood_co + hood_err, color="#F0C808",
               alpha=0.55, zorder=0)
    ax.axhline(hood_co, color="#C8A200", lw=1.0, zorder=1)
    # label the band in the clear right margin, not over the data
    ax.text(len(objs) + 1.5, hood_co, f"Hood et al.\nT8 C/O$=${hood_co}",
            ha="left", va="center", fontsize=7, color="#8A7200")

    # spectra below the training SNR support are shown, but de-emphasized:
    # they are out-of-distribution for the conditioning prior [10, 200]
    snrs = np.array([o["snr"] for o in objs])
    n_out = int((snrs < 10.0).sum())
    sup = np.arange(len(objs)) >= n_out
    for sel, alpha in ((~sup, 0.28), (sup, 1.0)):
        if not sel.any():
            continue
        ax.plot(xs[sel], np.asarray(ctrl_med)[sel], "s", color=C_CTRL,
                mec="white", mew=0.4, ms=4.0, alpha=alpha, zorder=2)
        ax.errorbar(xs[sel], np.asarray(med)[sel],
                    yerr=[np.asarray(lo)[sel], np.asarray(hi)[sel]], fmt="o",
                    color=C_ENS, ms=4.0, mec="white", mew=0.4, ecolor=C_ENS,
                    elinewidth=0.8, capsize=1.6, alpha=alpha, zorder=3)
    ax.axvline(n_out - 0.5, color=C_REF, ls="--", lw=0.9, zorder=1)
    ax.text(n_out - 1.5, 0.115, f"SNR $<$ 10 ({n_out})", ha="right",
            va="bottom", fontsize=7, color=C_REF)
    for ti in t8_idx:
        ax.axvline(ti, color=C_REF, ls=":", lw=1.0, zorder=1)
        ax.text(ti + 1.2, 1.68, "T8", ha="left", va="top", fontsize=7,
                color=C_REF)

    # direct labels in the headroom above the data, clear of every point
    ax.text(1.0, 1.68, "marg, 3-seed ens. (68% bars)", ha="left", va="top",
            fontsize=7.5, color=C_ENS)
    ax.text(1.0, 1.575, "control (median)", ha="left", va="top", fontsize=7.5,
            color="#B87D00")

    ax.set_xlabel("105 G395H spectral products of 13 late-T/Y dwarfs, ordered by SNR")
    ax.set_ylabel("Retrieved C/O")
    ax.set_xticks([])
    ax.set_xlim(-2, len(objs) + 13)
    ax.set_ylim(0.08, 1.72)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    ax.spines["left"].set_bounds(0.25, 1.5)
    ax.spines["bottom"].set_visible(False)
    _save(fig, "fig2_real_sample")


# --------------------------------------------------------------------------
# Figure 3 (appendix) -- C/O PIT histograms + TARP expected-coverage deviation
# --------------------------------------------------------------------------
def fig3_sbc_tarp():
    d = json.loads((JOURNAL / "item3_sbc_tarp.json").read_text())
    n_obj = d["n_obj"]
    models = [
        ("control_seed0", "control", C_CTRL),
        ("marg_seed0", "marg, 1 seed", C_SEED0),
        ("marg_ensemble3", "marg, 3-seed ens.", C_ENS),
    ]

    # This figure sits alone on the final appendix page, so it is drawn at the
    # full text width and roughly half again as tall as the in-line figures,
    # with type scaled to match. Drawn width stays COL_W, so \linewidth in
    # LaTeX includes it at scale 1.0 and the point sizes below are the point
    # sizes on the page.
    with mpl.rc_context({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }):
        _fig3_body(d, n_obj, models)


def _fig3_body(d, n_obj, models):
    fig, axes = plt.subplots(1, 4, figsize=(COL_W, 3.05),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.4]})

    # -- panels A-C: C/O PIT histograms -------------------------------------
    # The control's U-shape is so extreme that a shared y-scale flattens the two
    # marginalized histograms into featureless blocks and hides the residual
    # bias the text discusses. The control keeps its own scale; the two
    # marginalized panels share a second, expanded one.
    counts_by_key = {}
    for ax, (key, label, col) in zip(axes[:3], models):
        h = d["models"][key]["sbc_per_param"]["carbon_to_oxygen"]
        edges = np.asarray(h["pit_bin_edges"])
        counts = np.asarray(h["pit_hist_counts"], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        uniform = counts.sum() / len(counts)
        _tufte(ax)
        ax.bar(centers, counts, width=widths * 0.92, color=col,
               edgecolor="white", lw=0.4, zorder=2)
        ax.axhline(uniform, ls="--", lw=1.0, color=C_REF, zorder=3)
        # the panel titles double as the colour key for the TARP panel, so they
        # carry each model's colour and no curve needs a direct label
        ax.set_title(label, fontsize=10, loc="left", color=col)
        ax.set_xlabel("C/O PIT")
        ax.set_xlim(0, 1)
        # short tick strings: at this type size the default "0.0"/"0.5"/"1.0"
        # collide inside a panel this narrow
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        counts_by_key[key] = counts
        print(f"fig3 PIT {key}: rank_mean={h['rank_mean']:.3f} "
              f"chi2={h['chi2_stat']:.1f}")
    ctrl_max = counts_by_key["control_seed0"].max()
    marg_max = max(counts_by_key["marg_seed0"].max(),
                   counts_by_key["marg_ensemble3"].max())

    def _round_ticks(top, step):
        return [t for t in np.arange(0, top + 1, step) if t <= top]

    axes[0].set_ylim(0, ctrl_max * 1.10)
    axes[0].set_yticks(_round_ticks(ctrl_max, 200))
    axes[0].spines["left"].set_bounds(0, _round_ticks(ctrl_max, 200)[-1])
    for ax in axes[1:3]:
        ax.set_ylim(0, marg_max * 1.14)
        ax.set_yticks(_round_ticks(marg_max, 100))
        ax.spines["left"].set_bounds(0, _round_ticks(marg_max, 100)[-1])
    axes[0].set_ylabel(f"count ({n_obj} objects)")
    # panel 3 shares panel 2's scale: drop its labels AND its tick marks, so no
    # orphaned tick dash floats above the bars
    axes[2].tick_params(left=False, labelleft=False)

    # -- panel D: TARP expected-coverage deviation --------------------------
    axd = axes[3]
    _tufte(axd)
    axd.axhline(0.0, ls="--", lw=1.0, color=C_REF, zorder=1)
    for key, label, col in models:
        t = d["models"][key]["tarp"]
        alpha = np.asarray(t["alpha_grid"])
        dev = np.asarray(t["ecp"]) - alpha
        axd.plot(alpha, dev, color=col, lw=1.4, zorder=3)
        print(f"fig3 TARP {key}: max_dev={t['max_deviation']:.4f}")
    # right-aligned so the long "marg, 3-seed ens." title of the panel to its
    # left, which overhangs into this column, cannot run into it
    axd.set_title("TARP", fontsize=10, loc="right", color=C_REF)
    axd.set_xlabel("Credibility")
    axd.set_ylabel("ECP $-$ nominal")
    axd.set_xlim(0, 1)
    axd.set_xticks([0, 0.5, 1])
    axd.set_xticklabels(["0", "0.5", "1"])
    axd.set_ylim(-0.050, 0.030)
    axd.set_yticks([-0.04, -0.02, 0.0, 0.02])
    axd.spines["bottom"].set_bounds(0, 1)
    axd.spines["left"].set_bounds(-0.04, 0.02)
    axd.tick_params(labelsize=9)

    # direct labels where the three curves are best separated (mid-range),
    # kept short so they stay inside the narrow panel
    # No direct labels here: the panel is too narrow to carry three of them
    # without collisions, and the colour-coded PIT titles to the left already
    # identify every curve.

    fig.tight_layout(w_pad=1.1)
    _save(fig, "fig3_sbc_tarp")


def main():
    print(f"final/   : {DATA}")
    print(f"journal/ : {JOURNAL}")
    print(f"output   : {OUT}\n")
    fig1_deployment()
    fig2_real_sample()
    fig3_sbc_tarp()
    print("\nAll figures written.")


if __name__ == "__main__":
    main()
