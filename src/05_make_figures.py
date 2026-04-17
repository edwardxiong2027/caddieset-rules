"""
05_make_figures.py  —  Reproduce every figure in paper/figures/ from the artifacts.

Run in this order:
  python src/01_explore.py
  python src/02_effect_sizes.py
  python src/03_archetypes.py
  python src/04_rules.py
  python src/05_make_figures.py

Outputs 6 figures (PDF + PNG) under paper/figures/.
"""
from __future__ import annotations
import os
import pickle
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "figure.dpi": 130, "savefig.dpi": 200,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

PHASES = {
    "0": "P0\nAddress", "1": "P1\nTakeaway", "2": "P2\nMid-BS", "3": "P3\nTop",
    "4": "P4\nEarly-DS", "5": "P5\nPre-Imp", "6": "P6\nImpact", "7": "P7\nFinish",
}
ARCHETYPE_NAMES = {
    0: "A0 · Early Hip Release", 1: "A1 · Balanced Coil",
    2: "A2 · Aggressive Rotator", 3: "A3 · Compact Swing",
}
ARCH_COLORS = {0: "#d96c6c", 1: "#3a7ca5", 2: "#eaa94c", 3: "#7c9f3a"}


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    stages = [
        ("Multi-view\nswing videos", 0.5, "#3a7ca5"),
        ("8-phase\nsegmentation", 2.3, "#4a90b5"),
        ("68 body-posture\nfeatures", 4.2, "#56a0c5"),
        ("Effect-size\nranking", 6.0, "#eaa94c"),
        ("Archetype\nclustering", 7.7, "#d96c6c"),
        ("Coaching\nrule tree", 9.3, "#7c9f3a"),
    ]
    for i, (name, x, c) in enumerate(stages):
        ax.add_patch(plt.Rectangle((x - 0.55, 1.2), 1.4, 1.5, facecolor=c, edgecolor="black", lw=0.6))
        ax.text(x + 0.15, 1.95, name, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        if i < len(stages) - 1:
            nx = stages[i + 1][1]
            ax.annotate("", xy=(nx - 0.55, 1.95), xytext=(x + 0.85, 1.95),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.text(5.0, 3.55, "From Prediction to Prescription — Paper Pipeline",
            ha="center", fontsize=12, fontweight="bold")
    save(fig, "fig1_pipeline")


def fig2_heatmap() -> None:
    summary = pd.read_csv(os.path.join(ART, "phase_summary.csv"), index_col=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"wspace": 0.4})
    phase_order = ["0", "1", "2", "3", "4", "5", "6", "7"]
    target_order = ["Distance", "BallSpeed", "AbsDir", "AbsSpin"]
    target_labels = {"Distance": "Distance", "BallSpeed": "Ball Speed",
                     "AbsDir": "|Direction\nError|", "AbsSpin": "|Side Spin|"}
    im = None
    for ax, view in zip(axes, ["FACEON", "DTL"]):
        mat = summary.loc[view].reindex(target_order)[phase_order]
        im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=2.0)
        ax.set_xticks(range(len(phase_order)))
        ax.set_xticklabels([PHASES[p] for p in phase_order], fontsize=8)
        ax.set_yticks(range(len(target_order)))
        ax.set_yticklabels([target_labels[t] for t in target_order])
        for i in range(len(target_order)):
            for j in range(len(phase_order)):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8.5, color="white" if v > 1.2 else "black")
        ax.set_title(view)
        ax.set_xlabel("Swing phase")
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    cbar.set_label("max |Cohen's d|", fontsize=9)
    save(fig, "fig2_heatmap")


def fig3_archetypes() -> None:
    arch = pd.read_csv(os.path.join(ART, "archetypes.csv"))
    prof = pd.read_csv(os.path.join(ART, "archetype_profile.csv"), index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    for c in sorted(arch["cluster"].unique()):
        m = arch["cluster"] == c
        ax.scatter(arch.loc[m, "z1"], arch.loc[m, "z2"], s=18, alpha=0.65,
                   color=ARCH_COLORS[c],
                   label=f"{ARCHETYPE_NAMES[c]} (n={int(m.sum())})",
                   edgecolor="white", linewidth=0.4)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("(a) Swing archetypes in pose-feature space")
    ax.legend(loc="best", fontsize=8.5, frameon=True)

    ax = axes[1]
    metrics = ["dist_mean", "bs_mean", "spin_mean", "dir_err", "side_spin"]
    labels = ["Distance\n(yd)", "Ball Speed\n(m/s)", "Back Spin\n(rpm)",
              "|Dir Err|\n(°)", "|Side|\nSpin"]
    normed = prof[metrics].astype(float).copy()
    for col in metrics:
        lo, hi = normed[col].min(), normed[col].max()
        normed[col] = (normed[col] - lo) / (hi - lo + 1e-9)
    x = np.arange(len(metrics), dtype=float)
    w = 0.2
    for i, c in enumerate(prof.index):
        vals = normed.loc[c].values.astype(float)
        ax.bar(x + (i - 1.5) * w, vals, width=w,
               color=ARCH_COLORS[int(c)],
               label=ARCHETYPE_NAMES[int(c)],
               edgecolor="black", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Normalised score (0–1 across archetypes)")
    ax.set_title("(b) Performance profile per archetype")
    ax.legend(loc="upper center", fontsize=7.5, ncol=2, bbox_to_anchor=(0.5, 1.28))
    save(fig, "fig3_archetypes")


def fig4_tree() -> None:
    meta = pickle.load(open(os.path.join(ART, "tree.pkl"), "rb"))
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_tree(meta["tree"], feature_names=meta["features"],
              filled=True, rounded=True, fontsize=8, ax=ax,
              impurity=False, proportion=True, precision=1)
    save(fig, "fig4_tree")


def fig5_ceiling() -> None:
    cv = pd.read_csv(os.path.join(ART, "cv_ceiling.csv"))
    targets = ["Distance", "BallSpeed", "AbsDir", "AbsSpin"]
    tlabels = ["Distance", "Ball Speed", "|Direction Error|", "|Side Spin|"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(targets)); w = 0.35
    faceon = [cv[(cv["view"] == "FACEON") & (cv["target"] == t)]["r2"].values[0] for t in targets]
    dtl = [cv[(cv["view"] == "DTL") & (cv["target"] == t)]["r2"].values[0] for t in targets]
    for bars, vals, lab, col in [((x - w / 2), faceon, "Face-On view", "#3a7ca5"),
                                  ((x + w / 2), dtl, "Down-the-Line view", "#eaa94c")]:
        b = ax.bar(bars, vals, w, label=lab, color=col, edgecolor="black", lw=0.5)
        for bi in b:
            ax.text(bi.get_x() + bi.get_width() / 2, bi.get_height() + 0.012,
                    f"{bi.get_height():.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(tlabels)
    ax.set_ylabel("5-fold CV R²"); ax.set_ylim(0, 0.72)
    ax.axhline(0.5, linestyle=":", color="gray", lw=1)
    ax.legend(loc="upper right")
    save(fig, "fig5_ceiling")


def fig6_distributions() -> None:
    df = pd.read_csv(os.path.join(ART, "clean.csv"))
    sub = df[(df["View"] == "FACEON") & (df["ClubType"] == "W1")].copy()
    q_hi, q_lo = sub["Distance"].quantile(0.8), sub["Distance"].quantile(0.2)
    hi = sub[sub["Distance"] >= q_hi]; lo = sub[sub["Distance"] <= q_lo]
    key = [
        ("2-LEFT-ARM-ANGLE", "Mid-backswing lead arm angle (°)"),
        ("4-HIP-SHIFTED",    "Early-downswing hip shift"),
        ("5-HIP-SHIFTED",    "Pre-impact hip shift"),
        ("2-HIP-ROTATION",   "Mid-backswing hip rotation (°)"),
        ("6-LEFT-ARM-ANGLE", "Impact lead arm angle (°)"),
        ("7-FINISH-ANGLE",   "Finish body angle (°)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, (feat, lab) in zip(axes.ravel(), key):
        a = hi[feat].dropna(); b = lo[feat].dropna()
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 28)
        ax.hist(b, bins=bins, alpha=0.55, label="Bot 20% drives", color="#d96c6c", edgecolor="white")
        ax.hist(a, bins=bins, alpha=0.55, label="Top 20% drives", color="#3a7ca5", edgecolor="white")
        ax.axvline(a.median(), color="#3a7ca5", linestyle="--", lw=1.2)
        ax.axvline(b.median(), color="#d96c6c", linestyle="--", lw=1.2)
        pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
        d = (a.mean() - b.mean()) / pooled
        ax.set_title(f"{lab}  (d = {d:+.2f})", fontsize=9.5)
        ax.set_ylabel("count", fontsize=9)
        ax.legend(fontsize=7.5, loc="best")
    save(fig, "fig6_distributions")


def main() -> None:
    fig1_pipeline()
    fig2_heatmap()
    fig3_archetypes()
    fig4_tree()
    fig5_ceiling()
    fig6_distributions()
    print("Wrote 6 figures to", OUT)


if __name__ == "__main__":
    main()
