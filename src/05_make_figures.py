"""
05_make_figures.py  —  Reproduce every figure in paper/figures/ from the artifacts.

Revised per peer review (v2).  The new figure set is:

  fig1_pipeline         Schematic of the associative/descriptive pipeline
                        (no more ``prescription'' label).
  fig2_heatmap          Phase-x-outcome map.  Cells with BH-FDR q >= 0.05
                        are masked (diagonal hatch) so the eye does not
                        track pure noise.
  fig3_archetypes       (a) UMAP scatter with cluster-size callouts,
                        (b) per-golfer x per-cluster crosstab heatmap,
                        (c) silhouette-vs-K with null baseline.
                        Archetype *names* are suppressed (A0..A3 only)
                        because LOGO ARI shows 2/8 golfers destabilise them.
  fig4_tree             sklearn tree plot (unchanged structure, but the
                        caption in paper.tex is now honest about the
                        GroupKFold R^2).
  fig5_ceiling          R^2 bars with *both* random 5-fold and GroupKFold(8)
                        shown side-by-side per outcome.  The gap is the
                        headline reviewer finding.
  fig6_distributions    Top-20/bot-20 histograms (kept, with a caption
                        caveat about extreme-groups inflation).
  fig7_robustness       R^2 under three target transforms (signed, abs,
                        sqrt|.|) x two models (RF, HGB) x two CV schemes
                        (random, GroupKFold).  Shows the ceiling *gap*
                        is robust to modelling choices.
  fig8_tree_leaves      Per-leaf mean Distance with 95% bootstrap CI.

All figures are written as both PDF and PNG under paper/figures/.

Run order:
  python src/01_explore.py
  python src/02_effect_sizes.py
  python src/03_archetypes.py
  python src/04_rules.py
  python src/06_accuracy_robust.py
  python src/05_make_figures.py
"""
from __future__ import annotations
import ast
import os
import pickle
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
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
        ("Effect sizes +\nBH-FDR", 6.0, "#eaa94c"),
        ("Archetype\nvalidation", 7.7, "#d96c6c"),
        ("Shallow\nrule surrogate", 9.3, "#7c9f3a"),
    ]
    for i, (name, x, c) in enumerate(stages):
        ax.add_patch(plt.Rectangle((x - 0.55, 1.2), 1.4, 1.5,
                                    facecolor=c, edgecolor="black", lw=0.6))
        ax.text(x + 0.15, 1.95, name, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        if i < len(stages) - 1:
            nx = stages[i + 1][1]
            ax.annotate("", xy=(nx - 0.55, 1.95), xytext=(x + 0.85, 1.95),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.text(5.0, 3.55,
            "Associative pose $\\leftrightarrow$ outcome analysis on CaddieSet",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 0.4,
            "N = 8 golfers; all inference uses golfer-aware CV and FDR control",
            ha="center", fontsize=8.5, style="italic", color="#444")
    save(fig, "fig1_pipeline")


def fig2_heatmap() -> None:
    """Max |d| per (view, target, phase), masked where no cell in that
    (view, target, phase) tile survives BH-FDR q < 0.05."""
    eff = pd.read_csv(os.path.join(ART, "effect_by_phase.csv"))
    eff["phase"] = eff["phase"].astype(str)

    summary = (
        eff.groupby(["view", "target", "phase"])["abs_d"].max()
        .unstack("phase")
    )
    sig_any = (
        eff.assign(sig=eff["q_fdr_pearson"] < 0.05)
        .groupby(["view", "target", "phase"])["sig"].any()
        .unstack("phase")
    )

    phase_order = ["0", "1", "2", "3", "4", "5", "6", "7"]
    target_order = ["Distance", "BallSpeed", "AbsDir", "AbsSpin"]
    target_labels = {"Distance": "Distance", "BallSpeed": "Ball Speed",
                      "AbsDir": "|Direction\nError|",
                      "AbsSpin": "|Side Spin|"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9),
                              gridspec_kw={"wspace": 0.35})
    im = None
    for ax, view in zip(axes, ["FACEON", "DTL"]):
        mat = summary.loc[view].reindex(target_order)[phase_order]
        sig = sig_any.loc[view].reindex(target_order)[phase_order]
        im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=2.0)
        for i in range(len(target_order)):
            for j in range(len(phase_order)):
                v = mat.values[i, j]
                s = bool(sig.values[i, j]) if not pd.isna(sig.values[i, j]) else False
                if not np.isnan(v):
                    txt_color = "white" if v > 1.2 else "black"
                    label = f"{v:.2f}" if s else f"{v:.2f}*"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=8.5, color=txt_color)
                    if not s:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, hatch="///",
                            edgecolor="#555", linewidth=0.0,
                        ))
        ax.set_xticks(range(len(phase_order)))
        ax.set_xticklabels([PHASES[p] for p in phase_order], fontsize=8)
        ax.set_yticks(range(len(target_order)))
        ax.set_yticklabels([target_labels[t] for t in target_order])
        ax.set_title(view)
        ax.set_xlabel("Swing phase")
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    cbar.set_label("max |Cohen's d| (extreme-groups; inflated)", fontsize=9)
    fig.suptitle(
        "Phase $\\times$ outcome effect-size map   "
        "(*marker + hatch = no feature in that cell survives BH-FDR q<0.05)",
        fontsize=10, y=1.02,
    )
    save(fig, "fig2_heatmap")


def fig3_archetypes() -> None:
    arch = pd.read_csv(os.path.join(ART, "archetypes.csv"))
    xtab = pd.read_csv(
        os.path.join(ART, "archetype_golfer_table.csv"), index_col=0
    )
    # Drop the totals row/col if present.
    if "total" in xtab.index:
        xtab = xtab.drop(index="total")
    if "total" in xtab.columns:
        xtab = xtab.drop(columns="total")
    k_sweep = pd.read_csv(os.path.join(ART, "archetype_k_sweep.csv"))

    fig = plt.figure(figsize=(13.5, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.9], wspace=0.45)

    # (a) UMAP
    ax = fig.add_subplot(gs[0, 0])
    for c in sorted(arch["cluster"].unique()):
        m = arch["cluster"] == c
        ax.scatter(arch.loc[m, "z1"], arch.loc[m, "z2"], s=18, alpha=0.7,
                    color=ARCH_COLORS[int(c)],
                    label=f"A{int(c)} (n={int(m.sum())})",
                    edgecolor="white", linewidth=0.4)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("(a) UMAP of pose features, $K=4$")
    ax.legend(loc="best", fontsize=8.5, frameon=True)

    # (b) Per-golfer x cluster heatmap (row-proportion)
    ax = fig.add_subplot(gs[0, 1])
    rowprop = xtab.div(xtab.sum(axis=1), axis=0)
    im = ax.imshow(rowprop.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(rowprop.shape[0]):
        for j in range(rowprop.shape[1]):
            v = rowprop.values[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.55 else "black")
    ax.set_xticks(range(rowprop.shape[1]))
    ax.set_xticklabels([f"A{c}" for c in rowprop.columns])
    ax.set_yticks(range(rowprop.shape[0]))
    ax.set_yticklabels([f"G{g}" for g in rowprop.index])
    ax.set_title("(b) Per-golfer cluster share")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Golfer")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="row fraction")

    # (c) Silhouette-vs-K with null baseline
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(k_sweep["k"], k_sweep["silhouette"], marker="o",
            color="#3a7ca5", lw=1.6, label="Observed")
    ax.plot(k_sweep["k"], k_sweep["silhouette_null_mean"], marker="s",
            color="#999", lw=1.2, ls="--", label="Null (same sizes)")
    ax.axvline(4, color="#d96c6c", ls=":", lw=1.1)
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette")
    ax.set_title("(c) K sweep")
    ax.legend(fontsize=8, loc="lower left")

    save(fig, "fig3_archetypes")


def fig4_tree() -> None:
    meta = pickle.load(open(os.path.join(ART, "tree.pkl"), "rb"))
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_tree(meta["tree"], feature_names=meta["features"],
               filled=True, rounded=True, fontsize=8, ax=ax,
               impurity=False, proportion=True, precision=1)
    save(fig, "fig4_tree")


def fig5_ceiling() -> None:
    """Random KFold vs GroupKFold(8) R^2 per (view, outcome)."""
    cv = pd.read_csv(os.path.join(ART, "cv_ceiling.csv"))
    targets = ["Distance", "BallSpeed", "AbsDir", "AbsSpin"]
    tlabels = ["Distance", "Ball Speed", "|Dir. error|", "|Side spin|"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3),
                              gridspec_kw={"wspace": 0.25})
    x = np.arange(len(targets)); w = 0.38
    for ax, view, view_label in zip(
        axes, ["FACEON", "DTL"], ["Face-On", "Down-the-Line"]
    ):
        kf = [cv[(cv.view == view) & (cv.target == t)]["r2_kfold"].iat[0]
              for t in targets]
        gk = [cv[(cv.view == view) & (cv.target == t)]["r2_groupkfold"].iat[0]
              for t in targets]
        b1 = ax.bar(x - w / 2, kf, w, label="Random 5-fold",
                     color="#3a7ca5", edgecolor="black", lw=0.5)
        b2 = ax.bar(x + w / 2, gk, w, label="GroupKFold(8) – by golfer",
                     color="#d96c6c", edgecolor="black", lw=0.5)
        for bi in list(b1) + list(b2):
            h = bi.get_height()
            y = h + 0.08 if h >= 0 else h - 0.45
            ax.text(bi.get_x() + bi.get_width() / 2, y,
                     f"{h:.2f}", ha="center", fontsize=8.2)
        ax.axhline(0, color="#444", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(tlabels, fontsize=9)
        ax.set_ylabel("CV R$^2$")
        ax.set_title(view_label)
        ax.set_ylim(-5.2, 0.85)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Predictive ceiling collapses under golfer-aware CV   "
        "(N$_{\\mathrm{golfer}}$ = 8)",
        fontsize=11, y=1.02,
    )
    save(fig, "fig5_ceiling")


def fig6_distributions() -> None:
    df = pd.read_csv(os.path.join(ART, "clean.csv"))
    sub = df[(df["View"] == "FACEON") & (df["ClubType"] == "W1")].copy()
    q_hi, q_lo = sub["Distance"].quantile(0.8), sub["Distance"].quantile(0.2)
    hi = sub[sub["Distance"] >= q_hi]; lo = sub[sub["Distance"] <= q_lo]
    key = [
        ("2-LEFT-ARM-ANGLE", "Mid-BS lead arm angle (deg)"),
        ("4-HIP-SHIFTED",    "Early-DS hip shift (au)"),
        ("5-HIP-SHIFTED",    "Pre-impact hip shift (au)"),
        ("2-HIP-ROTATION",   "Mid-BS hip rotation (deg)"),
        ("6-LEFT-ARM-ANGLE", "Impact lead arm angle (deg)"),
        ("7-FINISH-ANGLE",   "Finish body angle (deg)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, (feat, lab) in zip(axes.ravel(), key):
        a = hi[feat].dropna(); b = lo[feat].dropna()
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 28)
        ax.hist(b, bins=bins, alpha=0.55, label="Bot 20% drives",
                 color="#d96c6c", edgecolor="white")
        ax.hist(a, bins=bins, alpha=0.55, label="Top 20% drives",
                 color="#3a7ca5", edgecolor="white")
        ax.axvline(a.median(), color="#3a7ca5", linestyle="--", lw=1.2)
        ax.axvline(b.median(), color="#d96c6c", linestyle="--", lw=1.2)
        pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
        d = (a.mean() - b.mean()) / pooled
        ax.set_title(f"{lab}  (d = {d:+.2f})", fontsize=9.5)
        ax.set_ylabel("count", fontsize=9)
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle(
        "Pose distributions, top-20% vs bot-20% drives    "
        "(extreme-groups d is an upper bound on the continuous effect)",
        fontsize=10, y=1.01,
    )
    save(fig, "fig6_distributions")


def fig7_robustness() -> None:
    """Robustness of the power/accuracy gap to target transform and model."""
    rob = pd.read_csv(os.path.join(ART, "accuracy_robustness.csv"))
    rob = rob[rob["view"] == "FACEON"].copy()
    rob["family"] = rob["target"].str.replace(
        "_signed", "", regex=False
    ).str.replace("_abs", "", regex=False).str.replace(
        "_sqrtabs", "", regex=False
    )
    rob["transform"] = (
        rob["target"].str.extract(r"_(signed|abs|sqrtabs)$")[0]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2),
                              gridspec_kw={"wspace": 0.3})
    for ax, fam in zip(axes, ["DirectionAngle", "SpinSide"]):
        sub = rob[rob["family"] == fam]
        models = ["rf", "hgb"]
        transforms = ["signed", "abs", "sqrtabs"]
        xlabs = [f"{t}\n({m.upper()})" for m in models for t in transforms]
        x = np.arange(len(xlabs)); w = 0.36
        kf, gk = [], []
        for m in models:
            for t in transforms:
                row = sub[(sub["model"] == m) & (sub["transform"] == t)]
                kf.append(row["r2_kfold"].iat[0])
                gk.append(row["r2_groupkfold"].iat[0])
        b1 = ax.bar(x - w / 2, kf, w, color="#3a7ca5",
                     label="Random 5-fold", edgecolor="black", lw=0.4)
        b2 = ax.bar(x + w / 2, gk, w, color="#d96c6c",
                     label="GroupKFold(8)", edgecolor="black", lw=0.4)
        for bi in list(b1) + list(b2):
            h = bi.get_height()
            y = h + 0.03 if h >= 0 else h - 0.12
            ax.text(bi.get_x() + bi.get_width() / 2, y,
                     f"{h:.2f}", ha="center", fontsize=7.5)
        ax.axhline(0, color="#444", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(xlabs, fontsize=7.5)
        ax.set_ylabel("CV R$^2$")
        ax.set_title(f"{fam}  (Face-On, driver)")
        ax.set_ylim(-0.95, 0.6)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Accuracy ceiling is robust to target transform and model family",
        fontsize=11, y=1.03,
    )
    save(fig, "fig7_robustness")


def fig8_tree_leaves() -> None:
    """Per-leaf mean Distance with 95% bootstrap CI."""
    lf = pd.read_csv(os.path.join(ART, "tree_leaves.csv"))
    lf = lf.sort_values("mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(lf))
    err_lo = lf["mean"] - lf["ci95_lo"]
    err_hi = lf["ci95_hi"] - lf["mean"]
    ax.errorbar(lf["mean"], y, xerr=[err_lo, err_hi],
                 fmt="o", color="#3a7ca5", ecolor="#999",
                 capsize=3, lw=1.4, markersize=7)
    for i, r in lf.iterrows():
        ax.text(r["ci95_hi"] + 2, i,
                 f"n={int(r['n'])}",
                 va="center", fontsize=8.5, color="#444")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Leaf {int(i)}" for i in lf["leaf_id"]])
    ax.set_xlabel("Mean drive distance, yards (95% bootstrap CI)")
    ax.set_title(
        "Tree leaves: point estimates carry wide CIs on small leaves"
    )
    save(fig, "fig8_tree_leaves")


def main() -> None:
    fig1_pipeline()
    fig2_heatmap()
    fig3_archetypes()
    fig4_tree()
    fig5_ceiling()
    fig6_distributions()
    fig7_robustness()
    fig8_tree_leaves()
    print("Wrote 8 figures to", OUT)


if __name__ == "__main__":
    main()
