"""
02_effect_sizes.py  —  Revised per peer review (v2).

Extreme-groups Cohen's d (as before) reported *alongside*:
  * point-biserial r and its back-transformed continuous Pearson r;
  * full-sample Pearson r and Spearman rho between the feature and
    the signed outcome;
  * Benjamini-Hochberg FDR q-value across the full grid.

Also fits Random-Forest predictive ceilings per (view, outcome) under
BOTH random 5-fold CV and golfer-aware GroupKFold(n_splits=8).

Outputs:
  artifacts/effect_by_phase.csv        long-form table of d, r_pb, r_cont, r_pearson, rho_spearman, q_fdr, p_perm
  artifacts/phase_summary.csv          max |d| per (view, target, phase) [legacy]
  artifacts/phase_summary_pearson.csv  max |r_pearson| per (view, target, phase) [FDR-significant only]
  artifacts/cv_ceiling.csv             R² under random-KFold and GroupKFold (per (view, target))
  artifacts/effect_meta.json           counts, thresholds, FDR survivors

Run:
  python src/02_effect_sizes.py
"""
from __future__ import annotations
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

SEED = 42


def load() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ART, "clean.csv"))


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna(); b = b.dropna()
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def point_biserial_to_pearson(r_pb: float, p_hi: float) -> float:
    """Back-transform a point-biserial r (extreme-groups) to an
    approximate continuous Pearson r using Fitzsimons (2008)'s
    standard correction.

        r_continuous ≈ r_pb * sqrt(p*(1-p)) / (dnorm(qnorm(p)))

    where p is the proportion in the high group (≈ 0.2 for a 20/80
    quantile split, noting that we discard the middle 60%)."""
    if np.isnan(r_pb):
        return float("nan")
    p = p_hi / (p_hi + (1 - p_hi))
    from math import sqrt, pi, exp
    z = stats.norm.ppf(p_hi)
    phi = (1.0 / sqrt(2 * pi)) * exp(-0.5 * z * z)
    # Dichotomised -> continuous correction (Cohen 1983; Fitzsimons 2008):
    # r_cont = r_pb * sqrt(p*(1-p)) / phi(z_{1-p})
    denom = phi if phi > 1e-6 else 1e-6
    return float(r_pb * sqrt(p_hi * (1 - p_hi)) / denom)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    q = np.full_like(p, np.nan)
    order = np.argsort(p[finite])
    ranked = p[finite][order]
    n = len(ranked)
    if n == 0:
        return q
    raw_q = ranked * n / (np.arange(n) + 1)
    raw_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_finite = np.empty_like(ranked)
    q_finite[order] = np.minimum(raw_q, 1.0)
    q[finite] = q_finite
    return q


def effect_by_phase(df: pd.DataFrame, view: str, target: str,
                     clubs: list[str]) -> pd.DataFrame:
    body_cols = [c for c in df.columns if c[0].isdigit() and "-" in c]
    sub = df[(df["View"] == view) & (df["ClubType"].isin(clubs))].copy()
    sub = sub[sub[target].notna()]
    view_cols = [c for c in body_cols if sub[c].notna().mean() > 0.9]
    q_hi, q_lo = sub[target].quantile(0.8), sub[target].quantile(0.2)
    hi = sub[sub[target] >= q_hi]
    lo = sub[sub[target] <= q_lo]
    p_hi_frac = len(hi) / (len(hi) + len(lo))

    rows = []
    for c in view_cols:
        d = cohens_d(hi[c], lo[c])
        if not np.isnan(d):
            # Point-biserial r using t statistic on two-sample split
            a, b = hi[c].dropna().to_numpy(), lo[c].dropna().to_numpy()
            tstat, p_ttest = stats.ttest_ind(a, b, equal_var=False)
            n1, n2 = len(a), len(b)
            # r_pb = t / sqrt(t^2 + df)
            dofu = n1 + n2 - 2
            r_pb = float(tstat / np.sqrt(tstat * tstat + dofu)) if dofu > 0 else np.nan
            r_cont = point_biserial_to_pearson(r_pb, p_hi_frac)

            # Full-sample Pearson r and Spearman rho against the signed target.
            full = sub[[c, target]].dropna()
            if len(full) > 10:
                pr, pp = stats.pearsonr(full[c], full[target])
                sr, sp = stats.spearmanr(full[c], full[target])
            else:
                pr = pp = sr = sp = np.nan

            rows.append(dict(
                feature=c,
                phase=c.split("-", 1)[0],
                family=c.split("-", 1)[1],
                cohens_d=d,
                abs_d=abs(d),
                r_pb=r_pb,
                r_cont_est=r_cont,
                r_pearson=float(pr),
                p_pearson=float(pp),
                rho_spearman=float(sr),
                p_spearman=float(sp),
                p_ttest=float(p_ttest),
                view=view,
                target=target,
            ))
    return pd.DataFrame(rows)


def cv_ceiling(
    df: pd.DataFrame, view: str, target: str, clubs: list[str]
) -> dict:
    """Return dict with r2_kfold, r2_groupkfold, n, mean_y, std_y."""
    body_cols = [c for c in df.columns if c[0].isdigit() and "-" in c]
    sub = df[(df["View"] == view) & (df["ClubType"].isin(clubs))].copy()
    cols = [c for c in body_cols if sub[c].notna().mean() > 0.9]
    X = sub[cols].copy()
    for c in X.columns:
        lo, hi = X[c].quantile([0.001, 0.999])
        X[c] = X[c].clip(lo, hi)
    X = X.fillna(X.median())
    y = sub[target]
    groups = sub["GolferId"].to_numpy()
    mask = y.notna()
    X, y, groups = X[mask], y[mask], groups[mask]
    if len(X) < 50:
        return dict(r2_kfold=float("nan"), r2_groupkfold=float("nan"),
                    n=int(len(X)))
    rf = RandomForestRegressor(
        n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=5
    )
    r2_kfold = float(cross_val_score(rf, X, y, cv=5, scoring="r2").mean())

    n_groups = len(np.unique(groups))
    n_splits = min(8, n_groups)
    r2_group = float(cross_val_score(
        rf, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits),
        scoring="r2"
    ).mean())

    return dict(
        r2_kfold=r2_kfold, r2_groupkfold=r2_group,
        n=int(len(X)), n_groups=int(n_groups),
        y_std=float(y.std()),
        y_mean=float(y.mean()),
    )


def main() -> None:
    df = load()

    # -- Effect sizes -----------------------------------------------------
    frames = []
    for view in ["FACEON", "DTL"]:
        for target in ["Distance", "BallSpeed", "AbsDir", "AbsSpin"]:
            frames.append(effect_by_phase(df, view, target, ["W1"]))
    eff = pd.concat(frames, ignore_index=True)

    # Benjamini-Hochberg FDR over the complete (view, target, feature) grid.
    # Use the Pearson-r p-value (comparable across features and signed outcomes).
    eff["q_fdr_pearson"] = bh_fdr(eff["p_pearson"].to_numpy())
    eff["q_fdr_ttest"] = bh_fdr(eff["p_ttest"].to_numpy())
    eff["sig_fdr_05"] = (eff["q_fdr_pearson"] < 0.05).astype(int)
    eff.to_csv(os.path.join(ART, "effect_by_phase.csv"), index=False)

    # Legacy summary: max |d| per (view, target, phase)
    summary = (
        eff.groupby(["view", "target", "phase"])["abs_d"]
        .max().unstack("phase").round(3)
    )
    summary.to_csv(os.path.join(ART, "phase_summary.csv"))
    print("Max |d| per (view, target, phase):")
    print(summary)

    # New: max |r_pearson| among FDR-significant cells per (view, target, phase)
    sig = eff[eff["sig_fdr_05"] == 1].copy()
    sig["abs_r"] = sig["r_pearson"].abs()
    pearson_summary = (
        sig.groupby(["view", "target", "phase"])["abs_r"]
        .max().unstack("phase").round(3)
    )
    pearson_summary.to_csv(os.path.join(ART, "phase_summary_pearson.csv"))

    meta = {
        "seed": SEED,
        "n_tests": int(len(eff)),
        "n_fdr_significant_05": int(eff["sig_fdr_05"].sum()),
        "extreme_groups_quantiles": [0.2, 0.8],
        "note_on_inflation": (
            "Extreme-groups Cohen's d is systematically larger than the "
            "underlying continuous standardised effect. The column "
            "`r_cont_est` back-transforms the point-biserial r to an "
            "approximate continuous Pearson r following Fitzsimons (2008)."
        ),
    }

    # -- CV ceilings ------------------------------------------------------
    rows = []
    for target in ["Distance", "BallSpeed", "AbsDir", "AbsSpin"]:
        for view in ["FACEON", "DTL"]:
            res = cv_ceiling(df, view, target, ["W1"])
            rows.append({"view": view, "target": target, **res})
    cv = pd.DataFrame(rows)
    cv.to_csv(os.path.join(ART, "cv_ceiling.csv"), index=False)
    print("\nRandom KFold vs GroupKFold R²:")
    print(cv[["view", "target", "r2_kfold", "r2_groupkfold", "n"]].round(3))

    meta["n_fdr_significant_05_at_pearson_q05"] = int(
        (eff["q_fdr_pearson"] < 0.05).sum()
    )
    with open(os.path.join(ART, "effect_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
