"""
02_effect_sizes.py  —  Cohen's d between top-20% and bottom-20% outcome groups,
                        for every (view, outcome, feature).

Also fits Random-Forest predictive ceilings per (view, outcome) as a reference.

Outputs:
  artifacts/effect_by_phase.csv   — long-form effect-size table
  artifacts/phase_summary.csv     — max |d| per (view, target, phase)
  artifacts/cv_ceiling.csv        — 5-fold CV R² per (view, target)

Run:
  python src/02_effect_sizes.py
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

SEED = 42


def load() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ART, "clean.csv"))
    return df


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def effect_by_phase(df: pd.DataFrame, view: str, target: str, clubs: list[str]) -> pd.DataFrame:
    body_cols = [c for c in df.columns if c[0].isdigit() and "-" in c]
    sub = df[(df["View"] == view) & (df["ClubType"].isin(clubs))].copy()
    sub = sub[sub[target].notna()]
    view_cols = [c for c in body_cols if sub[c].notna().mean() > 0.9]
    q_hi, q_lo = sub[target].quantile(0.8), sub[target].quantile(0.2)
    hi = sub[sub[target] >= q_hi]
    lo = sub[sub[target] <= q_lo]
    rows = []
    for c in view_cols:
        d = cohens_d(hi[c], lo[c])
        if not np.isnan(d):
            rows.append(
                {
                    "feature": c,
                    "phase": c.split("-", 1)[0],
                    "family": c.split("-", 1)[1],
                    "cohens_d": d,
                    "abs_d": abs(d),
                    "view": view,
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


def cv_ceiling(df: pd.DataFrame, view: str, target: str, clubs: list[str]) -> tuple[float, int]:
    body_cols = [c for c in df.columns if c[0].isdigit() and "-" in c]
    sub = df[(df["View"] == view) & (df["ClubType"].isin(clubs))].copy()
    cols = [c for c in body_cols if sub[c].notna().mean() > 0.9]
    # Winsorize to stabilize
    X = sub[cols].copy()
    for c in X.columns:
        lo, hi = X[c].quantile([0.001, 0.999])
        X[c] = X[c].clip(lo, hi)
    X = X.fillna(X.median())
    y = sub[target]
    mask = y.notna()
    X, y = X[mask], y[mask]
    if len(X) < 50:
        return float("nan"), len(X)
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=5)
    r2 = cross_val_score(rf, X, y, cv=5, scoring="r2").mean()
    return float(r2), int(len(X))


def main() -> None:
    df = load()

    # Effect sizes
    frames = []
    for view in ["FACEON", "DTL"]:
        for target in ["Distance", "BallSpeed", "AbsDir", "AbsSpin"]:
            frames.append(effect_by_phase(df, view, target, ["W1"]))
    eff = pd.concat(frames, ignore_index=True)
    eff.to_csv(os.path.join(ART, "effect_by_phase.csv"), index=False)

    summary = (
        eff.groupby(["view", "target", "phase"])["abs_d"].max().unstack("phase").round(3)
    )
    summary.to_csv(os.path.join(ART, "phase_summary.csv"))
    print("Max |d| per (view, target, phase):")
    print(summary)

    # CV ceilings
    rows = []
    for target in ["Distance", "BallSpeed", "AbsDir", "AbsSpin"]:
        for view in ["FACEON", "DTL"]:
            r2, n = cv_ceiling(df, view, target, ["W1"])
            rows.append({"view": view, "target": target, "r2": r2, "n": n})
    cv = pd.DataFrame(rows)
    cv.to_csv(os.path.join(ART, "cv_ceiling.csv"), index=False)
    print("\n5-fold CV R²:")
    print(cv.round(3))


if __name__ == "__main__":
    main()
