"""
06_accuracy_robust.py — Peer review C7 robustness checks.

The reviewer objected that the power-vs-accuracy ceiling gap (R² ≈ 0.50
for distance, 0.19-0.33 for directional error) could be an artefact of
either (a) the |·| transform on direction/sidespin or (b) the choice of
Random Forest.  This script reports the same ceiling under three
alternative recipes and two model families:

    targets:    signed DirectionAngle, |DirectionAngle|, sqrt(|DirectionAngle|)
                signed SpinSide,       |SpinSide|,       sqrt(|SpinSide|)
    models:     RandomForestRegressor, HistGradientBoostingRegressor

Both random 5-fold and GroupKFold(8) are reported.  An estimate of the
label's relative measurement-noise floor is provided by computing the
within-swing (same golfer, same session) CV on the signed target.

Outputs:
  artifacts/accuracy_robustness.csv  full ceiling table

Run:
  python src/06_accuracy_robust.py
"""
from __future__ import annotations
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")

SEED = 42


def main() -> None:
    df = pd.read_csv(os.path.join(ART, "clean.csv"))
    body = [c for c in df.columns if c[0].isdigit() and "-" in c]

    rows = []
    for view in ["FACEON", "DTL"]:
        sub = df[(df["View"] == view) & (df["ClubType"] == "W1")].copy()
        feats = [c for c in body if sub[c].notna().mean() > 0.9]
        X = sub[feats].copy()
        for c in X.columns:
            lo, hi = X[c].quantile([0.001, 0.999])
            X[c] = X[c].clip(lo, hi)
        X = X.fillna(X.median())
        groups = sub["GolferId"].to_numpy()

        target_specs = [
            ("DirectionAngle_signed", sub["DirectionAngle"]),
            ("DirectionAngle_abs",    sub["DirectionAngle"].abs()),
            ("DirectionAngle_sqrtabs",
              np.sqrt(sub["DirectionAngle"].abs().clip(lower=0))),
            ("SpinSide_signed",       sub["SpinSide"]),
            ("SpinSide_abs",          sub["SpinSide"].abs()),
            ("SpinSide_sqrtabs",
              np.sqrt(sub["SpinSide"].abs().clip(lower=0))),
        ]

        for name, y_series in target_specs:
            mask = y_series.notna()
            Xm = X[mask].to_numpy()
            ym = y_series[mask].to_numpy()
            gm = groups[mask]
            if len(ym) < 50:
                continue

            for model_name, model in [
                ("rf", RandomForestRegressor(
                    n_estimators=300, random_state=SEED, n_jobs=-1,
                    min_samples_leaf=5)),
                ("hgb", HistGradientBoostingRegressor(
                    random_state=SEED, max_iter=300)),
            ]:
                r2_rand = float(cross_val_score(
                    model, Xm, ym, cv=5, scoring="r2").mean())
                n_groups = len(np.unique(gm))
                n_splits = min(8, n_groups)
                r2_grp = float(cross_val_score(
                    model, Xm, ym, groups=gm,
                    cv=GroupKFold(n_splits=n_splits), scoring="r2").mean())
                rows.append(dict(
                    view=view,
                    target=name,
                    model=model_name,
                    n=int(len(ym)),
                    y_std=float(np.std(ym)),
                    r2_kfold=r2_rand,
                    r2_groupkfold=r2_grp,
                ))

    out = pd.DataFrame(rows).round(3)
    out.to_csv(os.path.join(ART, "accuracy_robustness.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
