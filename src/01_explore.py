"""
01_explore.py  —  Dataset profiling and cleaning.

Outputs:
  artifacts/clean.csv     — numeric-coerced, finite copy of the dataset
  artifacts/summary.json  — per-column missingness & per-club summary

Run:
  python src/01_explore.py
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_IN = os.path.join(ROOT, "data", "CaddieSet.csv")
OUT_DIR = os.path.join(ROOT, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA_IN)
    body_cols = [c for c in df.columns if c[0].isdigit() and "-" in c]

    # Numeric coercion + inf → NaN
    for c in body_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[body_cols] = df[body_cols].replace([np.inf, -np.inf], np.nan)

    # Derived absolute-magnitude targets
    df["AbsDir"] = df["DirectionAngle"].abs()
    df["AbsSpin"] = df["SpinSide"].abs()

    df.to_csv(os.path.join(OUT_DIR, "clean.csv"), index=False)

    summary = {
        "n_rows": int(len(df)),
        "n_golfers": int(df["GolferId"].nunique()),
        "view_counts": df["View"].value_counts().to_dict(),
        "club_counts": df["ClubType"].value_counts().to_dict(),
        "ball_flight_stats": df[
            ["Distance", "Carry", "BallSpeed", "SpinBack", "SpinSide", "DirectionAngle"]
        ]
        .describe()
        .round(2)
        .to_dict(),
        "per_view_feature_coverage": {
            v: int(sum(df[df["View"] == v][c].notna().mean() > 0.9 for c in body_cols))
            for v in ["FACEON", "DTL"]
        },
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {OUT_DIR}/clean.csv ({len(df)} rows)")
    print(f"Wrote {OUT_DIR}/summary.json")
    print(json.dumps(summary["per_view_feature_coverage"], indent=2))


if __name__ == "__main__":
    main()
