"""
04_rules.py  —  Fit a shallow (depth-3) regression tree on the Face-On, W1 subset
                predicting Distance, and export the rule list.

Outputs:
  artifacts/rules.txt       — human-readable rules (sklearn export_text)
  artifacts/tree.pkl        — pickled sklearn tree (reused by 05_make_figures.py)
  artifacts/tree_meta.json  — CV R² and feature list

Run:
  python src/04_rules.py
"""
from __future__ import annotations
import json
import os
import pickle
import warnings
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")

SEED = 42


def main() -> None:
    df = pd.read_csv(os.path.join(ART, "clean.csv"))
    body = [c for c in df.columns if c[0].isdigit() and "-" in c]
    feats = [c for c in body if df[df["View"] == "FACEON"][c].notna().mean() > 0.9]

    sub = df[(df["View"] == "FACEON") & (df["ClubType"] == "W1")].copy()
    X = sub[feats].fillna(sub[feats].median())
    y = sub["Distance"]
    mask = y.notna()
    X, y = X[mask], y[mask]

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=SEED)
    tree.fit(X, y)
    r2 = cross_val_score(tree, X, y, cv=5, scoring="r2").mean()
    print(f"Depth-3 tree CV R² = {r2:.3f}")

    rules = export_text(tree, feature_names=feats, max_depth=3)
    print("\n" + rules)

    with open(os.path.join(ART, "rules.txt"), "w") as f:
        f.write(rules)
    with open(os.path.join(ART, "tree.pkl"), "wb") as f:
        pickle.dump({"tree": tree, "features": feats, "r2": float(r2)}, f)
    with open(os.path.join(ART, "tree_meta.json"), "w") as f:
        json.dump({"r2": float(r2), "features": feats}, f, indent=2)


if __name__ == "__main__":
    main()
