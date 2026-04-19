"""
04_rules.py — Revised per peer review (v2).

Key changes:

  1. **No feature-selection leak.**  The original pipeline ranked
     features on the full dataset then reported cross-validated tree R².
     Here the tree sees *every* feature; the tree itself chooses its
     three splits inside each CV fold from scratch.  This is the
     standard way to obtain an honest CV estimate.

  2. **Golfer-aware CV.**  We report both random 5-fold and
     GroupKFold(n_splits = 8) grouped by golfer.

  3. **Bootstrap CIs on leaf means.**  Each leaf's mean drive distance
     is reported with a 95% bootstrap percentile CI so readers can see
     that the small leaves are not as sharp as the point estimates read.

  4. The depth-3 cap is retained, but a sensitivity sweep (depth 2-5)
     is written to disk.

Outputs:
  artifacts/rules.txt          human-readable export_text of the tree
  artifacts/tree.pkl           pickled fitted tree + metadata
  artifacts/tree_meta.json     random CV R², GroupKFold R², depth sweep
  artifacts/tree_leaves.csv    per-leaf n, mean, 95% bootstrap CI

Run:
  python src/04_rules.py
"""
from __future__ import annotations
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")

SEED = 42


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000,
                  q: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    if len(values) < 2:
        return float("nan"), float("nan")
    means = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    return float(np.percentile(means, q[0])), float(np.percentile(means, q[1]))


def main() -> None:
    df = pd.read_csv(os.path.join(ART, "clean.csv"))
    body = [c for c in df.columns if c[0].isdigit() and "-" in c]
    feats = [c for c in body if df[df["View"] == "FACEON"][c].notna().mean() > 0.9]

    sub = df[(df["View"] == "FACEON") & (df["ClubType"] == "W1")].copy()
    X = sub[feats].fillna(sub[feats].median())
    y = sub["Distance"]
    groups = sub["GolferId"].to_numpy()
    mask = y.notna()
    X, y, groups = X[mask], y[mask], groups[mask]

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=SEED)
    tree.fit(X, y)

    # Fresh tree object each fold; no feature preselection across folds.
    tree_cv = DecisionTreeRegressor(
        max_depth=3, min_samples_leaf=30, random_state=SEED
    )

    r2_kfold = float(cross_val_score(tree_cv, X, y, cv=5, scoring="r2").mean())
    n_groups = len(np.unique(groups))
    n_splits = min(8, n_groups)
    r2_group = float(cross_val_score(
        tree_cv, X, y, groups=groups,
        cv=GroupKFold(n_splits=n_splits), scoring="r2"
    ).mean())
    print(f"Random 5-fold CV R² = {r2_kfold:.3f}")
    print(f"GroupKFold(8)  CV R² = {r2_group:.3f}")

    # Depth sweep (random CV and GroupKFold)
    depth_rows = []
    for d in range(2, 6):
        t = DecisionTreeRegressor(max_depth=d, min_samples_leaf=30,
                                   random_state=SEED)
        r_rand = float(cross_val_score(t, X, y, cv=5, scoring="r2").mean())
        r_grp = float(cross_val_score(
            t, X, y, groups=groups,
            cv=GroupKFold(n_splits=n_splits), scoring="r2"
        ).mean())
        depth_rows.append({"depth": d, "r2_kfold": r_rand,
                           "r2_groupkfold": r_grp})
    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv(os.path.join(ART, "tree_depth_sweep.csv"), index=False)
    print("\nDepth sweep:")
    print(depth_df.round(3))

    rules = export_text(tree, feature_names=feats, max_depth=3)
    print("\n" + rules)
    with open(os.path.join(ART, "rules.txt"), "w") as f:
        f.write(rules)

    # Leaf-level CI
    leaves = tree.apply(X)
    leaf_rows = []
    for leaf in np.unique(leaves):
        vals = y[leaves == leaf].to_numpy()
        lo, hi = bootstrap_ci(vals)
        leaf_rows.append({
            "leaf_id": int(leaf),
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "ci95_lo": lo,
            "ci95_hi": hi,
        })
    leaf_df = pd.DataFrame(leaf_rows).sort_values("mean", ascending=False)
    leaf_df.to_csv(os.path.join(ART, "tree_leaves.csv"), index=False)
    print("\nLeaf-level bootstrap 95% CIs:")
    print(leaf_df.round(2))

    with open(os.path.join(ART, "tree.pkl"), "wb") as f:
        pickle.dump({"tree": tree, "features": feats,
                      "r2_kfold": r2_kfold,
                      "r2_groupkfold": r2_group}, f)
    with open(os.path.join(ART, "tree_meta.json"), "w") as f:
        json.dump({
            "r2_kfold": r2_kfold,
            "r2_groupkfold": r2_group,
            "depth_sweep": depth_rows,
            "n_total": int(len(y)),
            "n_leaves": int(tree.get_n_leaves()),
            "features": feats,
        }, f, indent=2)


if __name__ == "__main__":
    main()
