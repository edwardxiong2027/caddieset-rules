"""
03_archetypes.py  —  UMAP + K-means archetype discovery on the Face-On, W1 subset.

Outputs:
  artifacts/archetypes.csv          — per-swing embedding + cluster label
  artifacts/archetype_profile.csv   — per-cluster outcome summary
  artifacts/archetype_features.csv  — per-cluster z-score of each feature

Run:
  python src/03_archetypes.py
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

SEED = 42
K_CHOSEN = 4


def main() -> None:
    df = pd.read_csv(os.path.join(ART, "clean.csv"))
    body = [c for c in df.columns if c[0].isdigit() and "-" in c]
    faceon_feats = [c for c in body if df[df["View"] == "FACEON"][c].notna().mean() > 0.9]

    sub = df[(df["View"] == "FACEON") & (df["ClubType"] == "W1")].copy()
    X_df = sub[faceon_feats].copy()
    # Clip extreme outliers so they don't form singleton clusters
    for c in X_df.columns:
        lo, hi = X_df[c].quantile([0.01, 0.99])
        X_df[c] = X_df[c].clip(lo, hi)
    X_df = X_df.fillna(X_df.median())

    X = RobustScaler().fit_transform(X_df)

    # 2D embedding (for plotting only; clustering uses the full feature space)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=SEED)
    Z = reducer.fit_transform(X)

    # Silhouette sweep for reporting
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(X)
        s = silhouette_score(X, km.labels_)
        print(f"  k={k}  silhouette={s:.3f}  counts={np.bincount(km.labels_).tolist()}")

    km = KMeans(n_clusters=K_CHOSEN, random_state=SEED, n_init=30).fit(X)
    labels = km.labels_

    sub = sub.assign(cluster=labels, z1=Z[:, 0], z2=Z[:, 1])
    sub[["cluster", "z1", "z2", "GolferId", "Distance", "BallSpeed", "SpinBack",
         "DirectionAngle", "SpinSide"]].to_csv(
        os.path.join(ART, "archetypes.csv"), index=False
    )

    prof = (
        sub.groupby("cluster")
        .agg(
            n=("Distance", "count"),
            dist_mean=("Distance", "mean"),
            bs_mean=("BallSpeed", "mean"),
            spin_mean=("SpinBack", "mean"),
            dir_err=("DirectionAngle", lambda s: s.abs().mean()),
            side_spin=("SpinSide", lambda s: s.abs().mean()),
        )
        .round(2)
    )
    prof.to_csv(os.path.join(ART, "archetype_profile.csv"))
    print("\nCluster profile:")
    print(prof)

    # Per-cluster feature z-scores
    overall_mean = X_df.mean()
    overall_std = X_df.std()
    zmat = (X_df.groupby(labels).mean() - overall_mean) / overall_std
    zmat.index.name = "cluster"
    zmat.to_csv(os.path.join(ART, "archetype_features.csv"))


if __name__ == "__main__":
    main()
