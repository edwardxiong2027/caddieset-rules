"""
03_archetypes.py — Revised per peer review (v2).

Adds the following validation steps to the original UMAP + K-means pipeline:

  1. Full sweep K = 2..10 with silhouette, Davies-Bouldin, and a null
     silhouette baseline from permuted labels.
  2. Per-golfer × per-cluster cross-tabulation as a first-class artefact
     (moved from the Limitations discussion to the main analysis).
  3. Leave-one-golfer-out cluster-assignment stability (adjusted Rand
     index between the full-sample clustering and the k-means run with
     one golfer held out).

Outputs:
  artifacts/archetypes.csv              per-swing embedding + cluster label
  artifacts/archetype_profile.csv       per-cluster outcome summary
  artifacts/archetype_features.csv      per-cluster z-score of each feature
  artifacts/archetype_k_sweep.csv       K, silhouette, DB, null silhouette
  artifacts/archetype_golfer_table.csv  per-golfer × per-cluster crosstab
  artifacts/archetype_stability.csv     LOGO adjusted Rand per held-out golfer

Run:
  python src/03_archetypes.py
"""
from __future__ import annotations
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler
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
    faceon_feats = [
        c for c in body if df[df["View"] == "FACEON"][c].notna().mean() > 0.9
    ]

    sub = df[(df["View"] == "FACEON") & (df["ClubType"] == "W1")].copy()
    X_df = sub[faceon_feats].copy()
    for c in X_df.columns:
        lo, hi = X_df[c].quantile([0.01, 0.99])
        X_df[c] = X_df[c].clip(lo, hi)
    X_df = X_df.fillna(X_df.median())

    X = RobustScaler().fit_transform(X_df)

    # 1. Full K sweep with null baseline ----------------------------------
    rng = np.random.default_rng(SEED)
    sweep_rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(X)
        sil = silhouette_score(X, km.labels_)
        db = davies_bouldin_score(X, km.labels_)

        # Null baseline: random labels with the same cluster-size
        # distribution.
        counts = np.bincount(km.labels_, minlength=k)
        nulls = []
        for _ in range(5):
            null_labels = np.concatenate([
                np.full(int(n), i) for i, n in enumerate(counts)
            ])
            rng.shuffle(null_labels)
            nulls.append(silhouette_score(X, null_labels))
        null_sil = float(np.mean(nulls))

        sweep_rows.append(dict(
            k=k,
            silhouette=float(sil),
            silhouette_null_mean=null_sil,
            davies_bouldin=float(db),
            cluster_counts=counts.tolist(),
        ))
        print(
            f"  k={k}  silhouette={sil:.3f}  (null {null_sil:.3f})  "
            f"DB={db:.3f}  counts={counts.tolist()}"
        )

    pd.DataFrame(sweep_rows).to_csv(
        os.path.join(ART, "archetype_k_sweep.csv"), index=False
    )

    # 2. Chosen K -------------------------------------------------------
    km = KMeans(n_clusters=K_CHOSEN, random_state=SEED, n_init=30).fit(X)
    labels = km.labels_

    reducer = umap.UMAP(
        n_neighbors=20, min_dist=0.1, n_components=2, random_state=SEED
    )
    Z = reducer.fit_transform(X)

    sub = sub.assign(cluster=labels, z1=Z[:, 0], z2=Z[:, 1])
    sub[["cluster", "z1", "z2", "GolferId", "Distance", "BallSpeed",
          "SpinBack", "DirectionAngle", "SpinSide"]].to_csv(
        os.path.join(ART, "archetypes.csv"), index=False
    )

    prof = (
        sub.groupby("cluster")
        .agg(
            n=("Distance", "count"),
            dist_mean=("Distance", "mean"),
            dist_sd=("Distance", "std"),
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

    overall_mean = X_df.mean()
    overall_std = X_df.std()
    zmat = (X_df.groupby(labels).mean() - overall_mean) / overall_std
    zmat.index.name = "cluster"
    zmat.to_csv(os.path.join(ART, "archetype_features.csv"))

    # 3. Per-golfer × per-cluster crosstab ------------------------------
    xtab = pd.crosstab(sub["GolferId"], sub["cluster"], margins=True,
                        margins_name="total")
    xtab.to_csv(os.path.join(ART, "archetype_golfer_table.csv"))
    print("\nPer-golfer × per-cluster crosstab:")
    print(xtab)

    # Entropy of each cluster over golfers (0 = cluster is one golfer,
    # log(8) = uniformly spread across all 8).
    def entropy(row):
        p = row / row.sum()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    cluster_entropy = pd.crosstab(sub["GolferId"], sub["cluster"]).apply(
        entropy, axis=0
    )
    cluster_entropy.to_csv(
        os.path.join(ART, "archetype_golfer_entropy.csv"),
        header=["entropy_over_golfers"],
    )
    print("\nEntropy over golfers per cluster (bits, max = log2(8) = 3.0):")
    print(cluster_entropy.round(3))

    # 4. Leave-one-golfer-out stability via Adjusted Rand Index --------
    stability = []
    golfers = sub["GolferId"].unique()
    full_X = X
    full_labels = labels
    for g in golfers:
        keep = sub["GolferId"].to_numpy() != g
        Xg = full_X[keep]
        km_g = KMeans(n_clusters=K_CHOSEN, random_state=SEED, n_init=30).fit(Xg)
        # Map held-in labels onto the full clustering (nearest-centroid)
        # Simpler: run K-means again, then assign held-out rows to
        # closest centroid; then compare held-in labels directly.
        assigned = km_g.labels_
        ari = adjusted_rand_score(full_labels[keep], assigned)
        stability.append(dict(held_out_golfer=int(g), n_kept=int(keep.sum()),
                               ari=float(ari)))
    st_df = pd.DataFrame(stability).sort_values("held_out_golfer")
    st_df.to_csv(os.path.join(ART, "archetype_stability.csv"), index=False)
    print(f"\nLOGO cluster stability (ARI vs full-sample clustering):\n{st_df}")
    print(f"Mean LOGO ARI: {st_df['ari'].mean():.3f}")


if __name__ == "__main__":
    main()
