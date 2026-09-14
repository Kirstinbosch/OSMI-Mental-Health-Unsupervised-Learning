```python
"""
05_hierarchical_clustering.py

Hierarchical clustering of the OSMI Mental Health in Tech Survey 2016
using the Gower distance matrix.

Complete and average linkage methods are compared for 2–5 clusters.
Silhouette scores and cluster sizes are used to evaluate the resulting
solutions. Linkage-distance jumps and dendrograms are also examined.

The results and figures are saved to the results directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIGURATION
# ============================================================

GOWER_FILE = Path("results") / "gower_distance_matrix.npy"

RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

LINKAGE_METHODS = ["complete", "average"]
CLUSTER_RANGE = [2, 3, 4, 5]


# ============================================================
# 1. LOAD AND VALIDATE GOWER DISTANCE MATRIX
# ============================================================

print("=" * 60)
print("HIERARCHICAL CLUSTERING")
print("=" * 60)

gower_dist = np.load(GOWER_FILE)

print(f"\nGower matrix shape: {gower_dist.shape}")

# Check that the matrix is square
if gower_dist.shape[0] != gower_dist.shape[1]:
    raise ValueError(
        "The Gower distance matrix must be square."
    )

# Check for missing values
if np.isnan(gower_dist).any():
    raise ValueError(
        "The Gower distance matrix contains NaN values."
    )

# Check symmetry
if not np.allclose(gower_dist, gower_dist.T):
    raise ValueError(
        "The Gower distance matrix is not symmetric."
    )

# Check diagonal
if not np.allclose(np.diag(gower_dist), 0):
    raise ValueError(
        "The diagonal of the Gower matrix must contain zeros."
    )

# Check distance range
if gower_dist.min() < 0 or gower_dist.max() > 1:
    raise ValueError(
        "Gower distances must fall between 0 and 1."
    )

print("Gower matrix validation passed.")


# ============================================================
# 2. CONVERT TO CONDENSED DISTANCE FORMAT
# ============================================================

# scipy hierarchical clustering requires a condensed
# distance vector rather than a square distance matrix.

condensed_dist = squareform(
    gower_dist,
    checks=True
)

print(
    f"Condensed distance vector length: "
    f"{len(condensed_dist):,}"
)


# ============================================================
# 3. COMPARE LINKAGE METHODS AND NUMBER OF CLUSTERS
# ============================================================

results = []

for method in LINKAGE_METHODS:

    print("\n" + "=" * 60)
    print(f"LINKAGE METHOD: {method.upper()}")
    print("=" * 60)

    # Create hierarchical clustering tree
    Z = linkage(
        condensed_dist,
        method=method
    )

    for k in CLUSTER_RANGE:

        # Assign each respondent to a cluster
        labels = fcluster(
            Z,
            t=k,
            criterion="maxclust"
        )

        # Calculate silhouette score using the
        # original Gower distance matrix
        silhouette = silhouette_score(
            gower_dist,
            labels,
            metric="precomputed"
        )

        # Calculate cluster sizes
        cluster_sizes = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
        )

        results.append({
            "Linkage": method,
            "Clusters": k,
            "Silhouette": silhouette,
            "Minimum cluster size": cluster_sizes.min(),
            "Maximum cluster size": cluster_sizes.max()
        })

        print(
            f"k = {k}: "
            f"Silhouette = {silhouette:.4f}, "
            f"Cluster sizes = {cluster_sizes.tolist()}"
        )


# ============================================================
# 4. RESULTS SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CLUSTERING RESULTS SUMMARY")
print("=" * 60)

print(
    results_df[
        [
            "Linkage",
            "Clusters",
            "Silhouette",
            "Minimum cluster size",
            "Maximum cluster size"
        ]
    ].to_string(index=False)
)


# ============================================================
# 5. BEST SILHOUETTE SOLUTION
# ============================================================

best_result = results_df.loc[
    results_df["Silhouette"].idxmax()
]

print("\n" + "=" * 60)
print("BEST SILHOUETTE SOLUTION")
print("=" * 60)

print(
    "Linkage method:",
    best_result["Linkage"]
)

print(
    "Number of clusters:",
    int(best_result["Clusters"])
)

print(
    "Silhouette score:",
    f"{best_result['Silhouette']:.4f}"
)

print(
    "Minimum cluster size:",
    int(best_result["Minimum cluster size"])
)

print(
    "Maximum cluster size:",
    int(best_result["Maximum cluster size"])
)


# ============================================================
# 6. SAVE CLUSTERING RESULTS
# ============================================================

results_file = (
    RESULTS_DIR /
    "hierarchical_clustering_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print("\nClustering results saved to:")
print(results_file)


# ============================================================
# 7. LINKAGE-DISTANCE JUMPS
# ============================================================

for method in LINKAGE_METHODS:

    Z = linkage(
        condensed_dist,
        method=method
    )

    distances = Z[:, 2]

    # Difference between consecutive merge distances
    jumps = np.diff(distances)

    print("\n" + "=" * 60)
    print(
        f"{method.upper()} LINKAGE - "
        "LARGEST DISTANCE JUMPS"
    )
    print("=" * 60)

    largest_jumps = (
        np.argsort(jumps)[-5:][::-1]
    )

    for i in largest_jumps:

        print(
            f"Merge {i + 2}: "
            f"{distances[i]:.4f} -> "
            f"{distances[i + 1]:.4f} "
            f"(jump = {jumps[i]:.4f})"
        )


# ============================================================
# 8. CREATE COMPLETE-LINKAGE DENDROGRAM
# ============================================================

Z_complete = linkage(
    condensed_dist,
    method="complete"
)

plt.figure(figsize=(10, 6))

dendrogram(
    Z_complete,
    truncate_mode="lastp",
    p=15,
    show_leaf_counts=True
)

plt.title(
    "Hierarchical Clustering Dendrogram – "
    "Complete Linkage"
)

plt.xlabel("Cluster")
plt.ylabel("Distance")

plt.tight_layout()

complete_figure = (
    FIGURE_DIR /
    "dendrogram_complete.png"
)

plt.savefig(
    complete_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComplete-linkage dendrogram saved to:")
print(complete_figure)


# ============================================================
# 9. CREATE AVERAGE-LINKAGE DENDROGRAM
# ============================================================

Z_average = linkage(
    condensed_dist,
    method="average"
)

plt.figure(figsize=(10, 6))

dendrogram(
    Z_average,
    truncate_mode="lastp",
    p=15,
    show_leaf_counts=True
)

plt.title(
    "Hierarchical Clustering Dendrogram – "
    "Average Linkage"
)

plt.xlabel("Cluster")
plt.ylabel("Distance")

plt.tight_layout()

average_figure = (
    FIGURE_DIR /
    "dendrogram_average.png"
)

plt.savefig(
    average_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nAverage-linkage dendrogram saved to:")
print(average_figure)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("HIERARCHICAL CLUSTERING COMPLETE")
print("=" * 60)

print(
    f"Respondents analysed: "
    f"{gower_dist.shape[0]}"
)

print(
    f"Linkage methods compared: "
    f"{', '.join(LINKAGE_METHODS)}"
)

print(
    f"Cluster solutions evaluated: "
    f"k = {CLUSTER_RANGE[0]}–{CLUSTER_RANGE[-1]}"
)

print(
    f"Best silhouette solution: "
    f"{best_result['Linkage']} linkage, "
    f"k = {int(best_result['Clusters'])}, "
    f"silhouette = {best_result['Silhouette']:.4f}"
)

print("\nFiles created:")
print("- results/hierarchical_clustering_results.csv")
print("- results/figures/dendrogram_complete.png")
print("- results/figures/dendrogram_average.png")
```

This gives you a clean separation:

```text
03_gower_distance.py
        ↓
gower_distance_matrix.npy
        ↓
04_nmds.py
        ↓
nmds_coordinates.csv
        ↓
05_hierarchical_clustering.py
        ↓
hierarchical_clustering_results.csv
dendrogram_complete.png
dendrogram_average.png
        ↓
06_cluster_interpretation.py
```

And importantly, **`05` does not decide that the 3-cluster complete-linkage solution is the final answer**. It evaluates all the alternatives and reports the numerical best solution. Your `06_cluster_interpretation.py` can then explicitly recreate the **complete-linkage k=3 solution** that you selected as the
