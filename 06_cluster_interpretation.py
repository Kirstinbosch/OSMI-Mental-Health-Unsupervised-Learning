"""
06_cluster_interpretation.py

Interpretation of the selected hierarchical clustering solution
for the OSMI Mental Health in Tech Survey 2016.

Complete linkage with three clusters is used as the selected
exploratory solution. Cluster sizes, age distributions, and
differences in categorical and binary variables are examined.

The clustered dataset and summary tables are saved to the results
directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data") / "mental_health_cleaned.csv"
GOWER_FILE = Path("results") / "gower_distance_matrix.npy"
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LINKAGE_METHOD = "complete"
N_CLUSTERS = 3


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("CLUSTER INTERPRETATION")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
gower_dist = np.load(GOWER_FILE)

print(f"\nDataset shape: {df.shape}")
print(f"Gower matrix shape: {gower_dist.shape}")


# ============================================================
# 2. VALIDATE DATA
# ============================================================

if len(df) != gower_dist.shape[0]:
    raise ValueError(
        "The number of respondents in the dataset does not "
        "match the Gower distance matrix."
    )

if gower_dist.shape[0] != gower_dist.shape[1]:
    raise ValueError(
        "The Gower distance matrix must be square."
    )

if np.isnan(gower_dist).any():
    raise ValueError(
        "The Gower distance matrix contains NaN values."
    )

if not np.allclose(gower_dist, gower_dist.T):
    raise ValueError(
        "The Gower distance matrix is not symmetric."
    )

if not np.allclose(np.diag(gower_dist), 0):
    raise ValueError(
        "The diagonal of the Gower matrix must contain zeros."
    )

print("Data validation passed.")


# ============================================================
# 3. CREATE HIERARCHICAL CLUSTERING SOLUTION
# ============================================================

condensed_dist = squareform(
    gower_dist,
    checks=True
)

Z = linkage(
    condensed_dist,
    method=LINKAGE_METHOD
)

cluster_labels = fcluster(
    Z,
    t=N_CLUSTERS,
    criterion="maxclust"
)

df["Cluster"] = cluster_labels


# ============================================================
# 4. CLUSTER SIZES
# ============================================================

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 60)
print("CLUSTER SIZES")
print("=" * 60)

for cluster, size in cluster_sizes.items():

    percentage = (
        size / len(df) * 100
    )

    print(
        f"Cluster {cluster}: "
        f"{size} respondents "
        f"({percentage:.1f}%)"
    )


# ============================================================
# 5. AGE PROFILE
# ============================================================

AGE_COLUMN = "What is your age?"

age_summary = (
    df.groupby("Cluster")[AGE_COLUMN]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Minimum="min",
        Maximum="max"
    )
    .round(2)
)

print("\n" + "=" * 60)
print("AGE PROFILE BY CLUSTER")
print("=" * 60)

print(age_summary.to_string())


# ============================================================
# 6. IDENTIFY VARIABLES SUITABLE FOR COMPARISON
# ============================================================

# Variables with a very large number of unique values are
# excluded because they are not useful for a compact
# cluster-profile comparison.

profile_variables = []

for column in df.columns:

    if column in [AGE_COLUMN, "Cluster"]:
        continue

    unique_values = df[column].nunique(
        dropna=True
    )

    if unique_values <= 15:
        profile_variables.append(column)


print("\n" + "=" * 60)
print("CLUSTER PROFILE VARIABLES")
print("=" * 60)

print(
    f"Variables included: "
    f"{len(profile_variables)}"
)


# ============================================================
# 7. CALCULATE CLUSTER PROFILES
# ============================================================

profile_results = []

for column in profile_variables:

    percentages = (
        pd.crosstab(
            df["Cluster"],
            df[column],
            normalize="index"
        ) * 100
    )

    percentages = percentages.round(2)

    # Maximum difference between clusters for any response
    # category of the variable.
    max_difference = 0

    for category in percentages.columns:

        category_difference = (
            percentages[category].max()
            - percentages[category].min()
        )

        max_difference = max(
            max_difference,
            category_difference
        )

    profile_results.append({
        "Variable": column,
        "Maximum percentage difference":
            max_difference
    })


profile_differences = pd.DataFrame(
    profile_results
)

profile_differences = (
    profile_differences
    .sort_values(
        "Maximum percentage difference",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 8. DISPLAY VARIABLES WITH LARGEST DIFFERENCES
# ============================================================

print("\n" + "=" * 60)
print("VARIABLES WITH LARGEST CLUSTER DIFFERENCES")
print("=" * 60)

print(
    profile_differences
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 9. DISPLAY DETAILED CLUSTER PROFILES
# ============================================================

print("\n" + "=" * 60)
print("DETAILED CLUSTER PROFILES")
print("=" * 60)

top_variables = (
    profile_differences
    .head(10)["Variable"]
    .tolist()
)

for column in top_variables:

    print("\n" + "-" * 60)
    print(column)
    print("-" * 60)

    percentages = (
        pd.crosstab(
            df["Cluster"],
            df[column],
            normalize="index"
        ) * 100
    )

    percentages = percentages.round(1)

    print(percentages.to_string())


# ============================================================
# 10. SAVE CLUSTERED DATASET
# ============================================================

clustered_file = (
    RESULTS_DIR /
    "mental_health_clustered_k3.csv"
)

df.to_csv(
    clustered_file,
    index=False
)

print("\nClustered dataset saved to:")
print(clustered_file)


# ============================================================
# 11. SAVE VARIABLE DIFFERENCES
# ============================================================

differences_file = (
    RESULTS_DIR /
    "cluster_variable_differences.csv"
)

profile_differences.to_csv(
    differences_file,
    index=False
)

print("\nCluster variable differences saved to:")
print(differences_file)


# ============================================================
# 12. SAVE AGE SUMMARY
# ============================================================

age_file = (
    RESULTS_DIR /
    "cluster_age_summary.csv"
)

age_summary.to_csv(
    age_file
)

print("\nCluster age summary saved to:")
print(age_file)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER INTERPRETATION COMPLETE")
print("=" * 60)

print(
    f"Linkage method: {LINKAGE_METHOD}"
)

print(
    f"Number of clusters: {N_CLUSTERS}"
)

print(
    f"Respondents analysed: {len(df)}"
)

print("\nCluster sizes:")

for cluster, size in cluster_sizes.items():

    percentage = (
        size / len(df) * 100
    )

    print(
        f"  Cluster {cluster}: "
        f"{size} ({percentage:.1f}%)"
    )

print("\nFiles created:")
print("- results/mental_health_clustered_k3.csv")
print("- results/cluster_variable_differences.csv")
print("- results/cluster_age_summary.csv")