```python
"""
04_nmds.py

Non-Metric Multidimensional Scaling (NMDS) of the Gower distance
matrix for the OSMI Mental Health in Tech Survey 2016.

Multiple random initialisations are tested because NMDS solutions
can vary depending on the starting configuration. The solution
with the lowest stress is retained.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


# ============================================================
# CONFIGURATION
# ============================================================

GOWER_FILE = Path("results") / "gower_distance_matrix.npy"
RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD AND VALIDATE GOWER DISTANCE MATRIX
# ============================================================

print("=" * 60)
print("NMDS ANALYSIS")
print("=" * 60)

gower_dist = np.load(GOWER_FILE)

print("\nGower matrix shape:", gower_dist.shape)
print("Minimum distance:", round(gower_dist.min(), 4))
print("Maximum distance:", round(gower_dist.max(), 4))

# Check that the matrix is square
if gower_dist.shape[0] != gower_dist.shape[1]:
    raise ValueError("The Gower distance matrix must be square.")

# Check for missing values
if np.isnan(gower_dist).any():
    raise ValueError("The Gower distance matrix contains NaN values.")

# Check symmetry
if not np.allclose(gower_dist, gower_dist.T):
    raise ValueError("The Gower distance matrix is not symmetric.")

# Check diagonal
if not np.allclose(np.diag(gower_dist), 0):
    raise ValueError("The diagonal of the Gower matrix must contain zeros.")

# Check distance range
if gower_dist.min() < 0 or gower_dist.max() > 1:
    raise ValueError("Gower distances must fall between 0 and 1.")

print("\nGower matrix validation passed.")


# ============================================================
# 2. RUN NMDS WITH MULTIPLE RANDOM INITIALISATIONS
# ============================================================

seeds = [0, 1, 2, 3, 4, 5, 10, 20, 42, 100]

results = []

for seed in seeds:

    nmds = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=1,
        max_iter=1000,
        normalized_stress="auto"
    )

    coordinates = nmds.fit_transform(gower_dist)

    results.append({
        "seed": seed,
        "stress": nmds.stress_,
        "coordinates": coordinates
    })


# ============================================================
# 3. SELECT LOWEST-STRESS SOLUTION
# ============================================================

best_result = min(
    results,
    key=lambda result: result["stress"]
)

best_seed = best_result["seed"]
best_stress = best_result["stress"]
best_coordinates = best_result["coordinates"]


print("\n" + "=" * 60)
print("NMDS RESULTS")
print("=" * 60)

print("Number of dimensions:", 2)
print("Best random seed:", best_seed)
print("Best normalized stress:", round(best_stress, 4))


# ============================================================
# 4. COMPARE RANDOM INITIALISATIONS
# ============================================================

print("\nStress by random seed:")
print("-" * 30)

for result in sorted(results, key=lambda result: result["stress"]):

    print(
        f"Seed {result['seed']:>3}: "
        f"{result['stress']:.6f}"
    )


# ============================================================
# 5. SAVE NMDS COORDINATES
# ============================================================

nmds_results = pd.DataFrame(
    best_coordinates,
    columns=["NMDS1", "NMDS2"]
)

nmds_results.insert(
    0,
    "Respondent",
    range(len(nmds_results))
)

coordinates_file = RESULTS_DIR / "nmds_coordinates.csv"

nmds_results.to_csv(
    coordinates_file,
    index=False
)

print("\nNMDS coordinates saved to:")
print(coordinates_file)


# ============================================================
# 6. CREATE NMDS PLOT
# ============================================================

plt.figure(figsize=(9, 7))

plt.scatter(
    nmds_results["NMDS1"],
    nmds_results["NMDS2"],
    alpha=0.6,
    s=25
)

plt.xlabel("NMDS Dimension 1")
plt.ylabel("NMDS Dimension 2")

plt.title(
    "NMDS Representation of OSMI Respondents\n"
    f"Normalized Stress = {best_stress:.4f}"
)

plt.tight_layout()

nmds_figure = FIGURE_DIR / "nmds_plot.png"

plt.savefig(
    nmds_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("NMDS plot saved to:")
print(nmds_figure)


# ============================================================
# 7. NMDS COORDINATE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NMDS COORDINATE SUMMARY")
print("=" * 60)

print("\nCoordinate ranges:")

print(
    f"NMDS1: {nmds_results['NMDS1'].min():.4f} "
    f"to {nmds_results['NMDS1'].max():.4f}"
)

print(
    f"NMDS2: {nmds_results['NMDS2'].min():.4f} "
    f"to {nmds_results['NMDS2'].max():.4f}"
)

print("\nSummary statistics:")

print(
    nmds_results[["NMDS1", "NMDS2"]].describe().round(4)
)

print("\nCorrelation between NMDS dimensions:")

print(
    f"{nmds_results['NMDS1'].corr(nmds_results['NMDS2']):.4f}"
)


# ============================================================
# 8. DISTANCE FROM NMDS CENTRE
# ============================================================

distance_from_origin = np.sqrt(
    nmds_results["NMDS1"] ** 2
    + nmds_results["NMDS2"] ** 2
)

print("\nDistance from NMDS centre:")

print(
    f"Mean:   {distance_from_origin.mean():.4f}"
)

print(
    f"Median: {distance_from_origin.median():.4f}"
)

print(
    f"Maximum: {distance_from_origin.max():.4f}"
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NMDS ANALYSIS COMPLETE")
print("=" * 60)

print("Respondents analysed:", len(nmds_results))
print("Dimensions:", 2)
print("Best seed:", best_seed)
print("Normalized stress:", round(best_stress, 4))

print("\nFiles created:")
print("- results/nmds_coordinates.csv")
print("- results/figures/nmds_plot.png")
```
