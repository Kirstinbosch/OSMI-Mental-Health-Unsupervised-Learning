"""
03_gower_distance.py

Gower distance calculation for the cleaned OSMI Mental Health in Tech
Survey 2016 dataset.

Gower's distance is used because the dataset contains a mixture of
numerical, binary and categorical variables. The resulting pairwise
distance matrix is saved for use in NMDS and hierarchical clustering.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data") / "mental_health_cleaned.csv"
RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("GOWER DISTANCE ANALYSIS")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Number of respondents: {df.shape[0]}")
print(f"Number of variables: {df.shape[1]}")


# ============================================================
# 2. CLASSIFY VARIABLES
# ============================================================

# Numerical variable
numerical_columns = [
    "What is your age?"
]

# Binary variables from the original survey
binary_columns = [
    "Are you self-employed?",
    "Is your employer primarily a tech company/organization?",
    "Do you have previous employers?",
    "Have you ever sought treatment for a mental health issue from a mental health professional?"
]

# Binary role variables created during preprocessing
binary_role_columns = [
    column
    for column in df.columns
    if column.startswith("role_")
]

binary_columns.extend(binary_role_columns)

# Categorical variables
categorical_columns = [
    column
    for column in df.columns
    if df[column].dtype == "object"
]


print("\nVariable classification:")
print(f"Numerical variables: {len(numerical_columns)}")
print(f"Binary variables: {len(binary_columns)}")
print(f"Categorical variables: {len(categorical_columns)}")


# ============================================================
# 3. CHECK VARIABLE CLASSIFICATION
# ============================================================

classified_columns = (
    numerical_columns
    + binary_columns
    + categorical_columns
)

unclassified_columns = [
    column
    for column in df.columns
    if column not in classified_columns
]

if unclassified_columns:
    raise ValueError(
        "The following variables have not been classified:\n"
        + "\n".join(unclassified_columns)
    )

print("\nAll variables have been classified.")


# ============================================================
# 4. CALCULATE NUMERICAL VARIABLE RANGES
# ============================================================

numerical_ranges = {}

for column in numerical_columns:
    numerical_ranges[column] = (
        df[column].max() - df[column].min()
    )

print("\nNumerical variable ranges:")

for column, value in numerical_ranges.items():
    print(f"{column}: {value}")


# ============================================================
# 5. GOWER DISTANCE FUNCTION
# ============================================================

def calculate_gower_distance(row_a, row_b):
    """
    Calculate the Gower distance between two respondents.

    Numerical variables are normalized by their observed range.
    Binary and categorical variables receive a distance of:
        0 = same response
        1 = different response

    Features with missing values in either respondent are excluded
    from the calculation.
    """

    distances = []

    # Numerical variables
    for column in numerical_columns:

        value_a = row_a[column]
        value_b = row_b[column]

        if pd.isna(value_a) or pd.isna(value_b):
            continue

        variable_range = numerical_ranges[column]

        if variable_range == 0:
            distance = 0
        else:
            distance = (
                abs(value_a - value_b)
                / variable_range
            )

        distances.append(distance)

    # Binary variables
    for column in binary_columns:

        value_a = row_a[column]
        value_b = row_b[column]

        if pd.isna(value_a) or pd.isna(value_b):
            continue

        distance = 0 if value_a == value_b else 1
        distances.append(distance)

    # Categorical variables
    for column in categorical_columns:

        value_a = row_a[column]
        value_b = row_b[column]

        if pd.isna(value_a) or pd.isna(value_b):
            continue

        distance = 0 if value_a == value_b else 1
        distances.append(distance)

    if not distances:
        return np.nan

    return np.mean(distances)


# ============================================================
# 6. EXAMPLE GOWER DISTANCE
# ============================================================

print("\n" + "=" * 60)
print("EXAMPLE GOWER DISTANCE")
print("=" * 60)

# Select two of the most complete respondents
completeness = df.notna().sum(axis=1)

complete_indices = (
    completeness
    .sort_values(ascending=False)
    .index[:2]
)

respondent_a = df.loc[complete_indices[0]]
respondent_b = df.loc[complete_indices[1]]

print(f"\nRespondent A: {complete_indices[0]}")
print(f"Respondent B: {complete_indices[1]}")


# Features used to illustrate the calculation
example_features = [
    "What is your age?",
    "gender_clean",
    "Are you self-employed?",
    "Do you currently have a mental health disorder?",
    "Have you been diagnosed with a mental health condition by a medical professional?",
    "role_Back-end Developer"
]

comparison = []

for column in example_features:

    value_a = respondent_a[column]
    value_b = respondent_b[column]

    if pd.isna(value_a) or pd.isna(value_b):
        feature_distance = np.nan

    elif column in numerical_columns:

        feature_distance = (
            abs(value_a - value_b)
            / numerical_ranges[column]
        )

    else:

        feature_distance = (
            0 if value_a == value_b else 1
        )

    comparison.append({
        "Feature": column,
        "Respondent A": value_a,
        "Respondent B": value_b,
        "Distance": feature_distance
    })

comparison_df = pd.DataFrame(comparison)

print("\nFeature-level comparison:")
print(comparison_df.to_string(index=False))

example_distance = calculate_gower_distance(
    respondent_a,
    respondent_b
)

print(
    f"\nOverall Gower distance: "
    f"{example_distance:.4f}"
)

comparison_df.to_csv(
    RESULTS_DIR / "gower_feature_comparisons.csv",
    index=False
)


# ============================================================
# 7. CALCULATE FULL GOWER DISTANCE MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING GOWER DISTANCE MATRIX")
print("=" * 60)

n = len(df)

gower_matrix = np.zeros((n, n))

for i in range(n):

    if i % 100 == 0:
        print(
            f"Processing respondent "
            f"{i + 1} of {n}..."
        )

    for j in range(i + 1, n):

        distance = calculate_gower_distance(
            df.iloc[i],
            df.iloc[j]
        )

        gower_matrix[i, j] = distance
        gower_matrix[j, i] = distance

print("\nGower distance matrix calculated.")
print(f"Matrix shape: {gower_matrix.shape}")


# ============================================================
# 8. VALIDATE GOWER MATRIX
# ============================================================

print("\n" + "=" * 60)
print("VALIDATING GOWER DISTANCE MATRIX")
print("=" * 60)

if np.isnan(gower_matrix).any():
    raise ValueError(
        "The Gower matrix contains NaN values."
    )

if gower_matrix.min() < 0 or gower_matrix.max() > 1:
    raise ValueError(
        "Gower distances must be between 0 and 1."
    )

if not np.allclose(
    np.diag(gower_matrix),
    0
):
    raise ValueError(
        "The diagonal of the Gower matrix "
        "must contain zeros."
    )

if not np.allclose(
    gower_matrix,
    gower_matrix.T
):
    raise ValueError(
        "The Gower matrix is not symmetric."
    )

print("✓ No missing values")
print("✓ All distances are between 0 and 1")
print("✓ Diagonal contains zeros")
print("✓ Matrix is symmetric")
print("\nGower matrix validation passed.")


# ============================================================
# 9. DISTANCE SUMMARY
# ============================================================

upper_triangle = gower_matrix[
    np.triu_indices(n, k=1)
]

distance_summary = pd.Series(upper_triangle)

print("\n" + "=" * 60)
print("GOWER DISTANCE SUMMARY")
print("=" * 60)

print(
    f"\nNumber of unique respondent pairs: "
    f"{len(upper_triangle):,}"
)

print(
    f"Mean distance:   "
    f"{distance_summary.mean():.4f}"
)

print(
    f"Median distance: "
    f"{distance_summary.median():.4f}"
)

print(
    f"Minimum distance: "
    f"{distance_summary.min():.4f}"
)

print(
    f"Maximum distance: "
    f"{distance_summary.max():.4f}"
)


# ============================================================
# 10. SAVE GOWER MATRIX
# ============================================================

npy_file = RESULTS_DIR / "gower_distance_matrix.npy"
csv_file = RESULTS_DIR / "gower_distance_matrix.csv"

np.save(npy_file, gower_matrix)

pd.DataFrame(gower_matrix).to_csv(
    csv_file,
    index=False
)

print("\nGower matrix saved to:")
print(npy_file)
print(csv_file)


# ============================================================
# 11. PLOT DISTANCE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    upper_triangle,
    bins=30,
    alpha=0.7
)

plt.xlabel("Gower Distance")
plt.ylabel("Frequency")
plt.title("Distribution of Pairwise Gower Distances")

plt.tight_layout()

distribution_figure = (
    FIGURE_DIR /
    "gower_distance_distribution.png"
)

plt.savefig(
    distribution_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDistance distribution plot saved to:")
print(distribution_figure)


# ============================================================
# 12. PLOT GOWER DISTANCE MATRIX
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(
    gower_matrix,
    aspect="auto"
)

plt.xlabel("Respondent")
plt.ylabel("Respondent")
plt.title("Gower Distance Matrix")

plt.colorbar(label="Gower Distance")

plt.tight_layout()

matrix_figure = (
    FIGURE_DIR /
    "gower_distance_matrix.png"
)

plt.savefig(
    matrix_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Gower distance matrix plot saved to:")
print(matrix_figure)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("GOWER DISTANCE ANALYSIS COMPLETE")
print("=" * 60)

print(f"Respondents analysed: {n}")
print(f"Variables analysed: {df.shape[1]}")
print(f"Matrix shape: {gower_matrix.shape}")

print(
    f"Mean pairwise distance: "
    f"{distance_summary.mean():.4f}"
)

print(
    f"Median pairwise distance: "
    f"{distance_summary.median():.4f}"
)

print("\nFiles created:")
print("- results/gower_distance_matrix.npy")
print("- results/gower_distance_matrix.csv")
print("- results/gower_feature_comparisons.csv")
print("- results/figures/gower_distance_distribution.png")
print("- results/figures/gower_distance_matrix.png")