```python
"""
03_exploratory_data_analysis.py

Exploratory data analysis of the cleaned OSMI Mental Health in Tech Survey
2016 dataset.

The analysis examines:
1. Employment status
2. Mental-health experience
3. Workplace mental-health support

Figures are saved to the results/figures directory.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data") / "mental_health_cleaned.csv"
FIGURE_DIR = Path("results") / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Number of respondents: {df.shape[0]}")
print(f"Number of variables: {df.shape[1]}")


# ============================================================
# EDA 1 — EMPLOYMENT STATUS
# ============================================================

print("\n" + "=" * 60)
print("EDA 1 — EMPLOYMENT STATUS")
print("=" * 60)

employment_column = "Are you self-employed?"

employment_counts = df[employment_column].value_counts(dropna=False)

employment_percentages = (
    df[employment_column]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(1)
)

print("\nEmployment status:")
print(employment_counts)

print("\nEmployment status (%):")
print(employment_percentages)


# Create figure
plt.figure(figsize=(6, 4))

employment_counts.plot(kind="bar")

plt.title("Employment Status of Participants")
plt.xlabel("Self-employed")
plt.ylabel("Number of Participants")
plt.xticks(rotation=0)
plt.tight_layout()

employment_figure = FIGURE_DIR / "EDA1_employment_status.png"

plt.savefig(
    employment_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nFigure saved to: {employment_figure}")


# ============================================================
# EDA 2 — MENTAL-HEALTH EXPERIENCE
# ============================================================

print("\n" + "=" * 60)
print("EDA 2 — MENTAL-HEALTH EXPERIENCE")
print("=" * 60)


# ------------------------------------------------------------
# Identify mental-health-related variables
# ------------------------------------------------------------

print("\nMental-health-related columns in the cleaned dataset:")

for column in df.columns:
    if "mental health" in column.lower():
        print("-", column)


# ------------------------------------------------------------
# Define variables used in the analysis
# ------------------------------------------------------------

mental_health_columns = {
    "Current disorder":
        "Do you currently have a mental health disorder?",
    "Diagnosed":
        "Have you been diagnosed with a mental health condition by a medical professional?"
}


# ------------------------------------------------------------
# Check that required columns exist
# ------------------------------------------------------------

print("\nChecking selected mental-health variables:")

missing_columns = []

for label, column in mental_health_columns.items():

    if column in df.columns:
        print("✓", label)

    else:
        print("✗ MISSING:", column)
        missing_columns.append(column)


# ------------------------------------------------------------
# Analyse mental-health variables
# ------------------------------------------------------------

if missing_columns:

    print("\nThe following required columns could not be found:")

    for column in missing_columns:
        print("-", column)

    print("\nEDA 2 could not be completed.")

else:

    # Display counts and percentages
    for label, column in mental_health_columns.items():

        print("\n" + label)
        print("-" * 40)

        counts = df[column].value_counts(dropna=False)

        percentages = (
            df[column]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(1)
        )

        print("Counts:")
        print(counts)

        print("\nPercentages:")
        print(percentages)

    # --------------------------------------------------------
    # Calculate percentage answering "Yes"
    # --------------------------------------------------------

    yes_percentages = []

    for column in mental_health_columns.values():

        yes_percentage = (
            df[column]
            .eq("Yes")
            .mean()
            * 100
        )

        yes_percentages.append(yes_percentage)

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(figsize=(7, 4))

    plt.bar(
        mental_health_columns.keys(),
        yes_percentages
    )

    plt.title("Mental-Health Experience of Survey Participants")
    plt.ylabel("Percentage of Participants (%)")
    plt.ylim(0, 100)
    plt.tight_layout()

    mental_health_figure = (
        FIGURE_DIR / "EDA2_mental_health_experience.png"
    )

    plt.savefig(
        mental_health_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nFigure saved to: {mental_health_figure}")


# ============================================================
# EDA 3 — WORKPLACE MENTAL-HEALTH SUPPORT
# ============================================================

print("\n" + "=" * 60)
print("EDA 3 — WORKPLACE MENTAL-HEALTH SUPPORT")
print("=" * 60)


workplace_support_columns = {
    "Mental-health benefits":
        "Does your employer provide mental health benefits as part of healthcare coverage?",

    "Mental-health resources":
        "Does your employer offer resources to learn more about mental health concerns and options for seeking help?"
}


# ------------------------------------------------------------
# Exclude self-employed participants
# ------------------------------------------------------------

employed_df = df[
    df[employment_column] == 0
].copy()

print(
    "\nNumber of employed participants:",
    len(employed_df)
)


# ------------------------------------------------------------
# Check that workplace-support columns exist
# ------------------------------------------------------------

missing_support_columns = []

for label, column in workplace_support_columns.items():

    if column not in employed_df.columns:

        missing_support_columns.append(column)

        print("✗ MISSING:", column)

    else:

        print("✓", label)


# ------------------------------------------------------------
# Analyse workplace-support variables
# ------------------------------------------------------------

if missing_support_columns:

    print("\nThe following required columns could not be found:")

    for column in missing_support_columns:
        print("-", column)

    print("\nEDA 3 could not be completed.")

else:

    for label, column in workplace_support_columns.items():

        print("\n" + label)
        print("-" * 40)

        counts = employed_df[column].value_counts(
            dropna=False
        )

        percentages = (
            employed_df[column]
            .value_counts(
                normalize=True,
                dropna=False
            )
            .mul(100)
            .round(1)
        )

        print("Counts:")
        print(counts)

        print("\nPercentages:")
        print(percentages)

    # --------------------------------------------------------
    # Create percentage table for the figure
    # --------------------------------------------------------

    support_percentages = pd.DataFrame()

    for label, column in workplace_support_columns.items():

        percentages = (
            employed_df[column]
            .value_counts(normalize=True)
            .mul(100)
        )

        support_percentages[label] = percentages

    # Ensure a consistent response order
    support_percentages = support_percentages.reindex(
        ["Yes", "No", "I don't know"]
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    support_percentages.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Employer-Provided Mental-Health Support")
    plt.xlabel("Response")
    plt.ylabel("Percentage of Employed Participants (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Support measure")
    plt.tight_layout()

    support_figure = (
        FIGURE_DIR / "EDA3_workplace_support.png"
    )

    plt.savefig(
        support_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nFigure saved to: {support_figure}")


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("EDA SCRIPT COMPLETE")
print("=" * 60)
```
