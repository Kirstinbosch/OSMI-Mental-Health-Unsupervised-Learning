"""
OSMI Mental Health in Tech Survey 2016
Data preprocessing for unsupervised learning.

The preprocessing includes:
- Removal of respondents with less than 50% completion
- Removal of variables with at least 50% missing values
- Standardisation of gender responses
- Encoding of multiple work-position responses
- Correction and median imputation of invalid age values
- Removal of open-ended text variables
"""

from pathlib import Path

import kagglehub
import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET = "osmi/mental-health-in-tech-2016"
CSV_FILENAME = "mental-heath-in-tech-2016_20161114.csv"

path = kagglehub.dataset_download(DATASET)
csv_path = Path(path) / CSV_FILENAME

df = pd.read_csv(csv_path)

print(f"Original dataset shape: {df.shape}")


# ============================================================
# 2. CREATE CLEAN COPY
# ============================================================

df_clean = df.copy()


# ============================================================
# 3. REMOVE RESPONDENTS WITH LESS THAN 50% COMPLETENESS
# ============================================================

completion_rate = df_clean.notna().mean(axis=1)

low_completion = completion_rate < 0.50

print(
    f"Respondents removed due to <50% completion: "
    f"{low_completion.sum()}"
)

df_clean = df_clean.loc[~low_completion].copy()

print(f"Shape after respondent filtering: {df_clean.shape}")


# ============================================================
# 4. REMOVE VARIABLES WITH AT LEAST 50% MISSING VALUES
# ============================================================

missing_percentage = df_clean.isna().mean()

high_missing_columns = missing_percentage[
    missing_percentage >= 0.50
].index.tolist()

print(
    f"Variables removed due to >=50% missing values: "
    f"{len(high_missing_columns)}"
)

df_clean = df_clean.drop(columns=high_missing_columns)

print(f"Shape after variable filtering: {df_clean.shape}")


# ============================================================
# 5. CLEAN GENDER
# ============================================================

GENDER_COLUMN = "What is your gender?"


def clean_gender(value):
    """Standardise free-text gender responses into broad categories."""

    if pd.isna(value):
        return "Missing"

    value = str(value).strip().lower()

    male_values = [
        "male",
        "m",
        "man",
        "cis male",
        "cis man",
        "cisdude",
        "dude",
        "mail",
        "male.",
        "malr",
        "male (cis)",
        "male (trans, ftm)",
        "sex is male",
    ]

    female_values = [
        "female",
        "f",
        "woman",
        "fem",
        "cis female",
        "cis-woman",
        "cisgender female",
        "i identify as female.",
        "female/woman",
        "female assigned at birth",
    ]

    nonbinary_values = [
        "non-binary",
        "nonbinary",
        "enby",
        "genderfluid",
        "genderqueer",
        "genderflux demi-girl",
        "bigender",
        "agender",
        "nb masculine",
        "male/genderqueer",
        "genderqueer woman",
        "genderfluid (born female)",
        "other/transfeminine",
        "transitioned, m2f",
        "androgynous",
    ]

    if value in male_values:
        return "Male"

    if value in female_values:
        return "Female"

    if value in nonbinary_values:
        return "Non-binary / gender-diverse"

    return "Other / unclear"


df_clean["gender_clean"] = df_clean[GENDER_COLUMN].apply(clean_gender)

df_clean = df_clean.drop(columns=[GENDER_COLUMN])


# ============================================================
# 6. ENCODE WORK POSITION
# ============================================================

WORK_POSITION_COLUMN = (
    "Which of the following best describes your work position?"
)

ROLE_CATEGORIES = [
    "Back-end Developer",
    "Front-end Developer",
    "DevOps/SysAdmin",
    "Supervisor/Team Lead",
    "Other",
    "Support",
    "One-person shop",
    "Designer",
    "Executive Leadership",
    "Dev Evangelist/Advocate",
    "Sales",
    "HR",
]

for role in ROLE_CATEGORIES:
    column_name = f"role_{role}"

    df_clean[column_name] = (
        df_clean[WORK_POSITION_COLUMN]
        .fillna("")
        .str.split("|")
        .apply(lambda roles: int(role in roles))
    )

df_clean = df_clean.drop(columns=[WORK_POSITION_COLUMN])


# ============================================================
# 7. CLEAN AGE
# ============================================================

AGE_COLUMN = "What is your age?"

INVALID_AGE_VALUES = [3, 323]

df_clean.loc[
    df_clean[AGE_COLUMN].isin(INVALID_AGE_VALUES),
    AGE_COLUMN,
] = pd.NA

median_age = df_clean[AGE_COLUMN].median()

df_clean[AGE_COLUMN] = df_clean[AGE_COLUMN].fillna(median_age)

print(f"Median age used for imputation: {median_age}")


# ============================================================
# 8. REMOVE OPEN-ENDED TEXT VARIABLES
# ============================================================

TEXT_COLUMNS = [
    "Why or why not?",
    "Why or why not?.1",
]

text_columns_to_remove = [
    column
    for column in TEXT_COLUMNS
    if column in df_clean.columns
]

df_clean = df_clean.drop(columns=text_columns_to_remove)


# ============================================================
# 9. FINAL DATASET CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print(f"Number of observations: {df_clean.shape[0]}")
print(f"Number of variables: {df_clean.shape[1]}")

print("\nRemaining missing values:")
print(
    df_clean.isna()
    .sum()
    .sort_values(ascending=False)
    .head(20)
)


# ============================================================
# 10. SAVE CLEANED DATASET
# ============================================================

OUTPUT_FILE = "mental_health_cleaned.csv"

df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"\nCleaned dataset saved as: {OUTPUT_FILE}")