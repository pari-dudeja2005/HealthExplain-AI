
import sys
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

from src.config import (
    get_disease_config,
    get_available_diseases
)


# ==========================================
# BREAST CANCER DATASET
# ==========================================

def load_breast_cancer_data():

    dataset = load_breast_cancer(as_frame=True)

    df = dataset.data.copy()

    # sklearn: 0 = malignant, 1 = benign
    # Our project: 0 = benign, 1 = malignant
    df["target"] = (
        dataset.target == 0
    ).astype(int)

    return df


# ==========================================
# CLEAN CSV DATA
# ==========================================

def clean_csv_data(df):

    df = df.copy()

    # Remove whitespace from column names
    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found in dataset."
        )

    # Remove unnecessary index columns
    unnamed_columns = [
        col for col in df.columns
        if col.lower().startswith("unnamed:")
    ]

    df = df.drop(
        columns=unnamed_columns,
        errors="ignore"
    )

    # Clean text values
    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:

        df[column] = df[column].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    # Standard missing-value markers
    df = df.replace(
        ["?", "", "NA", "N/A", "null", "NULL"],
        np.nan
    )

    # Convert numeric-looking columns to numbers
    for column in df.columns:

        if not pd.api.types.is_object_dtype(
            df[column]
        ):
            continue

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        non_missing = df[column].notna()

        if (
            non_missing.any()
            and numeric_values[non_missing].notna().all()
        ):
            df[column] = numeric_values

    return df


# ==========================================
# STANDARDISE TARGET LABELS
# ==========================================

def standardise_target(series, disease):

    if series.isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    # Chronic Kidney Disease uses text labels
    if disease == "kidney_disease":

        labels = (
            series.astype(str)
            .str.lower()
            .str.replace(r"\s+", "", regex=True)
        )

        mapping = {
            "ckd": 1,
            "notckd": 0
        }

        target = labels.map(mapping)

        if target.isna().any():
            invalid = labels[target.isna()].unique()
            raise ValueError(
                f"Unknown kidney disease labels: {invalid}"
            )

        return target.astype(int)

    # Some heart disease datasets use
    # 0 = absence and 1-4 = presence
    if disease == "heart_disease":

        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        if not numeric.isin([0, 1, 2, 3, 4]).all():
            raise ValueError(
                "Unexpected heart disease target values."
            )

        return (numeric > 0).astype(int)

    # Diabetes and Parkinson's datasets
    # are expected to have binary targets
    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    if not numeric.isin([0, 1]).all():
        raise ValueError(
            f"Expected binary target for {disease}."
        )

    return numeric.astype(int)


# ==========================================
# LOAD CSV DATASET
# ==========================================

def load_csv_dataset(disease):

    config = get_disease_config(disease)

    file_path = config["raw_data_path"]

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nDataset not found: {file_path}\n"
            f"Place {config['dataset_file']} "
            f"inside data/raw/."
        )

    df = pd.read_csv(file_path)

    df = clean_csv_data(df)

    target_column = config["target_column"]

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            f"not found in {file_path.name}.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Remove dataset-specific identifiers
    if disease == "kidney_disease":
        df = df.drop(
            columns=["id"],
            errors="ignore"
        )

    if disease == "parkinsons":
        df = df.drop(
            columns=["name"],
            errors="ignore"
        )

    # Standardise target to 0 and 1
    target = standardise_target(
        df[target_column],
        disease
    )

    df = df.drop(columns=[target_column])

    # Common target name for all datasets
    df["target"] = target.to_numpy()

    return df


# ==========================================
# MAIN DATA LOADER
# ==========================================

def load_dataset(disease="breast_cancer"):

    config = get_disease_config(disease)

    if config["dataset_source"] == "sklearn":

        df = load_breast_cancer_data()

    elif config["dataset_source"] == "csv":

        df = load_csv_dataset(disease)

    else:

        raise ValueError(
            f"Unsupported dataset source: "
            f"{config['dataset_source']}"
        )

    if df.empty:
        raise ValueError(
            f"Dataset is empty: {disease}"
        )

    if df["target"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both target classes."
        )

    return df


# ==========================================
# DATASET INSPECTION
# ==========================================

def inspect_dataset(df):

    print("\n========== DATASET OVERVIEW ==========")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFeature Data Types:")
    print(df.dtypes.value_counts())

    print("\nMissing Values:")
    missing = df.isna().sum()
    print(missing[missing > 0])

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nTarget Distribution:")
    print(
        df["target"].value_counts().sort_index()
    )

    print("\nTarget Percentages:")
    print(
        (
            df["target"].value_counts(
                normalize=True
            ).sort_index() * 100
        ).round(2)
    )

    print("\n======================================")


# ==========================================
# TEST DATA LOADER
# ==========================================

if __name__ == "__main__":

    disease = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "breast_cancer"
    )

    config = get_disease_config(disease)

    print(f"\nLoading: {config['name']}")

    df = load_dataset(disease)

    inspect_dataset(df)

    print("\nDataset Loaded Successfully!")

