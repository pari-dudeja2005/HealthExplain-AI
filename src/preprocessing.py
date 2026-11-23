
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ==========================================
# DIABETES MISSING VALUE HANDLING
# ==========================================

class DiabetesZeroToMissing(BaseEstimator, TransformerMixin):

    # Supports both OpenML and common Pima column names
    ZERO_AS_MISSING = {
        "glucose",
        "bloodpressure",
        "skinthickness",
        "insulin",
        "bmi",
        "plas",
        "pres",
        "skin",
        "insu",
        "mass"
    }

    def fit(self, X, y=None):

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object
        )

        return self

    def transform(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            X = pd.DataFrame(
                X,
                columns=self.feature_names_in_
            )

        for column in X.columns:

            normalized_name = (
                str(column)
                .lower()
                .replace("_", "")
                .replace(" ", "")
            )

            if normalized_name in self.ZERO_AS_MISSING:

                X[column] = (
                    pd.to_numeric(
                        X[column],
                        errors="raise"
                    )
                    .replace(0, np.nan)
                )

        return X

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            return self.feature_names_in_

        return np.asarray(
            input_features,
            dtype=object
        )


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

def split_data(
    df,
    target_column="target",
    test_size=0.2,
    random_state=42
):

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    )

    return X_train, X_test, y_train, y_test


# ==========================================
# BUILD PREPROCESSING PIPELINE
# ==========================================

def build_preprocessor(disease="breast_cancer"):

    if disease not in ["breast_cancer", "diabetes"]:
        raise ValueError(
            f"Preprocessing is not yet configured "
            f"for {disease}."
        )

    steps = []

    if disease == "diabetes":

        steps.append(
            (
                "zero_to_missing",
                DiabetesZeroToMissing()
            )
        )

    steps.extend([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    return Pipeline(steps)


# ==========================================
# COMPLETE PREPROCESSING
# ==========================================

def preprocess_data(
    df,
    disease="breast_cancer",
    target_column="target",
    test_size=0.2,
    random_state=42
):

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state
    )

    preprocessor = build_preprocessor(disease)

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Apply training-fitted transformations to test data
    X_test_processed = preprocessor.transform(
        X_test
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


# ==========================================
# TEST PREPROCESSING
# ==========================================

if __name__ == "__main__":

    from src.data_loader import load_dataset

    for disease in ["breast_cancer", "diabetes"]:

        print(f"\n========== {disease.upper()} ==========")

        df = load_dataset(disease)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor
        ) = preprocess_data(
            df,
            disease=disease
        )

        print("Original Shape:", df.shape)
        print("Training Shape:", X_train.shape)
        print("Testing Shape:", X_test.shape)

        print(
            "Missing Values After Preprocessing:",
            int(
                X_train.isna().sum().sum()
                + X_test.isna().sum().sum()
            )
        )

        print(
            "Training Class Distribution:",
            y_train.value_counts().to_dict()
        )

        print(
            "Testing Class Distribution:",
            y_test.value_counts().to_dict()
        )

        print("Preprocessing Successful")

