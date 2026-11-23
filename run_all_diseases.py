
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data" / "processed" / "multidisease"
MODEL_DIR = ROOT / "models" / "multidisease"
REPORT_DIR = ROOT / "reports" / "multidisease"

SEED = 42

DISEASES = {
    "breast_cancer": {
        "name": "Breast Cancer",
        "negative": "Benign",
        "positive": "Malignant",
    },
    "diabetes": {
        "name": "Diabetes",
        "negative": "No Diabetes",
        "positive": "Diabetes",
    },
    "heart_disease": {
        "name": "Heart Disease",
        "negative": "No Heart Disease",
        "positive": "Heart Disease",
    },
    "kidney_disease": {
        "name": "Chronic Kidney Disease",
        "negative": "No CKD",
        "positive": "CKD",
    },
    "parkinsons": {
        "name": "Parkinson's Disease",
        "negative": "Healthy",
        "positive": "Parkinson's",
    },
}

UCI_IDS = {
    "heart_disease": 45,
    "kidney_disease": 336,
    "parkinsons": 174,
}

UCI_FILES = {
    "heart_disease": "heart_disease.csv",
    "kidney_disease": "kidney_disease.csv",
    "parkinsons": "parkinsons.csv",
}

DIABETES_MISSING_ZERO = {
    "glucose",
    "bloodpressure",
    "skinthickness",
    "insulin",
    "bmi",
    "plas",
    "pres",
    "skin",
    "insu",
    "mass",
}


# ==========================================
# DATASET UTILITIES
# ==========================================

def normalise_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
    )


def clean_dataframe(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    df = df.loc[
        :,
        ~df.columns.str.lower().str.startswith("unnamed:")
    ]

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names in dataset")

    for col in df.columns:
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            df[col] = df[col].map(
                lambda v: v.strip() if isinstance(v, str) else v
            )

            df[col] = df[col].replace(
                ["?", "", "NA", "N/A", "null", "NULL", "nan"],
                np.nan,
            )

            numeric = pd.to_numeric(df[col], errors="coerce")
            nonmissing = df[col].notna()

            if (
                nonmissing.any()
                and numeric[nonmissing].notna().all()
            ):
                df[col] = numeric

    return df


def download_uci(disease, path):
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Install ucimlrepo using: pip install ucimlrepo"
        ) from exc

    print(f"Downloading {DISEASES[disease]['name']}...")

    dataset = fetch_ucirepo(id=UCI_IDS[disease])

    features = dataset.data.features
    targets = dataset.data.targets

    if features is None or targets is None:
        raise RuntimeError("Dataset features or targets unavailable")

    features = features.reset_index(drop=True).copy()
    targets = targets.reset_index(drop=True).copy()

    if targets.shape[1] != 1:
        raise ValueError("Expected exactly one target column")

    parts = []

    ids = getattr(dataset.data, "ids", None)

    if ids is not None:
        ids = ids.reset_index(drop=True).copy()

        extra_ids = ids.loc[
            :,
            ~ids.columns.isin(features.columns)
            & ~ids.columns.isin(targets.columns),
        ]

        if not extra_ids.empty:
            parts.append(extra_ids)

    parts.extend([features, targets])

    df = pd.concat(parts, axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    return df


def load_diabetes_raw(path):
    if path.exists():
        return pd.read_csv(path)

    print("Downloading Diabetes dataset...")

    dataset = fetch_openml(
        data_id=37,
        as_frame=True,
        parser="auto",
    )

    df = dataset.data.copy()

    labels = (
        dataset.target.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "tested_negative": 0,
        "tested_positive": 1,
        "negative": 0,
        "positive": 1,
        "0": 0,
        "1": 1,
    }

    y = labels.map(mapping)

    if y.isna().any():
        raise ValueError("Unexpected Diabetes target labels")

    df["Outcome"] = y.to_numpy(dtype=int)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    return df


# ==========================================
# LOAD ANY DISEASE
# ==========================================

def load_disease(disease):

    if disease == "breast_cancer":
        dataset = load_breast_cancer(as_frame=True)

        X = dataset.data.copy()

        y = (
            dataset.target == 0
        ).astype(int).reset_index(drop=True)

        return X.reset_index(drop=True), y, None

    if disease == "diabetes":
        raw = load_diabetes_raw(
            RAW_DIR / "diabetes.csv"
        )
    else:
        path = RAW_DIR / UCI_FILES[disease]

        raw = (
            pd.read_csv(path)
            if path.exists()
            else download_uci(disease, path)
        )

    df = clean_dataframe(raw).reset_index(drop=True)

    target_candidates = {
        "diabetes": ["Outcome", "outcome", "class"],
        "heart_disease": ["num", "target"],
        "kidney_disease": ["classification", "class"],
        "parkinsons": ["status"],
    }[disease]

    target_col = next(
        (c for c in target_candidates if c in df.columns),
        None,
    )

    if target_col is None:
        raise ValueError(
            f"Target column missing for {disease}"
        )

    raw_y = df[target_col]

    if disease == "kidney_disease":
        labels = (
            raw_y.astype(str)
            .str.lower()
            .str.replace(r"\s+", "", regex=True)
        )

        y = labels.map({
            "ckd": 1,
            "notckd": 0,
        })

    elif disease == "diabetes":
        labels = (
            raw_y.astype(str)
            .str.strip()
            .str.lower()
        )

        y = labels.map({
            "tested_negative": 0,
            "tested_positive": 1,
            "negative": 0,
            "positive": 1,
            "0": 0,
            "1": 1,
        })

    else:
        numeric = pd.to_numeric(
            raw_y,
            errors="coerce",
        )

        if disease == "heart_disease":
            if not numeric.isin([0, 1, 2, 3, 4]).all():
                raise ValueError(
                    "Heart Disease target must be 0 through 4"
                )

            y = (numeric > 0).astype(int)

        else:
            if not numeric.isin([0, 1]).all():
                raise ValueError(
                    "Parkinson's target must contain 0 and 1"
                )

            y = numeric.astype(int)

    if y.isna().any():
        raise ValueError("Unexpected or missing target values")

    y = y.astype(int).reset_index(drop=True)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Both target classes are required")

    groups = None

    if disease == "parkinsons":
        if "name" not in df.columns:
            raise ValueError(
                "Parkinson's dataset requires subject identifiers"
            )

        names = df["name"].astype(str)

        groups = (
            names.str.replace(
                r"_\d+$",
                "",
                regex=True,
            )
            .reset_index(drop=True)
        )

        if groups.nunique() < 10:
            raise ValueError(
                "Insufficient Parkinson's subject groups"
            )

    X = df.drop(
        columns=[target_col, "id", "name"],
        errors="ignore",
    ).copy()

    if disease == "diabetes":
        for col in X.columns:
            if normalise_name(col) in DIABETES_MISSING_ZERO:
                X[col] = (
                    pd.to_numeric(X[col], errors="coerce")
                    .replace(0, np.nan)
                )

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = (
                pd.to_numeric(X[col], errors="coerce")
                .astype(float)
            )
        else:
            X[col] = X[col].map(
                lambda v: str(v) if pd.notna(v) else np.nan
            ).astype(object)

    return X.reset_index(drop=True), y, groups


# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================

def split_indices(y, groups=None):

    ids = np.arange(len(y))

    if groups is None:
        train_ids, test_ids = train_test_split(
            ids,
            test_size=0.2,
            random_state=SEED,
            stratify=y,
        )

        fit_ids, val_ids = train_test_split(
            train_ids,
            test_size=0.2,
            random_state=SEED,
            stratify=y.iloc[train_ids],
        )

    else:
        outer = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED,
        )

        train_pos, test_pos = next(
            outer.split(ids, y, groups)
        )

        train_ids = ids[train_pos]
        test_ids = ids[test_pos]

        inner = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED + 1,
        )

        fit_pos, val_pos = next(
            inner.split(
                train_ids,
                y.iloc[train_ids],
                groups.iloc[train_ids],
            )
        )

        fit_ids = train_ids[fit_pos]
        val_ids = train_ids[val_pos]

        assert not (
            set(groups.iloc[train_ids])
            & set(groups.iloc[test_ids])
        )

        assert not (
            set(groups.iloc[fit_ids])
            & set(groups.iloc[val_ids])
        )

    for label, part in [
        ("train", fit_ids),
        ("validation", val_ids),
        ("test", test_ids),
    ]:
        if y.iloc[part].nunique() < 2:
            raise ValueError(
                f"Only one target class in {label} split"
            )

    return fit_ids, val_ids, train_ids, test_ids


# ==========================================
# PREPROCESSING
# ==========================================

def make_preprocessor(X):

    numeric = list(
        X.select_dtypes(include=[np.number]).columns
    )

    categorical = [
        col for col in X.columns
        if col not in numeric
    ]

    transforms = []

    if numeric:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ])

        transforms.append(
            ("num", numeric_pipeline, numeric)
        )

    if categorical:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ])

        transforms.append(
            ("cat", categorical_pipeline, categorical)
        )

    preprocessor = ColumnTransformer(
        transforms,
        remainder="drop",
    )

    return preprocessor, numeric, categorical


# ==========================================
# MACHINE LEARNING MODELS
# ==========================================

def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=SEED,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def calculate_metrics(y, pred, proba):

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(
            average_precision_score(y, proba)
        ),
    }


def paths_for(disease):

    return {
        "model": MODEL_DIR / f"{disease}_model.joblib",
        "train": DATA_DIR / f"{disease}_train.csv",
        "test": DATA_DIR / f"{disease}_test.csv",
        "report": REPORT_DIR / disease,
    }


# ==========================================
# TRAIN A DISEASE
# ==========================================

def train_disease(disease):

    print(
        f"\n{'=' * 60}\n"
        f"{DISEASES[disease]['name']} - TRAINING\n"
        f"{'=' * 60}",
        flush=True,
    )

    X, y, groups = load_disease(disease)

    fit_ids, val_ids, train_ids, test_ids = (
        split_indices(y, groups)
    )

    X_fit = X.iloc[fit_ids]
    X_val = X.iloc[val_ids]
    X_train = X.iloc[train_ids]
    X_test = X.iloc[test_ids]

    y_fit = y.iloc[fit_ids]
    y_val = y.iloc[val_ids]
    y_train = y.iloc[train_ids]
    y_test = y.iloc[test_ids]

    print(
        f"Records: {len(X)} | "
        f"Train: {len(X_train)} | "
        f"Validation: {len(X_val)} | "
        f"Test: {len(X_test)}"
    )

    results = []

    for name, estimator in get_models().items():

        preprocessor, _, _ = make_preprocessor(X_fit)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        pipeline.fit(X_fit, y_fit)

        probabilities = pipeline.predict_proba(
            X_val
        )[:, 1]

        result = {
            "model": name,
            **calculate_metrics(
                y_val,
                (probabilities >= 0.5).astype(int),
                probabilities,
            ),
        }

        results.append(result)

        print(
            f"{name}: "
            f"ROC-AUC={result['roc_auc']:.4f}, "
            f"Recall={result['recall']:.4f}",
            flush=True,
        )

    comparison = (
        pd.DataFrame(results)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    best_model_name = str(
        comparison.iloc[0]["model"]
    )

    preprocessor, num_cols, cat_cols = (
        make_preprocessor(X_train)
    )

    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", get_models()[best_model_name]),
    ])

    final_model.fit(X_train, y_train)

    test_probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        predictions,
        test_probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).tolist()

    paths = paths_for(disease)

    for directory in [
        MODEL_DIR,
        DATA_DIR,
        paths["report"],
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    joblib.dump(
        final_model,
        paths["model"],
    )

    comparison.to_csv(
        paths["report"] / "model_comparison.csv",
        index=False,
    )

    train_df = X_train.copy().reset_index(drop=True)
    train_df["target"] = y_train.to_numpy()

    test_df = X_test.copy().reset_index(drop=True)
    test_df["target"] = y_test.to_numpy()

    train_df.to_csv(
        paths["train"],
        index=False,
    )

    test_df.to_csv(
        paths["test"],
        index=False,
    )

    report = {
        "disease": disease,
        "name": DISEASES[disease]["name"],
        "negative_label": DISEASES[disease]["negative"],
        "positive_label": DISEASES[disease]["positive"],
        "best_model": best_model_name,
        "selection_metric": "validation_roc_auc",
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "test_records": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "group_safe_split": groups is not None,
        "threshold": 0.5,
    }

    (
        paths["report"] / "test_metrics.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"Selected: {best_model_name} | "
        f"Test ROC-AUC={test_metrics['roc_auc']:.4f}",
        flush=True,
    )

    return report


# ==========================================
# MAP TRANSFORMED FEATURES
# ==========================================

def transformed_feature_map(preprocessor):

    feature_map = []

    for name, transformer, columns in (
        preprocessor.transformers_
    ):

        if name == "num":
            feature_map.extend(list(columns))

        elif name == "cat":
            encoder = transformer.named_steps["onehot"]

            for col, categories in zip(
                columns,
                encoder.categories_,
            ):
                feature_map.extend(
                    [col] * len(categories)
                )

    n_output = len(
        preprocessor.get_feature_names_out()
    )

    if len(feature_map) != n_output:
        raise ValueError(
            "Transformed feature mapping mismatch"
        )

    return feature_map


# ==========================================
# SHAP AND LIME EXPLAINERS
# ==========================================

def make_explainers(model, X_train):

    import shap
    from lime.lime_tabular import LimeTabularExplainer

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    background = X_train.sample(
        n=min(50, len(X_train)),
        random_state=SEED,
    )

    background_transformed = np.asarray(
        preprocessor.transform(background),
        dtype=float,
    )

    train_transformed = np.asarray(
        preprocessor.transform(X_train),
        dtype=float,
    )

    feature_map = transformed_feature_map(
        preprocessor
    )

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    try:
        shap_explainer = shap.Explainer(
            estimator,
            background_transformed,
            feature_names=feature_names,
        )

        shap_fallback = False

    except Exception as exc:
        warnings.warn(
            f"Using model-agnostic SHAP: {exc}"
        )

        masker = shap.maskers.Independent(
            background_transformed,
            max_samples=50,
        )

        shap_explainer = shap.Explainer(
            lambda array: estimator.predict_proba(
                np.asarray(array, dtype=float)
            )[:, 1],
            masker,
            algorithm="permutation",
            feature_names=feature_names,
        )

        shap_fallback = True

    lime_explainer = LimeTabularExplainer(
        training_data=train_transformed,
        feature_names=feature_names,
        class_names=["Negative", "Positive"],
        mode="classification",
        discretize_continuous=False,
        random_state=SEED,
    )

    return (
        shap_explainer,
        lime_explainer,
        feature_map,
        shap_fallback,
    )


def explain_record(
    patient,
    model,
    shap_explainer,
    lime_explainer,
    feature_map,
    shap_fallback=False,
    lime_samples=700,
):

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    transformed = np.asarray(
        preprocessor.transform(patient),
        dtype=float,
    )

    if shap_fallback:
        explanation = shap_explainer(
            transformed,
            max_evals=2 * transformed.shape[1] + 1,
        )
    else:
        explanation = shap_explainer(transformed)

    values = np.asarray(explanation.values)

    if values.ndim == 3:
        values = values[0, :, 1]
    else:
        values = values[0].reshape(-1)

    shap_series = (
        pd.Series(values)
        .groupby(feature_map, sort=False)
        .sum()
    )

    shap_df = pd.DataFrame({
        "feature": shap_series.index,
        "shap_value": shap_series.values,
    })

    shap_df["absolute_importance"] = (
        shap_df["shap_value"].abs()
    )

    shap_df = (
        shap_df.sort_values(
            "absolute_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    lime_explanation = lime_explainer.explain_instance(
        transformed[0],
        estimator.predict_proba,
        labels=(1,),
        num_features=transformed.shape[1],
        num_samples=lime_samples,
    )

    weights = dict(
        lime_explanation.as_map()[1]
    )

    lime_values = [
        float(weights.get(i, 0.0))
        for i in range(len(feature_map))
    ]

    lime_series = (
        pd.Series(lime_values)
        .groupby(feature_map, sort=False)
        .sum()
    )

    lime_df = pd.DataFrame({
        "feature": lime_series.index,
        "lime_weight": lime_series.values,
    })

    lime_df["absolute_importance"] = (
        lime_df["lime_weight"].abs()
    )

    lime_df = (
        lime_df.sort_values(
            "absolute_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        shap_df,
        lime_df,
        float(lime_explanation.score),
    )


# ==========================================
# XAI RELIABILITY METRICS
# ==========================================

def jaccard(a, b, k=5):

    first = set(list(a)[:k])
    second = set(list(b)[:k])

    union = first | second

    return (
        float(len(first & second) / len(union))
        if union else 1.0
    )


def perturb(patient, X_train, rng):

    modified = patient.copy()

    for col in X_train.select_dtypes(
        include=[np.number]
    ).columns:

        original = patient.iloc[0][col]

        if pd.isna(original):
            continue

        modified[col] = modified[col].astype(float)

        std = float(X_train[col].std())

        if not np.isfinite(std) or std == 0:
            continue

        minimum = X_train[col].min()
        maximum = X_train[col].max()

        new_value = np.clip(
            float(original) + rng.normal(
                0,
                0.01 * std,
            ),
            minimum,
            maximum,
        )

        if normalise_name(col) in {
            "preg",
            "pregnancies",
            "age",
        }:
            new_value = round(float(new_value))

        modified.at[
            modified.index[0],
            col,
        ] = float(new_value)

    return modified


def replace_features(
    patient,
    selected,
    X_train,
):

    modified = patient.copy()

    for col in selected:

        if pd.api.types.is_numeric_dtype(
            X_train[col]
        ):
            modified[col] = (
                modified[col].astype(float)
            )

            modified.at[
                modified.index[0],
                col,
            ] = float(X_train[col].median())

        else:
            modes = X_train[col].mode(
                dropna=True
            )

            if len(modes):
                modified.at[
                    modified.index[0],
                    col,
                ] = modes.iloc[0]

    return modified


def fidelity_ratio(
    patient,
    ranking,
    model,
    X_train,
    rng,
    random_comparisons,
):

    all_features = list(X_train.columns)
    k = min(5, len(all_features))

    important = list(ranking)[:k]

    original = float(
        model.predict_proba(patient)[0, 1]
    )

    samples = [
        replace_features(
            patient,
            important,
            X_train,
        )
    ]

    for _ in range(random_comparisons):

        random_set = rng.choice(
            all_features,
            size=k,
            replace=False,
        )

        samples.append(
            replace_features(
                patient,
                random_set,
                X_train,
            )
        )

    probabilities = model.predict_proba(
        pd.concat(
            samples,
            ignore_index=True,
        )
    )[:, 1]

    important_impact = abs(
        original - float(probabilities[0])
    )

    random_impact = float(
        np.mean(
            np.abs(
                original - probabilities[1:]
            )
        )
    )

    return float(
        important_impact /
        (random_impact + 1e-8)
    )


# ==========================================
# SAVE FEATURE IMPORTANCE PLOTS
# ==========================================

def plot_importance(
    df,
    column,
    path,
    title,
):

    top = df.head(10).iloc[::-1]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        top["feature"],
        top[column],
    )

    ax.axvline(
        0,
        linewidth=1,
        color="gray",
    )

    ax.set_title(title)
    ax.set_xlabel("Model explanation contribution")

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==========================================
# EVALUATE ONE DISEASE
# ==========================================

def evaluate_disease(disease, full=False):

    paths = paths_for(disease)

    model = joblib.load(
        paths["model"]
    )

    train_df = pd.read_csv(
        paths["train"]
    )

    test_df = pd.read_csv(
        paths["test"]
    )

    X_train = train_df.drop(
        columns="target"
    )

    X_test = test_df.drop(
        columns="target"
    )

    y_test = test_df["target"]

    print(
        f"\n{DISEASES[disease]['name']} "
        "- SHAP/LIME evaluation",
        flush=True,
    )

    explainers = make_explainers(
        model,
        X_train,
    )

    per_class = 3 if full else 2
    perturbations = 3 if full else 2
    lime_samples = 2000 if full else 700
    random_comparisons = 20 if full else 10

    selected = []

    for label in [0, 1]:

        candidates = y_test.index[
            y_test == label
        ].to_numpy()

        if len(candidates) == 0:
            raise ValueError(
                f"No class {label} in test set"
            )

        rng = np.random.default_rng(
            SEED + label
        )

        selected.extend(
            rng.choice(
                candidates,
                size=min(
                    per_class,
                    len(candidates),
                ),
                replace=False,
            )
        )

    results = []
    clinical_features = set()

    for number, index in enumerate(
        selected,
        start=1,
    ):

        patient = (
            X_test.loc[[index]]
            .copy()
        )

        original_shap, original_lime, lime_r2 = (
            explain_record(
                patient,
                model,
                *explainers,
                lime_samples=lime_samples,
            )
        )

        if number == 1:

            original_shap.to_csv(
                paths["report"] / "shap_explanation.csv",
                index=False,
            )

            original_lime.to_csv(
                paths["report"] / "lime_explanation.csv",
                index=False,
            )

            plot_importance(
                original_shap,
                "shap_value",
                paths["report"] / "figures"
                / "shap_patient_1.png",
                "SHAP",
            )

            plot_importance(
                original_lime,
                "lime_weight",
                paths["report"] / "figures"
                / "lime_patient_1.png",
                "LIME",
            )

        clinical_features.update(
            original_shap.head(5)["feature"]
        )

        clinical_features.update(
            original_lime.head(5)["feature"]
        )

        rng = np.random.default_rng(
            SEED + number
        )

        shap_stabilities = []
        lime_stabilities = []

        for _ in range(perturbations):

            changed = perturb(
                patient,
                X_train,
                rng,
            )

            new_shap, new_lime, _ = (
                explain_record(
                    changed,
                    model,
                    *explainers,
                    lime_samples=lime_samples,
                )
            )

            shap_stabilities.append(
                jaccard(
                    original_shap["feature"],
                    new_shap["feature"],
                )
            )

            lime_stabilities.append(
                jaccard(
                    original_lime["feature"],
                    new_lime["feature"],
                )
            )

        results.append({
            "record": number,
            "actual_class": int(
                y_test.loc[index]
            ),
            "positive_probability": float(
                model.predict_proba(patient)[0, 1]
            ),
            "top_k_jaccard": jaccard(
                original_shap["feature"],
                original_lime["feature"],
            ),
            "shap_stability": float(
                np.mean(shap_stabilities)
            ),
            "lime_stability": float(
                np.mean(lime_stabilities)
            ),
            "shap_fidelity_ratio": fidelity_ratio(
                patient,
                original_shap["feature"],
                model,
                X_train,
                np.random.default_rng(SEED + number),
                random_comparisons,
            ),
            "lime_fidelity_ratio": fidelity_ratio(
                patient,
                original_lime["feature"],
                model,
                X_train,
                np.random.default_rng(SEED + number),
                random_comparisons,
            ),
            "lime_local_r2": lime_r2,
        })

        print(
            f"Record {number}/{len(selected)} evaluated",
            flush=True,
        )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        paths["report"] / "xai_evaluation.csv",
        index=False,
    )

    review_path = (
        paths["report"]
        / "clinical_plausibility_review.csv"
    )

    if not review_path.exists():

        pd.DataFrame({
            "feature": sorted(clinical_features),
            "literature_evidence": "",
            "clinical_plausibility": "",
            "review_notes": "",
        }).to_csv(
            review_path,
            index=False,
        )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    columns = [
        "top_k_jaccard",
        "shap_stability",
        "lime_stability",
    ]

    ax.bar(
        [
            "Consistency",
            "SHAP Stability",
            "LIME Stability",
        ],
        [
            result_df[col].mean()
            for col in columns
        ],
    )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean top-five Jaccard similarity")

    ax.set_title(
        f"{DISEASES[disease]['name']} - XAI Reliability"
    )

    fig.tight_layout()

    figure_dir = (
        paths["report"] / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        figure_dir / "xai_reliability.png",
        dpi=140,
    )

    plt.close(fig)

    mean_scores = (
        result_df.select_dtypes(
            include=[np.number]
        )
        .drop(
            columns=[
                "record",
                "actual_class",
                "positive_probability",
            ]
        )
        .mean()
        .to_dict()
    )

    print(
        f"Consistency={mean_scores['top_k_jaccard']:.3f} | "
        f"SHAP Stability={mean_scores['shap_stability']:.3f} | "
        f"LIME Stability={mean_scores['lime_stability']:.3f}",
        flush=True,
    )

    return {
        "disease": disease,
        "name": DISEASES[disease]["name"],
        **mean_scores,
    }


# ==========================================
# RUN ALL FIVE DISEASES
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="HealthExplain-AI five-disease pipeline"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Use six XAI records per disease",
    )

    parser.add_argument(
        "--only",
        choices=list(DISEASES),
        help="Run a specific disease",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse saved models",
    )

    args = parser.parse_args()

    diseases = (
        [args.only]
        if args.only
        else list(DISEASES)
    )

    for directory in [
        RAW_DIR,
        DATA_DIR,
        MODEL_DIR,
        REPORT_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    summaries = []
    failures = []

    for disease in diseases:

        try:
            if not args.skip_training:
                train_disease(disease)

            summary = evaluate_disease(
                disease,
                full=args.full,
            )

            summaries.append(summary)

        except Exception as exc:

            failures.append(
                (disease, str(exc))
            )

            print(
                f"\nERROR in {disease}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            print(
                "Continuing with remaining diseases...",
                flush=True,
            )

    if summaries:

        summary_path = (
            REPORT_DIR / "xai_comparison.csv"
        )

        summary = pd.DataFrame(summaries)

        if args.only and summary_path.exists():

            existing = pd.read_csv(
                summary_path
            )

            existing = existing[
                existing["disease"] != args.only
            ]

            summary = pd.concat(
                [existing, summary],
                ignore_index=True,
            )

        summary.to_csv(
            summary_path,
            index=False,
        )

        print(
            "\n=== CROSS-DISEASE XAI COMPARISON ==="
        )

        print(
            summary.round(3).to_string(
                index=False
            )
        )

        print(
            f"\nSaved: {summary_path}"
        )

    if failures:

        print("\nFailed diseases:")

        for disease, error in failures:
            print(f"{disease}: {error}")

        raise SystemExit(1)

    print(
        "\nAll requested diseases completed successfully."
    )


if __name__ == "__main__":
    main()
