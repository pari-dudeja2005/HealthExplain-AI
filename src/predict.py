
import sys
import joblib
import numpy as np
import pandas as pd

from src.config import get_disease_config


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_trained_model(disease="breast_cancer"):

    config = get_disease_config(disease)
    model_path = config["model_path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run: python -m src.train {disease}"
        )

    return joblib.load(model_path)


# ==========================================
# PREPARE INPUT
# ==========================================

def prepare_input(data, model):

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    elif isinstance(data, pd.Series):
        data = data.to_frame().T

    elif isinstance(data, pd.DataFrame):
        data = data.copy()

    else:
        raise TypeError(
            "Input must be a dictionary, "
            "Pandas Series or DataFrame."
        )

    data = data.drop(
        columns=["target"],
        errors="ignore"
    )

    expected_features = list(
        model.feature_names_in_
    )

    missing_features = [
        feature for feature in expected_features
        if feature not in data.columns
    ]

    extra_features = [
        feature for feature in data.columns
        if feature not in expected_features
    ]

    if missing_features or extra_features:
        raise ValueError(
            f"Missing features: {missing_features}\n"
            f"Unexpected features: {extra_features}"
        )

    data = data[expected_features].copy()

    for feature in data.columns:
        data[feature] = pd.to_numeric(
            data[feature],
            errors="raise"
        )

    if np.isinf(data.to_numpy(dtype=float)).any():
        raise ValueError(
            "Input contains infinite values."
        )

    return data


# ==========================================
# PREDICTION PROBABILITIES
# ==========================================

def get_probabilities(data, model):

    prepared_data = prepare_input(
        data,
        model
    )

    return model.predict_proba(
        prepared_data
    )


# ==========================================
# SINGLE PREDICTION
# ==========================================

def predict_single(
    patient_data,
    threshold=0.5,
    disease="breast_cancer"
):

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    model = load_trained_model(disease)
    config = get_disease_config(disease)

    probabilities = get_probabilities(
        patient_data,
        model
    )

    positive_probability = float(
        probabilities[0, 1]
    )

    negative_probability = float(
        probabilities[0, 0]
    )

    predicted_class = int(
        positive_probability >= threshold
    )

    predicted_label = (
        config["positive_label"]
        if predicted_class == 1
        else config["negative_label"]
    )

    result = {
        "disease": disease,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "negative_probability": negative_probability,
        "positive_probability": positive_probability,
        "threshold": threshold
    }

    # Maintain compatibility with the existing dashboard
    if disease == "breast_cancer":
        result["benign_probability"] = (
            negative_probability
        )
        result["malignant_probability"] = (
            positive_probability
        )

    return result


# ==========================================
# BATCH PREDICTION
# ==========================================

def predict_batch(
    data,
    threshold=0.5,
    disease="breast_cancer"
):

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    model = load_trained_model(disease)
    config = get_disease_config(disease)

    prepared_data = prepare_input(
        data,
        model
    )

    probabilities = model.predict_proba(
        prepared_data
    )

    positive_probabilities = probabilities[:, 1]
    negative_probabilities = probabilities[:, 0]

    predicted_classes = (
        positive_probabilities >= threshold
    ).astype(int)

    predicted_labels = np.where(
        predicted_classes == 1,
        config["positive_label"],
        config["negative_label"]
    )

    results = prepared_data.copy()

    results["predicted_class"] = predicted_classes
    results["predicted_label"] = predicted_labels

    results["negative_probability"] = (
        negative_probabilities
    )

    results["positive_probability"] = (
        positive_probabilities
    )

    if disease == "breast_cancer":
        results["benign_probability"] = (
            negative_probabilities
        )
        results["malignant_probability"] = (
            positive_probabilities
        )

    return results


# ==========================================
# TEST PREDICTIONS
# ==========================================

if __name__ == "__main__":

    disease = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "breast_cancer"
    )

    config = get_disease_config(disease)

    if disease == "breast_cancer":
        test_path = (
            config["test_data_path"].parent
            / "test_data.csv"
        )
    else:
        test_path = config["test_data_path"]

    test_data = pd.read_csv(test_path)

    X_test = test_data.drop(
        columns=["target"]
    )

    print(f"\nDisease: {config['name']}")

    print("\n========== SINGLE PREDICTION ==========")

    result = predict_single(
        X_test.iloc[[0]],
        disease=disease
    )

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n========== BATCH PREDICTION ==========")

    batch_results = predict_batch(
        X_test.head(5),
        disease=disease
    )

    print(
        batch_results[
            [
                "predicted_class",
                "predicted_label",
                "negative_probability",
                "positive_probability"
            ]
        ].to_string(index=False)
    )

    print("\nPrediction Test Completed Successfully!")

