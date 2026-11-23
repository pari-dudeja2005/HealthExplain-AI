
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression

from src.config import get_disease_config
from src.predict import load_trained_model


def prepare_explanation_input(data, model):

    features = list(model.feature_names_in_)

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    elif isinstance(data, pd.Series):
        data = data.to_frame().T

    elif isinstance(data, pd.DataFrame):
        data = data.copy()

    else:
        raise TypeError("Input must be a DataFrame, Series or dict.")

    data = data.drop(columns=["target"], errors="ignore")

    missing = [f for f in features if f not in data.columns]
    extra = [f for f in data.columns if f not in features]

    if missing or extra:
        raise ValueError(
            f"Missing features: {missing}\n"
            f"Unexpected features: {extra}"
        )

    data = data[features].apply(pd.to_numeric, errors="raise")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError(
            "Explanation input contains missing or infinite values."
        )

    return data


# ==========================================
# CREATE SHAP AND LIME EXPLAINERS
# ==========================================

def create_explainers(model, X_train):

    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    if not isinstance(estimator, LogisticRegression):
        raise NotImplementedError(
            "This explanation module currently supports "
            "Logistic Regression models."
        )

    X_train = prepare_explanation_input(X_train, model)

    feature_names = list(X_train.columns)

    # Sample background data reproducibly
    background = X_train.sample(
        n=min(200, len(X_train)),
        random_state=42
    )

    # SHAP explains the fitted Logistic Regression model
    # using the same preprocessing as prediction.
    background_processed = preprocessor.transform(background)

    shap_explainer = shap.Explainer(
        estimator,
        background_processed,
        feature_names=feature_names
    )

    # LIME works with original feature values.
    lime_explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(dtype=float),
        feature_names=feature_names,
        class_names=["Negative", "Positive"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    return shap_explainer, lime_explainer


# ==========================================
# SHAP EXPLANATION
# ==========================================

def explain_with_shap(
    patient_data,
    model,
    shap_explainer
):

    patient = prepare_explanation_input(
        patient_data,
        model
    )

    if len(patient) != 1:
        raise ValueError(
            "SHAP explanation requires one record."
        )

    preprocessor = model.named_steps["preprocessor"]

    processed_patient = preprocessor.transform(patient)

    explanation = shap_explainer(processed_patient)

    shap_values = np.asarray(
        explanation.values[0]
    ).reshape(-1)

    feature_names = list(patient.columns)

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "feature_value": patient.iloc[0].to_numpy(dtype=float),
        "shap_value": shap_values
    })

    shap_df["absolute_importance"] = (
        shap_df["shap_value"].abs()
    )

    shap_df = shap_df.sort_values(
        "absolute_importance",
        ascending=False
    ).reset_index(drop=True)

    return shap_df


# ==========================================
# LIME EXPLANATION
# ==========================================

def explain_with_lime(
    patient_data,
    model,
    lime_explainer
):

    patient = prepare_explanation_input(
        patient_data,
        model
    )

    if len(patient) != 1:
        raise ValueError(
            "LIME explanation requires one record."
        )

    feature_names = list(patient.columns)

    def predict_fn(samples):

        samples_df = pd.DataFrame(
            samples,
            columns=feature_names
        )

        return model.predict_proba(samples_df)

    lime_explanation = lime_explainer.explain_instance(
        data_row=patient.iloc[0].to_numpy(dtype=float),
        predict_fn=predict_fn,
        labels=(1,),
        num_features=len(feature_names),
        num_samples=2000
    )

    weights = dict(
        lime_explanation.as_map()[1]
    )

    lime_df = pd.DataFrame({
        "feature": feature_names,
        "feature_value": patient.iloc[0].to_numpy(dtype=float),
        "lime_weight": [
            float(weights.get(index, 0.0))
            for index in range(len(feature_names))
        ]
    })

    lime_df["absolute_importance"] = (
        lime_df["lime_weight"].abs()
    )

    lime_df = lime_df.sort_values(
        "absolute_importance",
        ascending=False
    ).reset_index(drop=True)

    return lime_df, lime_explanation


# ==========================================
# SAVE FEATURE IMPORTANCE PLOT
# ==========================================

def save_feature_plot(
    df,
    value_column,
    title,
    xlabel,
    output_path
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_df = (
        df.sort_values(
            value_column,
            key=lambda values: values.abs(),
            ascending=False
        )
        .head(10)
        .iloc[::-1]
    )

    colors = [
        "#C65D5D" if value > 0 else "#417BA6"
        for value in plot_df[value_column]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        plot_df["feature"],
        plot_df[value_column],
        color=colors
    )

    ax.axvline(
        x=0,
        color="gray",
        linewidth=1
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ==========================================
# TEST EXPLANATIONS
# ==========================================

def main(disease="breast_cancer"):

    config = get_disease_config(disease)

    model = load_trained_model(disease)

    if disease == "breast_cancer":

        train_path = (
            config["train_data_path"].parent
            / "train_data.csv"
        )

        test_path = (
            config["test_data_path"].parent
            / "test_data.csv"
        )

        output_dir = (
            config["report_dir"].parent
            / "figures"
        )

        report_dir = config["report_dir"].parent

    else:

        train_path = config["train_data_path"]
        test_path = config["test_data_path"]

        report_dir = config["report_dir"]
        output_dir = report_dir / "figures"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])

    patient = X_test.iloc[[0]].copy()

    print(f"\nDisease: {config['name']}")
    print("Creating explainers...")

    shap_explainer, lime_explainer = create_explainers(
        model,
        X_train
    )

    print("\nGenerating SHAP explanation...")

    shap_df = explain_with_shap(
        patient,
        model,
        shap_explainer
    )

    print("\nGenerating LIME explanation...")

    lime_df, lime_explanation = explain_with_lime(
        patient,
        model,
        lime_explainer
    )

    probability = float(
        model.predict_proba(patient)[0, 1]
    )

    print(
        f"\n{config['positive_label']} probability: "
        f"{probability:.4f}"
    )

    print("\n========== TOP 5 SHAP FEATURES ==========")

    print(
        shap_df.head(5).to_string(index=False)
    )

    print("\n========== TOP 5 LIME FEATURES ==========")

    print(
        lime_df.head(5).to_string(index=False)
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    shap_df.to_csv(
        report_dir / "shap_explanation.csv",
        index=False
    )

    lime_df.to_csv(
        report_dir / "lime_explanation.csv",
        index=False
    )

    shap_path = save_feature_plot(
        shap_df,
        "shap_value",
        f"SHAP Explanation - {config['name']}",
        "Contribution to Positive-Class Log-Odds",
        output_dir / "shap_patient_1.png"
    )

    lime_path = save_feature_plot(
        lime_df,
        "lime_weight",
        f"LIME Explanation - {config['name']}",
        "Contribution to Positive-Class Probability",
        output_dir / "lime_patient_1.png"
    )

    print("\n========== SAVED FILES ==========")

    print("SHAP Plot:", shap_path)
    print("LIME Plot:", lime_path)

    print("\nExplanations Generated Successfully!")


if __name__ == "__main__":

    disease = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "breast_cancer"
    )

    main(disease)

