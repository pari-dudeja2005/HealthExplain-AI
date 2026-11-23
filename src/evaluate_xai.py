
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import get_disease_config
from src.predict import load_trained_model
from src.explain import (
    create_explainers,
    explain_with_shap,
    explain_with_lime
)


# ==========================================
# CONFIGURATION
# ==========================================

TOP_K = 5
PERTURBATIONS = 3
RANDOM_COMPARISONS = 20
RANDOM_STATE = 42


# ==========================================
# LOAD EVALUATION DATA
# ==========================================

def load_evaluation_data(disease):

    config = get_disease_config(disease)

    if disease == "breast_cancer":

        processed_dir = config["train_data_path"].parent

        train_path = processed_dir / "train_data.csv"
        test_path = processed_dir / "test_data.csv"

        report_dir = config["report_dir"].parent

    else:

        train_path = config["train_data_path"]
        test_path = config["test_data_path"]

        report_dir = config["report_dir"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    return X_train, X_test, y_test, report_dir


# ==========================================
# TOP-K FEATURE SIMILARITY
# ==========================================

def top_k_jaccard(features_a, features_b, k=TOP_K):

    set_a = set(list(features_a)[:k])
    set_b = set(list(features_b)[:k])

    union = set_a | set_b

    if not union:
        return 1.0

    return len(set_a & set_b) / len(union)


# ==========================================
# GENERATE EXPLANATIONS
# ==========================================

def get_explanations(
    patient,
    model,
    shap_explainer,
    lime_explainer
):

    shap_df = explain_with_shap(
        patient,
        model,
        shap_explainer
    )

    lime_df, lime_explanation = explain_with_lime(
        patient,
        model,
        lime_explainer
    )

    return shap_df, lime_df, lime_explanation


# ==========================================
# INPUT PERTURBATION
# ==========================================

def perturb_patient(
    patient,
    X_train,
    rng,
    disease
):

    # Convert integer columns to float so that
    # decimal perturbations can be assigned safely.
    perturbed = patient.astype(float).copy()

    std = X_train.std(numeric_only=True)
    minimum = X_train.min()
    maximum = X_train.max()

    zero_missing_features = {
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

    integer_features = {
        "preg",
        "pregnancies",
        "age"
    }

    for feature in patient.columns:

        original = float(patient.iloc[0][feature])

        normalized = (
            feature.lower()
            .replace("_", "")
            .replace(" ", "")
        )

        # Preserve zero values representing
        # missing measurements in Diabetes.
        if (
            disease == "diabetes"
            and normalized in zero_missing_features
            and original == 0
        ):
            continue

        feature_std = float(std[feature])

        noise = rng.normal(
            loc=0,
            scale=0.01 * feature_std
        )

        new_value = float(
            np.clip(
                original + noise,
                minimum[feature],
                maximum[feature]
            )
        )

        # Preserve integer-valued clinical features.
        if (
            disease == "diabetes"
            and normalized in integer_features
        ):
            new_value = float(round(new_value))

        perturbed.at[
            perturbed.index[0],
            feature
        ] = new_value

    return perturbed


# ==========================================
# EXPLANATION STABILITY
# ==========================================

def evaluate_stability(
    patient,
    X_train,
    model,
    shap_explainer,
    lime_explainer,
    original_shap,
    original_lime,
    disease,
    seed
):

    rng = np.random.default_rng(seed)

    shap_scores = []
    lime_scores = []

    for _ in range(PERTURBATIONS):

        perturbed = perturb_patient(
            patient,
            X_train,
            rng,
            disease
        )

        shap_new, lime_new, _ = get_explanations(
            perturbed,
            model,
            shap_explainer,
            lime_explainer
        )

        shap_similarity = top_k_jaccard(
            original_shap["feature"],
            shap_new["feature"]
        )

        lime_similarity = top_k_jaccard(
            original_lime["feature"],
            lime_new["feature"]
        )

        shap_scores.append(shap_similarity)
        lime_scores.append(lime_similarity)

    return (
        float(np.mean(shap_scores)),
        float(np.mean(lime_scores))
    )


# ==========================================
# FEATURE REPLACEMENT
# ==========================================

def replace_features(
    patient,
    features,
    medians
):

    # Convert to float to allow decimal median values.
    modified = patient.astype(float).copy()

    for feature in features:

        modified.at[
            modified.index[0],
            feature
        ] = float(medians[feature])

    return modified


# ==========================================
# EXPLANATION FIDELITY
# ==========================================

def evaluate_fidelity(
    patient,
    feature_ranking,
    model,
    X_train,
    rng
):

    all_features = list(patient.columns)

    k = min(
        TOP_K,
        len(all_features)
    )

    important_features = list(
        feature_ranking
    )[:k]

    medians = X_train.median()

    original_probability = float(
        model.predict_proba(patient)[0, 1]
    )

    candidates = [
        replace_features(
            patient,
            important_features,
            medians
        )
    ]

    for _ in range(RANDOM_COMPARISONS):

        random_features = rng.choice(
            all_features,
            size=k,
            replace=False
        )

        modified = replace_features(
            patient,
            random_features,
            medians
        )

        candidates.append(modified)

    modified_data = pd.concat(
        candidates,
        ignore_index=True
    )

    probabilities = model.predict_proba(
        modified_data
    )[:, 1]

    important_impact = abs(
        original_probability - probabilities[0]
    )

    random_impacts = abs(
        original_probability - probabilities[1:]
    )

    average_random_impact = float(
        np.mean(random_impacts)
    )

    fidelity_ratio = (
        important_impact
        / (average_random_impact + 1e-8)
    )

    return float(fidelity_ratio)


# ==========================================
# CLINICAL PLAUSIBILITY REVIEW
# ==========================================

def save_clinical_review(
    features,
    output_path
):

    if output_path.exists():

        print(
            "Existing clinical review preserved:",
            output_path
        )

        return

    review_df = pd.DataFrame({
        "feature": sorted(features),
        "literature_evidence": "",
        "clinical_plausibility": "",
        "review_notes": ""
    })

    review_df.to_csv(
        output_path,
        index=False
    )


# ==========================================
# RELIABILITY VISUALISATION
# ==========================================

def save_reliability_plot(
    results,
    output_path
):

    metrics = {
        "SHAP-LIME Consistency":
            results["top_k_jaccard"].mean(),

        "SHAP Stability":
            results["shap_stability"].mean(),

        "LIME Stability":
            results["lime_stability"].mean()
    }

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        list(metrics.keys()),
        list(metrics.values()),
        color=[
            "#587C9E",
            "#417BA6",
            "#C18B50"
        ]
    )

    ax.set_ylabel(
        "Mean Jaccard Similarity"
    )

    ax.set_ylim(0, 1.1)

    ax.set_title(
        "XAI Explanation Reliability"
    )

    for bar, value in zip(
        bars,
        metrics.values()
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center"
        )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# MAIN XAI EVALUATION
# ==========================================

def evaluate_xai(disease="breast_cancer"):

    config = get_disease_config(disease)

    print("\n======================================")
    print("XAI RELIABILITY EVALUATION")
    print("======================================")

    print("Disease:", config["name"])

    model = load_trained_model(disease)

    X_train, X_test, y_test, report_dir = (
        load_evaluation_data(disease)
    )

    print("\nCreating SHAP and LIME explainers...")

    shap_explainer, lime_explainer = (
        create_explainers(
            model,
            X_train
        )
    )

    # Select three records from each class.
    selected_indices = []

    for class_value in [0, 1]:

        class_indices = y_test[
            y_test == class_value
        ].index

        sampled = pd.Series(
            class_indices
        ).sample(
            n=min(3, len(class_indices)),
            random_state=RANDOM_STATE
        )

        selected_indices.extend(
            sampled.tolist()
        )

    results = []
    clinical_features = set()

    total_records = len(selected_indices)

    for record_number, index in enumerate(
        selected_indices,
        start=1
    ):

        print(
            f"\nEvaluating record "
            f"{record_number}/{total_records}..."
        )

        # Use float data during explanation evaluation.
        patient = (
            X_test.loc[[index]]
            .astype(float)
            .copy()
        )

        actual_class = int(
            y_test.loc[index]
        )

        probability = float(
            model.predict_proba(patient)[0, 1]
        )

        # ----------------------------------
        # SHAP AND LIME EXPLANATIONS
        # ----------------------------------

        shap_df, lime_df, lime_explanation = (
            get_explanations(
                patient,
                model,
                shap_explainer,
                lime_explainer
            )
        )

        # ----------------------------------
        # CONSISTENCY
        # ----------------------------------

        consistency = top_k_jaccard(
            shap_df["feature"],
            lime_df["feature"]
        )

        # ----------------------------------
        # STABILITY
        # ----------------------------------

        shap_stability, lime_stability = (
            evaluate_stability(
                patient,
                X_train,
                model,
                shap_explainer,
                lime_explainer,
                shap_df,
                lime_df,
                disease,
                RANDOM_STATE + record_number
            )
        )

        # ----------------------------------
        # FIDELITY
        # ----------------------------------

        rng = np.random.default_rng(
            RANDOM_STATE + record_number
        )

        shap_fidelity = evaluate_fidelity(
            patient,
            shap_df["feature"],
            model,
            X_train,
            rng
        )

        lime_fidelity = evaluate_fidelity(
            patient,
            lime_df["feature"],
            model,
            X_train,
            rng
        )

        # ----------------------------------
        # LIME LOCAL FIT
        # ----------------------------------

        lime_r2 = float(
            lime_explanation.score
        )

        # ----------------------------------
        # CLINICAL REVIEW FEATURES
        # ----------------------------------

        clinical_features.update(
            shap_df.head(TOP_K)["feature"]
        )

        clinical_features.update(
            lime_df.head(TOP_K)["feature"]
        )

        # ----------------------------------
        # STORE RESULTS
        # ----------------------------------

        results.append({
            "record": record_number,
            "actual_class": actual_class,
            "positive_probability": probability,
            "top_k_jaccard": consistency,
            "shap_stability": shap_stability,
            "lime_stability": lime_stability,
            "shap_fidelity_ratio": shap_fidelity,
            "lime_fidelity_ratio": lime_fidelity,
            "lime_local_r2": lime_r2
        })

        print(
            f"Consistency: {consistency:.4f}"
        )

        print(
            f"SHAP Stability: {shap_stability:.4f}"
        )

        print(
            f"LIME Stability: {lime_stability:.4f}"
        )

    # ======================================
    # SAVE EVALUATION RESULTS
    # ======================================

    results_df = pd.DataFrame(results)

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation_path = (
        report_dir / "xai_evaluation.csv"
    )

    results_df.to_csv(
        evaluation_path,
        index=False
    )

    # ======================================
    # SAVE RELIABILITY PLOT
    # ======================================

    figure_path = (
        report_dir
        / "figures"
        / "xai_reliability.png"
    )

    save_reliability_plot(
        results_df,
        figure_path
    )

    # ======================================
    # SAVE CLINICAL REVIEW TEMPLATE
    # ======================================

    clinical_path = (
        report_dir
        / "clinical_plausibility_review.csv"
    )

    save_clinical_review(
        clinical_features,
        clinical_path
    )

    # ======================================
    # DISPLAY AVERAGE RESULTS
    # ======================================

    print("\n========== AVERAGE RESULTS ==========")

    metrics = [
        "top_k_jaccard",
        "shap_stability",
        "lime_stability",
        "shap_fidelity_ratio",
        "lime_fidelity_ratio",
        "lime_local_r2"
    ]

    for metric in metrics:

        print(
            f"{metric}: "
            f"{results_df[metric].mean():.4f}"
        )

    print("\n========== SAVED FILES ==========")

    print("Evaluation:", evaluation_path)
    print("Reliability Plot:", figure_path)
    print("Clinical Review:", clinical_path)

    print("\nEvaluation Completed Successfully!")

    return results_df


# ==========================================
# RUN EVALUATION
# ==========================================

if __name__ == "__main__":

    disease = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "breast_cancer"
    )

    evaluate_xai(disease)


