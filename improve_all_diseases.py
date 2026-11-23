
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from run_all_diseases import (
    DISEASES,
    REPORT_DIR,
    load_disease,
    split_indices,
    make_preprocessor,
    get_models,
    paths_for,
)


THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.01),
    2,
)


def calculate_scores(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "specificity": (
            tn / (tn + fp)
            if tn + fp else 0.0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "f2": fbeta_score(
            y_true, predictions,
            beta=2,
            zero_division=0,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(disease, validation_results):

    # Prioritise sensitivity while maintaining
    # minimum precision for Diabetes.
    if disease == "diabetes":

        candidates = validation_results[
            validation_results["precision"] >= 0.50
        ].copy()

        if candidates.empty:
            candidates = validation_results.copy()

        candidates["distance"] = abs(
            candidates["threshold"] - 0.5
        )

        candidates = candidates.sort_values(
            ["f2", "recall", "distance"],
            ascending=[False, False, True],
        )

        return float(
            candidates.iloc[0]["threshold"]
        )

    # Parkinson's needs better discrimination
    # between positive and negative cases.
    if disease == "parkinsons":

        candidates = validation_results.copy()

        candidates["distance"] = abs(
            candidates["threshold"] - 0.5
        )

        candidates = candidates.sort_values(
            [
                "balanced_accuracy",
                "specificity",
                "distance",
            ],
            ascending=[False, False, True],
        )

        return float(
            candidates.iloc[0]["threshold"]
        )

    # Keep the established threshold for
    # the three better-performing diseases.
    return 0.5


def count_cross_split_duplicates(train_df, test_df):

    features = [
        col for col in train_df.columns
        if col != "target"
    ]

    train_hashes = set(
        pd.util.hash_pandas_object(
            train_df[features],
            index=False,
        ).tolist()
    )

    test_hashes = pd.util.hash_pandas_object(
        test_df[features],
        index=False,
    )

    return int(
        test_hashes.isin(train_hashes).sum()
    )


def evaluate_disease(disease):

    paths = paths_for(disease)

    report_path = (
        paths["report"] / "test_metrics.json"
    )

    if not report_path.exists():
        raise FileNotFoundError(report_path)

    report = json.loads(
        report_path.read_text()
    )

    selected_model = report["best_model"]

    print("\n" + "=" * 55)
    print(DISEASES[disease]["name"])
    print("=" * 55)

    # Recreate the original deterministic split.
    X, y, groups = load_disease(disease)

    fit_ids, val_ids, train_ids, test_ids = (
        split_indices(y, groups)
    )

    X_fit = X.iloc[fit_ids]
    y_fit = y.iloc[fit_ids]

    X_val = X.iloc[val_ids]
    y_val = y.iloc[val_ids]

    # Train a validation-only model.
    # The final saved model is not modified.
    preprocessor, _, _ = make_preprocessor(X_fit)

    validation_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", get_models()[selected_model]),
    ])

    validation_model.fit(X_fit, y_fit)

    validation_probabilities = (
        validation_model.predict_proba(X_val)[:, 1]
    )

    # Evaluate candidate thresholds only
    # on validation data.
    validation_results = pd.DataFrame([
        calculate_scores(
            y_val,
            validation_probabilities,
            threshold,
        )
        for threshold in THRESHOLDS
    ])

    selected_threshold = select_threshold(
        disease,
        validation_results,
    )

    paths["report"].mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_results.to_csv(
        paths["report"]
        / "validation_threshold_scan.csv",
        index=False,
    )

    # Use the original saved test dataset.
    train_df = pd.read_csv(paths["train"])
    test_df = pd.read_csv(paths["test"])

    X_test = test_df.drop(columns="target")
    y_test = test_df["target"]

    final_model = joblib.load(
        paths["model"]
    )

    probabilities = (
        final_model.predict_proba(X_test)[:, 1]
    )

    baseline = calculate_scores(
        y_test,
        probabilities,
        0.5,
    )

    candidate = calculate_scores(
        y_test,
        probabilities,
        selected_threshold,
    )

    duplicates = count_cross_split_duplicates(
        train_df,
        test_df,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print(f"Model: {selected_model}")
    print(f"Validation records: {len(y_val)}")
    print(f"Test records: {len(y_test)}")

    print(
        f"Default threshold: 0.50 | "
        f"Candidate threshold: {selected_threshold:.2f}"
    )

    print("\nDEFAULT THRESHOLD (0.50)")
    print(f"Accuracy: {baseline['accuracy']:.4f}")
    print(f"Recall: {baseline['recall']:.4f}")
    print(f"Specificity: {baseline['specificity']:.4f}")
    print(f"F1: {baseline['f1']:.4f}")
    print(f"Balanced Accuracy: {baseline['balanced_accuracy']:.4f}")
    print(f"False Positives: {baseline['fp']}")
    print(f"False Negatives: {baseline['fn']}")

    print("\nVALIDATION-SELECTED THRESHOLD")
    print(f"Accuracy: {candidate['accuracy']:.4f}")
    print(f"Recall: {candidate['recall']:.4f}")
    print(f"Specificity: {candidate['specificity']:.4f}")
    print(f"F1: {candidate['f1']:.4f}")
    print(f"Balanced Accuracy: {candidate['balanced_accuracy']:.4f}")
    print(f"False Positives: {candidate['fp']}")
    print(f"False Negatives: {candidate['fn']}")

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(
        "Cross-split duplicate test records:",
        duplicates,
    )

    return {
        "Disease": DISEASES[disease]["name"],
        "Model": selected_model,
        "Default Threshold": 0.5,
        "Candidate Threshold": selected_threshold,
        "Default Recall": baseline["recall"],
        "Candidate Recall": candidate["recall"],
        "Default Specificity": baseline["specificity"],
        "Candidate Specificity": candidate["specificity"],
        "Default Balanced Accuracy":
            baseline["balanced_accuracy"],
        "Candidate Balanced Accuracy":
            candidate["balanced_accuracy"],
        "Default F1": baseline["f1"],
        "Candidate F1": candidate["f1"],
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Duplicate Test Records": duplicates,
    }


def main():

    summaries = []
    failures = []

    for disease in DISEASES:

        try:
            result = evaluate_disease(disease)
            summaries.append(result)

        except Exception as exc:
            print(
                f"\nERROR in {disease}: "
                f"{type(exc).__name__}: {exc}"
            )

            failures.append(disease)

    if summaries:

        summary = pd.DataFrame(summaries)

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = (
            REPORT_DIR
            / "threshold_optimisation_comparison.csv"
        )

        summary.to_csv(
            output,
            index=False,
        )

        print("\n" + "=" * 55)
        print("FINAL FIVE-DISEASE COMPARISON")
        print("=" * 55)

        columns = [
            "Disease",
            "Default Threshold",
            "Candidate Threshold",
            "Default Recall",
            "Candidate Recall",
            "Default Specificity",
            "Candidate Specificity",
            "Default Balanced Accuracy",
            "Candidate Balanced Accuracy",
        ]

        print(
            summary[columns]
            .round(3)
            .to_string(index=False)
        )

        print(f"\nSaved: {output}")

    if failures:
        print("\nFailed:", ", ".join(failures))
    else:
        print("\nAll five diseases evaluated.")


if __name__ == "__main__":
    main()
