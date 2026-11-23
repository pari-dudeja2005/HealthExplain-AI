
import sys
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.config import (
    get_disease_config,
    RANDOM_STATE,
    TEST_SIZE
)

from src.data_loader import load_dataset
from src.preprocessing import (
    split_data,
    build_preprocessor
)


# ==========================================
# DEFINE MODELS
# ==========================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    return models


# ==========================================
# EVALUATION METRICS
# ==========================================

def calculate_metrics(y_true, y_pred, y_prob):

    return {

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                y_prob
            )
        )
    }


# ==========================================
# CREATE MODEL PIPELINE
# ==========================================

def create_model_pipeline(disease, estimator):

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(disease)
        ),
        (
            "model",
            estimator
        )
    ])


# ==========================================
# TRAIN AND COMPARE MODELS
# ==========================================

def train_models(disease="breast_cancer"):

    config = get_disease_config(disease)

    print("\n======================================")
    print("HEALTHEXPLAIN-AI MODEL TRAINING")
    print("======================================")

    print(f"\nDisease: {config['name']}")

    # --------------------------------------
    # LOAD DATASET
    # --------------------------------------

    df = load_dataset(disease)

    print("\nDataset Shape:", df.shape)

    print("\nTarget Distribution:")
    print(df["target"].value_counts().sort_index())

    # --------------------------------------
    # TRAIN-TEST SPLIT
    # --------------------------------------

    X_train_full, X_test, y_train_full, y_test = (
        split_data(
            df,
            target_column="target",
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    )

    # Validation split from training data only
    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y_train_full
        )
    )

    print("\nData Split:")
    print("Training Records:", len(X_train))
    print("Validation Records:", len(X_val))
    print("Testing Records:", len(X_test))

    # --------------------------------------
    # MODEL COMPARISON
    # --------------------------------------

    models = get_models()

    validation_results = []

    print("\n========== MODEL COMPARISON ==========")

    for model_name, estimator in models.items():

        print(f"\nTraining {model_name}...")

        pipeline = create_model_pipeline(
            disease,
            estimator
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(X_val)

        probabilities = pipeline.predict_proba(
            X_val
        )[:, 1]

        metrics = calculate_metrics(
            y_val,
            predictions,
            probabilities
        )

        validation_results.append({
            "model": model_name,
            **metrics
        })

        print(
            f"Accuracy:  {metrics['accuracy']:.4f}"
        )
        print(
            f"Precision: {metrics['precision']:.4f}"
        )
        print(
            f"Recall:    {metrics['recall']:.4f}"
        )
        print(
            f"F1 Score:  {metrics['f1']:.4f}"
        )
        print(
            f"ROC-AUC:   {metrics['roc_auc']:.4f}"
        )
        print(
            f"PR-AUC:    {metrics['pr_auc']:.4f}"
        )

    comparison_df = pd.DataFrame(
        validation_results
    )

    comparison_df = comparison_df.sort_values(
        by="roc_auc",
        ascending=False
    ).reset_index(drop=True)

    print("\n========== VALIDATION RESULTS ==========")

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------

    best_model_name = comparison_df.iloc[0][
        "model"
    ]

    print("\nBest Model:", best_model_name)

    # --------------------------------------
    # RETRAIN ON FULL TRAINING DATA
    # --------------------------------------

    print("\nRetraining Best Model...")

    best_estimator = get_models()[
        best_model_name
    ]

    final_pipeline = create_model_pipeline(
        disease,
        best_estimator
    )

    final_pipeline.fit(
        X_train_full,
        y_train_full
    )

    # --------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------

    test_predictions = final_pipeline.predict(
        X_test
    )

    test_probabilities = (
        final_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities
    )

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )

    print("\n========== FINAL TEST RESULTS ==========")

    for metric, value in test_metrics.items():

        print(
            f"{metric.upper()}: {value:.4f}"
        )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                config["negative_label"],
                config["positive_label"]
            ],
            zero_division=0
        )
    )

    # --------------------------------------
    # OUTPUT PATHS
    # --------------------------------------

    # Preserve existing breast cancer file paths
    if disease == "breast_cancer":

        model_path = config["model_path"]

        comparison_path = (
            config["report_dir"].parent
            / "model_comparison.csv"
        )

        metrics_path = (
            config["report_dir"].parent
            / "test_metrics.json"
        )

        train_path = (
            config["train_data_path"].parent
            / "train_data.csv"
        )

        test_path = (
            config["test_data_path"].parent
            / "test_data.csv"
        )

    else:

        model_path = config["model_path"]

        comparison_path = (
            config["report_dir"]
            / "model_comparison.csv"
        )

        metrics_path = (
            config["report_dir"]
            / "test_metrics.json"
        )

        train_path = config["train_data_path"]

        test_path = config["test_data_path"]

    # --------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------

    for path in [
        model_path,
        comparison_path,
        metrics_path,
        train_path,
        test_path
    ]:

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------
    # SAVE MODEL
    # --------------------------------------

    joblib.dump(
        final_pipeline,
        model_path
    )

    # --------------------------------------
    # SAVE VALIDATION RESULTS
    # --------------------------------------

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # --------------------------------------
    # SAVE TEST METRICS
    # --------------------------------------

    report = {

        "disease": disease,

        "best_model": best_model_name,

        "selection_metric": "validation_roc_auc",

        "training_records": int(
            len(X_train_full)
        ),

        "testing_records": int(
            len(X_test)
        ),

        "test_metrics": test_metrics,

        "confusion_matrix": cm.tolist(),

        "class_labels": {
            "0": config["negative_label"],
            "1": config["positive_label"]
        }
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------
    # SAVE TRAIN AND TEST DATA
    # --------------------------------------

    train_data = X_train_full.copy()

    train_data["target"] = (
        y_train_full.to_numpy()
    )

    test_data = X_test.copy()

    test_data["target"] = (
        y_test.to_numpy()
    )

    train_data.to_csv(
        train_path,
        index=False
    )

    test_data.to_csv(
        test_path,
        index=False
    )

    print("\n========== FILES SAVED ==========")

    print("Model:", model_path)
    print("Comparison:", comparison_path)
    print("Metrics:", metrics_path)
    print("Train Data:", train_path)
    print("Test Data:", test_path)

    print("\nTraining Completed Successfully!")

    return final_pipeline, comparison_df, report


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    disease = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "breast_cancer"
    )

    train_models(disease)

