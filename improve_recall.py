
"""Improve recall for Diabetes and Heart Disease.

Run from the HealthExplain-AI project folder:

    python improve_recall.py
    python improve_recall.py --apply

The script:
1. Compares class-weighted ML models using training CV.
2. Selects a candidate only when validation safeguards pass.
3. Evaluates the selected candidate on held-out test data.
4. Backs up existing models before replacement.

The classification threshold remains 0.50.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from run_all_diseases import (
    DISEASES,
    SEED,
    make_preprocessor,
    paths_for,
)


MIN_SPECIFICITY = {
    "diabetes": 0.70,
    "heart_disease": 0.70,
}

MIN_RECALL_GAIN = 0.03
MAX_BALANCED_ACCURACY_DROP = 0.02


def evaluate(y, prob):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)

    pred = (prob >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "specificity": (
            float(tn / (tn + fp))
            if tn + fp else 0.0
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y, prob)
        ),
        "pr_auc": float(
            average_precision_score(y, prob)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def candidates(X_train, y_train, quick=False):
    """Create class-weighted candidate pipelines."""

    prep, _, _ = make_preprocessor(X_train)

    negative, positive = np.bincount(
        np.asarray(y_train, dtype=int),
        minlength=2,
    )

    imbalance = negative / max(positive, 1)

    models = {}

    # Logistic Regression
    if quick:
        logistic_specs = [
            (1.0, "balanced"),
            (1.0, {0: 1, 1: 2.0}),
        ]
    else:
        logistic_specs = [
            (c, weight)
            for c in (0.3, 1.0, 3.0)
            for weight in (
                "balanced",
                {0: 1, 1: 1.5},
                {0: 1, 1: 2.5},
            )
        ]

    for c, weight in logistic_specs:

        name = (
            f"Logistic Regression C={c} "
            f"weights={weight}"
        )

        models[name] = LogisticRegression(
            C=c,
            class_weight=weight,
            max_iter=4000,
            solver="lbfgs",
            random_state=SEED,
        )

    # Random Forest
    if quick:
        forest_specs = [
            (3, "balanced_subsample"),
        ]
    else:
        forest_specs = [
            (leaf, weight)
            for leaf in (2, 5)
            for weight in (
                "balanced_subsample",
                {0: 1, 1: 2.0},
            )
        ]

    for leaf, weight in forest_specs:

        name = (
            f"Random Forest leaf={leaf} "
            f"weights={weight}"
        )

        models[name] = RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=leaf,
            class_weight=weight,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=2,
        )

    # XGBoost
    if quick:
        xgb_specs = [(2, 1.5)]
    else:
        xgb_specs = [
            (depth, multiplier)
            for depth in (2, 3)
            for multiplier in (1.0, 1.5, 2.0)
        ]

    for depth, multiplier in xgb_specs:

        weight = round(
            imbalance * multiplier, 3
        )

        name = (
            f"XGBoost depth={depth} "
            f"pos_weight={weight}"
        )

        models[name] = XGBClassifier(
            n_estimators=160,
            max_depth=depth,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            scale_pos_weight=weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=2,
        )

    return {
        name: Pipeline([
            ("preprocessor", clone(prep)),
            ("model", estimator),
        ])
        for name, estimator in models.items()
    }


def cv_metrics(model, X, y, cv):
    probabilities = cross_val_predict(
        clone(model),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    return evaluate(y, probabilities)


def format_metrics(metrics):
    return (
        f"recall={metrics['recall']:.3f}, "
        f"specificity={metrics['specificity']:.3f}, "
        f"balanced_acc="
        f"{metrics['balanced_accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}"
    )


def backup_file(path, stamp):
    if path.exists():

        backup = (
            path.parent
            / "backups"
            / f"{path.stem}_{stamp}{path.suffix}"
        )

        backup.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(path, backup)

        return str(backup)

    return None


def run_one(disease, apply=False, quick=False):

    paths = paths_for(disease)

    required = [
        paths["model"],
        paths["train"],
        paths["test"],
    ]

    missing = [
        str(p)
        for p in required
        if not Path(p).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing files for {disease}: {missing}. "
            "First run python run_all_diseases.py."
        )

    original = joblib.load(
        paths["model"]
    )

    train = pd.read_csv(
        paths["train"]
    )

    test = pd.read_csv(
        paths["test"]
    )

    if (
        "target" not in train
        or "target" not in test
    ):
        raise ValueError(
            "Saved CSVs must contain target column"
        )

    features = list(
        original.feature_names_in_
    )

    X_train = train[features].copy()
    y_train = train["target"].astype(int)

    X_test = test[features].copy()
    y_test = test["target"].astype(int)

    if (
        y_train.value_counts().min() < 5
        or y_test.nunique() != 2
    ):
        raise ValueError(
            "Insufficient examples of both classes"
        )

    folds = 3 if quick else 5

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=SEED,
    )

    print(
        f"\n=== {DISEASES[disease]['name']} "
        f"| {folds}-fold training CV ===",
        flush=True,
    )

    print(
        "All models evaluated at threshold 0.50.",
        flush=True,
    )

    baseline_cv = cv_metrics(
        original,
        X_train,
        y_train,
        cv,
    )

    print(
        "Current model CV:",
        format_metrics(baseline_cv),
        flush=True,
    )

    rows = [
        {
            "model": "Current saved model",
            **baseline_cv,
        }
    ]

    eligible = []

    candidate_map = candidates(
        X_train,
        y_train,
        quick=quick,
    )

    for number, (name, model) in enumerate(
        candidate_map.items(), 1
    ):

        try:
            result = cv_metrics(
                model,
                X_train,
                y_train,
                cv,
            )

        except Exception as exc:
            print(
                f"[{number}/{len(candidate_map)}] "
                f"{name}: ERROR {exc}",
                flush=True,
            )
            continue

        rows.append({
            "model": name,
            **result,
        })

        meets = (
            result["recall"]
            >= baseline_cv["recall"]
            + MIN_RECALL_GAIN - 1e-12

            and result["specificity"]
            >= MIN_SPECIFICITY[disease] - 1e-12

            and result["balanced_accuracy"]
            >= baseline_cv["balanced_accuracy"]
            - MAX_BALANCED_ACCURACY_DROP - 1e-12
        )

        if meets:
            eligible.append(
                (name, result)
            )

        status = "PASS" if meets else "----"

        print(
            f"[{number}/{len(candidate_map)}] "
            f"{status} {name}: "
            f"{format_metrics(result)}",
            flush=True,
        )

    paths["report"].mkdir(
        parents=True,
        exist_ok=True,
    )

    cv_table = pd.DataFrame(rows)

    cv_table.to_csv(
        paths["report"]
        / "recall_training_cv_comparison.csv",
        index=False,
    )

    baseline_test = evaluate(
        y_test,
        original.predict_proba(X_test)[:, 1],
    )

    summary = {
        "disease": disease,
        "threshold": 0.50,
        "selection": (
            f"{folds}-fold stratified CV "
            "on saved training data"
        ),
        "cv_current": baseline_cv,
        "test_current": baseline_test,
        "minimum_specificity": (
            MIN_SPECIFICITY[disease]
        ),
        "minimum_cv_recall_gain": (
            MIN_RECALL_GAIN
        ),
        "applied": False,
    }

    if not eligible:

        print(
            "No candidate met the CV safeguards. "
            "Original model preserved."
        )

        summary["outcome"] = (
            "No eligible training-CV improvement"
        )

    else:

        best_name, best_cv = max(
            eligible,
            key=lambda item: (
                item[1]["recall"],
                item[1]["balanced_accuracy"],
                item[1]["specificity"],
                item[1]["roc_auc"],
            ),
        )

        print(
            f"\nSelected on TRAINING CV: {best_name}"
        )

        print(
            "CV result:",
            format_metrics(best_cv),
        )

        improved = clone(
            candidate_map[best_name]
        ).fit(
            X_train,
            y_train,
        )

        candidate_path = (
            paths["model"].with_name(
                f"{disease}_recall_candidate.joblib"
            )
        )

        joblib.dump(
            improved,
            candidate_path,
        )

        candidate_test = evaluate(
            y_test,
            improved.predict_proba(X_test)[:, 1],
        )

        print(
            "HELD-OUT TEST current:",
            format_metrics(baseline_test),
        )

        print(
            "HELD-OUT TEST candidate:",
            format_metrics(candidate_test),
        )

        if (
            candidate_test["recall"]
            <= baseline_test["recall"]
        ):
            print(
                "WARNING: Held-out test recall "
                "did not improve."
            )

        summary.update({
            "outcome": (
                "Candidate selected by training CV"
            ),
            "candidate_name": best_name,
            "candidate_path": str(candidate_path),
            "cv_candidate": best_cv,
            "test_candidate": candidate_test,
        })

        if apply:

            stamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            report_path = (
                paths["report"]
                / "test_metrics.json"
            )

            comparison_path = (
                paths["report"]
                / "model_comparison.csv"
            )

            backup_model = backup_file(
                paths["model"], stamp
            )

            backup_report = backup_file(
                report_path, stamp
            )

            backup_comparison = backup_file(
                comparison_path, stamp
            )

            shutil.copy2(
                candidate_path,
                paths["model"],
            )

            if report_path.exists():
                report = json.loads(
                    report_path.read_text()
                )
            else:
                report = {}

            report.update({
                "disease": disease,
                "name": DISEASES[disease]["name"],
                "best_model": best_name,
                "selection_metric": (
                    "training_cv_recall_"
                    "with_specificity_safeguards"
                ),
                "test_metrics": {
                    key: candidate_test[key]
                    for key in (
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "pr_auc",
                        "balanced_accuracy",
                        "specificity",
                    )
                },
                "confusion_matrix": [
                    [
                        candidate_test["tn"],
                        candidate_test["fp"],
                    ],
                    [
                        candidate_test["fn"],
                        candidate_test["tp"],
                    ],
                ],
                "threshold": 0.50,
                "training_cv_metrics": best_cv,
            })

            report_path.write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )

            cv_table.to_csv(
                comparison_path,
                index=False,
            )

            summary.update({
                "applied": True,
                "backup_model": backup_model,
                "backup_report": backup_report,
                "backup_comparison": backup_comparison,
            })

            print(
                "Applied candidate model and "
                "refreshed dashboard report."
            )

            print(
                "Original files backed up."
            )

        else:

            print(
                "Candidate saved but NOT deployed. "
                "Rerun with --apply to deploy."
            )

    summary_path = (
        paths["report"]
        / "recall_improvement_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only",
        choices=[
            "diabetes",
            "heart_disease",
        ],
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Deploy if training CV safeguards pass",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer models and 3-fold CV",
    )

    args = parser.parse_args()

    diseases = (
        [args.only]
        if args.only
        else [
            "diabetes",
            "heart_disease",
        ]
    )

    summaries = []

    for disease in diseases:

        summaries.append(
            run_one(
                disease,
                apply=args.apply,
                quick=args.quick,
            )
        )

    print("\n=== FINISHED ===")

    for item in summaries:
        print(
            f"{item['disease']}: "
            f"{item['outcome']} | "
            f"deployed={item['applied']}"
        )

    print(
        "After deployment, regenerate "
        "SHAP/LIME evaluation files:"
    )

    print(
        "python run_all_diseases.py "
        "--only diabetes --skip-training"
    )

    print(
        "python run_all_diseases.py "
        "--only heart_disease --skip-training"
    )

    print(
        "Then restart: "
        "streamlit run app_multi.py"
    )


if __name__ == "__main__":
    main()
