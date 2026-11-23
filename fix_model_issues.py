
"""
HealthExplain-AI: Model Quality Audit and Improvement

Checks:
1. CKD duplicate records and target conflicts
2. CKD missingness-related prediction signals
3. CKD performance after deduplication
4. Parkinson's subject-level model comparison
5. Saves an improved Parkinson's candidate if justified

Existing trained models are NOT overwritten unless --apply is used.

Run:
    python fix_model_issues.py

Optional:
    python fix_model_issues.py --only kidney_disease
    python fix_model_issues.py --only parkinsons
"""

import argparse
import json
import shutil

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
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from run_all_diseases import (
    SEED,
    load_disease,
    make_preprocessor,
    paths_for,
    split_indices,
)


# ============================================================
# METRIC CALCULATION
# ============================================================

def scores(y_true, y_pred, y_prob):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),
        "pr_auc": float(
            average_precision_score(y_true, y_prob)
        ),
    }


# ============================================================
# UTILITIES
# ============================================================

def save_json(path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False
        ),
        encoding="utf-8"
    )


def fingerprints(X):

    return pd.util.hash_pandas_object(
        X.reset_index(drop=True),
        index=False
    )


def cv_pipeline(
    X,
    y,
    estimator,
    n_splits=5,
    repeats=3
):

    rows = []

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=repeats,
        random_state=SEED
    )

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        preprocessor, _, _ = make_preprocessor(
            X.iloc[train_idx]
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", clone(estimator))
        ])

        pipeline.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        probabilities = pipeline.predict_proba(
            X.iloc[val_idx]
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        rows.append({
            "fold": fold,
            "roc_auc": float(
                roc_auc_score(
                    y.iloc[val_idx],
                    probabilities
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y.iloc[val_idx],
                    predictions
                )
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# CKD DATA QUALITY AUDIT
# ============================================================

def audit_ckd(repeats=5):

    disease = "kidney_disease"

    paths = paths_for(disease)

    output_dir = paths["report"]

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y, _ = load_disease(disease)

    _, _, train_ids, test_ids = split_indices(y)

    X_train = X.iloc[train_ids].reset_index(
        drop=True
    )

    y_train = y.iloc[train_ids].reset_index(
        drop=True
    )

    X_test = X.iloc[test_ids].reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # DUPLICATE AUDIT
    # --------------------------------------------------------

    train_hash = fingerprints(X_train)

    test_hash = fingerprints(X_test)

    full_hash = fingerprints(X)

    label_table = pd.DataFrame({
        "fingerprint": full_hash,
        "target": y
    })

    conflicts = int(
        (
            label_table.groupby(
                "fingerprint"
            )["target"].nunique() > 1
        ).sum()
    )

    shared = (
        set(train_hash.to_numpy())
        &
        set(test_hash.to_numpy())
    )

    duplicate_info = {
        "total_records": len(X),
        "unique_feature_rows": int(
            full_hash.nunique()
        ),
        "duplicate_feature_rows": int(
            len(X) - full_hash.nunique()
        ),
        "distinct_feature_patterns_shared_between_train_and_test": len(
            shared
        ),
        "test_rows_matching_training_features": int(
            test_hash.isin(shared).sum()
        ),
        "feature_identical_rows_with_conflicting_labels": conflicts,
    }

    save_json(
        output_dir / "ckd_duplicate_audit_v4.json",
        duplicate_info
    )

    # --------------------------------------------------------
    # MISSINGNESS AUDIT
    # --------------------------------------------------------

    missing_rows = []

    for feature in X_train.columns:

        missing = X_train[feature].isna()

        negative_rate = float(
            missing[y_train == 0].mean()
        )

        positive_rate = float(
            missing[y_train == 1].mean()
        )

        missing_rows.append({
            "feature": feature,
            "missing_rate_no_ckd": negative_rate,
            "missing_rate_ckd": positive_rate,
            "absolute_difference": abs(
                positive_rate - negative_rate
            )
        })

    missing_df = pd.DataFrame(
        missing_rows
    ).sort_values(
        "absolute_difference",
        ascending=False
    )

    missing_df.to_csv(
        output_dir / "ckd_missingness_audit_v4.csv",
        index=False
    )

    # --------------------------------------------------------
    # MISSINGNESS-ONLY MODEL
    # --------------------------------------------------------

    missing_only = X_train.isna().astype(int)

    missing_only = missing_only.loc[
        :,
        missing_only.nunique() > 1
    ]

    missing_cv = []

    if missing_only.shape[1] > 0:

        cv = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=repeats,
            random_state=SEED
        )

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(missing_only, y_train),
            start=1
        ):

            estimator = LogisticRegression(
                max_iter=2000,
                random_state=SEED
            )

            estimator.fit(
                missing_only.iloc[train_idx],
                y_train.iloc[train_idx]
            )

            probabilities = estimator.predict_proba(
                missing_only.iloc[val_idx]
            )[:, 1]

            missing_cv.append({
                "fold": fold,
                "roc_auc": float(
                    roc_auc_score(
                        y_train.iloc[val_idx],
                        probabilities
                    )
                )
            })

        pd.DataFrame(
            missing_cv
        ).to_csv(
            output_dir / "ckd_missingness_only_cv_v4.csv",
            index=False
        )

    # --------------------------------------------------------
    # DEDUPLICATED CROSS-VALIDATION
    # --------------------------------------------------------

    dedup_summary = None

    if conflicts == 0:

        unique_ids = ~train_hash.duplicated(
            keep="first"
        )

        X_unique = X_train.loc[
            unique_ids
        ].reset_index(drop=True)

        y_unique = y_train.loc[
            unique_ids
        ].reset_index(drop=True)

        if (
            len(X_unique) >= 50
            and y_unique.value_counts().min() >= 5
        ):

            saved_model = joblib.load(
                paths["model"]
            )

            estimator = saved_model.named_steps[
                "model"
            ]

            cv_results = cv_pipeline(
                X_unique,
                y_unique,
                estimator,
                repeats=repeats
            )

            cv_results.to_csv(
                output_dir / "ckd_deduplicated_cv_v4.csv",
                index=False
            )

            dedup_summary = {
                "unique_training_rows": len(
                    X_unique
                ),
                "mean_roc_auc": float(
                    cv_results["roc_auc"].mean()
                ),
                "mean_balanced_accuracy": float(
                    cv_results[
                        "balanced_accuracy"
                    ].mean()
                )
            }

    # --------------------------------------------------------
    # CKD SUMMARY
    # --------------------------------------------------------

    result = {
        "duplicate_audit": duplicate_info,
        "missingness_only_mean_cv_roc_auc": (
            float(
                np.mean([
                    row["roc_auc"]
                    for row in missing_cv
                ])
            )
            if missing_cv
            else None
        ),
        "deduplicated_cv": dedup_summary,
        "interpretation": (
            "Duplicate overlap, conflicting labels, "
            "and strong missingness signals require "
            "further dataset investigation. "
            "Perfect ROC-AUC alone does not prove leakage."
        )
    }

    save_json(
        output_dir / "ckd_quality_summary_v4.json",
        result
    )

    print("\n" + "=" * 60)
    print("CKD AUDIT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    if shared or conflicts:

        print(
            "\nWARNING: Duplicate overlap or label "
            "conflicts were detected. Review the "
            "dataset before trusting the test score."
        )

    return result


# ============================================================
# PARKINSON'S CANDIDATE MODELS
# ============================================================

def parkinsons_candidates(current):

    return {

        "Current saved estimator": (
            "existing",
            clone(current)
        ),

        "Logistic Regression C=0.1 balanced": (
            "Logistic Regression",
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=4000,
                random_state=SEED
            )
        ),

        "Logistic Regression C=1 balanced": (
            "Logistic Regression",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=4000,
                random_state=SEED
            )
        ),

        "Random Forest leaf=5 balanced": (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED
            )
        ),

        "XGBoost shallow regularised": (
            "XGBoost",
            XGBClassifier(
                n_estimators=100,
                max_depth=2,
                learning_rate=0.04,
                min_child_weight=4,
                reg_lambda=10,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=2
            )
        )
    }


# ============================================================
# SUBJECT-LEVEL METRICS
# ============================================================

def subject_metrics(y, probabilities, groups):

    df = pd.DataFrame({
        "subject": np.asarray(groups),
        "target": np.asarray(y, dtype=int),
        "probability": np.asarray(
            probabilities,
            dtype=float
        )
    })

    if (
        df.groupby("subject")["target"]
        .nunique()
        .max() > 1
    ):

        raise ValueError(
            "Inconsistent Parkinson's labels "
            "for the same subject."
        )

    subject_df = df.groupby(
        "subject"
    ).agg(
        target=("target", "first"),
        probability=("probability", "mean")
    )

    if subject_df["target"].nunique() < 2:

        raise ValueError(
            "Subject-level evaluation requires "
            "both target classes."
        )

    predictions = (
        subject_df["probability"] >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        subject_df["target"],
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "subjects": int(len(subject_df)),
        "subject_roc_auc": float(
            roc_auc_score(
                subject_df["target"],
                subject_df["probability"]
            )
        ),
        "subject_balanced_accuracy": float(
            balanced_accuracy_score(
                subject_df["target"],
                predictions
            )
        ),
        "subject_recall": (
            float(tp / (tp + fn))
            if tp + fn
            else None
        ),
        "subject_specificity": (
            float(tn / (tn + fp))
            if tn + fp
            else None
        )
    }


# ============================================================
# PARKINSON'S CROSS-VALIDATION
# ============================================================

def evaluate_parkinsons_candidates(
    X,
    y,
    groups,
    candidates,
    repeats
):

    rows = []

    for name, (_, estimator) in candidates.items():

        print(
            f"\nTesting: {name}",
            flush=True
        )

        for repeat in range(repeats):

            cv = StratifiedGroupKFold(
                n_splits=3,
                shuffle=True,
                random_state=SEED + repeat
            )

            oof = np.full(
                len(X),
                np.nan,
                dtype=float
            )

            for train_idx, val_idx in cv.split(
                X, y, groups
            ):

                if y.iloc[train_idx].nunique() < 2:

                    raise RuntimeError(
                        "A training fold has only one class."
                    )

                train_groups = set(
                    groups.iloc[train_idx]
                )

                val_groups = set(
                    groups.iloc[val_idx]
                )

                if train_groups & val_groups:

                    raise RuntimeError(
                        "Subject leakage detected."
                    )

                preprocessor, _, _ = make_preprocessor(
                    X.iloc[train_idx]
                )

                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", clone(estimator))
                ])

                pipeline.fit(
                    X.iloc[train_idx],
                    y.iloc[train_idx]
                )

                oof[val_idx] = pipeline.predict_proba(
                    X.iloc[val_idx]
                )[:, 1]

            if not np.isfinite(oof).all():

                raise RuntimeError(
                    "Incomplete out-of-fold predictions."
                )

            result = subject_metrics(
                y,
                oof,
                groups
            )

            result.update({
                "candidate": name,
                "repeat": repeat + 1,
                "record_roc_auc": float(
                    roc_auc_score(y, oof)
                )
            })

            rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# PARKINSON'S MODEL IMPROVEMENT
# ============================================================

def fix_parkinsons(repeats=5, apply=False):

    disease = "parkinsons"

    paths = paths_for(disease)

    paths["report"].mkdir(
        parents=True,
        exist_ok=True
    )

    X_all, y_all, groups_all = load_disease(
        disease
    )

    if groups_all is None:

        raise ValueError(
            "Subject IDs are required "
            "for Parkinson's evaluation."
        )

    _, _, train_ids, test_ids = split_indices(
        y_all,
        groups_all
    )

    X = X_all.iloc[train_ids].reset_index(
        drop=True
    )

    y = y_all.iloc[train_ids].reset_index(
        drop=True
    )

    groups = groups_all.iloc[
        train_ids
    ].reset_index(drop=True)

    current_model = joblib.load(
        paths["model"]
    )

    current_estimator = current_model.named_steps[
        "model"
    ]

    candidates = parkinsons_candidates(
        current_estimator
    )

    details = evaluate_parkinsons_candidates(
        X,
        y,
        groups,
        candidates,
        repeats
    )

    details.to_csv(
        paths["report"] / "parkinsons_subject_cv_v4.csv",
        index=False
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    summary = details.groupby(
        "candidate",
        as_index=False
    ).agg(
        mean_subject_auc=(
            "subject_roc_auc",
            "mean"
        ),
        std_subject_auc=(
            "subject_roc_auc",
            "std"
        ),
        mean_subject_balanced_accuracy=(
            "subject_balanced_accuracy",
            "mean"
        ),
        mean_subject_recall=(
            "subject_recall",
            "mean"
        ),
        mean_subject_specificity=(
            "subject_specificity",
            "mean"
        ),
        mean_record_auc=(
            "record_roc_auc",
            "mean"
        )
    )

    summary["std_subject_auc"] = (
        summary["std_subject_auc"].fillna(0.0)
    )

    summary["selection_score"] = (
        0.75 * summary["mean_subject_auc"]
        + 0.25 * summary[
            "mean_subject_balanced_accuracy"
        ]
        - 0.10 * summary["std_subject_auc"]
    )

    summary = summary.sort_values(
        "selection_score",
        ascending=False
    )

    summary.to_csv(
        paths["report"] / "parkinsons_model_comparison_v4.csv",
        index=False
    )

    baseline = summary.loc[
        summary["candidate"] == "Current saved estimator"
    ].iloc[0]

    best = summary.iloc[0]

    # --------------------------------------------------------
    # IMPROVEMENT CRITERIA
    # --------------------------------------------------------

    better_discrimination = (
        best["mean_subject_auc"]
        >= baseline["mean_subject_auc"] + 0.02
        and
        best["mean_subject_balanced_accuracy"]
        >= baseline[
            "mean_subject_balanced_accuracy"
        ] - 0.03
    )

    better_balance = (
        best["mean_subject_balanced_accuracy"]
        >= baseline[
            "mean_subject_balanced_accuracy"
        ] + 0.05
        and
        best["mean_subject_auc"]
        >= baseline["mean_subject_auc"] - 0.03
    )

    improved = bool(
        best["candidate"] != "Current saved estimator"
        and
        best["selection_score"]
        >= baseline["selection_score"] + 0.015
        and
        (
            better_discrimination
            or better_balance
        )
    )

    print("\n" + "=" * 60)
    print("PARKINSON'S SUBJECT-LEVEL COMPARISON")
    print("=" * 60)

    print(
        summary.round(3).to_string(
            index=False
        )
    )

    result = {
        "training_subjects": int(
            groups.nunique()
        ),
        "cv_repeats": repeats,
        "selected_candidate": str(
            best["candidate"]
        ),
        "meets_improvement_rule": improved,
        "baseline_mean_subject_auc": float(
            baseline["mean_subject_auc"]
        ),
        "candidate_mean_subject_auc": float(
            best["mean_subject_auc"]
        ),
        "note": (
            "Exploratory model selection using "
            "the original training partition only. "
            "Independent patient-level validation "
            "is still needed."
        )
    }

    # --------------------------------------------------------
    # SAVE CANDIDATE
    # --------------------------------------------------------

    if improved:

        family, estimator = candidates[
            str(best["candidate"])
        ]

        preprocessor, _, _ = make_preprocessor(X)

        new_model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", clone(estimator))
        ])

        new_model.fit(X, y)

        candidate_path = paths["model"].with_name(
            "parkinsons_candidate_model.joblib"
        )

        joblib.dump(
            new_model,
            candidate_path
        )

        result["candidate_model"] = str(
            candidate_path
        )

        print(
            f"\nCandidate saved: {candidate_path}"
        )

        if apply:

            # Back up the active model and report.

            backup_model = paths["model"].with_name(
                "parkinsons_model_before_v4.joblib"
            )

            shutil.copy2(
                paths["model"],
                backup_model
            )

            report_path = (
                paths["report"] / "test_metrics.json"
            )

            backup_report = (
                paths["report"]
                / "test_metrics_before_v4.json"
            )

            shutil.copy2(
                report_path,
                backup_report
            )

            # Promote candidate.

            joblib.dump(
                new_model,
                paths["model"]
            )

            # Recalculate test metrics.

            X_test = X_all.iloc[test_ids]

            y_test = y_all.iloc[test_ids]

            probabilities = new_model.predict_proba(
                X_test
            )[:, 1]

            predictions = new_model.predict(
                X_test
            ).astype(int)

            report = json.loads(
                report_path.read_text(
                    encoding="utf-8"
                )
            )

            report["best_model"] = family

            report["model_variant"] = str(
                best["candidate"]
            )

            report["selection_metric"] = (
                "repeated_subject_group_cv_training_only"
            )

            report["test_metrics"] = scores(
                y_test,
                predictions,
                probabilities
            )

            report["confusion_matrix"] = (
                confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1]
                ).tolist()
            )

            report["validation_note"] = (
                "Model revised after earlier test "
                "results were reviewed. These test "
                "metrics are exploratory."
            )

            save_json(
                report_path,
                report
            )

            result["applied"] = True

            print(
                "Candidate applied. Original model "
                "and report backed up."
            )

        else:

            result["applied"] = False

            print(
                "Active model unchanged. "
                "Review results before applying."
            )

    else:

        result["applied"] = False

        print(
            "No candidate met the improvement rule. "
            "Existing model remains unchanged."
        )

    save_json(
        paths["report"]
        / "parkinsons_quality_summary_v4.json",
        result
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "HealthExplain-AI CKD and "
            "Parkinson's model audit"
        )
    )

    parser.add_argument(
        "--only",
        choices=[
            "kidney_disease",
            "parkinsons"
        ]
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply the Parkinson's candidate "
            "only if the improvement criteria pass"
        )
    )

    args = parser.parse_args()

    if args.repeats < 2:

        parser.error(
            "--repeats must be at least 2"
        )

    if args.apply and args.only != "parkinsons":

        parser.error(
            "--apply requires --only parkinsons"
        )

    if args.only in (
        None,
        "kidney_disease"
    ):

        audit_ckd(
            repeats=args.repeats
        )

    if args.only in (
        None,
        "parkinsons"
    ):

        fix_parkinsons(
            repeats=args.repeats,
            apply=args.apply
        )

    print(
        "\nAll requested model audits completed."
    )

    print(
        "Review the v4 reports before "
        "changing any trained model."
    )


if __name__ == "__main__":
    main()

