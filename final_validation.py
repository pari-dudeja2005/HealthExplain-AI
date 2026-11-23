
"""HealthExplain-AI final validation: XAI robustness and model audits.

Run: python final_validation.py
Optional: python final_validation.py --only diabetes
Does not modify trained models, original reports, or Streamlit app.
"""
import argparse
import json
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, roc_auc_score,
    average_precision_score, recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from run_all_diseases import (
    DISEASES, REPORT_DIR, SEED, paths_for, make_explainers,
    explain_record, jaccard, load_disease, split_indices, make_preprocessor,
)
from reliable_xai import make_explainers


def selected_records(y, per_class):
    indices = []
    for cls in (0, 1):
        available = y.index[y == cls].to_numpy()
        if not len(available):
            raise ValueError(f"No test examples for class {cls}")
        rng = np.random.default_rng(SEED + cls)
        indices.extend(rng.choice(available, min(per_class, len(available)), replace=False).tolist())
    return indices


def explain_with_seed(patient, model, explainers, seed, samples):
    # LIME random sampling must be reset before every explanation.
    explainers[1].random_state = np.random.RandomState(seed)
    return explain_record(patient, model, *explainers, lime_samples=samples)


def neighbor_perturb(patient, X_train, rng, fraction):
    """Interpolate continuous features toward one nearby training example.

    Discrete/categorical fields are held fixed. This is bounded to the
    observed feature domain, but does not guarantee clinical validity.
    """
    modified = patient.copy()
    numeric = []
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            if X_train[col].nunique(dropna=True) > 12:
                numeric.append(col)
    if not numeric:
        return modified, 0

    # Choose a similar training example using standardized continuous features.
    data = X_train[numeric].apply(pd.to_numeric, errors="coerce")
    ref = patient[numeric].apply(pd.to_numeric, errors="coerce").iloc[0]
    med = data.median()
    std = data.std().replace(0, 1).fillna(1)
    distance = (((data.fillna(med) - ref.fillna(med)) / std) ** 2).mean(axis=1)
    nearest = distance.nsmallest(min(15, len(distance))).index.to_numpy()
    donor = data.loc[rng.choice(nearest)]

    changed = 0
    for col in numeric:
        original = ref[col]
        replacement = donor[col]
        if not np.isfinite(original) or not np.isfinite(replacement):
            continue
        value = float(original + fraction * (replacement - original))
        minimum, maximum = data[col].min(), data[col].max()
        value = float(np.clip(value, minimum, maximum))
        # Preserve integer-valued clinical fields (e.g. age, counts).
        observed = data[col].dropna().to_numpy(dtype=float)
        if len(observed) and np.all(np.isclose(observed, np.round(observed))):
            value = float(np.round(value))
        if not np.isclose(value, original, atol=1e-12, rtol=1e-10):
            modified[col] = modified[col].astype(float)
            modified.at[modified.index[0], col] = value
            changed += 1
    return modified, changed


def ablation_effects(patient, model, X_train, shap_features, lime_features, seed, n=25):
    """Prediction-change proxy, not causal feature importance.

    Replace the top-k original features with values from the same training
    donor row, comparing against equally sized random feature sets.
    No unstable division by a near-zero random baseline.
    """
    features = list(X_train.columns)
    k = min(5, len(features))
    shap_top = [f for f in shap_features if f in features][:k]
    lime_top = [f for f in lime_features if f in features][:k]
    if len(shap_top) != k or len(lime_top) != k:
        raise ValueError("Explanation features do not match original feature names")

    rng = np.random.default_rng(seed)
    donor_ids = rng.choice(len(X_train), size=n, replace=True)
    donors = X_train.iloc[donor_ids].reset_index(drop=True)
    base = pd.concat([patient] * n, ignore_index=True)
    baseline = float(model.predict_proba(patient)[0, 1])

    def effect(feature_sets):
        frames = []
        for i, chosen in enumerate(feature_sets):
            row = base.iloc[[i]].copy()
            for f in chosen:
                row.at[i, f] = donors.at[i, f]
            frames.append(row)
        changed = pd.concat(frames, ignore_index=True)
        return float(np.mean(np.abs(model.predict_proba(changed)[:, 1] - baseline)))

    shap_effect = effect([shap_top] * n)
    lime_effect = effect([lime_top] * n)
    random_sets = [rng.choice(features, size=k, replace=False).tolist() for _ in range(n)]
    random_effect = effect(random_sets)
    return shap_effect, lime_effect, random_effect


def evaluate_xai(disease, args):
    p = paths_for(disease)
    model = joblib.load(p["model"])
    train = pd.read_csv(p["train"])
    test = pd.read_csv(p["test"])
    X_train = train.drop(columns="target")
    X_test = test.drop(columns="target")
    y_test = test["target"]
    indices = selected_records(y_test, args.per_class)
    print(f"\n{DISEASES[disease]['name']} — XAI robustness ({len(indices)} records)", flush=True)
    explainers = make_explainers(model, X_train)
    rows = []

    for count, idx in enumerate(indices, 1):
        patient = X_test.loc[[idx]].copy()
        seed = SEED + idx * 31
        shap_df, lime_df, r2 = explain_with_seed(patient, model, explainers, seed, args.samples)
        lime_rankings = [lime_df["feature"].tolist()]
        r2_scores = [r2]
        for j in range(1, args.seeds):
            _, other_lime, other_r2 = explain_with_seed(
                patient, model, explainers, seed + j, args.samples
            )
            lime_rankings.append(other_lime["feature"].tolist())
            r2_scores.append(other_r2)
        seed_agreement = np.mean([
            jaccard(a, b, k=5) for a, b in combinations(lime_rankings, 2)
        ])
        row = {
            "record": count,
            "test_index": int(idx),
            "actual_class": int(y_test.loc[idx]),
            "positive_probability": float(model.predict_proba(patient)[0, 1]),
            "top5_agreement": jaccard(shap_df["feature"], lime_df["feature"], k=5),
            "top10_agreement": jaccard(shap_df["feature"], lime_df["feature"], k=10),
            "lime_seed_stability": float(seed_agreement),
            "lime_local_r2": float(np.mean(r2_scores)),
            "shap_top5": "; ".join(shap_df["feature"].head(5)),
            "lime_top5": "; ".join(lime_df["feature"].head(5)),
        }
        for fraction in args.strengths:
            shap_changes, lime_changes, actual_changes = [], [], []
            for repeat in range(args.perturbations):
                rng = np.random.default_rng(seed + 10000 + int(fraction * 1000) * 10 + repeat)
                changed, n_changed = neighbor_perturb(patient, X_train, rng, fraction)
                if n_changed == 0:
                    continue
                new_shap, new_lime, _ = explain_with_seed(
                    changed, model, explainers, seed, args.samples
                )
                shap_changes.append(jaccard(shap_df["feature"], new_shap["feature"], k=5))
                lime_changes.append(jaccard(lime_df["feature"], new_lime["feature"], k=5))
                actual_changes.append(n_changed)
            tag = f"{int(fraction * 100)}pct"
            row[f"shap_stability_{tag}"] = float(np.mean(shap_changes)) if shap_changes else np.nan
            row[f"lime_stability_{tag}"] = float(np.mean(lime_changes)) if lime_changes else np.nan
            row[f"changed_features_{tag}"] = float(np.mean(actual_changes)) if actual_changes else 0.0
        shap_delta, lime_delta, random_delta = ablation_effects(
            patient, model, X_train,
            shap_df["feature"].tolist(), lime_df["feature"].tolist(),
            seed=seed + 20000, n=args.donors,
        )
        row.update({
            "shap_ablation_delta": shap_delta,
            "lime_ablation_delta": lime_delta,
            "random_ablation_delta": random_delta,
            "shap_ablation_excess": shap_delta - random_delta,
            "lime_ablation_excess": lime_delta - random_delta,
        })
        rows.append(row)
        print(f"  {count}/{len(indices)} | agreement={row['top5_agreement']:.3f} | LIME R2={row['lime_local_r2']:.3f}", flush=True)

    out_dir = p["report"]
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "xai_final_v3.csv", index=False)
    numeric = detail.select_dtypes(include=[np.number]).drop(
        columns=["record", "test_index", "actual_class", "positive_probability"]
    )
    summary = {"disease": disease, "name": DISEASES[disease]["name"],
               "records_evaluated": len(detail),
               "negative_records": int((detail["actual_class"] == 0).sum()),
               "positive_records": int((detail["actual_class"] == 1).sum())}
    summary.update({key: float(value) for key, value in numeric.mean().items()})
    return summary


def cv_model_audit(disease, repeats=3):
    """Additional model audit. Fixed model family; not nested model selection."""
    X_all, y_all, groups_all = load_disease(disease)
    # Restrict cross-validation to the original training partition.
    # The held-out test partition must not be reused in CV.
    _, _, train_ids, test_ids = split_indices(y_all, groups_all)
    X = X_all.iloc[train_ids].reset_index(drop=True)
    y = y_all.iloc[train_ids].reset_index(drop=True)
    groups = (groups_all.iloc[train_ids].reset_index(drop=True)
              if groups_all is not None else None)
    report = json.loads((paths_for(disease)["report"] / "test_metrics.json").read_text())
    chosen = report["best_model"]
    saved_estimator = joblib.load(paths_for(disease)["model"]).named_steps["model"]
    indices = np.arange(len(y))
    rows = []
    if groups is None:
        splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
        splits = splitter.split(X, y)
    else:
        splits = (
            (a, b)
            for repeat in range(repeats)
            for a, b in StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=SEED + repeat
            ).split(indices, y, groups)
        )

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        if y.iloc[train_idx].nunique() != 2 or y.iloc[test_idx].nunique() != 2:
            print(f"  Skipping fold {fold}: single-class fold")
            continue
        if groups is not None:
            overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
            if overlap:
                raise RuntimeError("Subject leakage detected in grouped CV")
        preprocessor, _, _ = make_preprocessor(X.iloc[train_idx])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", clone(saved_estimator)),
        ])
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y.iloc[test_idx], pred, labels=[0, 1]).ravel()
        result = {
            "fold": fold, "n_test": len(test_idx),
            "positive_prevalence": float(y.iloc[test_idx].mean()),
            "roc_auc": float(roc_auc_score(y.iloc[test_idx], prob)),
            "pr_auc": float(average_precision_score(y.iloc[test_idx], prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y.iloc[test_idx], pred)),
            "recall": float(recall_score(y.iloc[test_idx], pred)),
            "specificity": float(tn / (tn + fp)),
        }
        if groups is not None:
            # Group-level evaluation avoids treating repeated recordings as independent patients.
            grouped = pd.DataFrame({
                "subject": groups.iloc[test_idx].to_numpy(),
                "label": y.iloc[test_idx].to_numpy(),
                "probability": prob,
            })
            if grouped.groupby("subject")["label"].nunique().max() != 1:
                raise ValueError("Inconsistent subject labels")
            subject = grouped.groupby("subject").agg(
                label=("label", "first"), probability=("probability", "mean")
            )
            if subject["label"].nunique() == 2:
                subject_pred = (subject["probability"] >= 0.5).astype(int)
                result["subject_count"] = len(subject)
                result["subject_roc_auc"] = float(roc_auc_score(subject["label"], subject["probability"]))
                result["subject_balanced_accuracy"] = float(balanced_accuracy_score(subject["label"], subject_pred))
        rows.append(result)
        print(f"  Fold {fold}: ROC-AUC={result['roc_auc']:.3f}, balanced accuracy={result['balanced_accuracy']:.3f}", flush=True)

    if not rows:
        raise ValueError("No valid CV folds")
    detail = pd.DataFrame(rows)
    detail.to_csv(paths_for(disease)["report"] / "cross_validation_audit_v3.csv", index=False)
    print("  CV means:", detail.select_dtypes(include=[np.number]).drop(columns=["fold", "n_test"]).mean().round(3).to_dict())

    if disease == "kidney_disease":
        # Training-only missingness audit; no test-set information is used.
        # Class-dependent missingness alone does not prove leakage.
        records = []
        for col in X.columns:
            missing = X[col].isna()
            records.append({
                "feature": col,
                "missing_rate_negative": float(missing[y == 0].mean()),
                "missing_rate_positive": float(missing[y == 1].mean()),
                "absolute_difference": float(abs(missing[y == 0].mean() - missing[y == 1].mean())),
            })
        pd.DataFrame(records).sort_values("absolute_difference", ascending=False).to_csv(
            paths_for(disease)["report"] / "missingness_audit_v3.csv", index=False
        )
    return {"disease": disease, "valid_folds": len(detail),
            "mean_roc_auc": float(detail["roc_auc"].mean()),
            "mean_balanced_accuracy": float(detail["balanced_accuracy"].mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(DISEASES))
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--perturbations", type=int, default=2)
    parser.add_argument("--samples", type=int, default=700)
    parser.add_argument("--donors", type=int, default=25)
    args = parser.parse_args()
    args.strengths = (0.05, 0.15)
    if min(args.per_class, args.seeds - 1, args.perturbations, args.samples, args.donors) < 1:
        parser.error("All evaluation settings must be positive, and seeds must be >= 2")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    diseases = [args.only] if args.only else list(DISEASES)
    (REPORT_DIR / "final_validation_settings_v3.json").write_text(
        json.dumps({
            "diseases": diseases,
            "per_class": args.per_class,
            "lime_seeds": args.seeds,
            "perturbations_per_strength": args.perturbations,
            "lime_samples": args.samples,
            "background_donors": args.donors,
            "neighbor_interpolation_fractions": args.strengths,
            "cv_repeats": 3,
            "cv_partition": "original training set only; held-out test excluded",
            "note": "Exploratory evaluation; no clinical validity claims",
        }, indent=2), encoding="utf-8"
    )
    summaries, failures = [], []
    for disease in diseases:
        try:
            summaries.append(evaluate_xai(disease, args))
        except Exception as exc:
            print(f"ERROR in XAI {disease}: {type(exc).__name__}: {exc}", flush=True)
            failures.append(disease)
    if summaries:
        output = REPORT_DIR / "xai_final_summary_v3.csv"
        result = pd.DataFrame(summaries)
        if args.only and output.exists():
            previous = pd.read_csv(output)
            result = pd.concat([previous[previous["disease"] != args.only], result], ignore_index=True)
        result.to_csv(output, index=False)
        print("\nFINAL XAI SUMMARY")
        cols = ["name", "records_evaluated", "top5_agreement", "shap_stability_5pct",
                "shap_stability_15pct", "lime_stability_15pct", "lime_seed_stability",
                "shap_ablation_delta", "lime_ablation_delta", "random_ablation_delta", "lime_local_r2"]
        print(result[cols].round(3).to_string(index=False))
        print(f"Saved: {output}")

    audit_results = []
    for disease in diseases:
        print(f"\n{DISEASES[disease]['name']} — cross-validation audit", flush=True)
        try:
            audit_results.append(cv_model_audit(disease))
        except Exception as exc:
            print(f"ERROR in CV audit {disease}: {type(exc).__name__}: {exc}", flush=True)
            failures.append(f"{disease} CV")
    if audit_results:
        pd.DataFrame(audit_results).to_csv(REPORT_DIR / "model_audit_summary_v3.csv", index=False)
    if failures:
        raise SystemExit("Incomplete: " + ", ".join(failures))
    print("\nAll requested evaluations completed. Existing models were not changed.")


if __name__ == "__main__":
    main()

