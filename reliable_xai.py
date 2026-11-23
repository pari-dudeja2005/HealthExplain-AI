
"""Training-only LIME kernel tuning and configured SHAP/LIME explainers.

Usage:
    python reliable_xai.py
    python reliable_xai.py --only parkinsons

To use the selected configuration in app_multi.py and final_validation.py,
add AFTER the existing imports:
    from reliable_xai import make_explainers

This overrides the previous make_explainers name. It does not retrain models.
Tuning maximises local surrogate fit/stability on training records; it does
not prove explanation correctness or clinical validity.
"""
import argparse
import json
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from run_all_diseases import (
    DISEASES, REPORT_DIR, SEED, jaccard, make_explainers as original_make_explainers,
    paths_for,
)

CONFIG_PATH = REPORT_DIR / "lime_kernel_config_v4.json"
KERNEL_SCALES = (0.5, 0.75, 1.0, 1.25)


def load_settings():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def lime_explainer(model, X_train, scale):
    preprocessor = model.named_steps["preprocessor"]
    transformed = np.asarray(preprocessor.transform(X_train), dtype=float)
    width = float(scale * np.sqrt(transformed.shape[1]))
    return LimeTabularExplainer(
        training_data=transformed,
        feature_names=list(preprocessor.get_feature_names_out()),
        class_names=["Negative", "Positive"],
        mode="classification",
        discretize_continuous=False,
        kernel_width=width,
        random_state=SEED,
    )


def make_explainers(model, X_train):
    """Same return signature as run_all_diseases.make_explainers."""
    shap_exp, default_lime, feature_map, fallback = original_make_explainers(
        model, X_train
    )
    disease = next(
        (key for key in DISEASES
         if paths_for(key)["model"].exists()
         and paths_for(key)["model"].stem == getattr(model, "_health_model_name", "")),
        None,
    )
    # Pipelines loaded with joblib normally have no disease name. Infer from
    # the exact expected original feature schema, without modifying the model.
    if disease is None:
        features = tuple(model.feature_names_in_)
        for key in DISEASES:
            train_path = paths_for(key)["train"]
            if not train_path.exists():
                continue
            headers = tuple(pd.read_csv(train_path, nrows=0).drop(columns="target").columns)
            if features == headers:
                disease = key
                break
    settings = load_settings().get(disease, {}) if disease else {}
    scale = float(settings.get("kernel_scale", 0.75))
    if np.isclose(scale, 0.75):
        return shap_exp, default_lime, feature_map, fallback
    return shap_exp, lime_explainer(model, X_train, scale), feature_map, fallback


def evaluate_scale(model, X_train, sample_indices, feature_map, scale,
                   seeds=3, samples=1200):
    exp = lime_explainer(model, X_train, scale)
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    records = []
    for index in sample_indices:
        row = X_train.iloc[[index]]
        transformed = np.asarray(preprocessor.transform(row), dtype=float)[0]
        rankings, r2s = [], []
        for seed in range(seeds):
            exp.random_state = np.random.RandomState(SEED + int(index) * 19 + seed)
            explanation = exp.explain_instance(
                transformed, estimator.predict_proba, labels=(1,),
                num_features=len(feature_map), num_samples=samples,
            )
            weights = dict(explanation.as_map()[1])
            grouped = pd.Series(
                [float(weights.get(i, 0.0)) for i in range(len(feature_map))]
            ).groupby(feature_map, sort=False).sum()
            ranking = grouped.abs().sort_values(ascending=False).index.tolist()
            rankings.append(ranking)
            r2s.append(float(explanation.score))
        stability = float(np.mean([
            jaccard(a, b, k=5) for a, b in combinations(rankings, 2)
        ]))
        records.append({"mean_r2": float(np.mean(r2s)), "seed_stability": stability})
    result = pd.DataFrame(records).mean().to_dict()
    return {"kernel_scale": float(scale), **{k: float(v) for k, v in result.items()}}


def tune(disease, per_class=3, seeds=3, samples=1200):
    p = paths_for(disease)
    model = joblib.load(p["model"])
    train = pd.read_csv(p["train"])
    X = train.drop(columns="target")
    y = train["target"]
    sample_indices = []
    for label in (0, 1):
        ids = np.flatnonzero(y.to_numpy() == label)
        if not len(ids):
            raise ValueError(f"Missing class {label} for {disease}")
        rng = np.random.default_rng(SEED + label)
        sample_indices.extend(rng.choice(ids, min(per_class, len(ids)), replace=False).tolist())
    preprocessor = model.named_steps["preprocessor"]
    from run_all_diseases import transformed_feature_map
    feature_map = transformed_feature_map(preprocessor)
    results = []
    for scale in KERNEL_SCALES:
        print(f"  {DISEASES[disease]['name']}: kernel scale={scale}", flush=True)
        results.append(evaluate_scale(
            model, X, sample_indices, feature_map, scale, seeds, samples
        ))
    detail = pd.DataFrame(results)
    detail["selection_score"] = detail["mean_r2"] + 0.10 * detail["seed_stability"]
    baseline = detail.loc[np.isclose(detail["kernel_scale"], 0.75)].iloc[0]
    ranked = detail.sort_values("selection_score", ascending=False)
    best = ranked.iloc[0]
    accept = (
        best["selection_score"] >= baseline["selection_score"] + 0.02
        and best["mean_r2"] >= baseline["mean_r2"] + 0.01
        and best["seed_stability"] >= baseline["seed_stability"] - 0.05
    )
    chosen = best if accept else baseline
    p["report"].mkdir(parents=True, exist_ok=True)
    detail.to_csv(p["report"] / "lime_kernel_audit_v4.csv", index=False)
    settings = load_settings()
    settings[disease] = {
        "kernel_scale": float(chosen["kernel_scale"]),
        "training_mean_local_r2": float(chosen["mean_r2"]),
        "training_seed_stability": float(chosen["seed_stability"]),
        "training_records": len(sample_indices),
        "note": "Tuned on training examples only; external fidelity not established",
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(detail.round(3).to_string(index=False))
    print(f"  Selected kernel scale: {chosen['kernel_scale']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=list(DISEASES))
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=1200)
    args = parser.parse_args()
    if min(args.per_class, args.seeds - 1, args.samples) < 1:
        parser.error("Invalid sample counts or seeds < 2")
    diseases = [args.only] if args.only else [
        "breast_cancer", "kidney_disease", "parkinsons"
    ]
    for disease in diseases:
        tune(disease, args.per_class, args.seeds, args.samples)
    print(f"\nSaved LIME settings: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
