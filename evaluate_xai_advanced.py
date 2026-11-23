
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run_all_diseases import (
    DISEASES,
    REPORT_DIR,
    SEED,
    paths_for,
    make_explainers,
    explain_record,
    jaccard,
    perturb,
    fidelity_ratio,
)


def lime_explanation_with_seed(
    patient,
    model,
    lime_explainer,
    feature_map,
    seed,
    lime_samples,
):
    """Generate a LIME explanation with a controlled seed."""

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    transformed = np.asarray(
        preprocessor.transform(patient),
        dtype=float,
    )

    lime_explainer.random_state = np.random.RandomState(seed)

    explanation = lime_explainer.explain_instance(
        transformed[0],
        estimator.predict_proba,
        labels=(1,),
        num_features=transformed.shape[1],
        num_samples=lime_samples,
    )

    weights = dict(explanation.as_map()[1])

    values = [
        float(weights.get(i, 0.0))
        for i in range(len(feature_map))
    ]

    grouped = pd.Series(values).groupby(
        feature_map,
        sort=False,
    ).sum()

    result = pd.DataFrame({
        "feature": grouped.index,
        "lime_weight": grouped.values,
    })

    result["absolute_importance"] = (
        result["lime_weight"].abs()
    )

    result = result.sort_values(
        "absolute_importance",
        ascending=False,
    ).reset_index(drop=True)

    return result, float(explanation.score)


def select_test_records(y_test, per_class):
    """Select a balanced evaluation sample."""

    selected = []

    for label in [0, 1]:

        candidates = y_test.index[
            y_test == label
        ].to_numpy()

        if len(candidates) == 0:
            raise ValueError(
                f"No test records for class {label}"
            )

        rng = np.random.default_rng(
            SEED + label
        )

        chosen = rng.choice(
            candidates,
            size=min(per_class, len(candidates)),
            replace=False,
        )

        selected.extend(chosen.tolist())

    return selected


def evaluate_one_disease(disease, args):

    paths = paths_for(disease)

    model = joblib.load(paths["model"])

    train_df = pd.read_csv(paths["train"])
    test_df = pd.read_csv(paths["test"])

    X_train = train_df.drop(columns="target")
    X_test = test_df.drop(columns="target")
    y_test = test_df["target"]

    selected = select_test_records(
        y_test,
        args.per_class,
    )

    print("\n" + "=" * 60)
    print(DISEASES[disease]["name"])
    print("=" * 60)

    print(f"Evaluation records: {len(selected)}")
    print(f"LIME random seeds: {args.seeds}")
    print(f"Input perturbations: {args.perturbations}")
    print("Preparing SHAP and LIME explainers...", flush=True)

    explainers = make_explainers(
        model,
        X_train,
    )

    (
        shap_explainer,
        lime_explainer,
        feature_map,
        shap_fallback,
    ) = explainers

    rows = []

    for number, index in enumerate(selected, start=1):

        patient = X_test.loc[[index]].copy()

        base_seed = SEED + int(index) * 17

        # Original explanations
        lime_explainer.random_state = (
            np.random.RandomState(base_seed)
        )

        shap_df, lime_df, original_r2 = explain_record(
            patient,
            model,
            *explainers,
            lime_samples=args.lime_samples,
        )

        # SHAP-LIME feature agreement
        consistency_top5 = jaccard(
            shap_df["feature"],
            lime_df["feature"],
            k=5,
        )

        consistency_top10 = jaccard(
            shap_df["feature"],
            lime_df["feature"],
            k=10,
        )

        # LIME random-seed stability
        seed_stabilities = []
        lime_r2_scores = [original_r2]

        for seed_offset in range(1, args.seeds):

            alternative_lime, alternative_r2 = (
                lime_explanation_with_seed(
                    patient,
                    model,
                    lime_explainer,
                    feature_map,
                    base_seed + seed_offset,
                    args.lime_samples,
                )
            )

            seed_stabilities.append(
                jaccard(
                    lime_df["feature"],
                    alternative_lime["feature"],
                    k=5,
                )
            )

            lime_r2_scores.append(
                alternative_r2
            )

        # Input perturbation stability
        shap_input_stabilities = []
        lime_input_stabilities = []

        for repeat in range(args.perturbations):

            rng = np.random.default_rng(
                base_seed + 10000 + repeat
            )

            changed_patient = perturb(
                patient,
                X_train,
                rng,
            )

            # Reset LIME to the original seed.
            # This isolates the effect of input changes.
            lime_explainer.random_state = (
                np.random.RandomState(base_seed)
            )

            new_shap, new_lime, _ = explain_record(
                changed_patient,
                model,
                *explainers,
                lime_samples=args.lime_samples,
            )

            shap_input_stabilities.append(
                jaccard(
                    shap_df["feature"],
                    new_shap["feature"],
                    k=5,
                )
            )

            lime_input_stabilities.append(
                jaccard(
                    lime_df["feature"],
                    new_lime["feature"],
                    k=5,
                )
            )

        # Fidelity: use matching random comparisons
        # for SHAP and LIME.
        fidelity_seed = base_seed + 20000

        shap_fidelity = fidelity_ratio(
            patient,
            shap_df["feature"],
            model,
            X_train,
            np.random.default_rng(fidelity_seed),
            args.random_comparisons,
        )

        lime_fidelity = fidelity_ratio(
            patient,
            lime_df["feature"],
            model,
            X_train,
            np.random.default_rng(fidelity_seed),
            args.random_comparisons,
        )

        probability = float(
            model.predict_proba(patient)[0, 1]
        )

        rows.append({
            "record": number,
            "test_index": int(index),
            "actual_class": int(y_test.loc[index]),
            "positive_probability": probability,
            "top5_consistency": consistency_top5,
            "top10_consistency": consistency_top10,
            "shap_input_stability": float(
                np.mean(shap_input_stabilities)
            ),
            "lime_input_stability": float(
                np.mean(lime_input_stabilities)
            ),
            "lime_seed_stability": float(
                np.mean(seed_stabilities)
            ),
            "shap_fidelity_ratio": shap_fidelity,
            "lime_fidelity_ratio": lime_fidelity,
            "lime_local_r2": float(
                np.mean(lime_r2_scores)
            ),
        })

        print(
            f"  Record {number}/{len(selected)} completed "
            f"| Agreement: {consistency_top5:.3f} "
            f"| LIME R2: {np.mean(lime_r2_scores):.3f}",
            flush=True,
        )

    result_df = pd.DataFrame(rows)

    report_dir = paths["report"]
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        report_dir / "xai_robustness_v2.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    metrics = [
        "top5_consistency",
        "top10_consistency",
        "shap_input_stability",
        "lime_input_stability",
        "lime_seed_stability",
        "shap_fidelity_ratio",
        "lime_fidelity_ratio",
        "lime_local_r2",
    ]

    summary = {
        "disease": disease,
        "name": DISEASES[disease]["name"],
        "records_evaluated": len(result_df),
        "negative_records": int(
            (result_df["actual_class"] == 0).sum()
        ),
        "positive_records": int(
            (result_df["actual_class"] == 1).sum()
        ),
    }

    for metric in metrics:
        summary[metric] = float(
            result_df[metric].mean()
        )

    print("\nRESULTS")
    print(
        result_df[metrics]
        .mean()
        .round(4)
        .to_string()
    )

    print(f"\nSaved: {output_path}")

    return summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--per-class",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--perturbations",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lime-samples",
        type=int,
        default=700,
    )

    parser.add_argument(
        "--random-comparisons",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--only",
        choices=list(DISEASES),
        default=None,
    )

    args = parser.parse_args()

    if (
        args.per_class < 1
        or args.seeds < 2
        or args.perturbations < 1
        or args.lime_samples < 100
        or args.random_comparisons < 1
    ):
        parser.error(
            "Invalid evaluation settings."
        )

    diseases = (
        [args.only]
        if args.only
        else list(DISEASES)
    )

    summaries = []
    failures = []

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "per_class": args.per_class,
        "seeds": args.seeds,
        "perturbations": args.perturbations,
        "lime_samples": args.lime_samples,
        "random_comparisons": args.random_comparisons,
        "diseases": diseases,
    }

    config_path = (
        REPORT_DIR / "xai_robustness_config_v2.json"
    )

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    for disease in diseases:

        try:
            result = evaluate_one_disease(
                disease,
                args,
            )

            summaries.append(result)

        except Exception as exc:

            failures.append(disease)

            print(
                f"\nERROR in {disease}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    if summaries:

        summary_df = pd.DataFrame(summaries)

        output_path = (
            REPORT_DIR
            / "xai_robustness_summary_v2.csv"
        )

        if args.only and output_path.exists():
            existing = pd.read_csv(output_path)
            existing = existing[
                existing["disease"] != args.only
            ]
            summary_df = pd.concat(
                [existing, summary_df],
                ignore_index=True,
            )

        summary_df.to_csv(
            output_path,
            index=False,
        )

        display_columns = [
            "name",
            "records_evaluated",
            "top5_consistency",
            "shap_input_stability",
            "lime_input_stability",
            "lime_seed_stability",
            "lime_local_r2",
        ]

        print("\n" + "=" * 60)
        print("FINAL FIVE-DISEASE XAI RELIABILITY COMPARISON")
        print("=" * 60)

        print(
            summary_df[display_columns]
            .round(3)
            .to_string(index=False)
        )

        print(f"\nSaved: {output_path}")

    if failures:
        print("\nFailed diseases:", ", ".join(failures))
        raise SystemExit(1)

    print("\nXAI evaluation completed successfully.")


if __name__ == "__main__":
    main()
