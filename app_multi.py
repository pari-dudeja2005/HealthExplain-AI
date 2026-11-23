
"""HealthExplain-AI: Five-Disease Prediction and Explainability Dashboard.

Run:
    streamlit run app_multi.py
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from run_all_diseases import (
    DISEASES,
    REPORT_DIR,
    explain_record,
    paths_for,
)

from reliable_xai import (
    CONFIG_PATH,
    make_explainers,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HealthExplain-AI",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 650 !important;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(120,120,120,.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_resource
def load_model(disease, model_version):
    return joblib.load(
        paths_for(disease)["model"]
    )


@st.cache_data
def load_records(disease):

    paths = paths_for(disease)

    train_df = pd.read_csv(
        paths["train"]
    )

    test_df = pd.read_csv(
        paths["test"]
    )

    return train_df, test_df


@st.cache_data
def load_report(disease, report_version):

    path = (
        paths_for(disease)["report"]
        / "test_metrics.json"
    )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


@st.cache_resource
def load_explainers(
    disease,
    model_version,
    lime_config_version
):

    model = load_model(
        disease,
        model_version
    )

    train_df, _ = load_records(disease)

    X_train = train_df.drop(
        columns="target"
    )

    return make_explainers(
        model,
        X_train
    )


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def validate_features(df, model):

    expected = list(
        model.feature_names_in_
    )

    features = df.drop(
        columns=["target"],
        errors="ignore"
    ).copy()

    missing = sorted(
        set(expected) - set(features.columns)
    )

    extra = sorted(
        set(features.columns) - set(expected)
    )

    if missing or extra:

        raise ValueError(
            f"Missing columns: {missing}. "
            f"Unexpected columns: {extra}."
        )

    return features[expected]


def predict_table(df, model, info):

    features = validate_features(
        df,
        model
    )

    probabilities = model.predict_proba(
        features
    )[:, 1]

    predicted = model.predict(
        features
    ).astype(int)

    output = features.copy()

    output["positive_probability"] = (
        probabilities
    )

    output["negative_probability"] = (
        1.0 - probabilities
    )

    output["predicted_class"] = predicted

    output["predicted_label"] = np.where(
        predicted == 1,
        info["positive"],
        info["negative"]
    )

    return output


# ============================================================
# VISUALISATION
# ============================================================

def explanation_chart(
    df,
    value_column,
    heading
):

    top = df.head(10).iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=top[value_column],
            y=top["feature"],
            orientation="h",
            marker_color=[
                "#B86161"
                if value > 0
                else "#507B9D"
                for value in top[value_column]
            ]
        )
    )

    fig.add_vline(
        x=0,
        line_color="gray",
        line_width=1
    )

    fig.update_layout(
        title=heading,
        height=420,
        margin=dict(
            l=5,
            r=5,
            t=45,
            b=35
        ),
        xaxis_title="Contribution",
        yaxis_title=""
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

available = [
    key
    for key in DISEASES
    if paths_for(key)["model"].exists()
]

if not available:

    st.error(
        "No trained models found. "
        "Run python run_all_diseases.py first."
    )

    st.stop()


st.sidebar.title("HealthExplain-AI")

disease = st.sidebar.selectbox(
    "Disease",
    available,
    format_func=lambda key: DISEASES[key]["name"]
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Risk Prediction",
        "SHAP & LIME",
        "XAI Reliability"
    ]
)


# ============================================================
# SELECTED DISEASE
# ============================================================

info = DISEASES[disease]

paths = paths_for(disease)

model_version = (
    paths["model"].stat().st_mtime_ns
)

report_version = (
    (
        paths["report"]
        / "test_metrics.json"
    ).stat().st_mtime_ns
)

lime_config_version = (
    CONFIG_PATH.stat().st_mtime_ns
    if CONFIG_PATH.exists()
    else 0
)

model = load_model(
    disease,
    model_version
)

train_df, test_df = load_records(disease)

X_train = train_df.drop(
    columns="target"
)

X_test = test_df.drop(
    columns="target"
)

report = load_report(
    disease,
    report_version
)

st.title(info["name"])


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Dataset and Model")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        len(train_df) + len(test_df)
    )

    c2.metric(
        "Features",
        X_train.shape[1]
    )

    c3.metric(
        "Training Records",
        len(train_df)
    )

    c4.metric(
        "Test Records",
        len(test_df)
    )

    st.write(
        f"Selected Model: **{report['best_model']}**"
    )

    # --------------------------------------------------------
    # HELD-OUT TEST PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Held-Out Test Performance")

    metrics = report["test_metrics"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}"
    )

    c2.metric(
        "Precision",
        f"{metrics['precision']:.2%}"
    )

    c3.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

    c4.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.3f}"
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Model Comparison")

    comparison_path = (
        paths["report"]
        / "model_comparison.csv"
    )

    if comparison_path.exists():

        comparison = pd.read_csv(
            comparison_path
        )

        st.dataframe(
            comparison.round(4),
            hide_index=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Target Distribution")

    counts = pd.concat(
        [
            train_df["target"],
            test_df["target"]
        ]
    ).value_counts()

    fig = go.Figure(
        go.Bar(
            x=[
                info["negative"],
                info["positive"]
            ],
            y=[
                int(counts.get(0, 0)),
                int(counts.get(1, 0))
            ],
            marker_color=[
                "#507B9D",
                "#B86161"
            ]
        )
    )

    fig.update_layout(
        height=300,
        yaxis_title="Records",
        xaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CROSS-VALIDATION
    # --------------------------------------------------------

    st.subheader("Model Performance Validation")

    cv_path = (
        paths["report"]
        / "cross_validation_audit_v3.csv"
    )

    if cv_path.exists():

        audit = pd.read_csv(cv_path)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Valid CV Folds",
            len(audit)
        )

        c2.metric(
            "Mean CV ROC-AUC",
            f"{audit['roc_auc'].mean():.3f}"
        )

        c3.metric(
            "Mean CV Balanced Accuracy",
            f"{audit['balanced_accuracy'].mean():.3f}"
        )

        with st.expander(
            "View Cross-Validation Results"
        ):

            st.dataframe(
                audit.round(4),
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Cross-validation uses the original "
                "training partition and the selected "
                "model family. Held-out test records "
                "are excluded."
            )

        if disease == "parkinsons":

            if "subject_roc_auc" in audit.columns:

                subject_auc = audit[
                    "subject_roc_auc"
                ].mean()

                st.metric(
                    "Mean Subject-Level ROC-AUC",
                    f"{subject_auc:.3f}"
                    if pd.notna(subject_auc)
                    else "N/A"
                )

            st.caption(
                "Parkinson's cross-validation separates "
                "recordings by subject."
            )

    else:

        st.info(
            "Run python final_validation.py "
            "to generate cross-validation results."
        )

    # --------------------------------------------------------
    # PARKINSON'S MODEL AUDIT
    # --------------------------------------------------------

    if disease == "parkinsons":

        candidate_path = (
            paths["report"]
            / "parkinsons_model_comparison_v4.csv"
        )

        if candidate_path.exists():

            with st.expander(
                "Parkinson's Subject-Level Model Audit"
            ):

                candidate_df = pd.read_csv(
                    candidate_path
                )

                st.dataframe(
                    candidate_df.round(4),
                    hide_index=True,
                    use_container_width=True
                )

                st.caption(
                    "Candidate results are exploratory. "
                    "The comparison does not mean that "
                    "the active model was replaced."
                )

    # --------------------------------------------------------
    # CKD DATA QUALITY AUDIT
    # --------------------------------------------------------

    if disease == "kidney_disease":

        summary_path = (
            paths["report"]
            / "ckd_quality_summary_v4.json"
        )

        if summary_path.exists():

            with st.expander(
                "CKD Data Quality Audit"
            ):

                audit_summary = json.loads(
                    summary_path.read_text(
                        encoding="utf-8"
                    )
                )

                st.json(audit_summary)

        missing_path = (
            paths["report"]
            / "ckd_missingness_audit_v4.csv"
        )

        if not missing_path.exists():

            missing_path = (
                paths["report"]
                / "missingness_audit_v3.csv"
            )

        if missing_path.exists():

            with st.expander(
                "CKD Missingness Analysis"
            ):

                missing_df = pd.read_csv(
                    missing_path
                )

                st.dataframe(
                    missing_df.round(4),
                    hide_index=True,
                    use_container_width=True
                )


# ============================================================
# PAGE 2: RISK PREDICTION
# ============================================================

elif page == "Risk Prediction":

    single_tab, batch_tab = st.tabs(
        [
            "Single Prediction",
            "Batch Prediction"
        ]
    )

    # --------------------------------------------------------
    # SINGLE PREDICTION
    # --------------------------------------------------------

    with single_tab:

        source = st.radio(
            "Input Source",
            [
                "Test Dataset",
                "Manual Entry"
            ],
            horizontal=True
        )

        with st.form("single_prediction"):

            if source == "Test Dataset":

                number = st.number_input(
                    "Record Number",
                    min_value=1,
                    max_value=len(X_test),
                    value=1,
                    step=1
                )

                patient = X_test.iloc[
                    [int(number) - 1]
                ].copy()

                with st.expander(
                    "Input Features"
                ):

                    st.dataframe(
                        patient.T,
                        use_container_width=True
                    )

            else:

                numeric = set(
                    report["numeric_features"]
                )

                categorical = set(
                    report["categorical_features"]
                )

                values = {}

                columns = st.columns(3)

                for i, feature in enumerate(
                    X_train.columns
                ):

                    with columns[i % 3]:

                        if feature in numeric:

                            median = pd.to_numeric(
                                X_train[feature],
                                errors="coerce"
                            ).median()

                            default = (
                                float(median)
                                if pd.notna(median)
                                else 0.0
                            )

                            values[feature] = (
                                st.number_input(
                                    feature,
                                    value=default,
                                    format="%.4f",
                                    key=f"{disease}_{feature}"
                                )
                            )

                        elif feature in categorical:

                            options = sorted(
                                X_train[feature]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            ) or [""]

                            values[feature] = (
                                st.selectbox(
                                    feature,
                                    options,
                                    key=f"{disease}_{feature}"
                                )
                            )

                        else:

                            st.error(
                                f"Unrecognised feature type: "
                                f"{feature}"
                            )

                patient = pd.DataFrame(
                    [values]
                )

            submitted = st.form_submit_button(
                "Predict",
                type="primary"
            )

        if submitted:

            try:

                prediction = predict_table(
                    patient,
                    model,
                    info
                ).iloc[0]

                st.session_state[
                    f"latest_{disease}"
                ] = patient.copy()

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Predicted Class",
                    prediction["predicted_label"]
                )

                c2.metric(
                    f"{info['negative']} Probability",
                    f"{prediction['negative_probability']:.2%}"
                )

                c3.metric(
                    f"{info['positive']} Probability",
                    f"{prediction['positive_probability']:.2%}"
                )

                st.caption(
                    "Predictions use the trained model's "
                    "classification rule. They are not "
                    "clinical diagnoses."
                )

            except Exception as exc:

                st.error(str(exc))

    # --------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------

    with batch_tab:

        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"]
        )

        if uploaded is None:

            count = st.slider(
                "Sample Test Records",
                min_value=1,
                max_value=min(50, len(X_test)),
                value=min(5, len(X_test))
            )

            batch = X_test.head(count)

        else:

            batch = pd.read_csv(
                uploaded
            )

        st.dataframe(
            batch.head(),
            use_container_width=True
        )

        if st.button(
            "Run Batch Prediction",
            type="primary"
        ):

            try:

                result = predict_table(
                    batch,
                    model,
                    info
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Records",
                    len(result)
                )

                c2.metric(
                    info["negative"],
                    int(
                        (
                            result["predicted_class"] == 0
                        ).sum()
                    )
                )

                c3.metric(
                    info["positive"],
                    int(
                        (
                            result["predicted_class"] == 1
                        ).sum()
                    )
                )

                st.dataframe(
                    result,
                    use_container_width=True
                )

                st.download_button(
                    "Download Results",
                    data=result.to_csv(index=False),
                    file_name=f"{disease}_predictions.csv",
                    mime="text/csv"
                )

            except Exception as exc:

                st.error(str(exc))


# ============================================================
# PAGE 3: SHAP & LIME
# ============================================================

elif page == "SHAP & LIME":

    st.subheader("Patient-Level Explanations")

    sources = ["Test Record"]

    if f"latest_{disease}" in st.session_state:

        sources.append(
            "Latest Prediction"
        )

    source = st.radio(
        "Input Source",
        sources,
        horizontal=True
    )

    if source == "Test Record":

        number = st.number_input(
            "Record Number",
            min_value=1,
            max_value=len(X_test),
            value=1,
            step=1
        )

        patient = X_test.iloc[
            [int(number) - 1]
        ].copy()

    else:

        patient = st.session_state[
            f"latest_{disease}"
        ].copy()

    if st.button(
        "Generate Explanations",
        type="primary"
    ):

        with st.spinner(
            "Calculating explanations..."
        ):

            try:

                explainers = load_explainers(
                    disease,
                    model_version,
                    lime_config_version
                )

                shap_df, lime_df, lime_r2 = (
                    explain_record(
                        patient,
                        model,
                        *explainers,
                        lime_samples=1200
                    )
                )

                probability = (
                    model.predict_proba(
                        patient
                    )[0, 1]
                )

                st.metric(
                    f"{info['positive']} Probability",
                    f"{probability:.2%}"
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.plotly_chart(
                        explanation_chart(
                            shap_df,
                            "shap_value",
                            "SHAP"
                        ),
                        use_container_width=True
                    )

                with c2:

                    st.plotly_chart(
                        explanation_chart(
                            lime_df,
                            "lime_weight",
                            "LIME"
                        ),
                        use_container_width=True
                    )

                st.metric(
                    "LIME Local R²",
                    f"{lime_r2:.3f}"
                )

                # Explanation reliability warning

                if (
                    not np.isfinite(lime_r2)
                    or lime_r2 < 0.70
                ):

                    st.warning(
                        "LIME has a low local approximation "
                        "fit for this prediction. Interpret "
                        "its feature contributions cautiously."
                    )

                st.caption(
                    "SHAP and LIME explain model behaviour, "
                    "not clinical causation. Their contribution "
                    "values may use different scales."
                )

            except Exception as exc:

                st.error(str(exc))


# ============================================================
# PAGE 4: XAI RELIABILITY
# ============================================================

elif page == "XAI Reliability":

    evaluation_path = (
        paths["report"]
        / "xai_final_v3.csv"
    )

    if not evaluation_path.exists():

        st.warning(
            "Run python final_validation.py "
            "to generate XAI reliability results."
        )

        st.stop()

    evaluation = pd.read_csv(
        evaluation_path
    )

    # --------------------------------------------------------
    # EXPLANATION AGREEMENT
    # --------------------------------------------------------

    st.subheader("Explanation Agreement")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Top-5 Agreement",
        f"{evaluation['top5_agreement'].mean():.3f}"
    )

    c2.metric(
        "Top-10 Agreement",
        f"{evaluation['top10_agreement'].mean():.3f}"
    )

    c3.metric(
        "LIME Seed Stability",
        f"{evaluation['lime_seed_stability'].mean():.3f}"
    )

    # --------------------------------------------------------
    # INPUT STABILITY
    # --------------------------------------------------------

    st.subheader("Input Perturbation Stability")

    stability_metrics = [
        (
            "SHAP Stability (5%)",
            "shap_stability_5pct"
        ),
        (
            "LIME Stability (5%)",
            "lime_stability_5pct"
        ),
        (
            "SHAP Stability (15%)",
            "shap_stability_15pct"
        ),
        (
            "LIME Stability (15%)",
            "lime_stability_15pct"
        )
    ]

    columns = st.columns(4)

    for column, (title, key) in zip(
        columns,
        stability_metrics
    ):

        value = evaluation[key].mean()

        column.metric(
            title,
            f"{value:.3f}"
            if pd.notna(value)
            else "N/A"
        )

    st.caption(
        "Perturbation percentages describe interpolation "
        "strengths, not changes in clinical risk."
    )

    # --------------------------------------------------------
    # FEATURE REPLACEMENT
    # --------------------------------------------------------

    st.subheader("Feature Replacement Analysis")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "SHAP Prediction Change",
        f"{evaluation['shap_ablation_delta'].mean():.3f}"
    )

    c2.metric(
        "LIME Prediction Change",
        f"{evaluation['lime_ablation_delta'].mean():.3f}"
    )

    c3.metric(
        "Random Feature Change",
        f"{evaluation['random_ablation_delta'].mean():.3f}"
    )

    c4.metric(
        "LIME Local R²",
        f"{evaluation['lime_local_r2'].mean():.3f}"
    )

    st.caption(
        "Feature replacement measures prediction changes, "
        "not causality. This evaluation is exploratory."
    )

    # --------------------------------------------------------
    # RECORD-LEVEL RESULTS
    # --------------------------------------------------------

    st.subheader("Record-Level Evaluation")

    st.dataframe(
        evaluation.round(4),
        hide_index=True,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FIVE-DISEASE COMPARISON
    # --------------------------------------------------------

    comparison_path = (
        REPORT_DIR
        / "xai_final_summary_v3.csv"
    )

    if comparison_path.exists():

        st.subheader(
            "Five-Disease XAI Comparison"
        )

        summary = pd.read_csv(
            comparison_path
        )

        columns = [
            "name",
            "records_evaluated",
            "top5_agreement",
            "shap_stability_15pct",
            "lime_stability_15pct",
            "lime_seed_stability",
            "shap_ablation_delta",
            "lime_ablation_delta",
            "lime_local_r2"
        ]

        st.dataframe(
            summary[columns].round(4),
            hide_index=True,
            use_container_width=True
        )

        fig = go.Figure(
            go.Bar(
                x=summary["name"],
                y=summary["top5_agreement"],
                marker_color="#507B9D"
            )
        )

        fig.update_layout(
            title="SHAP-LIME Feature Agreement",
            xaxis_title="Disease",
            yaxis_title="Top-5 Jaccard Similarity",
            yaxis_range=[0, 1],
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.caption(
        "These results are based on an exploratory "
        "evaluation and do not establish clinical reliability."
    )



