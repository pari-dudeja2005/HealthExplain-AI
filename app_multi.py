
"""HealthExplain-AI: Final Five-Disease Streamlit Dashboard.

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
    paths_for,
    explain_record,
    make_explainers,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HealthExplain-AI",
    layout="wide"
)

st.markdown("""
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
""", unsafe_allow_html=True)


# ============================================================
# DATA AND MODEL LOADING
# ============================================================

def file_version(path):
    """Refresh cached files when models or reports change."""
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_resource
def load_model(disease, version):
    return joblib.load(paths_for(disease)["model"])


@st.cache_data
def load_records(disease, train_version, test_version):
    p = paths_for(disease)

    train = pd.read_csv(p["train"])
    test = pd.read_csv(p["test"])

    return train, test


@st.cache_data
def load_report(disease, version):
    p = paths_for(disease)

    return json.loads(
        (p["report"] / "test_metrics.json").read_text(
            encoding="utf-8"
        )
    )


@st.cache_resource
def load_explainers(
    disease,
    model_version,
    train_version,
    test_version
):
    model = load_model(disease, model_version)

    train_df, _ = load_records(
        disease,
        train_version,
        test_version
    )

    return make_explainers(
        model,
        train_df.drop(columns="target")
    )


# ============================================================
# COMPACT MODEL COMPARISON
# ============================================================

def model_family(name):
    """Convert detailed model names to simple algorithm names."""

    for family in [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ]:
        if str(name).startswith(family):
            return family

    return str(name)


def compact_model_comparison(comparison, deployed_name):
    """
    Display one configuration per algorithm.

    For the deployed algorithm, use its actual CV results.
    For the other algorithms, show the configuration with
    the highest cross-validation balanced accuracy.

    Original comparison files are not modified.
    """

    if "model" not in comparison.columns:
        return comparison

    choices = comparison.copy()

    choices["family"] = choices["model"].map(
        model_family
    )

    families = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ]

    choices = choices[
        choices["family"].isin(families)
    ]

    if choices.empty:
        return comparison

    displayed = []

    for family in families:

        group = choices[
            choices["family"] == family
        ]

        if group.empty:
            continue

        deployed = group[
            group["model"] == deployed_name
        ]

        if not deployed.empty:
            row = deployed.iloc[0]

        else:
            ranking = [
                column
                for column in [
                    "balanced_accuracy",
                    "roc_auc",
                    "recall"
                ]
                if column in group.columns
            ]

            if ranking:
                row = group.sort_values(
                    ranking,
                    ascending=False
                ).iloc[0]
            else:
                row = group.iloc[0]

        summary = {
            "model": family
        }

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc"
        ]:
            if metric in group.columns:
                summary[metric] = row[metric]

        displayed.append(summary)

    return pd.DataFrame(displayed)


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def validated_features(df, model):

    expected = list(model.feature_names_in_)

    df = df.drop(
        columns=["target"],
        errors="ignore"
    ).copy()

    missing = sorted(
        set(expected) - set(df.columns)
    )

    extra = sorted(
        set(df.columns) - set(expected)
    )

    if missing or extra:
        raise ValueError(
            f"Missing columns: {missing}; "
            f"Unexpected columns: {extra}"
        )

    return df[expected]


def probability_results(df, model, info):

    df = validated_features(df, model)

    probability = model.predict_proba(df)[:, 1]

    result = df.copy()

    result["positive_probability"] = probability

    result["negative_probability"] = 1 - probability

    result["predicted_class"] = (
        probability >= 0.50
    ).astype(int)

    result["predicted_label"] = np.where(
        result["predicted_class"] == 1,
        info["positive"],
        info["negative"]
    )

    return result


# ============================================================
# EXPLANATION CHARTS
# ============================================================

def importance_chart(df, value_col, title):

    subset = df.head(10).iloc[::-1]

    colours = [
        "#B86161" if value > 0 else "#507B9D"
        for value in subset[value_col]
    ]

    fig = go.Figure(
        go.Bar(
            x=subset[value_col],
            y=subset["feature"],
            orientation="h",
            marker_color=colours
        )
    )

    fig.add_vline(
        x=0,
        line_color="gray",
        line_width=1
    )

    fig.update_layout(
        title=title,
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
    disease
    for disease in DISEASES
    if paths_for(disease)["model"].exists()
]

if not available:

    st.error(
        "No trained multi-disease models found."
    )

    st.stop()


st.sidebar.title("HealthExplain-AI")

disease = st.sidebar.selectbox(
    "Disease",
    available,
    format_func=lambda d: DISEASES[d]["name"]
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
# LOAD SELECTED DISEASE
# ============================================================

info = DISEASES[disease]

p = paths_for(disease)

model_version = file_version(p["model"])

train_version = file_version(p["train"])

test_version = file_version(p["test"])

report_version = file_version(
    p["report"] / "test_metrics.json"
)

model = load_model(
    disease,
    model_version
)

train_df, test_df = load_records(
    disease,
    train_version,
    test_version
)

X_train = train_df.drop(columns="target")

X_test = test_df.drop(columns="target")

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
        f"Selected Model: "
        f"**{model_family(report['best_model'])}**"
    )

    # --------------------------------------------------------
    # TEST PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Held-out Test Performance")

    metrics = report["test_metrics"]

    tn, fp = report["confusion_matrix"][0]

    specificity = metrics.get(
        "specificity",
        tn / (tn + fp) if tn + fp else 0.0
    )

    balanced_accuracy = metrics.get(
        "balanced_accuracy",
        (metrics["recall"] + specificity) / 2
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

    c2.metric(
        "Specificity",
        f"{specificity:.2%}"
    )

    c3.metric(
        "Balanced Accuracy",
        f"{balanced_accuracy:.2%}"
    )

    c4.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.3f}"
    )

    with st.expander("Additional Test Metrics"):

        a, b, c, d = st.columns(4)

        a.metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}"
        )

        b.metric(
            "Precision",
            f"{metrics['precision']:.2%}"
        )

        c.metric(
            "F1 Score",
            f"{metrics['f1']:.3f}"
        )

        d.metric(
            "PR-AUC",
            f"{metrics['pr_auc']:.3f}"
        )

    # --------------------------------------------------------
    # COMPACT MODEL COMPARISON
    # --------------------------------------------------------

    is_cv_comparison = report.get(
        "selection_metric", ""
    ).startswith("training_cv_")

    if is_cv_comparison:

        st.subheader(
            "Training Cross-Validation Model Comparison"
        )

    else:

        st.subheader(
            "Validation Model Comparison"
        )

    comparison_path = (
        p["report"] / "model_comparison.csv"
    )

    if comparison_path.exists():

        comparison = pd.read_csv(
            comparison_path
        )

        if is_cv_comparison:

            comparison = compact_model_comparison(
                comparison,
                report["best_model"]
            )

        st.dataframe(
            comparison.round(4),
            hide_index=True,
            width="stretch"
        )

        if is_cv_comparison:

            st.caption(
                "One training cross-validation configuration "
                "per algorithm is displayed. "
                "The selected algorithm uses its deployed "
                "configuration. Other algorithms show their "
                "highest balanced-accuracy configurations. "
                "These scores are separate from the "
                "held-out test results."
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
            ]
        )
    )

    fig.update_layout(
        height=300,
        yaxis_title="Records"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    if report.get("group_safe_split"):

        st.caption(
            "Parkinson's training and test data "
            "are separated by subject."
        )


# ============================================================
# PAGE 2: RISK PREDICTION
# ============================================================

elif page == "Risk Prediction":

    tab_single, tab_batch = st.tabs(
        [
            "Single Prediction",
            "Batch Prediction"
        ]
    )

    # --------------------------------------------------------
    # SINGLE PREDICTION
    # --------------------------------------------------------

    with tab_single:

        source = st.radio(
            "Input",
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
                    value=1
                )

                patient = X_test.iloc[
                    [int(number) - 1]
                ].copy()

                with st.expander("Input Features"):

                    st.dataframe(
                        patient.T,
                        width="stretch"
                    )

            else:

                numeric = report["numeric_features"]

                categorical = report["categorical_features"]

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

                            default_value = (
                                float(median)
                                if pd.notna(median)
                                else 0.0
                            )

                            values[feature] = st.number_input(
                                feature,
                                value=default_value,
                                format="%.4f",
                                key=f"{disease}_{feature}"
                            )

                        elif feature in categorical:

                            options = sorted(
                                X_train[feature]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )

                            if not options:
                                options = [""]

                            values[feature] = st.selectbox(
                                feature,
                                options,
                                key=f"{disease}_{feature}"
                            )

                patient = pd.DataFrame([values])

            submitted = st.form_submit_button(
                "Predict",
                type="primary"
            )

        if submitted:

            try:

                result = probability_results(
                    patient,
                    model,
                    info
                ).iloc[0]

                st.session_state[
                    f"latest_{disease}"
                ] = patient.copy()

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Prediction",
                    result["predicted_label"]
                )

                c2.metric(
                    info["negative"],
                    f"{result['negative_probability']:.2%}"
                )

                c3.metric(
                    info["positive"],
                    f"{result['positive_probability']:.2%}"
                )

            except Exception as exc:

                st.error(str(exc))

    # --------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------

    with tab_batch:

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

            batch = pd.read_csv(uploaded)

        st.dataframe(
            batch.head(),
            width="stretch"
        )

        if st.button(
            "Run Batch Prediction",
            type="primary"
        ):

            try:

                result = probability_results(
                    batch,
                    model,
                    info
                )

                st.session_state[
                    f"batch_{disease}"
                ] = result

            except Exception as exc:

                st.error(str(exc))

                st.session_state.pop(
                    f"batch_{disease}",
                    None
                )

        result = st.session_state.get(
            f"batch_{disease}"
        )

        if result is not None:

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Records",
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
                width="stretch"
            )

            st.download_button(
                "Download Results",
                result.to_csv(index=False),
                file_name=f"{disease}_predictions.csv",
                mime="text/csv"
            )


# ============================================================
# PAGE 3: SHAP AND LIME
# ============================================================

elif page == "SHAP & LIME":

    st.subheader("Patient-Level Explanations")

    sources = ["Test Record"]

    if f"latest_{disease}" in st.session_state:
        sources.append("Latest Prediction")

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
            value=1
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
                    train_version,
                    test_version
                )

                shap_df, lime_df, lime_r2 = (
                    explain_record(
                        patient,
                        model,
                        *explainers,
                        lime_samples=1000
                    )
                )

                positive_probability = (
                    model.predict_proba(patient)[0, 1]
                )

                st.metric(
                    f"{info['positive']} Probability",
                    f"{positive_probability:.2%}"
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.plotly_chart(
                        importance_chart(
                            shap_df,
                            "shap_value",
                            "SHAP"
                        ),
                        width="stretch"
                    )

                with c2:

                    st.plotly_chart(
                        importance_chart(
                            lime_df,
                            "lime_weight",
                            "LIME"
                        ),
                        width="stretch"
                    )

                st.metric(
                    "LIME Local R²",
                    f"{lime_r2:.3f}"
                )

                st.caption(
                    "Contributions describe model behaviour, "
                    "not clinical causation. "
                    "SHAP units depend on the selected model; "
                    "LIME uses a local surrogate."
                )

            except Exception as exc:

                st.error(str(exc))


# ============================================================
# PAGE 4: XAI RELIABILITY
# ============================================================

elif page == "XAI Reliability":

    evaluation_path = (
        p["report"] / "xai_evaluation.csv"
    )

    if not evaluation_path.exists():

        st.warning(
            "XAI results not found. Run: "
            f"python run_all_diseases.py "
            f"--only {disease} --skip-training"
        )

        st.stop()

    evaluation = pd.read_csv(
        evaluation_path
    )

    st.subheader("Reliability Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "SHAP-LIME Consistency",
        f"{evaluation['top_k_jaccard'].mean():.3f}"
    )

    c2.metric(
        "SHAP Stability",
        f"{evaluation['shap_stability'].mean():.3f}"
    )

    c3.metric(
        "LIME Stability",
        f"{evaluation['lime_stability'].mean():.3f}"
    )

    st.subheader("Fidelity and Local Fit")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "SHAP Fidelity Ratio",
        f"{evaluation['shap_fidelity_ratio'].mean():.3f}"
    )

    c2.metric(
        "LIME Fidelity Ratio",
        f"{evaluation['lime_fidelity_ratio'].mean():.3f}"
    )

    c3.metric(
        "LIME Local R²",
        f"{evaluation['lime_local_r2'].mean():.3f}"
    )

    st.subheader("Record-Level Evaluation")

    st.dataframe(
        evaluation.round(4),
        hide_index=True,
        width="stretch"
    )

    comparison_path = (
        REPORT_DIR / "xai_comparison.csv"
    )

    if comparison_path.exists():

        st.subheader("Across-Disease Comparison")

        comparison = pd.read_csv(
            comparison_path
        )

        st.dataframe(
            comparison.round(4),
            hide_index=True,
            width="stretch"
        )

    st.caption(
        "These preliminary scores reflect a limited "
        "number of test records. "
        "Clinical plausibility requires a separate "
        "literature-based review."
    )






