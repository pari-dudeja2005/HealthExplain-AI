
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import (
    load_trained_model,
    predict_single,
    predict_batch
)

from src.explain import (
    create_explainers,
    explain_with_shap,
    explain_with_lime
)


# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="HealthExplain-AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "data/processed/train_data.csv"
TEST_PATH = BASE_DIR / "data/processed/test_data.csv"

COMPARISON_PATH = BASE_DIR / "reports/model_comparison.csv"
METRICS_PATH = BASE_DIR / "reports/test_metrics.json"
EVALUATION_PATH = BASE_DIR / "reports/xai_evaluation.csv"
CLINICAL_PATH = BASE_DIR / "reports/clinical_plausibility_review.csv"


# ==========================================
# STYLING
# ==========================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 650 !important;
        letter-spacing: -0.5px;
    }

    h2, h3 {
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] {
        padding: 16px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# DATA LOADING
# ==========================================

@st.cache_resource
def get_model():
    return load_trained_model()


@st.cache_data
def get_train_data():
    return pd.read_csv(TRAIN_PATH)


@st.cache_data
def get_test_data():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def get_comparison():
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data
def get_test_metrics():
    with open(METRICS_PATH, "r") as file:
        return json.load(file)


try:
    model = get_model()
    train_df = get_train_data()
    test_df = get_test_data()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.info("Run python -m src.train first.")
    st.stop()


X_train = train_df.drop(columns=["target"])
X_test = test_df.drop(columns=["target"])


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def contribution_chart(
    dataframe,
    value_column,
    title,
    axis_title
):

    top_features = dataframe.head(10).iloc[::-1]

    values = top_features[value_column]

    colors = [
        "#C65D5D" if value > 0 else "#417BA6"
        for value in values
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=top_features["feature"],
            orientation="h",
            marker_color=colors
        )
    )

    fig.add_vline(
        x=0,
        line_color="gray",
        line_width=1
    )

    fig.update_layout(
        title=title,
        xaxis_title=axis_title,
        yaxis_title="",
        height=450,
        margin=dict(l=10, r=10, t=50, b=40)
    )

    return fig


def show_prediction_result(result):

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Class",
        result["predicted_label"]
    )

    col2.metric(
        "Benign Probability",
        f"{result['benign_probability']:.2%}"
    )

    col3.metric(
        "Malignant Probability",
        f"{result['malignant_probability']:.2%}"
    )

    st.progress(
        result["malignant_probability"],
        text="Malignant class probability"
    )


# ==========================================
# NAVIGATION
# ==========================================

st.sidebar.title("HealthExplain-AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Risk Prediction",
        "SHAP & LIME",
        "XAI Reliability"
    ]
)


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================

if page == "Overview":

    st.title("Overview")

    st.subheader("Dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        len(train_df) + len(test_df)
    )

    col2.metric(
        "Features",
        X_train.shape[1]
    )

    col3.metric(
        "Training Records",
        len(train_df)
    )

    col4.metric(
        "Test Records",
        len(test_df)
    )

    st.divider()

    st.subheader("Model Performance")

    if METRICS_PATH.exists():

        report = get_test_metrics()
        metrics = report["test_metrics"]

        st.write(
            f"Selected Model: **{report['best_model']}**"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}"
        )

        col2.metric(
            "Precision",
            f"{metrics['precision']:.2%}"
        )

        col3.metric(
            "Recall",
            f"{metrics['recall']:.2%}"
        )

        col4.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.4f}"
        )

        st.caption("Performance on the held-out test set.")

    st.divider()

    st.subheader("Model Comparison")

    if COMPARISON_PATH.exists():

        comparison = get_comparison()

        st.dataframe(
            comparison.round(4),
            use_container_width=True,
            hide_index=True
        )

        fig = go.Figure()

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]:

            fig.add_trace(
                go.Bar(
                    name=metric.upper(),
                    x=comparison["model"],
                    y=comparison[metric]
                )
            )

        fig.update_layout(
            barmode="group",
            height=420,
            yaxis_title="Score",
            yaxis_range=[0, 1.05],
            legend_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Model comparison uses validation-set results."
        )

    st.divider()

    st.subheader("Class Distribution")

    complete_data = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    counts = (
        complete_data["target"]
        .value_counts()
        .sort_index()
    )

    fig = go.Figure(
        go.Bar(
            x=["Benign", "Malignant"],
            y=[counts.get(0, 0), counts.get(1, 0)],
            marker_color=["#417BA6", "#C65D5D"],
            text=[counts.get(0, 0), counts.get(1, 0)],
            textposition="outside"
        )
    )

    fig.update_layout(
        height=350,
        xaxis_title="Class",
        yaxis_title="Number of Records",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# PAGE 2: RISK PREDICTION
# ==========================================

elif page == "Risk Prediction":

    st.title("Risk Prediction")

    single_tab, batch_tab = st.tabs([
        "Single Prediction",
        "Batch Prediction"
    ])

    # --------------------------------------
    # SINGLE PREDICTION
    # --------------------------------------

    with single_tab:

        input_mode = st.radio(
            "Input Method",
            [
                "Use Test Dataset",
                "Manual Entry"
            ],
            horizontal=True
        )

        with st.form("single_prediction_form"):

            if input_mode == "Use Test Dataset":

                patient_number = st.number_input(
                    "Record Number",
                    min_value=1,
                    max_value=len(X_test),
                    value=1,
                    step=1
                )

                # Corrected indexing
                patient_data = X_test.iloc[
                    [int(patient_number) - 1]
                ].copy()

                with st.expander("View Input Features"):

                    st.dataframe(
                        patient_data.T,
                        use_container_width=True
                    )

            else:

                patient_values = {}

                defaults = X_train.median()

                with st.expander(
                    "Enter Feature Values",
                    expanded=True
                ):

                    columns = st.columns(3)

                    for index, feature in enumerate(
                        X_train.columns
                    ):

                        with columns[index % 3]:

                            patient_values[feature] = (
                                st.number_input(
                                    feature,
                                    value=float(
                                        defaults[feature]
                                    ),
                                    format="%.4f",
                                    key=f"manual_{feature}"
                                )
                            )

                patient_data = pd.DataFrame(
                    [patient_values]
                )

            threshold = st.slider(
                "Classification Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01
            )

            submitted = st.form_submit_button(
                "Predict",
                type="primary"
            )

        if submitted:

            try:

                result = predict_single(
                    patient_data,
                    threshold=threshold
                )

                st.session_state[
                    "latest_patient"
                ] = patient_data.copy()

                st.session_state[
                    "latest_result"
                ] = result

                st.subheader("Prediction Result")

                show_prediction_result(result)

            except Exception as error:

                st.error(str(error))

    # --------------------------------------
    # BATCH PREDICTION
    # --------------------------------------

    with batch_tab:

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"]
        )

        if uploaded_file is not None:

            batch_data = pd.read_csv(
                uploaded_file
            )

        else:

            sample_count = st.slider(
                "Number of Test Records",
                min_value=1,
                max_value=min(50, len(X_test)),
                value=5
            )

            batch_data = X_test.head(
                sample_count
            ).copy()

        st.dataframe(
            batch_data.head(),
            use_container_width=True
        )

        batch_threshold = st.slider(
            "Batch Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )

        if st.button(
            "Run Batch Prediction",
            type="primary"
        ):

            try:

                results = predict_batch(
                    batch_data,
                    threshold=batch_threshold
                )

                total = len(results)

                malignant = int(
                    (
                        results["predicted_class"] == 1
                    ).sum()
                )

                benign = total - malignant

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Records",
                    total
                )

                col2.metric(
                    "Predicted Benign",
                    benign
                )

                col3.metric(
                    "Predicted Malignant",
                    malignant
                )

                st.subheader("Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                st.download_button(
                    "Download Results",
                    data=results.to_csv(
                        index=False
                    ),
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

            except Exception as error:

                st.error(str(error))


# ==========================================
# PAGE 3: SHAP & LIME
# ==========================================

elif page == "SHAP & LIME":

    st.title("SHAP & LIME Explanations")

    options = ["Test Record"]

    if "latest_patient" in st.session_state:
        options.append("Latest Prediction")

    source = st.radio(
        "Input Source",
        options,
        horizontal=True
    )

    if source == "Latest Prediction":

        patient = st.session_state[
            "latest_patient"
        ].copy()

    else:

        record_number = st.number_input(
            "Record Number",
            min_value=1,
            max_value=len(X_test),
            value=1,
            step=1
        )

        patient = X_test.iloc[
            [int(record_number) - 1]
        ].copy()

    if st.button(
        "Generate Explanations",
        type="primary"
    ):

        with st.spinner(
            "Generating SHAP and LIME explanations..."
        ):

            try:

                shap_explainer, lime_explainer = (
                    create_explainers(
                        model,
                        X_train
                    )
                )

                shap_df = explain_with_shap(
                    patient,
                    model,
                    shap_explainer
                )

                lime_df, lime_explanation = (
                    explain_with_lime(
                        patient,
                        model,
                        lime_explainer
                    )
                )

                probability = float(
                    model.predict_proba(
                        patient
                    )[0, 1]
                )

                st.metric(
                    "Malignant Probability",
                    f"{probability:.2%}"
                )

                col1, col2 = st.columns(2)

                with col1:

                    fig = contribution_chart(
                        shap_df,
                        "shap_value",
                        "SHAP",
                        "Contribution to Malignant Log-Odds"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                with col2:

                    fig = contribution_chart(
                        lime_df,
                        "lime_weight",
                        "LIME",
                        "Contribution to Malignant Probability"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                st.divider()

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("SHAP Feature Importance")

                    st.dataframe(
                        shap_df.head(10).round(4),
                        use_container_width=True,
                        hide_index=True
                    )

                with col2:

                    st.subheader("LIME Feature Importance")

                    st.dataframe(
                        lime_df.head(10).round(4),
                        use_container_width=True,
                        hide_index=True
                    )

                st.caption(
                    "Positive contributions support the "
                    "malignant class; negative contributions "
                    "support the benign class. SHAP and LIME "
                    "use different numerical scales."
                )

            except Exception as error:

                st.error(str(error))


# ==========================================
# PAGE 4: XAI RELIABILITY
# ==========================================

elif page == "XAI Reliability":

    st.title("XAI Reliability")

    if not EVALUATION_PATH.exists():

        st.warning(
            "Evaluation results are not available."
        )

        st.code(
            "python -m src.evaluate_xai"
        )

        st.stop()

    evaluation = pd.read_csv(
        EVALUATION_PATH
    )

    # --------------------------------------
    # CONSISTENCY AND STABILITY
    # --------------------------------------

    st.subheader("Consistency and Stability")

    consistency = evaluation[
        "top_k_jaccard"
    ].mean()

    shap_stability = evaluation[
        "shap_stability"
    ].mean()

    lime_stability = evaluation[
        "lime_stability"
    ].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "SHAP-LIME Consistency",
        f"{consistency:.3f}"
    )

    col2.metric(
        "SHAP Stability",
        f"{shap_stability:.3f}"
    )

    col3.metric(
        "LIME Stability",
        f"{lime_stability:.3f}"
    )

    fig = go.Figure(
        go.Bar(
            x=[
                "Consistency",
                "SHAP Stability",
                "LIME Stability"
            ],
            y=[
                consistency,
                shap_stability,
                lime_stability
            ],
            marker_color=[
                "#587C9E",
                "#417BA6",
                "#C18B50"
            ],
            text=[
                f"{consistency:.3f}",
                f"{shap_stability:.3f}",
                f"{lime_stability:.3f}"
            ],
            textposition="outside"
        )
    )

    fig.update_layout(
        height=380,
        yaxis_title="Similarity Score",
        yaxis_range=[0, 1.1],
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------
    # FIDELITY
    # --------------------------------------

    st.divider()

    st.subheader("Fidelity")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "SHAP Fidelity Ratio",
        f"{evaluation['shap_fidelity_ratio'].mean():.3f}"
    )

    col2.metric(
        "LIME Fidelity Ratio",
        f"{evaluation['lime_fidelity_ratio'].mean():.3f}"
    )

    col3.metric(
        "LIME Local R²",
        f"{evaluation['lime_local_r2'].mean():.3f}"
    )

    # --------------------------------------
    # DETAILED RESULTS
    # --------------------------------------

    st.divider()

    st.subheader("Evaluation Results")

    st.dataframe(
        evaluation.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Evaluation Results",
        data=evaluation.to_csv(index=False),
        file_name="xai_evaluation.csv",
        mime="text/csv"
    )

    # --------------------------------------
    # CLINICAL REVIEW
    # --------------------------------------

    st.divider()

    st.subheader("Clinical Plausibility")

    if CLINICAL_PATH.exists():

        clinical = pd.read_csv(
            CLINICAL_PATH
        )

        st.dataframe(
            clinical,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Clinical plausibility requires a separate "
            "literature-based review. The entries are "
            "not yet validated."
        )

    else:

        st.info(
            "Clinical plausibility review is not available."
        )

