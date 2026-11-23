
# 🩺 HealthExplain-AI

### Explainable Multi-Disease Risk Prediction with Machine Learning, SHAP & LIME

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-1565C0)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/XAI-LIME-00897B)](https://github.com/marcotcr/lime)

**HealthExplain-AI** is an end-to-end machine learning and explainable AI (XAI) project that estimates risk across **five disease datasets** and helps users understand which input features influence individual predictions.

It combines disease-specific classification pipelines, SHAP and LIME explanations, reliability experiments, and an interactive **four-page Streamlit dashboard**.

**Repository:** [HealthExplain-AI](https://github.com/pari-dudeja2005/HealthExplain-AI)

> **Important:** This is a research and educational prototype. Predictions and explanations are not clinically validated and must not be used for diagnosis, treatment, or medical decisions.

---

## ✨ Key Features

- **Multi-Disease Prediction:** Supports Breast Cancer, Diabetes, Heart Disease, Chronic Kidney Disease (CKD), and Parkinson's Disease.
- **Machine Learning Models:** Compares Logistic Regression, Random Forest, and XGBoost.
- **Data Preprocessing:** Handles missing values, numerical scaling, categorical encoding, and feature preparation.
- **Single & Batch Predictions:** Supports individual records and CSV-based batch analysis.
- **Explainable AI:** Integrates SHAP and LIME to explain individual predictions.
- **XAI Reliability Evaluation:** Measures explanation agreement, stability, local fidelity, and feature-replacement impact.
- **Cross-Validation:** Evaluates predictive performance across multiple folds.
- **Model Auditing:** Includes CKD data-quality checks and subject-aware Parkinson's evaluation.
- **Interactive Dashboard:** Built using Streamlit.

---

## 🧬 Diseases Covered

| Disease | Dataset | Prediction Target |
|---|---|---|
| Breast Cancer | Wisconsin Diagnostic Breast Cancer | Malignant / Benign |
| Diabetes | Pima Indians Diabetes | Diabetes / No Diabetes |
| Heart Disease | UCI Heart Disease | Heart Disease / No Heart Disease |
| Chronic Kidney Disease | UCI CKD | CKD / No CKD |
| Parkinson's Disease | UCI Parkinson's | Parkinson's / Healthy |

The datasets are publicly available benchmark datasets used for research and machine learning experimentation.

---

## 🖥️ Streamlit Dashboard

The application consists of four interactive pages.

### 1. Overview

Provides a comprehensive summary of the project, including:

- Supported diseases and datasets
- Selected machine learning models
- Model evaluation metrics
- Cross-validation performance
- Model comparisons and audit results

### 2. Risk Prediction

Allows users to generate predictions using trained machine learning models.

**Single Prediction**
- Enter disease-specific feature values.
- Generate a prediction.
- View the estimated probability and classification.

**Batch Prediction**
- Upload CSV datasets.
- Generate predictions for multiple records.
- Download the results for further analysis.

Predictions use the saved model's decision rule without a user-adjustable threshold slider.

### 3. SHAP & LIME

Provides individual prediction explanations using two explainable AI techniques.

**SHAP (SHapley Additive exPlanations)**

- Estimates feature contributions to model predictions.
- Identifies features influencing the model's output.
- Helps interpret predictions at the individual-record level.

**LIME (Local Interpretable Model-Agnostic Explanations)**

- Approximates model behaviour around an individual prediction.
- Provides local feature importance.
- Reports local surrogate R² to assess explanation fidelity.

### 4. XAI Reliability

Evaluates how reliable and consistent the generated explanations are.

Includes:

- SHAP–LIME feature agreement
- Explanation stability under input perturbations
- LIME random-seed stability
- Local surrogate fidelity
- Feature-replacement analysis
- Cross-disease explanation comparisons

---

## 🤖 Machine Learning Models

Three machine learning algorithms are evaluated.

### Logistic Regression

A statistical classification algorithm that models the probability of a binary outcome.

\[
P(y=1|X)=\frac{1}{1+e^{-(w^TX+b)}}
\]

### Random Forest

An ensemble learning algorithm that combines predictions from multiple decision trees.

It uses bootstrap sampling and random feature selection to improve predictive robustness.

### XGBoost

An optimised gradient-boosting algorithm that builds decision trees sequentially to minimise prediction errors.

The best-performing model is selected using validation performance, with ROC-AUC as a primary selection metric.

---

## 📊 Model Performance

The following results were obtained from the project's final cross-validation evaluation.

| Disease | ROC-AUC | Balanced Accuracy | Recall | Specificity |
|---|---:|---:|---:|---:|
| Breast Cancer | **0.996** | **97.2%** | 95.7% | 98.7% |
| Diabetes | 0.842 | 73.4% | 57.3% | 89.5% |
| Heart Disease | 0.884 | 80.5% | 75.0% | 86.0% |
| Chronic Kidney Disease | **1.000** | **99.4%** | 99.7% | 99.2% |
| Parkinson's Disease | 0.709 | 69.5% | 91.5% | 47.5% |

### Performance Interpretation

**Breast Cancer:** Demonstrates strong classification performance on the benchmark dataset.

**Diabetes:** Achieves reasonable discrimination but has lower recall, indicating that some positive cases are missed.

**Heart Disease:** Demonstrates good predictive performance with relatively balanced sensitivity and specificity.

**Chronic Kidney Disease:** Achieves near-perfect benchmark performance. However, further investigation is necessary to assess generalisation beyond the dataset.

**Parkinson's Disease:** Shows greater variability across subject-grouped validation folds, highlighting the challenges associated with small datasets and repeated recordings.

These results represent cross-validation performance, not external clinical validation.

---

## 🔍 Explainable AI: SHAP & LIME

Explainability is a central component of HealthExplain-AI.

The project uses SHAP and LIME to interpret predictions and evaluate the reliability of explanations.

### SHAP

SHAP uses concepts from cooperative game theory to estimate the contribution of individual features.

A simplified additive explanation is:

\[
f(x)=\phi_0+\sum_{i=1}^{M}\phi_i
\]

Where:

- \(f(x)\) represents the model output.
- \(\phi_0\) represents the baseline output.
- \(\phi_i\) represents the contribution of feature \(i\).

### LIME

LIME approximates the behaviour of a complex model around a selected observation using a simpler interpretable model.

Its optimisation objective is:

\[
\xi(x)=\arg\min_{g\in G}
\left[L(f,g,\pi_x)+\Omega(g)\right]
\]

Where:

- \(f\) is the original machine learning model.
- \(g\) is the interpretable surrogate model.
- \(\pi_x\) defines the local neighbourhood.
- \(L\) measures approximation error.
- \(\Omega(g)\) penalises model complexity.

---

## 📈 XAI Reliability Results

The final explanation evaluation used eight records per disease.

| Disease | SHAP–LIME Top-5 Agreement | Mean LIME R² |
|---|---:|---:|
| Breast Cancer | 0.369 | 0.834 |
| Diabetes | **0.917** | 0.937 |
| Heart Disease | 0.577 | **0.949** |
| Chronic Kidney Disease | 0.427 | 0.655 |
| Parkinson's Disease | 0.649 | 0.452 |

### Explanation Reliability Metrics

**1. SHAP–LIME Agreement**

Measures the overlap between the top five features identified by SHAP and LIME using Jaccard similarity.

**2. Explanation Stability**

Measures whether feature rankings remain consistent after small changes to input values.

**3. LIME Seed Stability**

Measures the consistency of LIME explanations across different random seeds.

**4. Local Surrogate Fidelity**

Uses LIME R² to evaluate how closely the local surrogate approximates the original model.

**5. Feature-Replacement Analysis**

Compares changes in model output after replacing highly ranked features with changes after replacing randomly selected features.

### Key Observations

- Diabetes demonstrates high SHAP–LIME agreement.
- Heart Disease demonstrates strong LIME local fidelity.
- Breast Cancer achieves good predictive performance but lower explanation agreement.
- CKD has moderate-to-weak LIME fidelity.
- Parkinson's Disease has relatively low LIME fidelity and requires cautious interpretation.

Differences between SHAP and LIME are not necessarily errors. Both techniques use different approximation approaches.

---

## 🧪 Model Validation and Data Quality Audits

### Cross-Validation

Repeated cross-validation was used to evaluate the consistency of model performance.

The evaluation includes:

- ROC-AUC
- PR-AUC
- Balanced accuracy
- Recall
- Specificity

### CKD Data Audit

The CKD dataset was investigated for duplicates, conflicting labels, and missingness-related predictive signals.

| Audit Metric | Result |
|---|---:|
| Total Records | 400 |
| Unique Feature Rows | 400 |
| Duplicate Feature Rows | 0 |
| Train–Test Duplicate Overlap | 0 |
| Conflicting Labels | 0 |
| Missingness-Only ROC-AUC | 0.862 |

The results indicate that missingness patterns contain substantial predictive information.

Although no exact duplicate leakage was identified, near-perfect CKD performance requires cautious interpretation.

### Parkinson's Subject-Level Evaluation

Parkinson's recordings were evaluated using subject-aware cross-validation to reduce overlap between recordings belonging to the same subject.

Additional candidate models were compared, including regularised Logistic Regression, Random Forest, and XGBoost.

No alternative satisfied the predefined improvement criteria, so the existing trained model was retained.

---

## 🏗️ Project Structure

```text
HealthExplain-AI/
│
├── app_multi.py
│   └── Multi-page Streamlit dashboard
│
├── run_all_diseases.py
│   └── Data processing, model training and evaluation
│
├── final_validation.py
│   └── Cross-validation and XAI robustness testing
│
├── reliable_xai.py
│   └── LIME reliability evaluation and configuration
│
├── fix_model_issues.py
│   └── CKD and Parkinson's model audits
│
├── data/
│   ├── raw/
│   └── processed/
│       └── multidisease/
│
├── models/
│   └── multidisease/
│       └── Trained machine learning models
│
├── reports/
│   └── multidisease/
│       └── Evaluation metrics and XAI reports
│
├── requirements.txt
│
└── README.md
```

Some data, model, and report files are generated during the training and evaluation processes.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pari-dudeja2005/HealthExplain-AI.git

cd HealthExplain-AI
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment.

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app_multi.py
```

The Streamlit application will open in your browser, usually at:

```text
http://localhost:8501
```

### 5. Train Models (If Required)

If the trained model files are unavailable, run:

```bash
python run_all_diseases.py
```

### 6. Run Final Evaluation

```bash
python final_validation.py
```

### 7. Run Additional Audits

```bash
python fix_model_issues.py

python reliable_xai.py
```

These scripts generate additional model-quality and explanation-reliability reports.

---

## 🚀 Deployment

HealthExplain-AI can be deployed using Streamlit Community Cloud.

### Deployment Steps

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using GitHub.
4. Select the `HealthExplain-AI` repository.
5. Choose the `main` branch.
6. Set the application entry point to `app_multi.py`.
7. Deploy the application.

The deployment requires the trained models, relevant processed datasets, evaluation reports, and Python dependencies to be available.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | NumPy, pandas |
| Machine Learning | scikit-learn, XGBoost |
| Explainable AI | SHAP, LIME |
| Dashboard | Streamlit |
| Visualisation | Plotly, Matplotlib |
| Model Persistence | joblib |
| Dataset Sources | scikit-learn, OpenML, UCI Repository |
| Version Control | Git, GitHub |

---

## ⚠️ Limitations

Although HealthExplain-AI provides an integrated predictive and explainability framework, several limitations remain.

- The models are trained on public benchmark datasets rather than independently collected clinical data.
- Dataset size, sampling, and missingness patterns may affect generalisation.
- CKD's near-perfect classification results require further external validation.
- Parkinson's model performance varies substantially between subject-grouped validation folds.
- LIME local fidelity is relatively weak for CKD and Parkinson's Disease.
- SHAP and LIME explanations describe model behaviour and do not establish medical causality.
- The application is intended for educational and research purposes only.

---

## 🔮 Future Enhancements

- Integration of additional disease datasets.
- External validation using independently collected clinical datasets.
- Improved model calibration and risk estimation.
- More robust explanation techniques.
- Improved LIME fidelity for CKD and Parkinson's Disease.
- Fairness and subgroup analysis where suitable data is available.
- Clinician-reviewed model interpretation.
- Enhanced visualisation and user experience.

---

## 👩‍💻 Author

**Pari Dudeja**

B.Tech — Artificial Intelligence & Machine Learning

**GitHub:** [@pari-dudeja2005](https://github.com/pari-dudeja2005)

**Project Repository:** [HealthExplain-AI](https://github.com/pari-dudeja2005/HealthExplain-AI)

---

### ⭐ Project Summary

HealthExplain-AI demonstrates how machine learning and explainable artificial intelligence can be integrated into a multi-disease prediction system.

The project focuses not only on predictive performance but also on **explanation reliability, model transparency, cross-validation, and responsible interpretation of AI-generated results**.

*Built with Python, Machine Learning, SHAP, LIME, and Streamlit.*
