# 🩺 HealthExplain-AI

### Explainable Multi-Disease Risk Prediction with Machine Learning, SHAP & LIME

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EC6B23)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-1565C0)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/XAI-LIME-00897B)](https://github.com/marcotcr/lime)

**HealthExplain-AI** is an end-to-end machine learning and explainable AI (XAI) project designed to predict outcomes across **five disease datasets** and explain how individual input features influence model predictions.

The project combines disease-specific machine learning pipelines, model evaluation, recall-focused improvements, SHAP and LIME explanations, and XAI reliability analysis through an interactive **four-page Streamlit dashboard**.

**GitHub Repository:** [HealthExplain-AI](https://github.com/pari-dudeja2005/HealthExplain-AI)

**Streamlit Application:** https://healthexplain-ai.streamlit.app/

**Important:** HealthExplain-AI is an educational and research prototype. Its predictions and explanations are not clinically validated and must not be used for medical diagnosis, treatment or clinical decision-making.

---

## ✨ Key Features

- **Multi-Disease Prediction:** Supports Breast Cancer, Diabetes, Heart Disease, Chronic Kidney Disease (CKD), and Parkinson's Disease.
- **Machine Learning Models:** Compares Logistic Regression, Random Forest and XGBoost.
- **Recall-Focused Model Improvement:** Improves Diabetes detection through class-weighted model selection while evaluating specificity and balanced accuracy.
- **Data Preprocessing:** Handles missing values, numerical scaling, categorical encoding and feature preparation.
- **Single and Batch Prediction:** Supports individual predictions, manual feature entry and CSV-based batch analysis.
- **Explainable AI:** Uses SHAP and LIME to interpret individual predictions.
- **XAI Reliability Evaluation:** Measures explanation agreement, stability, fidelity and local surrogate fit.
- **Cross-Validation:** Evaluates model performance across multiple training folds.
- **Model Auditing:** Includes CKD data-quality checks and subject-aware Parkinson's evaluation.
- **Interactive Dashboard:** Provides a consistent four-page Streamlit interface without threshold optimisation controls.

---

## 🧬 Diseases Covered

| Disease | Dataset | Prediction Target |
|---|---|---|
| Breast Cancer | Wisconsin Diagnostic Breast Cancer | Benign / Malignant |
| Diabetes | Pima Indians Diabetes | Diabetes / No Diabetes |
| Heart Disease | UCI Heart Disease | Heart Disease / No Heart Disease |
| Chronic Kidney Disease | UCI CKD | CKD / No CKD |
| Parkinson's Disease | UCI Parkinson's | Parkinson's / Healthy |

These datasets are publicly available benchmark datasets used for machine learning experimentation.

---

## 🖥️ Streamlit Dashboard

The application consists of four interactive pages.

### 1. Overview

Displays information about the selected disease and trained model.

Features include:

- Dataset size and feature count
- Training and test record distribution
- Selected machine learning model
- Held-out test performance
- Training cross-validation or validation model comparison
- Target-class distribution

The dashboard presents model comparison results in a compact format.

For Diabetes and Heart Disease, the updated models are evaluated using training cross-validation and independently tested on held-out data.

### 2. Risk Prediction

Allows users to generate disease-specific predictions using trained machine learning models.

**Single Prediction**

- Select a record from the test dataset.
- Enter feature values manually.
- Generate a binary classification.
- View estimated probabilities for both classes.

**Batch Prediction**

- Upload CSV datasets.
- Generate predictions for multiple records.
- View predicted classes and probabilities.
- Download prediction results.

The application uses a fixed internal classification cutoff of **0.50**, without displaying a threshold optimisation section or user-adjustable threshold controls.

### 3. SHAP & LIME

Provides individual prediction explanations using two explainable AI techniques.

**SHAP (SHapley Additive exPlanations)**

- Estimates feature contributions to model outputs.
- Identifies influential input features.
- Provides individual-level explanations.

**LIME (Local Interpretable Model-Agnostic Explanations)**

- Approximates model behaviour around a selected prediction.
- Generates local feature importance values.
- Reports local surrogate R² as a measure of explanation fit.

### 4. XAI Reliability

Evaluates the consistency and reliability of explanations.

Includes:

- SHAP–LIME top-feature agreement
- SHAP explanation stability
- LIME explanation stability
- SHAP and LIME fidelity ratios
- LIME local R²
- Record-level explanation evaluation
- Cross-disease reliability comparison

---

## 🤖 Machine Learning Models

Three primary machine learning algorithms are evaluated.

### Logistic Regression

Logistic Regression is a statistical classification algorithm that estimates the probability of a binary outcome.

Its probability function is:

\[
P(y=1|X)=\frac{1}{1+e^{-(w^TX+b)}}
\]

### Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

It uses bootstrap sampling and random feature selection to improve predictive robustness.

### XGBoost

XGBoost is a gradient-boosting algorithm that constructs decision trees sequentially to minimise prediction errors.

It supports regularisation and class weighting, making it suitable for imbalanced classification tasks.

### Model Selection

The original model development workflow compares algorithms using validation performance, including ROC-AUC.

For Diabetes and Heart Disease, an additional recall-focused model selection experiment was conducted using five-fold stratified training cross-validation.

Candidate models were evaluated using:

- Recall
- Specificity
- Balanced accuracy
- Precision
- ROC-AUC

The selected models were subsequently evaluated on held-out test data.

---

## 📊 Model Performance

### 1. Final Held-out Test Performance

Held-out testing evaluates the selected model on records excluded from model training and candidate selection.

The updated Diabetes and Heart Disease results are:

| Metric | Diabetes (XGBoost) | Heart Disease (Logistic Regression) |
|---|---:|---:|
| Recall | **83.33%** | **92.9%** |
| Specificity | 70.00% | 81.8% |
| Balanced Accuracy | 76.67% | 87.3% |
| Precision | 60.00% | 81.2% |
| Accuracy | 74.68% | — |
| F1 Score | 0.698 | — |
| ROC-AUC | 0.825 | — |
| PR-AUC | 0.707 | — |

The remaining Heart Disease metrics are available in its saved `test_metrics.json` report.

### 2. Diabetes Model Improvement

The final Diabetes dataset contains:

| Dataset Property | Value |
|---|---:|
| Total Records | 768 |
| Features | 8 |
| Training Records | 614 |
| Test Records | 154 |
| Selected Model | XGBoost |

The recall-focused improvement produced the following held-out test results:

| Metric | Previous Model | Updated Model |
|---|---:|---:|
| Recall | 50.00% | **83.33%** |
| Specificity | 82.00% | 70.00% |
| Balanced Accuracy | 66.00% | **76.67%** |
| Precision | 60.00% | 60.00% |

The updated XGBoost model identifies a larger proportion of positive Diabetes cases.

However, this improvement involves a trade-off: specificity decreased from 82% to 70%, indicating an increased false-positive rate.

### 3. Heart Disease Model Evaluation

The updated Heart Disease model uses Logistic Regression with class weighting.

Its training cross-validation recall improved, while held-out test recall remained approximately **92.9%**.

This distinction is important because improvements during model selection do not necessarily translate into improvements on unseen test data.

### 4. Training Cross-Validation Model Comparison

Cross-validation is used to compare candidate algorithms and configurations before final evaluation.

| Disease | Selected Model | CV Recall | CV Specificity | CV Balanced Accuracy |
|---|---|---:|---:|---:|
| Diabetes | XGBoost | 79.9% | 70.8% | 75.3% |
| Heart Disease | Logistic Regression | 82.0% | 82.4% | 82.2% |

These results come from training cross-validation and should not be confused with held-out test performance.

The dashboard presents a compact comparison of Logistic Regression, Random Forest and XGBoost, while detailed candidate results remain in the evaluation reports.

### 5. Earlier Baseline Cross-Validation Results

The following table records the project's earlier cross-validation evaluation before the recall-focused model updates.

| Disease | ROC-AUC | Balanced Accuracy | Recall | Specificity |
|---|---:|---:|---:|---:|
| Breast Cancer | 0.996 | 97.2% | 95.7% | 98.7% |
| Diabetes | 0.842 | 73.4% | 57.3% | 89.5% |
| Heart Disease | 0.884 | 80.5% | 75.0% | 86.0% |
| Chronic Kidney Disease | 1.000 | 99.4% | 99.7% | 99.2% |
| Parkinson's Disease | 0.709 | 69.5% | 91.5% | 47.5% |

These historical baseline results use a different evaluation stage from the final held-out metrics. They are retained for comparison and should not be interpreted as the performance of the updated Diabetes and Heart Disease models.

---

## 🔍 Explainable AI: SHAP & LIME

Explainability is a central component of HealthExplain-AI.

The project uses SHAP and LIME to interpret model predictions and investigate explanation reliability.

### SHAP

SHAP uses concepts from cooperative game theory to estimate the contribution of individual features.

Its additive explanation can be expressed as:

\[
f(x)=\phi_0+\sum_{i=1}^{M}\phi_i
\]

Where:

- \(f(x)\) represents the model output.
- \(\phi_0\) represents the baseline output.
- \(\phi_i\) represents the contribution of feature \(i\).

### LIME

LIME approximates the behaviour of a complex model around an individual observation using a simpler interpretable surrogate model.

Its optimisation objective is:

\[
\xi(x)=\arg\min_{g\in G}
\left[L(f,g,\pi_x)+\Omega(g)\right]
\]

Where:

- \(f\) is the original model.
- \(g\) is the interpretable surrogate model.
- \(\pi_x\) defines the local neighbourhood.
- \(L\) measures approximation error.
- \(\Omega(g)\) penalises model complexity.

SHAP and LIME use different explanation approaches, so differences between their feature rankings do not necessarily indicate an error.

---

## 📈 XAI Reliability Results

The latest SHAP and LIME evaluation was regenerated for the updated models.

The following results are based on **four sampled test records per disease**, making them preliminary rather than comprehensive reliability estimates.

| Disease | SHAP–LIME Top-5 Agreement | SHAP Stability | LIME Stability | Mean LIME R² |
|---|---:|---:|---:|---:|
| Breast Cancer | 0.190 | 1.000 | 0.607 | 0.835 |
| Diabetes | **0.607** | 0.875 | **0.917** | 0.738 |
| Heart Disease | **0.607** | 1.000 | **0.917** | **0.936** |
| Chronic Kidney Disease | 0.646 | 0.958 | 0.637 | 0.622 |
| Parkinson's Disease | 0.488 | 1.000 | 0.589 | 0.454 |

### Explanation Reliability Metrics

**1. SHAP–LIME Agreement**

Measures the overlap between the top five features identified by SHAP and LIME using Jaccard similarity.

**2. SHAP Stability**

Evaluates the consistency of SHAP explanations under the evaluation's perturbation procedure.

**3. LIME Stability**

Evaluates whether LIME explanations remain consistent across repeated evaluations.

**4. Local Surrogate Fidelity**

Uses LIME local R² to measure how closely the interpretable surrogate approximates the original model around a selected observation.

**5. Feature-Replacement Fidelity**

Examines how model predictions change when influential features are replaced compared with reference feature replacements.

### Key Observations

- **Breast Cancer:** Shows strong predictive performance in baseline evaluation, but relatively low SHAP–LIME agreement.
- **Diabetes:** Achieves a SHAP–LIME agreement score of 0.607, with relatively stable LIME explanations.
- **Heart Disease:** Achieves high SHAP stability and strong LIME local fidelity.
- **Chronic Kidney Disease:** Shows moderate explanation agreement, while local surrogate fidelity requires further improvement.
- **Parkinson's Disease:** Has relatively low LIME local R², indicating that local surrogate explanations should be interpreted cautiously.

These metrics evaluate explanation behaviour, not clinical correctness.

---

## 🧪 Model Validation and Data Quality Audits

### Cross-Validation

Cross-validation evaluates the consistency of predictive performance across different training partitions.

The evaluation considers:

- ROC-AUC
- PR-AUC
- Balanced accuracy
- Recall
- Specificity
- Precision

The updated recall-focused experiment uses five-fold stratified training cross-validation.

### CKD Data Audit

The Chronic Kidney Disease dataset was investigated for duplicate records, conflicting labels and missingness-related predictive signals.

| Audit Metric | Result |
|---|---:|
| Total Records | 400 |
| Unique Feature Rows | 400 |
| Duplicate Feature Rows | 0 |
| Train–Test Duplicate Overlap | 0 |
| Conflicting Labels | 0 |
| Missingness-Only ROC-AUC | 0.862 |

The audit found no exact duplicate leakage.

However, missingness patterns contain substantial predictive information, meaning that near-perfect CKD benchmark performance requires cautious interpretation.

### Parkinson's Subject-Level Evaluation

Parkinson's recordings were evaluated using subject-aware validation to reduce overlap between recordings belonging to the same subject.

Additional candidate models were compared, including regularised Logistic Regression, Random Forest and XGBoost.

No alternative satisfied the predefined improvement criteria, so the existing trained model was retained.

---

## 🏗️ Project Structure

```text
HealthExplain-AI/
│
├── app_multi.py
│   └── Four-page Streamlit dashboard
│
├── run_all_diseases.py
│   └── Data preparation, model training and XAI evaluation
│
├── improve_recall.py
│   └── Diabetes and Heart Disease recall improvement
│
├── final_validation.py
│   └── Cross-validation and robustness experiments
│
├── reliable_xai.py
│   └── Explanation reliability evaluation
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

Some data, model and report files are generated during training and evaluation. The required saved artifacts must be available when running the dashboard.

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

The Streamlit application will normally open at:

```text
http://localhost:8501
```

### 5. Train Models (Only If Required)

If trained model files are unavailable, the original training workflow can be executed using:

```bash
python run_all_diseases.py
```

**Important:** Running this command without `--skip-training` can overwrite the updated recall-improved models.

### 6. Regenerate XAI Evaluations Without Retraining

```bash
python run_all_diseases.py --only diabetes --skip-training
python run_all_diseases.py --only heart_disease --skip-training
```

These commands regenerate explanation evaluation results using the saved models.

### 7. Run Additional Validation

```bash
python final_validation.py
```

Additional auditing scripts:

```bash
python fix_model_issues.py
python reliable_xai.py
```

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

The deployment requires:

- Trained model files
- Processed datasets
- Evaluation reports
- Python dependencies

The Streamlit application link is provided at the beginning of this README. The live deployment should be checked to confirm that it uses the latest committed code.

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

Although HealthExplain-AI provides an integrated prediction and explainability framework, several limitations remain.

- The models are trained on public benchmark datasets rather than independently collected clinical data.
- Dataset size, sampling and missingness patterns may affect generalisation.
- The improved Diabetes recall is accompanied by lower specificity.
- CKD's near-perfect benchmark classification results require external validation.
- Parkinson's model performance varies across subject-grouped validation folds.
- LIME local fidelity is relatively weak for CKD and Parkinson's Disease.
- The latest XAI reliability results are based on a small number of sampled records.
- SHAP and LIME explanations describe model behaviour and do not establish medical causality.
- The application is intended for educational and research purposes only.

---

## 🔮 Future Enhancements

- Integration of additional disease datasets.
- External validation using independently collected clinical datasets.
- Improved probability calibration and risk estimation.
- More robust explanation techniques.
- Expanded XAI reliability evaluation using larger test samples.
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
