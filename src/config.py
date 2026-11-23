
from pathlib import Path


# ==========================================
# PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ==========================================
# DISEASE CONFIGURATIONS
# ==========================================

DISEASE_CONFIG = {

    "breast_cancer": {
        "name": "Breast Cancer",
        "dataset_source": "sklearn",
        "dataset_file": None,
        "target_column": "target",
        "negative_label": "Benign",
        "positive_label": "Malignant",
        "task": "classification"
    },

    "diabetes": {
        "name": "Diabetes",
        "dataset_source": "csv",
        "dataset_file": "diabetes.csv",
        "target_column": "Outcome",
        "negative_label": "No Diabetes",
        "positive_label": "Diabetes",
        "task": "classification"
    },

    "heart_disease": {
        "name": "Heart Disease",
        "dataset_source": "csv",
        "dataset_file": "heart_disease.csv",
        "target_column": "target",
        "negative_label": "No Heart Disease",
        "positive_label": "Heart Disease",
        "task": "classification"
    },

    "kidney_disease": {
        "name": "Chronic Kidney Disease",
        "dataset_source": "csv",
        "dataset_file": "kidney_disease.csv",
        "target_column": "classification",
        "negative_label": "No CKD",
        "positive_label": "CKD",
        "task": "classification"
    },

    "parkinsons": {
        "name": "Parkinson's Disease",
        "dataset_source": "csv",
        "dataset_file": "parkinsons.csv",
        "target_column": "status",
        "negative_label": "Healthy",
        "positive_label": "Parkinson's",
        "task": "classification"
    }
}


# ==========================================
# GET DISEASE CONFIGURATION
# ==========================================

def get_disease_config(disease):

    if disease not in DISEASE_CONFIG:
        raise ValueError(
            f"Unsupported disease: {disease}. "
            f"Available diseases: "
            f"{list(DISEASE_CONFIG.keys())}"
        )

    config = DISEASE_CONFIG[disease].copy()

    # Add paths specific to this disease
    if config["dataset_file"] is not None:
        config["raw_data_path"] = (
            RAW_DATA_DIR / config["dataset_file"]
        )
    else:
        config["raw_data_path"] = None

    config["model_path"] = (
        MODEL_DIR / f"{disease}_model.joblib"
    )

    config["train_data_path"] = (
        PROCESSED_DATA_DIR / f"{disease}_train.csv"
    )

    config["test_data_path"] = (
        PROCESSED_DATA_DIR / f"{disease}_test.csv"
    )

    config["report_dir"] = (
        REPORT_DIR / disease
    )

    return config


# ==========================================
# LIST AVAILABLE DISEASES
# ==========================================

def get_available_diseases():

    return {
        disease: config["name"]
        for disease, config in DISEASE_CONFIG.items()
    }


# ==========================================
# CREATE REQUIRED DIRECTORIES
# ==========================================

def create_project_directories():

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        REPORT_DIR,
        FIGURE_DIR
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    for disease in DISEASE_CONFIG:

        disease_report_dir = (
            REPORT_DIR / disease
        )

        disease_report_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ==========================================
# TEST CONFIGURATION
# ==========================================

if __name__ == "__main__":

    create_project_directories()

    print("\n========== DISEASE CONFIGURATION ==========")

    diseases = get_available_diseases()

    for disease_id, disease_name in diseases.items():

        config = get_disease_config(disease_id)

        print(f"\nDisease: {disease_name}")
        print(f"ID: {disease_id}")
        print(f"Dataset Source: {config['dataset_source']}")
        print(f"Target Column: {config['target_column']}")
        print(f"Model Path: {config['model_path']}")

    print("\nTotal Diseases:", len(diseases))

    print("\nConfiguration Loaded Successfully!")
