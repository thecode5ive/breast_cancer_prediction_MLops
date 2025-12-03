Breast Cancer Prediction MLOps Project
Project Overview

This project implements an end-to-end machine learning pipeline for predicting breast cancer malignancy using supervised learning (Logistic Regression). The pipeline is designed following MLOps best practices: from data ingestion → model training → versioning → CI/CD → deployment via API and frontend.

Pipeline Architecture
+------------------+        +----------------+        +-----------------+        +-----------------+        +-------------------+
|                  |        |                |        |                 |        |                 |        |                   |
|  Data Ingestion  +------->|  Training      +------->|  Model Version  +------->|  CI/CD Pipeline |------->|  Inference API &  |
|  (CSV, Excel)    |        |  & Evaluation  |        |  & Artifact     |        |  (GitHub Actions)|        |  Streamlit UI     |
|                  |        |  (train.py)    |        |  Management     |        |                 |        |                   |
+------------------+        +----------------+        +-----------------+        +-----------------+        +-------------------+

Pipeline Components & Tool Choices
| Component                            | Tool / Library                                        | Justification                                                                                                              |
| ------------------------------------ | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Data Ingestion**                   | `pandas`                                              | Handles structured CSV/Excel data efficiently. Easy integration with preprocessing and model training.                     |
| **Data Preprocessing**               | `scikit-learn` (`StandardScaler`, `train_test_split`) | Standard ML preprocessing methods, ensures feature scaling and train-test split reproducibility.                           |
| **Model Training**                   | `scikit-learn` (`LogisticRegression`)                 | Lightweight, interpretable, and sufficient for binary classification (benign vs malignant).                                |
| **Metrics & Logging**                | `joblib`, `json`, `numpy`                             | Saves trained models, experiment metrics, and results in a structured, reusable format.                                    |
| **Versioning / Experiment Tracking** | GitHub + Git branches                                 | Enables reproducible experiments, version control for code, and collaboration.                                             |
| **CI/CD**                            | GitHub Actions                                        | Automates training, evaluation, artifact storage, and ensures reproducibility whenever code is pushed or merged.           |
| **Inference Service**                | `FastAPI`                                             | Lightweight REST API framework for serving the model; supports JSON input/output for integration with frontend.            |
| **Frontend Interface**               | `Streamlit`                                           | Simple and interactive UI for non-technical users to input features and get predictions without running the model locally. |
| **Artifacts / Model Storage**        | Local `models/` folder (can extend to MLflow)         | Keeps trained model versions and results for reproducibility and deployment.                                               |


How the Pipeline Works

Data Ingestion & Preprocessing:

Raw breast cancer dataset (CSV/XLSX) is loaded.

Preprocessing includes scaling features and splitting into training and test sets.

Model Training & Evaluation:

train.py trains a Logistic Regression model.

Metrics such as accuracy, log-loss, and RMSE are calculated.

Model and metrics are saved locally.

Versioning & Experiment Tracking:

GitHub tracks code changes and experiment scripts.

Separate branches can track experimental models (experiment/add-rmse).

CI/CD Automation:

GitHub Actions workflow triggers on push/merge.

Installs dependencies, runs train.py, and uploads model and metrics as workflow artifacts.

Inference Service & Frontend:

FastAPI exposes /predict and /health endpoints.

Streamlit frontend interacts with the API, allowing users to input features and receive real-time predictions.

Project Structure
breast_cancer_prediction_MLops/
│
├─ backend/
│   └─ app.py          # FastAPI inference service
├─ frontend/
│   └─ app.py          # Streamlit UI
├─ src/
│   ├─ train.py        # Model training script
│   └─ utils.py        # Data loading & preprocessing
├─ models/             # Saved trained models
├─ results/            # Metrics JSON files
├─ logs/               # Experiment logs
├─ .github/workflows/  # GitHub Actions CI/CD pipelines
├─ requirements.txt
├─ README.md
└─ .gitignore
