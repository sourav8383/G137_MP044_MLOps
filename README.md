# G137_MP044_MLOps

## Village Milk Collection — MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) project for predicting daily milk yield at village milk collection centres.

## Project Overview

This project focuses on building a reliable machine learning pipeline to predict milk yield using historical milk collection and sensor-related data.

The project follows a staged MLOps workflow covering data auditing, preprocessing, model training, experiment tracking, model deployment, and monitoring.

## Problem Statement

Accurately predicting milk collection quantities can help collection centres plan operations, manage resources, and identify unusual collection patterns.

The objective is to develop a regression model that predicts milk yield using the available input features.

## Dataset

The project uses three datasets:

| Dataset | Purpose |
|---|---|
| `train.csv` | Historical training data |
| `later.csv` | Later-period data for evaluation and drift monitoring |
| `holdout_features.csv` | Holdout feature data |

The raw training and later-period datasets are managed using DVC.

## Project Workflow

### Stage 1 — Data Audit

The initial stage involved examining the datasets for quality issues, including:

- Duplicate records
- Missing and sentinel values
- Data leakage
- Stuck sensor readings
- Inconsistent values and data formats

The audit identified 32 duplicate records and 35 sentinel values. The cleaning analysis also examined potential sensor and data consistency issues.

The original findings are documented in `findings.yaml`.

### Stage 2 — Model Development and MLflow Tracking

This stage focuses on preprocessing, model training, experiment tracking, and model registration.

**Data preprocessing**
- Applied the approved cleaning operations.
- Handled missing and sentinel values.
- Selected approved input features.
- Used chronological data splitting for model validation.

**Models evaluated**

Five regression configurations were evaluated:

| Model | Validation MAE |
|---|---:|
| Linear Regression | 29.289 |
| Ridge Regression (alpha = 1) | 29.305 |
| Ridge Regression (alpha = 10) | 29.513 |
| Gradient Boosting Regressor | 35.271 |
| Random Forest Regressor | 44.844 |

MAE represents Mean Absolute Error. Lower values indicate better prediction accuracy.

**Best model**

Linear Regression achieved the lowest validation MAE of approximately 29.289.

**MLflow experiment tracking**

- Experiment name: `mp044-milk-collection`
- Tracking backend: SQLite (`mlflow.db`)
- Total experiments runs: 5
- Registered model: `mp044-milk-model`
- Registered version: 1
- Model alias: `champion`

The training pipeline logs model parameters, evaluation metrics, and the SHA-256 hash of the training dataset.

**Testing**

The Stage 2 test suite completed successfully:

`10 passed`

### Stage 3 — Model Serving and Deployment

Planned work includes:
- FastAPI prediction service
- Input validation
- Docker containerization
- Continuous integration
- Kubernetes deployment

### Stage 4 — Monitoring and Retraining

Planned work includes:
- Data drift monitoring using PSI
- Model performance evaluation
- Retraining with later-period data
- Holdout prediction and final evaluation

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Testing | Pytest |
| Configuration | YAML |
| Model Storage | Joblib |
| Version Control | Git, GitHub |

## Repository Structure

```text
G137_MP044_MLOps/
│
├── data/
│   ├── raw/
│   │   ├── train.csv.dvc
│   │   ├── later.csv.dvc
│   │   └── holdout_features.csv
│   │
│   └── .gitignore
│
├── notebooks/
│
├── src/
│   └── mp044/
│       ├── __init__.py
│       ├── data_audit.py
│       ├── cleaning.py
│       ├── config.py
│       ├── modeling.py
│       └── train.py
│
├── tests/
│   ├── conftest.py
│   ├── test_cleaning.py
│   └── test_modeling.py
│
├── models/
│   └── model.joblib
│
├── reports/
│   └── v1_training_summary.json
│
├── .dvc/
├── .gitignore
├── .dvcignore
├── findings.yaml
├── params.yaml
├── requirements.txt
├── pytest.ini
├── mlflow.db
└── README.md
```

## Running the Stage 2 Pipeline

### 1. Clone the repository

```bash
git clone https://github.com/sourav8383/G137_MP044_MLOps.git
cd G137_MP044_MLOps
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Restore DVC-managed data

```bash
dvc pull
```

This requires the project data to be available through a configured DVC remote.

### 5. Run the training pipeline

```bash
$env:PYTHONPATH = "src"
python -m mp044.train
```

### 6. Run tests

```bash
python -m pytest tests -v
```

## Team Contributions

The project is developed collaboratively, with responsibilities divided across the four project stages.

- **Stage 1:** Data auditing and quality assessment
- **Stage 2:** Data preprocessing, model development, MLflow experiment tracking, and model registration
- **Stage 3:** API development, containerization, and deployment
- **Stage 4:** Monitoring, drift detection, and retraining

## Project Status

| Stage | Status |
|---|---|
| Stage 1 — Data Audit | Completed |
| Stage 2 — Model Development and MLflow | Completed |
| Stage 3 — Serving and Deployment | In progress / pending |
| Stage 4 — Monitoring and Retraining | Pending |

## Repository

[GitHub Repository — G137_MP044_MLOps](https://github.com/sourav8383/G137_MP044_MLOps)