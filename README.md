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

| Part | Files |
|---|---|
| FastAPI prediction service | `src/mp044/api.py`, `src/mp044/serving.py` |
| Input validation | `src/mp044/schemas.py` |
| Docker | `Dockerfile`, `.dockerignore`, `requirements-api.txt` |
| Continuous integration | `.github/workflows/ci.yml` |
| Kubernetes | `k8s/deployment.yaml`, `k8s/service.yaml` |
| Tests | `tests/test_serving.py`, `tests/test_api.py` |

The API serves `models/model.joblib` (the promoted champion from Stage 4).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | 200 when the model is loaded, otherwise 503 |
| GET | `/model-info` | Model file, SHA-256 and feature list |
| POST | `/predict` | Prediction for one record |
| POST | `/predict/batch` | Predictions for 1 to 1000 records |

Invalid input returns HTTP 422: negative counts, unknown breed or cluster, missing fields, and extra fields such as the leaking column `litres_paid_in_passbook`. `fodder_price_rs` may be `null` or `-999` (treated as missing).

**Run locally**

```bash
pip install -r requirements.txt
uvicorn mp044.api:app --app-dir src --port 8000
```

Then open `http://127.0.0.1:8000/docs` and use **Try it out** on `/predict`.

**Run with Docker**

```bash
docker build -t mp044-api:1.0.0 .
docker run --rm -p 8000:8000 mp044-api:1.0.0
```

**Deploy on Kubernetes (minikube)**

```bash
minikube start
minikube image load mp044-api:1.0.0
kubectl apply -f k8s/
kubectl rollout status deployment/mp044-api
kubectl port-forward service/mp044-api 8000:80
```

**CI:** every push and pull request runs the tests, builds the image, deploys it to a temporary `kind` cluster and calls `/health` and `/predict`.

### Stage 4 — Monitoring and Retraining

Implemented in `src/mp044/monitoring.py` (pure helpers) and `src/mp044/monitor.py` (pipeline):
- **Drift:** PSI (10 quantile bins of the cleaned `train.csv`) for the six numeric features against `later.csv`; the feature with the highest PSI is `most_drifted_feature`. Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major.
- **Performance:** MAE of v1 on all of `later.csv`, per `village_cluster`; the cluster with the highest MAE is `worst_served_group`.
- **Retraining:** v2 = same pipeline recipe as v1, fitted on cleaned train + the oldest 70% of `later.csv` (by date).
- **Decision:** v1 and v2 are both scored on the newest 30% of `later.csv`; `promote` only if v2's MAE is at least 8% lower than v1's (the client's rule), otherwise `keep`. v2 is always registered in MLflow (version 2); the `champion` alias moves to it only when promoted. A promoted v2 is registered in MLflow with the `champion` alias and replaces `models/model.joblib`.
- **Holdout:** predictions from the winning model are written to `predictions/holdout_predictions.csv` (columns `record_id,prediction`).
- Results are written to `reports/stage4_monitoring.json`, logged to MLflow, and copied into the `monitoring:` block of `findings.yaml`.

**Stage 4 results (v1 = champion v1, v2 = train + first 70% of later.csv):**

| Item | Value |
|---|---|
| Most drifted feature | `buffaloes` (PSI 4.276; every other feature below 0.07) |
| v1 MAE on later.csv | Cluster A 61.61, Cluster B 59.61, Cluster C 342.43 |
| Worst-served group | Cluster C |
| MAE on last 30% of later.csv | v1 159.83, v2 97.43 (about 39% lower) |
| Decision | promote v2 (champion version 2) |

Run (after placing `train.csv` and `later.csv` in `data/raw/`):

```bash
PYTHONPATH=src python -m mp044.monitor
pytest
```

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
| Stage 3 — Serving and Deployment | Implemented |
| Stage 4 — Monitoring and Retraining | Completed (see results below) |

## Repository

[GitHub Repository — G137_MP044_MLOps](https://github.com/sourav8383/G137_MP044_MLOps)