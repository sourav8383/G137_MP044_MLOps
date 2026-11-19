# G137_MP044_MLOps

# Village Milk Collection – MLOps Pipeline

## 1. Project Information

| Field | Details |
|---|---|
| Course | SCSE3040 – MLOps |
| Group ID | G137 |
| Problem ID | MP044 |
| Project Domain | Milk Collection and Dairy Management |
| Client | Indore District Dairy Cooperative |
| Location | Indore, Madhya Pradesh, India |
| Problem Type | Supervised Machine Learning – Regression |
| Evaluation Metric | Mean Absolute Error (MAE) |

## 2. Team Members

| Enrolment Number | Name |
|---|---|
| S24CSEU0569 | Sukhbeer Kaur |
| S24CSEU0754 | Hanu Singhal |
| S24CSEU0557 | Sourav Dash |
| S24CSEU0749 | Hardik Garg |

## 3. Project Overview

The **Village Milk Collection (MP044)** project focuses on developing an end-to-end Machine Learning Operations (MLOps) pipeline for the Indore District Dairy Cooperative, a fictional dairy organisation near Indore, Madhya Pradesh.

The cooperative collects milk from different village centres every morning. It needs to estimate the daily milk collection at each centre so that trucks and chilling capacity can be planned one day in advance.

Accurate predictions can help the cooperative select appropriate truck sizes, reduce unnecessary fuel costs, utilise chilling facilities efficiently, and prevent milk spoilage.

The project uses historical milk collection and operational data to develop a machine learning regression solution. It also incorporates data versioning, model development, experiment tracking, deployment, automated testing, and monitoring to establish a reproducible MLOps workflow.

## 4. Problem Statement

The primary objective is to predict `litres_per_day`, representing the total quantity of milk collected at a village centre in litres.

The prediction is based on operational features such as:

- `cows` – Number of cows registered at the collection centre.
- `buffaloes` – Number of buffaloes registered at the collection centre.
- `fodder_price_rs` – Fodder price in rupees.
- `temp_c` – Temperature in degrees Celsius.
- `vet_visits` – Number of veterinary visits.
- `feed_kg` – Quantity of cattle feed sold at the centre.
- `breed_mix` – Main cattle breed category.
- `village_cluster` – Village cluster associated with the collection centre.

This is a supervised machine learning regression problem.

Model performance is evaluated using **Mean Absolute Error (MAE)**, which measures the average absolute difference between actual and predicted milk collection values.

The final objective is to generate reliable daily milk collection estimates that support operational planning and decision-making.

## 5. Project Objectives

The main objectives of this project are:

1. Understand and analyse the supplied milk collection datasets.
2. Identify and handle data-quality issues such as missing values, duplicate records, sentinel values, and data leakage.
3. Maintain dataset version control using DVC.
4. Develop and evaluate a machine learning regression model.
5. Track model experiments and performance using MLflow.
6. Build a prediction API for serving model predictions.
7. Containerise the application using Docker.
8. Automate testing and integration using GitHub Actions.
9. Configure application deployment using Kubernetes.
10. Monitor data drift and model performance and support model retraining.

## 6. Dataset Description

The project uses three datasets supplied for problem MP044.

| Dataset | Period | Records | Description |
|---|---|---:|---|
| `train.csv` | January–June 2026 | 1,032 | Training dataset containing historical operational features and the target variable. |
| `later.csv` | July–August 2026 | 300 | Later-period dataset used for subsequent evaluation and monitoring. |
| `holdout_features.csv` | September 2026 | 250 | Final prediction dataset containing input features without the target variable. |

### Dataset Features

The datasets contain information related to:

- Cattle population (cows and buffaloes).
- Fodder prices.
- Temperature.
- Veterinary visits.
- Cattle feed quantity.
- Breed information.
- Village cluster.
- Milk collection and passbook records.

The target variable is:

`litres_per_day`

It represents the daily milk collected at each village centre.

**Important:** The `litres_paid_in_passbook` column is identified as a potential data leakage feature because it is highly correlated with the target and is not available in the holdout dataset. It must be excluded from model input features.

## 7. Stage 1 – Data Management and Data Quality Audit

The first stage of the project focuses on understanding the datasets, identifying data-quality issues, cleaning the training data, and establishing dataset version control.

### 7.1 Data Quality Checks

Seven data-quality problems were investigated using Python-based audit functions.

| Data Issue | Finding | Description |
|---|---|---|
| Data Leakage | Detected | `litres_paid_in_passbook` has a correlation of approximately 0.9988 with the target and is unavailable in the holdout dataset. |
| Unit Mix-up | Not Detected | No inconsistent measurement units were identified. |
| Sentinel Values | Detected | The `fodder_price_rs` column contains 35 occurrences of `-999`. |
| Duplicate Rows | Detected | 32 extra duplicate records were identified in the training dataset. |
| Stuck Sensor | Detected | The `cows` column repeatedly reports 162 for clusters A, B, and C during 23 February–18 March 2026. |
| Target Typos | Not Detected | No suspicious target-value typo pattern was identified using the implemented check. |
| Spelling Variants | Not Detected | No spelling variants were identified in the checked categorical values. |

### 7.2 Data Cleaning

The training dataset was cleaned using Python and Pandas.

The following operations were performed:

- Removed duplicate records.
- Replaced the sentinel value `-999` in `fodder_price_rs` with missing values (`NaN`).
- Preserved the original raw dataset.
- Validated the cleaned dataset to ensure the identified issues were handled correctly.

**Cleaning Results**

| Description | Before Cleaning | After Cleaning |
|---|---:|---:|
| Total training records | 1,032 | 1,000 |
| Duplicate records | 32 | 0 |
| Sentinel values in `fodder_price_rs` | 35 | 0 |
| Missing values in `fodder_price_rs` after sentinel replacement | — | 34 |

The sentinel count decreases from 35 to 34 missing values after cleaning because one sentinel-containing row was also a duplicate and was removed during deduplication.

### 7.3 Data Version Control Using DVC

DVC (Data Version Control) was implemented to track the training and later-period datasets.

The following datasets are tracked:

- `data/train.csv`
- `data/later.csv`

DVC metadata files:

- `data/train.csv.dvc`
- `data/later.csv.dvc`

The original training dataset was also verified against the supplied ZIP archive using file hashes.

**Verification Result:** The original ZIP dataset, current training CSV, and DVC-recorded hash matched successfully.

This confirms that the raw training dataset was preserved without unintended modification.

### 7.4 Stage 1 Deliverables

- Dataset inspection and analysis.
- Seven data-quality audit checks.
- Data-cleaning functions.
- Updated `findings.yaml`.
- DVC dataset tracking.
- Stage 1 audit notebook.
- Original dataset integrity verification.

## 8. Overall MLOps Workflow

The project follows an end-to-end MLOps workflow consisting of the following stages:

| Stage | Activity | Technology |
|---|---|---|
| 1 | Data inspection, cleaning, validation, and version control | Python, Pandas, DVC |
| 2 | Model development, evaluation, and experiment tracking | Scikit-learn, MLflow |
| 3 | Prediction API development | FastAPI |
| 4 | Application containerisation | Docker |
| 5 | Automated testing and continuous integration | Pytest, GitHub Actions |
| 6 | Application deployment | Kubernetes |
| 7 | Data drift monitoring, performance evaluation, and retraining | Python, MLflow |

The stages are developed progressively as part of the group project.

## 9. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Version Control | DVC |
| Experiment Tracking | MLflow |
| API Development | FastAPI |
| Containerisation | Docker |
| Version Control | Git, GitHub |
| Continuous Integration | GitHub Actions |
| Deployment | Kubernetes |
| Testing | Pytest |
| Development Environment | Jupyter Notebook, VS Code |

## 10. Repository Structure

The repository is organised to maintain a clear separation between datasets, notebooks, source code, and project configuration files.

```text
G137_MP044_MLOps/
│
├── .dvc/
│
├── data/
│   ├── train.csv
│   ├── train.csv.dvc
│   ├── later.csv
│   ├── later.csv.dvc
│   └── holdout_features.csv
│
├── notebooks/
│   └── stage1_data_audit.ipynb
│
├── src/
│   └── mp044/
│       └── data_audit.py
│
├── findings.yaml
├── .dvcignore
├── .gitignore
├── README.md
└── requirements.txt
```

Additional directories and configuration files for model development, API serving, testing, and deployment can be added as the remaining pipeline stages are implemented.

## 11. Team Contributions

The project is divided among four team members to cover different components of the MLOps pipeline.

The major responsibility areas include:

- Data management and data-quality analysis.
- Machine learning model development and evaluation.
- API and application development.
- MLOps automation, deployment, and monitoring.

Each member contributes to the development and integration of the overall project.

## 12. Current Project Status

| Component | Status |
|---|---|
| Repository setup | Completed |
| Dataset inspection | Completed |
| Seven data-quality checks | Completed |
| Data cleaning implementation | Completed |
| DVC dataset tracking | Completed |
| Stage 1 audit notebook | Completed |
| Original dataset integrity verification | Completed |
| Model development | Remaining project work |
| Experiment tracking | Remaining project work |
| API development | Remaining project work |
| Docker and CI | Remaining project work |
| Kubernetes deployment | Remaining project work |
| Monitoring and retraining | Remaining project work |

*Status reflects the verified Stage 1 data work and the project progress documented at the time of writing. Update it as the group completes subsequent stages.*

## 13. GitHub Repository

**Repository:** https://github.com/sourav8383/G137_MP044_MLOps

The repository contains the project's source code, data audit notebook, data-quality findings, and DVC tracking metadata. It serves as the central location for collaboration, version control, and development of the MLOps pipeline.
