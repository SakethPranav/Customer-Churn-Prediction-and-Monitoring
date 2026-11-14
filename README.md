
# Customer Churn Prediction & MLOps Platform

An end-to-end machine learning and MLOps project for predicting customer churn, serving predictions through an API, monitoring model and data quality, and supporting controlled model retraining and promotion.

The project demonstrates the complete lifecycle of a machine learning application — from data preparation and experiment tracking to deployment, feedback collection, drift detection, and retraining.

## 1. Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to discontinue their services allows businesses to prioritize retention efforts.

This project builds a customer churn prediction platform using the Telco Customer Churn dataset.

The system provides:

- Data cleaning and feature engineering.
- Distributed feature processing using Ray.
- Model training and experiment tracking using MLflow.
- Model registration and champion model management.
- A FastAPI prediction service.
- A Streamlit user interface for single and batch predictions.
- Prediction logging and feedback collection.
- Data drift monitoring using Population Stability Index (PSI).
- Live model performance monitoring using labelled predictions.
- Automated retraining with statistical promotion safeguards.
- Metrics collection and visualization using Prometheus and Grafana.
- Automated tests and GitHub Actions CI.

## 2. Model Performance

The training pipeline evaluates Logistic Regression, Random Forest, and XGBoost using five-fold stratified cross-validation.

The model with the highest cross-validated PR-AUC is selected as the champion candidate.

### Final model: Random Forest

| Metric | Result |
|---|---:|
| Cross-validation PR-AUC | 0.6640 ± 0.0193 |
| Test ROC-AUC | 0.8430 |
| Test PR-AUC | 0.6508 |
| Test Recall | 0.7995 |
| Test Precision | 0.5301 |
| Test F1-score | 0.6375 |

The model achieved approximately 80% recall at the default classification threshold of 0.5.

This means it identifies a substantial proportion of actual churners, while its precision indicates that retention interventions should be prioritized rather than applied indiscriminately.

The reported test metrics come from the project's held-out test split. They should not be interpreted as guaranteed production performance.

## 3. System Architecture

The application consists of five containerized services.

| Component | Technology | Responsibility |
|---|---|---|
| Frontend | Streamlit | User interface and prediction workflows |
| Backend | FastAPI | Prediction API, logging, feedback and monitoring |
| Model Registry | MLflow | Experiment tracking and champion model management |
| Metrics Collection | Prometheus | Scraping application metrics |
| Monitoring Dashboard | Grafana | Visualization of operational and model metrics |

### Architecture Flow

```text
                  User
                   |
                   v
            Streamlit Frontend
              Port 8501
                   |
                   v
             FastAPI Backend
              Port 8000
                   |
          +--------+--------+
          |                 |
          v                 v
     MLflow Registry    Prediction Log
       Port 5000          SQLite
          |
          v
     Champion Model

FastAPI Metrics
      |
      v
  Prometheus
  Port 9091
      |
      v
   Grafana
  Port 3001
```

### Training Pipeline

```text
Raw Telco Dataset
        |
        v
   Clean Data
        |
        v
  Feature Engineering
     (Ray)
        |
        v
   Train/Test Split
        |
        v
 Model Training
        |
        v
 Cross-validation
        |
        v
 MLflow Experiment Tracking
        |
        v
 Model Registration
        |
        v
 Champion Alias
```

The training pipeline is managed using DVC.

## 4. Technology Stack

**Programming and data processing**
- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Ray

**Machine Learning and MLOps**
- MLflow
- DVC
- Joblib

**Application and deployment**
- FastAPI
- Streamlit
- Docker
- Docker Compose

**Monitoring**
- Prometheus
- Grafana
- SQLite

**Testing and automation**
- Pytest
- GitHub Actions

## 5. Repository Structure

```text
churn-mlops/
|
├── .github/
│   └── workflows/
│       └── ci.yml
|
├── data/
│   ├── raw/
│   └── processed/
|
├── docker/
│   ├── backend/
│   ├── frontend/
│   ├── mlflow/
│   ├── prometheus/
│   └── grafana/
|
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── requirements.txt
|
├── src/
│   ├── api/
│   ├── common/
│   ├── monitoring/
│   ├── pipeline/
│   └── training/
|
├── scripts/
│   ├── retrain_trigger.py
│   ├── concept_shift.py
│   ├── concept_shift_demo.py
│   └── simulate_traffic.py
|
├── tests/
│
├── metrics/
├── models/
├── monitoring/
├── docs/
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── requirements-dev.txt
├── docker-compose.yml
└── README.md
```

## 6. Getting Started

### Prerequisites

Install the following:

- Python 3.12
- Docker
- Docker Compose
- Git
- DVC

### Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd churn-mlops
```

### Set Up the Python Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Dataset Setup

The project uses the Telco Customer Churn dataset.

The raw dataset is managed using DVC. The Git repository contains the DVC pointer and pipeline metadata, but the configured DVC remote in the development environment is local to the original machine.

Consequently, a fresh clone cannot retrieve the dataset from that local remote.

Before running the pipeline, obtain the compatible dataset and place it at:

```text
data/raw/telco_churn.csv
```

Alternatively, configure an accessible DVC remote containing the required data and retrieve it using:

```bash
dvc pull
```

The dataset must have the columns expected by the cleaning and feature engineering stages.

### Start MLflow

The training pipeline registers its selected model in MLflow. Start the MLflow service first:

```bash
docker compose up -d mlflow
```

Wait for the MLflow service to become healthy.

### Run the DVC Pipeline

```bash
dvc repro
```

The pipeline executes:

1. Data cleaning.
2. Feature engineering.
3. Train/test splitting.
4. Model training and evaluation.
5. Reference statistics generation.

The training stage logs experiments to MLflow, registers the selected model as `churn-model`, and assigns the `champion` alias.

The initial model must be registered before the prediction backend can serve predictions.

### Start the Complete Application

```bash
docker compose up --build -d
```

Check service status:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f backend
```

## 7. Application URLs

| Service | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Swagger API Documentation | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9091 |
| Grafana | http://localhost:3001 |

Grafana uses local development credentials configured in Docker Compose:

- Username: `admin`
- Password: `admin`

Change these credentials before exposing the application outside a trusted local environment.

## 8. Prediction API

The FastAPI backend provides endpoints for prediction, feedback, monitoring, and model management.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Application and model health |
| POST | `/predict` | Single customer prediction |
| POST | `/predict_batch` | Batch predictions |
| POST | `/feedback` | Submit actual churn outcome |
| GET | `/labelled` | Retrieve labelled prediction data |
| GET | `/performance` | Live model performance |
| GET | `/drift` | Data drift status |
| GET | `/metrics` | Prometheus metrics |
| POST | `/reload` | Reload the champion model |

Interactive API documentation is available at `/docs`.

Predictions are logged with unique identifiers. These identifiers allow later feedback to be associated with the original prediction.

## 9. Data Drift Monitoring

The application monitors changes in incoming customer data using Population Stability Index (PSI).

The reference distributions are generated from training data and stored in:

```text
monitoring/reference_stats.json
```

The monitor evaluates numerical and categorical features.

### PSI Thresholds

| PSI | Status |
|---|---|
| Below 0.10 | Normal |
| 0.10 to below 0.25 | Warning |
| 0.25 or above | Alert |

The configured drift window is 300 rows, with a minimum of 300 observations required before a drift assessment is considered ready.

An insufficient number of observations is reported separately rather than being treated as evidence of healthy data.

## 10. Live Model Performance Monitoring

The system supports feedback-based monitoring.

When actual churn outcomes become available, they can be submitted through the `/feedback` endpoint.

The application calculates:

- Recall
- Precision
- PR-AUC
- Number of labelled predictions
- Positive and negative class counts

The configured performance monitoring settings are:

| Parameter | Value |
|---|---:|
| Evaluation window | 500 labelled predictions |
| Minimum labelled predictions | 500 |
| Minimum samples per class | 10 |
| PR-AUC warning floor | 0.54 |

Performance is not evaluated until the minimum data requirements are satisfied.

Prediction logs are stored in SQLite using a persistent Docker volume, allowing feedback and prediction history to survive container recreation.

## 11. Automated Retraining and Model Promotion

The project includes a retraining workflow that evaluates whether a new model should replace the current champion.

The retraining process:

1. Collects labelled prediction data.
2. Checks minimum data requirements.
3. Combines historical training data with weighted live data.
4. Trains challenger models.
5. Evaluates the champion and challenger on a holdout set.
6. Uses paired bootstrap comparisons to assess the improvement.
7. Promotes the challenger only if the promotion conditions are satisfied.
8. Reloads the model in the API after successful promotion.

### Promotion Safeguards

| Parameter | Configuration |
|---|---:|
| Minimum labelled rows | 500 |
| Live-data weight | 10 |
| Holdout rows | 200 |
| Required PR-AUC gain | 0.02 |
| Bootstrap samples | 1,000 |
| Cooldown | 300 new labels |

The promotion gate requires:

- A PR-AUC improvement of at least 0.02.
- A positive fifth percentile of the paired bootstrap improvement distribution.

This prevents a model from being promoted solely because of a small or statistically uncertain improvement.

The retraining workflow has been tested for both acceptance and rejection paths. The latest controlled retraining run rejected the challenger because it did not demonstrate the required improvement. The existing champion remained unchanged.

A successful production promotion has not yet been demonstrated.

## 12. Monitoring Dashboard

Prometheus collects metrics exposed by the FastAPI backend.

Grafana visualizes operational and model monitoring information.

The dashboard includes 13 panels covering application health, prediction activity, latency, model status, drift, and live performance.

Access the dashboard at:

```text
http://localhost:3001
```

## 13. Automated Testing

The project contains automated tests covering data processing, feature engineering, drift monitoring, prediction logging, API behavior, retraining logic, and retraining workflow integration.

Run the complete test suite:

```bash
python -m pytest -v
```

Latest local test result:

```text
66 passed
```

The test modules include:

- `test_clean_data.py`
- `test_features.py`
- `test_drift.py`
- `test_prediction_log.py`
- `test_api.py`
- `test_retraining.py`
- `test_retrain_workflow.py`

GitHub Actions is configured to run the test suite on pushes and pull requests targeting `master` or `main`.

## 14. Important Implementation Notes and Limitations

- The current DVC remote is local to the original development environment. A shared remote is required for complete data reproducibility by other users.
- The initial MLflow model must be registered before the backend can serve predictions.
- The reported test metrics are from the held-out test set and do not guarantee future production performance.
- The latest retraining experiment correctly rejected a challenger; successful production promotion remains to be demonstrated.
- The concept-shift demonstration is a controlled experiment, not an independent production evaluation.
- The application is designed as a local development and demonstration environment. Production deployment would require stronger authentication, secrets management, access controls, and operational hardening.
- Grafana's default development credentials must be changed before external exposure.

## 15. Future Improvements

- Configure a shared remote DVC storage backend.
- Add a fully reproducible fresh-install deployment workflow.
- Demonstrate a successful model promotion using an independent evaluation dataset.
- Add authentication and role-based access control.
- Introduce scheduled or event-driven retraining.
- Add richer model explainability and feature-level monitoring.
- Improve dashboard coverage and deployment automation.

## 16. Author

Developed as an individual end-to-end MLOps project demonstrating machine learning engineering, model deployment, monitoring, and controlled retraining.
