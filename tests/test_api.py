
import json
import os
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Configure isolated files BEFORE importing the API module.
TEST_DIR = tempfile.mkdtemp(prefix="churn_api_tests_")

REFERENCE_FILE = os.path.join(TEST_DIR, "reference_stats.json")
PREDICTION_DB = os.path.join(TEST_DIR, "predictions.db")

with open(REFERENCE_FILE, "w") as f:
    json.dump({
        "features": {
            "Contract": {
                "type": "categorical",
                "proportions": {
                    "Month-to-month": 0.5,
                    "One year": 0.5
                }
            }
        },
        "settings": {
            "window_size": 10,
            "min_window": 1,
            "psi_warn": 0.1,
            "psi_alert": 0.25,
            "perf_window": 10,
            "perf_min_labelled": 2,
            "perf_min_per_class": 1,
            "perf_pr_auc_floor": 0.54
        }
    }, f)

os.environ["REFERENCE_PATH"] = REFERENCE_FILE
os.environ["PREDICTION_LOG_PATH"] = PREDICTION_DB

from src.api import main as api


class MockModel:
    """A predictable model for testing API behaviour."""

    def predict_proba(self, df):
        probabilities = np.full(len(df), 0.8)
        return np.column_stack([1 - probabilities, probabilities])


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create an isolated API test environment."""

    api.prediction_log = api.PredictionLog(
        str(tmp_path / "predictions.db")
    )
    api.monitor = api.DriftMonitor()

    api.state["model"] = MockModel()
    api.state["version"] = "test-v1"

    # Prevent the real MLflow model from loading during startup.
    monkeypatch.setattr(api, "load_model", lambda: None)

    with TestClient(api.app) as test_client:
        yield test_client

    api.state["model"] = None
    api.state["version"] = None


def valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.5,
        "TotalCharges": 906.0
    }


def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "test-v1"


def test_single_prediction(client):
    response = client.post(
        "/predict",
        json=valid_customer()
    )

    assert response.status_code == 200
    data = response.json()

    assert data["model_version"] == "test-v1"
    assert data["prediction"]["churn_probability"] == 0.8
    assert data["prediction"]["churn_prediction"] == 1
    assert data["prediction"]["risk_tier"] == "high"
    assert data["prediction"]["prediction_id"] is not None


def test_batch_prediction(client):
    customers = [valid_customer(), valid_customer()]

    response = client.post(
        "/predict_batch",
        json={"customers": customers}
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["predictions"]) == 2
    assert data["model_version"] == "test-v1"
    assert all(
        p["prediction_id"] is not None
        for p in data["predictions"]
    )


def test_invalid_customer_is_rejected(client):
    customer = valid_customer()
    customer["Contract"] = "Invalid contract"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_empty_batch_is_rejected(client):
    response = client.post(
        "/predict_batch",
        json={"customers": []}
    )

    assert response.status_code == 422


def test_prediction_fails_when_model_is_not_loaded(client):
    api.state["model"] = None

    response = client.post(
        "/predict",
        json=valid_customer()
    )

    assert response.status_code == 503


def test_feedback_updates_prediction(client):
    prediction = client.post(
        "/predict",
        json=valid_customer()
    ).json()["prediction"]

    response = client.post(
        "/feedback",
        json={
            "items": [{
                "prediction_id": prediction["prediction_id"],
                "churned": 1
            }]
        }
    )

    assert response.status_code == 200
    assert response.json() == {
        "updated": 1,
        "unknown": []
    }


def test_feedback_reports_unknown_id(client):
    response = client.post(
        "/feedback",
        json={
            "items": [{
                "prediction_id": "unknown-id",
                "churned": 1
            }]
        }
    )

    assert response.status_code == 200
    assert response.json()["updated"] == 0
    assert response.json()["unknown"] == ["unknown-id"]


def test_performance_endpoint(client):
    # Generate two predictions and attach one positive and one negative label.
    first = client.post(
        "/predict", json=valid_customer()
    ).json()["prediction"]

    second = client.post(
        "/predict", json=valid_customer()
    ).json()["prediction"]

    client.post("/feedback", json={
        "items": [
            {"prediction_id": first["prediction_id"], "churned": 1},
            {"prediction_id": second["prediction_id"], "churned": 0}
        ]
    })

    response = client.get("/performance")

    assert response.status_code == 200
    data = response.json()

    assert data["ready"] is True
    assert data["n_labelled"] == 2
    assert data["positives"] == 1
    assert data["status"] in ["ok", "degraded"]


def test_drift_endpoint(client):
    # A prediction adds a row to the drift monitor.
    client.post("/predict", json=valid_customer())

    response = client.get("/drift")

    assert response.status_code == 200
    data = response.json()

    assert data["ready"] is True
    assert data["window_size"] == 1
    assert data["status"] in ["ok", "warn", "alert"]
    assert "Contract" in data["feature_psi"]


def test_labelled_endpoint(client):
    prediction = client.post(
        "/predict", json=valid_customer()
    ).json()["prediction"]

    client.post("/feedback", json={
        "items": [{
            "prediction_id": prediction["prediction_id"],
            "churned": 1
        }]
    })

    response = client.get("/labelled")

    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 1
    assert len(data["rows"]) == 1
    assert data["rows"][0]["label"] == 1


def test_prometheus_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "churn_api_requests_total" in response.text


def test_reload_endpoint(client, monkeypatch):
    def fake_load_model():
        api.state["model"] = MockModel()
        api.state["version"] = "test-v2"

    monkeypatch.setattr(api, "load_model", fake_load_model)

    response = client.post("/reload")

    assert response.status_code == 200
    assert response.json() == {
        "status": "reloaded",
        "model_version": "test-v2"
    }
