"""FastAPI service that serves the champion churn model from MLflow."""
import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.tracking import MlflowClient
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)

from src.api.drift_monitor import DriftMonitor
from src.api.prediction_log import PredictionLog
from src.api.schemas import (BatchRequest, BatchResponse, Customer,
                             DriftResponse, FeedbackRequest, FeedbackResponse,
                             HealthResponse, LabelledResponse,
                             PerformanceResponse, Prediction,
                             PredictionResponse)
from src.common.features import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("churn-api")

# Configuration comes from environment variables (set by docker-compose).
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
DECISION_THRESHOLD = 0.5
MEDIUM_RISK, HIGH_RISK = 0.3, 0.6
PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "prediction_log/predictions.db")

state: dict = {"model": None, "version": None}

# --- Prometheus metrics ---------------------------------------------------
REQUESTS = Counter("churn_api_requests_total", "HTTP requests handled",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("churn_api_request_latency_seconds", "Request latency in seconds",
                    ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
PREDICTIONS = Counter("churn_predictions_total", "Predictions made, by risk tier",
                      ["risk_tier"])
PROBABILITY = Histogram("churn_prediction_probability",
                        "Distribution of predicted churn probabilities",
                        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
MODEL_LOADED = Gauge("churn_model_loaded", "1 if a model is loaded, else 0")
MODEL_VERSION = Gauge("churn_model_version", "Registered version of the loaded model")
MODEL_LOADED.set(0)
PSI_GAUGE = Gauge("churn_feature_psi",
                  "PSI of live inputs against the training reference", ["feature"])
PSI_MAX = Gauge("churn_drift_max_psi", "Largest per-feature PSI in the live window")
WINDOW_ROWS = Gauge("churn_drift_window_rows", "Customers currently in the drift window")
LIVE_RECALL = Gauge("churn_live_recall", "Recall on recently labelled predictions")
LIVE_PRECISION = Gauge("churn_live_precision",
                       "Precision on recently labelled predictions")
LIVE_PR_AUC = Gauge("churn_live_pr_auc", "PR-AUC on recently labelled predictions")
LABELLED_ROWS = Gauge("churn_labelled_predictions",
                      "Predictions that have a reported outcome")
monitor = DriftMonitor()
prediction_log = PredictionLog(PREDICTION_LOG_PATH)


def load_model() -> None:
    """Load the champion model from the MLflow registry into memory."""
    mlflow.set_tracking_uri(TRACKING_URI)
    state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    state["version"] = str(mv.version)
    MODEL_LOADED.set(1)
    MODEL_VERSION.set(float(mv.version))
    log.info("Loaded %s v%s (alias %s)", MODEL_NAME, state["version"], MODEL_ALIAS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception:
        # Stay up so /health can report the problem instead of crashing.
        log.exception("Could not load model from %s", TRACKING_URI)
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    """Record request count and latency for every call except /metrics."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        endpoint = route.path if route else "unmatched"
        if endpoint != "/metrics":
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(time.perf_counter() - start)


def risk_tier(p: float) -> str:
    """Map a churn probability to a business-friendly tier."""
    if p >= HIGH_RISK:
        return "high"
    return "medium" if p >= MEDIUM_RISK else "low"


def score(customers: list[Customer]) -> list[Prediction]:
    """Apply feature engineering and the model to a list of customers."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        df = engineer_features(pd.DataFrame([c.model_dump() for c in customers]))
        text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        df = df.astype({c: "object" for c in text_cols})
        probs = state["model"].predict_proba(df)[:, 1]
    except Exception:
        log.exception("Scoring failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    rows = [c.model_dump() for c in customers]
    monitor.add(rows)
    try:
        ids = prediction_log.add(rows, probs, state["version"])
    except Exception:
        log.exception("Could not write to the prediction log")
        ids = [None] * len(rows)
    results = []
    for pid, p in zip(ids, probs):
        tier = risk_tier(float(p))
        PROBABILITY.observe(float(p))
        PREDICTIONS.labels(tier).inc()
        results.append(Prediction(prediction_id=pid,
                                  churn_probability=round(float(p), 4),
                                  churn_prediction=int(p >= DECISION_THRESHOLD),
                                  risk_tier=tier))
    return results


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Report whether the service is up and the model is loaded."""
    loaded = state["model"] is not None
    return HealthResponse(status="ok" if loaded else "degraded",
                          model_loaded=loaded, model_name=MODEL_NAME,
                          model_version=state["version"])


def publish_drift() -> dict:
    """Compute the current drift report and publish it as Prometheus gauges."""
    report = monitor.report()
    for name, value in report["feature_psi"].items():
        PSI_GAUGE.labels(name).set(value)
    PSI_MAX.set(report["max_psi"])
    WINDOW_ROWS.set(report["window_size"])
    return report


def publish_performance() -> dict:
    """Compute live performance of the served model and publish it as gauges."""
    cfg = monitor.settings
    report = prediction_log.performance(
        cfg["perf_window"], DECISION_THRESHOLD, cfg["perf_min_labelled"],
        cfg["perf_min_per_class"], model_version=state["version"])
    counts = prediction_log.counts()
    nan = float("nan")  # shown as "no data" until enough labels exist
    LIVE_RECALL.set(report["recall"] if report["ready"] else nan)
    LIVE_PRECISION.set(report["precision"] if report["ready"] else nan)
    LIVE_PR_AUC.set(report["pr_auc"] if report["ready"] else nan)
    LABELLED_ROWS.set(counts["labelled"])
    if not report["ready"]:
        status = "insufficient_data"
    else:
        status = "degraded" if report["pr_auc"] < cfg["perf_pr_auc_floor"] else "ok"
    return {**report, "status": status, "window": cfg["perf_window"],
            "pr_auc_floor": cfg["perf_pr_auc_floor"],
            "predictions_logged": counts["predictions"],
            "labelled_total": counts["labelled"]}


@app.get("/drift", response_model=DriftResponse, tags=["monitoring"])
def drift() -> DriftResponse:
    """Input drift (PSI) of the most recent requests against the training data."""
    return DriftResponse(**publish_drift())


@app.get("/metrics", tags=["system"])
def metrics() -> Response:
    """Prometheus metrics in the text exposition format."""
    publish_drift()
    try:
        publish_performance()
    except Exception:
        log.exception("Could not publish live performance gauges")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: Customer) -> PredictionResponse:
    """Predict churn for a single customer."""
    t0 = time.perf_counter()
    result = score([customer])[0]
    log.info("predict prob=%.4f tier=%s latency_ms=%.1f", result.churn_probability,
             result.risk_tier, (time.perf_counter() - t0) * 1000)
    return PredictionResponse(prediction=result, model_version=state["version"])


@app.post("/predict_batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Predict churn for up to 1000 customers at once."""
    t0 = time.perf_counter()
    results = score(request.customers)
    log.info("predict_batch n=%d latency_ms=%.1f", len(results),
             (time.perf_counter() - t0) * 1000)
    return BatchResponse(predictions=results, model_version=state["version"])


@app.post("/feedback", response_model=FeedbackResponse, tags=["monitoring"])
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Report true outcomes (1 = churned) for earlier predictions, by prediction ID."""
    try:
        result = prediction_log.label(
            [(item.prediction_id, item.churned) for item in request.items])
    except Exception:
        log.exception("Saving feedback failed")
        raise HTTPException(status_code=500, detail="Could not save feedback")
    publish_performance()
    return FeedbackResponse(**result)


@app.get("/labelled", response_model=LabelledResponse, tags=["monitoring"])
def labelled(limit: int = 10000) -> LabelledResponse:
    """Labelled predictions with their features, oldest outcome first (for retraining)."""
    limit = max(1, min(limit, 100000))
    try:
        rows = prediction_log.export(limit)
    except Exception:
        log.exception("Exporting labelled rows failed")
        raise HTTPException(status_code=500, detail="Could not read the prediction log")
    return LabelledResponse(count=len(rows), rows=rows)


@app.get("/performance", response_model=PerformanceResponse, tags=["monitoring"])
def performance() -> PerformanceResponse:
    """Recall, precision and PR-AUC of the served model on labelled predictions."""
    return PerformanceResponse(**publish_performance())


@app.post("/reload", tags=["system"])
def reload_model() -> dict:
    """Reload the champion model (used after retraining promotes a new one)."""
    try:
        load_model()
        monitor.reload_reference()
    except Exception:
        log.exception("Reload failed")
        raise HTTPException(status_code=500, detail="Model reload failed")
    return {"status": "reloaded", "model_version": state["version"]}
