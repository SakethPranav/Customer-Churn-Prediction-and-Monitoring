"""FastAPI service that serves the champion churn model from MLflow."""
import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from src.api.schemas import (BatchRequest, BatchResponse, Customer,
                             HealthResponse, Prediction, PredictionResponse)
from src.common.features import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("churn-api")

# Configuration comes from environment variables (set by docker-compose later).
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
DECISION_THRESHOLD = 0.5
MEDIUM_RISK, HIGH_RISK = 0.3, 0.6

state: dict = {"model": None, "version": None}


def load_model() -> None:
    """Load the champion model from the MLflow registry into memory."""
    mlflow.set_tracking_uri(TRACKING_URI)
    state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    state["version"] = str(mv.version)
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
    return [Prediction(churn_probability=round(float(p), 4),
                       churn_prediction=int(p >= DECISION_THRESHOLD),
                       risk_tier=risk_tier(float(p))) for p in probs]


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Report whether the service is up and the model is loaded."""
    loaded = state["model"] is not None
    return HealthResponse(status="ok" if loaded else "degraded",
                          model_loaded=loaded, model_name=MODEL_NAME,
                          model_version=state["version"])


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


@app.post("/reload", tags=["system"])
def reload_model() -> dict:
    """Reload the champion model (used after retraining promotes a new one)."""
    try:
        load_model()
    except Exception:
        log.exception("Reload failed")
        raise HTTPException(status_code=500, detail="Model reload failed")
    return {"status": "reloaded", "model_version": state["version"]}
