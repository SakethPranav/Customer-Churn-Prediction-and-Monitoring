"""REST client for the churn prediction API.

This is the ONLY place the frontend talks to the backend. The address comes
from the API_URL environment variable, so no code changes are needed when
the services move into Docker.
"""
import os

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 30
BATCH_LIMIT = 1000  # must match the API's max batch size


class ApiError(Exception):
    """An error whose message is safe to show directly to the end user."""


def _explain_validation(detail: list, row_offset: int = 0) -> str:
    """Turn FastAPI's 422 detail into one readable sentence."""
    if not detail:
        return "The data was rejected by the service."
    first = detail[0]
    loc = [str(p) for p in first.get("loc", []) if p != "body"]
    where = ""
    if len(loc) >= 2 and loc[0] == "customers" and loc[1].isdigit():
        where = f"row {int(loc[1]) + row_offset + 1}, field '{loc[-1]}': "
    elif loc:
        where = f"field '{loc[-1]}': "
    return f"Invalid data ({where}{first.get('msg', 'invalid value')})."


def _call(method: str, path: str, row_offset: int = 0, **kwargs) -> dict:
    try:
        resp = requests.request(method, f"{API_URL}{path}",
                                timeout=TIMEOUT_SECONDS, **kwargs)
    except requests.ConnectionError:
        raise ApiError(f"Cannot reach the prediction service at {API_URL}. "
                       "Please check that it is running.")
    except requests.Timeout:
        raise ApiError("The prediction service took too long to respond.")
    if resp.status_code == 422:
        raise ApiError(_explain_validation(resp.json().get("detail", []), row_offset))
    if resp.status_code == 503:
        raise ApiError("The prediction model is not loaded yet. Try again shortly.")
    if not resp.ok:
        raise ApiError(f"The service reported an error (HTTP {resp.status_code}).")
    return resp.json()


def health() -> dict:
    return _call("GET", "/health")


def predict(customer: dict) -> dict:
    """Returns {'prediction': {...}, 'model_version': '...'}."""
    return _call("POST", "/predict", json=customer)


def predict_batch(customers: list[dict]) -> tuple[list[dict], str]:
    """Score any number of customers, sending them in chunks of BATCH_LIMIT."""
    predictions, version = [], ""
    for start in range(0, len(customers), BATCH_LIMIT):
        chunk = customers[start:start + BATCH_LIMIT]
        data = _call("POST", "/predict_batch", row_offset=start,
                     json={"customers": chunk})
        predictions.extend(data["predictions"])
        version = data["model_version"]
    return predictions, version
