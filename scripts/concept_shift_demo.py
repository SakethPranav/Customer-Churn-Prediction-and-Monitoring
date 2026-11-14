"""Concept-shift demo: normal inputs, shifted outcomes.

Sends distinct held-out test rows through the real API, then reports outcome
labels that follow a shifted behaviour (scripts/concept_shift.py). The inputs
stay normal, so only the live quality metrics should react.
Run with --dry-run first: it samples and shifts but does not call the API.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from scripts.concept_shift import SHIFTS, apply_shift  # noqa: E402

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TARGET = "Churn"
CHUNK = 100
FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"]


def post(path: str, payload: dict) -> dict:
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def fail(message: str) -> int:
    print("ERROR:", message)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", required=True, choices=sorted(SHIFTS))
    ap.add_argument("--rows", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv("data/processed/test.csv")
    if args.rows > len(df):
        return fail(f"--rows {args.rows} exceeds the {len(df)} test rows")
    pos = np.random.default_rng(args.seed).permutation(len(df))[:args.rows]
    assert len(set(pos.tolist())) == len(pos), "sampled rows must be distinct"
    rows = df.iloc[pos].reset_index(drop=True)
    X = rows[FEATURES]
    y_true = rows[TARGET].to_numpy().astype(int)
    y_new = apply_shift(args.shift, X, y_true, seed=args.seed)
    print(f"shift={args.shift} rows={len(rows)} distinct test positions={len(set(pos.tolist()))}")
    print(f"churn rate: unshifted {y_true.mean():.3f}, shifted {y_new.mean():.3f}; "
          f"labels changed: {np.mean(y_true != y_new):.3f}")
    if args.dry_run:
        print("dry run: the API was not called")
        return 0

    try:
        perf = requests.get(f"{API_URL}/performance", timeout=30).json()
    except requests.RequestException as err:
        return fail(f"cannot read the API at {API_URL}: {err}")
    if perf["labelled_total"] != 0:
        return fail(f"the prediction log already holds {perf['labelled_total']} "
        "labelled predictions; use a clean experiment log first")

    records = X.to_dict("records")
    ids = []
    for start in range(0, len(records), CHUNK):
        data = post("/predict_batch", {"customers": records[start:start + CHUNK]})
        ids.extend(p["prediction_id"] for p in data["predictions"])
    if len(ids) != len(records) or None in ids or len(set(ids)) != len(ids):
        return fail("prediction IDs are missing or not unique; nothing was labelled")
    print(f"sent {len(ids)} predictions, {len(set(ids))} unique IDs")

    labelled = 0
    for start in range(0, len(ids), CHUNK):
        items = [{"prediction_id": i, "churned": int(label)}
                 for i, label in zip(ids[start:start + CHUNK], y_new[start:start + CHUNK])]
        res = post("/feedback", {"items": items})
        if res["updated"] != len(items) or res["unknown"]:
            return fail(f"feedback mismatch: updated {res['updated']} of {len(items)}, "
                        f"unknown {res['unknown'][:3]}")
        labelled += res["updated"]
    print(f"reported {labelled} shifted outcomes")

    perf = requests.get(f"{API_URL}/performance", timeout=30).json()
    keys = ("status", "n_labelled", "labelled_total", "positives", "recall",
            "precision", "pr_auc", "pr_auc_floor")
    print("performance:", {k: perf[k] for k in keys})
    return 0


if __name__ == "__main__":
    sys.exit(main())
