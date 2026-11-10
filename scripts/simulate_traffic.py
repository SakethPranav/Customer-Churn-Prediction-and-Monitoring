"""Send simulated traffic to the API: normal, or deliberately drifted.

Rows come from the held-out test set. In drift mode customers are made
newer, pricier and all on month-to-month contracts.
"""
import argparse
import os
import sys

import pandas as pd
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"]


def load(n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv("data/processed/test.csv")
    return (df[FEATURES].sample(n=n, replace=n > len(df), random_state=seed)
            .reset_index(drop=True))


def drift(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["tenure"] = (d["tenure"] * 0.4).round().astype(int)
    d["MonthlyCharges"] = (d["MonthlyCharges"] * 1.3).round(2)
    d["Contract"] = "Month-to-month"
    d["TotalCharges"] = (d["tenure"] * d["MonthlyCharges"]).round(2)
    return d


def send(df: pd.DataFrame) -> None:
    for start in range(0, len(df), 100):
        chunk = df.iloc[start:start + 100].to_dict("records")
        resp = requests.post(f"{API_URL}/predict_batch",
                             json={"customers": chunk}, timeout=30)
        resp.raise_for_status()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["normal", "drift"], default="normal")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = load(args.n, args.seed)
    if args.mode == "drift":
        df = drift(df)
    try:
        send(df)
        rep = requests.get(f"{API_URL}/drift", timeout=10).json()
    except requests.RequestException as err:
        print("API error:", err)
        sys.exit(1)
    top = sorted(rep["feature_psi"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    print(f"sent {len(df)} {args.mode} rows | window={rep['window_size']} "
          f"status={rep['status']} max_psi={rep['max_psi']}")
    print("top features:", ", ".join(f"{k}={v}" for k, v in top))


if __name__ == "__main__":
    main()
