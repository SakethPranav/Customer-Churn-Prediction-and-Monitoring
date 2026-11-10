"""Stateless feature engineering stage, parallelised with Ray."""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level features. Each row is processed independently."""
    df = df.copy()
    # Average spend per month of tenure (tenure 0 treated as 1 month).
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    # Number of add-on services the customer has.
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLUMNS)
    df["num_services"] += (df["InternetService"] != "No").astype(int)
    # Tenure buckets.
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)
    # Whether the customer pays automatically.
    df["is_autopay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    return df


@ray.remote
def process_partition(part: pd.DataFrame) -> pd.DataFrame:
    """Ray task: engineer features for one chunk of rows."""
    return engineer_features(part)


def run_ray(df: pd.DataFrame, n_partitions: int) -> pd.DataFrame:
    """Split df into chunks, process them in parallel, and reassemble."""
    bounds = np.linspace(0, len(df), n_partitions + 1, dtype=int)
    parts = [df.iloc[bounds[i]:bounds[i + 1]] for i in range(n_partitions)]
    results = ray.get([process_partition.remote(p) for p in parts])
    return pd.concat(results, ignore_index=True)


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["featurize"]

    try:
        df = pd.read_csv(params["input_path"])
        log.info("Loaded cleaned data: %s", df.shape)

        # Baseline: single-process pandas.
        t0 = time.perf_counter()
        expected = engineer_features(df).reset_index(drop=True)
        pandas_s = time.perf_counter() - t0

        # Ray version.
        t0 = time.perf_counter()
        ray.init(num_cpus=params["num_cpus"], include_dashboard=False,
                 logging_level="ERROR", ignore_reinit_error=True)
        init_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = run_ray(df, params["n_partitions"])
        ray_s = time.perf_counter() - t0
        ray.shutdown()

        # Both paths must give identical output.
        pd.testing.assert_frame_equal(expected, result)
        log.info("Ray output matches pandas output")
    except Exception:
        log.exception("Feature engineering failed")
        sys.exit(1)

    metrics = {
        "rows": len(df),
        "n_partitions": params["n_partitions"],
        "pandas_seconds": round(pandas_s, 4),
        "ray_processing_seconds": round(ray_s, 4),
        "ray_init_seconds": round(init_s, 4),
        "ray_rows_per_second": round(len(df) / ray_s, 1),
    }
    Path(params["metrics_path"]).write_text(json.dumps(metrics, indent=2))
    log.info("Timing metrics: %s", metrics)

    result.to_csv(params["output_path"], index=False)
    log.info("Saved features: %s -> %s", result.shape, params["output_path"])


if __name__ == "__main__":
    main()
