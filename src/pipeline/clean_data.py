"""Data validation and cleaning stage for the churn pipeline."""
import logging
import sys

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
}


def validate_schema(df: pd.DataFrame) -> None:
    """Fail loudly if the input does not have the expected columns."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {sorted(missing)}")


def clean(df: pd.DataFrame, target: str, id_column: str) -> pd.DataFrame:
    """Clean the raw Telco data."""
    df = df.copy()

    # TotalCharges is text with blanks; convert and fill blanks with 0
    # (these are customers with tenure 0 who have not been billed yet).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = int(df["TotalCharges"].isna().sum())
    log.info("Blank TotalCharges rows: %d", n_blank)
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target as 1/0.
    df[target] = (df[target].str.strip() == "Yes").astype(int)

    # Remove duplicate customers (same customerID), before dropping the ID.
    before = len(df)
    df = df.drop_duplicates(subset=[id_column])
    log.info("Dropped %d duplicate customer rows", before - len(df))

    # Drop the identifier; it carries no predictive signal.
    df = df.drop(columns=[id_column])
    return df


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["data"]

    try:
        df = pd.read_csv(params["raw_path"])
        log.info("Loaded raw data: %s", df.shape)
        validate_schema(df)
        cleaned = clean(df, params["target"], params["id_column"])
    except Exception:
        log.exception("Data cleaning failed")
        sys.exit(1)

    cleaned.to_csv(params["processed_path"], index=False)
    log.info("Saved cleaned data: %s -> %s", cleaned.shape, params["processed_path"])


if __name__ == "__main__":
    main()
