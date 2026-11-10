"""Feature engineering shared by the training pipeline and the prediction API."""
import pandas as pd

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
