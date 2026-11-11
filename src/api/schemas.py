"""Request and response schemas for the churn prediction API."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """Raw customer attributes, as recorded by the business."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5, "TotalCharges": 191.0}})

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    """Churn prediction for one customer."""

    churn_probability: float = Field(description="Probability between 0 and 1")
    churn_prediction: Literal[0, 1] = Field(description="1 if probability >= 0.5")
    risk_tier: Literal["low", "medium", "high"]
    prediction_id: str | None = Field(
        default=None,
        description="ID to quote when reporting the true outcome; null if not logged")


class PredictionResponse(BaseModel):
    prediction: Prediction
    model_version: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    model_version: str | None = None


class DriftResponse(BaseModel):
    """Input drift of recent requests against the training data."""

    window_size: int = Field(description="Requests currently in the window")
    min_window: int = Field(description="Rows needed before PSI is reported")
    ready: bool
    max_psi: float = Field(description="Largest per-feature PSI")
    status: Literal["insufficient_data", "ok", "warn", "alert"]
    feature_psi: dict[str, float]


class FeedbackItem(BaseModel):
    prediction_id: str
    churned: Literal[0, 1] = Field(description="1 if the customer churned")


class FeedbackRequest(BaseModel):
    """True outcomes for earlier predictions, reported later."""

    items: list[FeedbackItem] = Field(min_length=1, max_length=1000)


class FeedbackResponse(BaseModel):
    updated: int = Field(description="Predictions that received an outcome")
    unknown: list[str] = Field(description="IDs that were not found")


class PerformanceResponse(BaseModel):
    """Live model quality on the most recently labelled predictions."""

    ready: bool = Field(description="False until enough labelled rows of both classes")
    n_labelled: int = Field(description="Labelled rows in the window")
    positives: int
    min_labelled: int
    threshold: float
    recall: float | None
    precision: float | None
    pr_auc: float | None
    predictions_logged: int
    labelled_total: int
    window: int = Field(description="Labelled rows the metrics are computed over")
    pr_auc_floor: float = Field(description="Live PR-AUC below this means degraded")
    status: Literal["insufficient_data", "ok", "degraded"]


class LabelledRow(BaseModel):
    """A served prediction together with the outcome reported for it."""

    prediction_id: str
    model_version: str | None
    probability: float = Field(description="Churn probability that was served")
    features: Customer
    label: Literal[0, 1] = Field(description="1 if the customer churned")
    labelled_at: float = Field(description="Unix time the outcome was reported")


class LabelledResponse(BaseModel):
    """Labelled predictions for retraining, oldest outcome first."""

    count: int
    rows: list[LabelledRow]
