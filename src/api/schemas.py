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
