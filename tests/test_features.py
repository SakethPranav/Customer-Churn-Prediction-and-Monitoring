
import pandas as pd
import pytest

from src.common.features import engineer_features


@pytest.fixture
def sample_customers():
    return pd.DataFrame({
        "tenure": [12, 0, 30],
        "TotalCharges": [1200.0, 50.0, 3000.0],

        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["Yes", "No phone service", "No"],
        "OnlineSecurity": ["Yes", "No internet service", "No"],
        "OnlineBackup": ["No", "No internet service", "Yes"],
        "DeviceProtection": ["Yes", "No internet service", "No"],
        "TechSupport": ["No", "No internet service", "Yes"],
        "StreamingTV": ["Yes", "No internet service", "No"],
        "StreamingMovies": ["No", "No internet service", "Yes"],

        "InternetService": ["Fiber optic", "No", "DSL"],
        "PaymentMethod": [
            "Bank transfer (automatic)",
            "Electronic check",
            "Credit card (automatic)"
        ]
    })


def test_engineered_columns_are_created(sample_customers):
    result = engineer_features(sample_customers)

    expected = {
        "avg_monthly_spend",
        "num_services",
        "tenure_group",
        "is_autopay"
    }

    assert expected.issubset(result.columns)


def test_average_monthly_spend(sample_customers):
    result = engineer_features(sample_customers)

    assert result["avg_monthly_spend"].tolist() == pytest.approx(
        [100.0, 50.0, 100.0]
    )


def test_zero_tenure_does_not_cause_division_by_zero(sample_customers):
    result = engineer_features(sample_customers)

    assert result.loc[1, "avg_monthly_spend"] == 50.0
    assert pd.notna(result.loc[1, "avg_monthly_spend"])


def test_tenure_groups(sample_customers):
    result = engineer_features(sample_customers)

    assert result["tenure_group"].iloc[0] == "0-12"
    assert result["tenure_group"].iloc[1] == "0-12"
    assert result["tenure_group"].iloc[2] == "25-48"


def test_autopay_feature(sample_customers):
    result = engineer_features(sample_customers)

    assert result["is_autopay"].tolist() == [1, 0, 1]


def test_num_services(sample_customers):
    result = engineer_features(sample_customers)

    assert result["num_services"].tolist() == [6, 0, 5]


def test_original_dataframe_is_not_modified(sample_customers):
    original = sample_customers.copy(deep=True)

    engineer_features(sample_customers)

    pd.testing.assert_frame_equal(sample_customers, original)
