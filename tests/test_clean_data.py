
import pandas as pd
import pytest

from src.pipeline.clean_data import EXPECTED_COLUMNS, clean, validate_schema


def make_raw_data():
    """Create a small raw dataset containing all expected source columns."""
    data = {
        column: ["value_1", "value_2", "value_3"]
        for column in EXPECTED_COLUMNS
    }

    data["customerID"] = ["C001", "C002", "C003"]
    data["TotalCharges"] = ["100.50", " ", "250.00"]
    data["Churn"] = ["Yes", "No", " Yes "]

    return pd.DataFrame(data)


def test_valid_schema_passes():
    df = make_raw_data()

    # Should not raise an exception.
    validate_schema(df)


def test_missing_column_raises_error():
    df = make_raw_data()
    df = df.drop(columns=["MonthlyCharges"])

    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df)


def test_total_charges_are_numeric_and_missing_values_filled():
    df = make_raw_data()

    result = clean(df, target="Churn", id_column="customerID")

    assert result["TotalCharges"].tolist() == [100.50, 0.0, 250.00]
    assert pd.api.types.is_numeric_dtype(result["TotalCharges"])


def test_target_is_encoded_as_binary():
    df = make_raw_data()

    result = clean(df, target="Churn", id_column="customerID")

    assert result["Churn"].tolist() == [1, 0, 1]


def test_duplicate_customers_are_removed():
    df = make_raw_data()

    # Add a duplicate of the first customer.
    duplicate = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate], ignore_index=True)

    result = clean(df, target="Churn", id_column="customerID")

    assert len(result) == 3


def test_customer_id_is_removed():
    df = make_raw_data()

    result = clean(df, target="Churn", id_column="customerID")

    assert "customerID" not in result.columns


def test_clean_does_not_modify_original_dataframe():
    df = make_raw_data()
    original = df.copy(deep=True)

    clean(df, target="Churn", id_column="customerID")

    pd.testing.assert_frame_equal(df, original)
