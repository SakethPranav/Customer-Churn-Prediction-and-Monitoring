
import json

import numpy as np
import pandas as pd
import pytest

from src.api.prediction_log import PredictionLog


@pytest.fixture
def log(tmp_path):
    """Create an isolated temporary prediction database."""
    return PredictionLog(str(tmp_path / "test_predictions.db"))


def add_sample_predictions(log, n=20, model_version="v1"):
    """Insert a predictable set of test predictions."""
    rows = [
        {"tenure": i + 1, "MonthlyCharges": 50.0 + i}
        for i in range(n)
    ]

    # Alternate between low and high churn probabilities.
    probabilities = [
        0.9 if i % 2 == 0 else 0.1
        for i in range(n)
    ]

    ids = log.add(rows, probabilities, model_version)

    # Even indices are actual churn cases.
    labels = [
        (pid, 1 if i % 2 == 0 else 0)
        for i, pid in enumerate(ids)
    ]

    log.label(labels)

    return ids, rows, probabilities, labels


def test_database_is_created(log):
    assert log.counts() == {
        "predictions": 0,
        "labelled": 0,
    }


def test_add_creates_unique_prediction_ids(log):
    rows = [
        {"tenure": 12},
        {"tenure": 24},
        {"tenure": 36},
    ]

    ids = log.add(rows, [0.2, 0.5, 0.8], "v1")

    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_add_stores_predictions_and_counts(log):
    rows = [
        {"tenure": 12},
        {"tenure": 24},
    ]

    log.add(rows, [0.2, 0.8], "v1")

    assert log.counts() == {
        "predictions": 2,
        "labelled": 0,
    }


def test_label_updates_known_predictions(log):
    ids = log.add(
        [{"tenure": 12}, {"tenure": 24}],
        [0.2, 0.8],
        "v1",
    )

    result = log.label([
        (ids[0], 1),
        (ids[1], 0),
    ])

    assert result == {"updated": 2, "unknown": []}
    assert log.counts()["labelled"] == 2


def test_label_reports_unknown_prediction_ids(log):
    ids = log.add([{"tenure": 12}], [0.8], "v1")

    result = log.label([
        (ids[0], 1),
        ("unknown_prediction_id", 0),
    ])

    assert result["updated"] == 1
    assert result["unknown"] == ["unknown_prediction_id"]


def test_latest_feedback_wins(log):
    ids = log.add([{"tenure": 12}], [0.8], "v1")

    log.label([(ids[0], 1)])
    log.label([(ids[0], 0)])

    X, y = log.labelled()

    assert len(y) == 1
    assert y.iloc[0] == 0


def test_performance_is_not_ready_with_insufficient_labels(log):
    ids = log.add(
        [{"tenure": i} for i in range(5)],
        [0.8] * 5,
        "v1",
    )

    log.label([(pid, 1) for pid in ids])

    result = log.performance(
        window=100,
        threshold=0.5,
        min_labelled=10,
        min_per_class=2,
    )

    assert result["ready"] is False
    assert result["recall"] is None
    assert result["precision"] is None
    assert result["pr_auc"] is None


def test_performance_calculates_metrics(log):
    add_sample_predictions(log, n=20)

    result = log.performance(
        window=100,
        threshold=0.5,
        min_labelled=10,
        min_per_class=5,
    )

    assert result["ready"] is True
    assert result["n_labelled"] == 20
    assert result["positives"] == 10
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert result["pr_auc"] == 1.0


def test_performance_filters_by_model_version(log):
    add_sample_predictions(log, n=20, model_version="v1")

    rows = [{"tenure": i} for i in range(20)]
    probabilities = [0.9 if i % 2 == 0 else 0.1 for i in range(20)]
    ids = log.add(rows, probabilities, "v2")
    log.label([(pid, 1 if i % 2 == 0 else 0) for i, pid in enumerate(ids)])

    result = log.performance(
        window=100,
        threshold=0.5,
        min_labelled=10,
        min_per_class=5,
        model_version="v1",
    )

    assert result["ready"] is True
    assert result["n_labelled"] == 20
    assert result["positives"] == 10


def test_export_returns_labelled_records(log):
    ids, rows, probabilities, labels = add_sample_predictions(log, n=5)

    exported = log.export()

    assert len(exported) == 5
    assert exported[0]["prediction_id"] in ids
    assert exported[0]["model_version"] == "v1"
    assert exported[0]["label"] in [0, 1]
    assert isinstance(exported[0]["features"], dict)


def test_labelled_returns_features_and_targets(log):
    ids, rows, probabilities, labels = add_sample_predictions(log, n=10)

    X, y = log.labelled()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 10
    assert len(y) == 10
    assert list(X.columns) == ["tenure", "MonthlyCharges"]
    assert set(y.unique()) == {0, 1}


def test_empty_labelled_data_returns_empty_objects(log):
    X, y = log.labelled()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.empty
    assert y.empty
