
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.training import retrain


@pytest.fixture
def workflow(monkeypatch):
    """Isolate retrain.main() from real data, MLflow and API services."""

    cfg = {
        "data": {"target": "Churn"},
        "monitoring": {"perf_min_per_class": 10},
        "split": {"train_path": "unused.csv"},
        "train": {
            "tracking_uri": "unused",
            "experiment_name": "test",
            "registered_model_name": "test-model",
            "random_state": 42,
            "cv_folds": 2,
            "models": {"logistic_regression": {}},
        },
        "retrain": {
            "api_url": "http://unused",
            "min_labelled": 220,
            "holdout_rows": 200,
            "live_weight": 10.0,
            "promote_margin": 0.02,
            "bootstrap_samples": 20,
            "max_live_rows": 1000,
        },
    }

    monkeypatch.setattr(retrain.yaml, "safe_load", lambda _: cfg)

    # 20 live training rows followed by 200 holdout rows.
    rows = (
    [{"label": i % 2, "features": {}} for i in range(20)]
    + [{"label": 0, "features": {}} for _ in range(100)]
    + [{"label": 1, "features": {}} for _ in range(100)])
    monkeypatch.setattr(retrain, "fetch_labelled", lambda *a: rows)

    X_base = pd.DataFrame({"feature": np.arange(40)})
    y_base = pd.Series([0, 1] * 20)
    monkeypatch.setattr(
        retrain, "load_xy", lambda *a: (X_base.copy(), y_base.copy())
    )

    def fake_to_frame(selected_rows, columns, text_cols):
        n = len(selected_rows)
        X = pd.DataFrame({"feature": np.arange(n)})
        y = pd.Series([int(row["label"]) for row in selected_rows])
        return X, y

    monkeypatch.setattr(retrain, "to_frame", fake_to_frame)

    monkeypatch.setattr(
        retrain, "weighted_cv_pr_auc", lambda *a: (0.7, 0.01)
    )
    monkeypatch.setattr(retrain, "make_estimator", lambda *a: object())

    # Holdout labels are balanced: 100 negatives and 100 positives.
    y_hold = np.array([0] * 100 + [1] * 100)

    # Champion predictions are deliberately poor.
    old_proba = np.where(y_hold == 1, 0.1, 0.9)

    class FakeChampion:
        def predict_proba(self, X):
            return np.column_stack([1 - old_proba, old_proba])

    class FakeChallenger:
        def fit(self, X, y, **kwargs):
            return self

        def predict_proba(self, X):
            if scenario["name"] == "training_error":
                raise RuntimeError("simulated training error")

            if scenario["name"] == "reject":
                new_proba = old_proba
            else:
                new_proba = np.where(y_hold == 1, 0.9, 0.1)

            return np.column_stack([1 - new_proba, new_proba])

    scenario = {"name": "promote"}

    monkeypatch.setattr(
        retrain, "build_pipeline", lambda *a: FakeChallenger()
    )
    monkeypatch.setattr(
        retrain, "paired_bootstrap",
        lambda *a: np.ones(20) if scenario["name"] != "reject"
        else np.zeros(20),
    )

    # Fake MLflow client and model registry.
    class FakeClient:
        def get_model_version_by_alias(self, *a):
            return SimpleNamespace(version="3")

        def set_registered_model_alias(self, *a):
            scenario["alias_updated"] = True

    monkeypatch.setattr(retrain, "MlflowClient", FakeClient)
    monkeypatch.setattr(retrain.mlflow, "set_tracking_uri", lambda *a: None)
    monkeypatch.setattr(retrain.mlflow, "set_experiment", lambda *a: None)
    monkeypatch.setattr(retrain.mlflow, "start_run", lambda **a: nullcontext())
    monkeypatch.setattr(retrain.mlflow, "set_tags", lambda *a: None)
    monkeypatch.setattr(retrain.mlflow, "log_params", lambda *a: None)
    monkeypatch.setattr(retrain.mlflow, "log_metrics", lambda *a: None)

    monkeypatch.setattr(
        retrain.mlflow.sklearn, "load_model", lambda *a: FakeChampion()
    )
    monkeypatch.setattr(
        retrain.mlflow.sklearn, "log_model",
        lambda *a, **k: SimpleNamespace(model_uri="runs:/fake/model"),
    )
    monkeypatch.setattr(
        retrain, "infer_signature", lambda *a: None
    )
    monkeypatch.setattr(retrain, "git_info", lambda: {})
    monkeypatch.setattr(retrain, "file_md5", lambda *a: "fake-md5")

    def fake_register(*a):
        scenario["registered"] = True
        return SimpleNamespace(version="4")

    monkeypatch.setattr(retrain.mlflow, "register_model", fake_register)

    class FakeResponse:
        def raise_for_status(self):
            if scenario["name"] == "reload_failure":
                raise RuntimeError("simulated reload failure")

    monkeypatch.setattr(
        retrain.requests, "post", lambda *a, **k: FakeResponse()
    )

    return scenario


def get_result(capsys):
    output = capsys.readouterr().out
    result_line = next(
        line for line in output.splitlines() if line.startswith("RESULT ")
    )
    return json.loads(result_line.removeprefix("RESULT "))


def test_insufficient_labelled_data(monkeypatch, capsys, workflow):
    workflow["name"] = "insufficient"
    monkeypatch.setattr(retrain, "fetch_labelled", lambda *a: [])

    code = retrain.main()

    assert code == retrain.EXIT_NOT_ENOUGH_DATA
    assert get_result(capsys)["decision"] == "not_enough_data"
    assert "registered" not in workflow


def test_challenger_rejected_without_champion_change(capsys, workflow):
    workflow["name"] = "reject"

    code = retrain.main()
    result = get_result(capsys)

    assert code == retrain.EXIT_OK
    assert result["decision"] == "rejected"
    assert result["champion_version_before"] == 3
    assert "champion_version_after" not in result
    assert "registered" not in workflow
    assert "alias_updated" not in workflow


def test_challenger_promoted(capsys, workflow):
    workflow["name"] = "promote"

    code = retrain.main()
    result = get_result(capsys)

    assert code == retrain.EXIT_OK
    assert result["decision"] == "promoted"
    assert result["champion_version_after"] == 4
    assert result["reloaded"] is True
    assert workflow["registered"] is True
    assert workflow["alias_updated"] is True


def test_reload_failure_after_promotion(capsys, workflow):
    workflow["name"] = "reload_failure"

    code = retrain.main()
    result = get_result(capsys)

    assert code == retrain.EXIT_RELOAD_FAILED
    assert result["decision"] == "promoted"
    assert result["reloaded"] is False
    assert workflow["registered"] is True
    assert workflow["alias_updated"] is True


def test_unexpected_training_error(workflow):
    workflow["name"] = "training_error"

    code = retrain.main()

    assert code == retrain.EXIT_ERROR