
import numpy as np
import pytest

from src.training.retrain import paired_bootstrap
from scripts.retrain_trigger import decide


def test_paired_bootstrap_detects_better_challenger():
    y = np.array([0, 1] * 100)

    # Challenger perfectly separates both classes.
    p_new = y.astype(float)

    # Champion assigns higher scores to the wrong class.
    p_old = 1.0 - p_new

    diffs = paired_bootstrap(
        y, p_new, p_old, n=500, seed=42
    )

    assert len(diffs) > 0
    assert np.all(diffs > 0)
    assert np.percentile(diffs, 5) > 0


def test_paired_bootstrap_returns_zero_for_identical_predictions():
    y = np.array([0, 1] * 100)
    probabilities = np.linspace(0.1, 0.9, len(y))

    diffs = paired_bootstrap(
        y, probabilities, probabilities, n=500, seed=42
    )

    assert len(diffs) > 0
    assert np.allclose(diffs, 0.0)


def test_paired_bootstrap_detects_worse_challenger():
    y = np.array([0, 1] * 100)

    # Champion is perfect; challenger is inverted.
    p_old = y.astype(float)
    p_new = 1.0 - p_old

    diffs = paired_bootstrap(
        y, p_new, p_old, n=500, seed=42
    )

    assert len(diffs) > 0
    assert np.all(diffs < 0)


def test_paired_bootstrap_skips_single_class_samples():
    # Every possible bootstrap sample has only one class.
    y = np.ones(20, dtype=int)
    p_new = np.full(20, 0.9)
    p_old = np.full(20, 0.5)

    diffs = paired_bootstrap(
        y, p_new, p_old, n=100, seed=42
    )

    assert len(diffs) == 0


@pytest.fixture
def trigger_config():
    return {
        "min_labelled": 500,
        "cooldown_new_labels": 300,
    }


@pytest.fixture
def healthy_performance():
    return {
        "status": "ok",
        "labelled_total": 1000,
    }


@pytest.fixture
def healthy_drift():
    return {
        "status": "ok",
    }


def test_trigger_skips_when_everything_is_healthy(
    trigger_config, healthy_performance, healthy_drift
):
    action, signal, reason = decide(
        healthy_performance,
        healthy_drift,
        trigger_config,
        {},
    )

    assert action == "skip"
    assert signal is None
    assert "no trigger" in reason


def test_degraded_performance_triggers_retraining(
    trigger_config, healthy_drift
):
    performance = {
        "status": "degraded",
        "labelled_total": 1000,
    }

    action, signal, reason = decide(
        performance,
        healthy_drift,
        trigger_config,
        {},
    )

    assert action == "retrain"
    assert signal == "performance_degraded"


def test_drift_alert_triggers_retraining(
    trigger_config, healthy_performance
):
    drift = {
        "status": "alert",
    }

    action, signal, reason = decide(
        healthy_performance,
        drift,
        trigger_config,
        {},
    )

    assert action == "retrain"
    assert signal == "drift_alert"


def test_performance_degradation_has_priority_over_drift_alert(
    trigger_config
):
    performance = {
        "status": "degraded",
        "labelled_total": 1000,
    }
    drift = {
        "status": "alert",
    }

    action, signal, reason = decide(
        performance,
        drift,
        trigger_config,
        {},
    )

    assert action == "retrain"
    assert signal == "performance_degraded"


def test_insufficient_labelled_data_prevents_retraining(
    trigger_config, healthy_drift
):
    performance = {
        "status": "degraded",
        "labelled_total": 300,
    }

    action, signal, reason = decide(
        performance,
        healthy_drift,
        trigger_config,
        {},
    )

    assert action == "skip"
    assert signal == "performance_degraded"
    assert "only 300 labelled rows" in reason


def test_cooldown_prevents_repeated_retraining(
    trigger_config, healthy_drift
):
    performance = {
        "status": "degraded",
        "labelled_total": 1100,
    }

    previous_state = {
        "labelled_total_at_last_attempt": 1000,
    }

    action, signal, reason = decide(
        performance,
        healthy_drift,
        trigger_config,
        previous_state,
    )

    assert action == "skip"
    assert signal == "performance_degraded"
    assert "cooling down" in reason


def test_retraining_allowed_after_cooldown(
    trigger_config, healthy_drift
):
    performance = {
        "status": "degraded",
        "labelled_total": 1300,
    }

    previous_state = {
        "labelled_total_at_last_attempt": 1000,
    }

    action, signal, reason = decide(
        performance,
        healthy_drift,
        trigger_config,
        previous_state,
    )

    assert action == "retrain"
    assert signal == "performance_degraded"
