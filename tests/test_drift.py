
import numpy as np
import pytest

from src.monitoring.drift import (
    _bin_proportions,
    numeric_reference,
    categorical_reference,
    psi_from_proportions,
    feature_psi,
)


def test_bin_proportions_sum_to_one():
    values = np.array([1, 2, 3, 4, 5, 6])
    edges = np.array([2.5, 4.5])

    proportions = _bin_proportions(values, edges)

    assert np.isclose(proportions.sum(), 1.0)
    assert len(proportions) == 3


def test_numeric_reference_creates_quantile_bins():
    values = np.arange(1, 101)

    reference = numeric_reference(values, n_bins=10)

    assert reference["type"] == "numeric"
    assert len(reference["edges"]) == 9
    assert len(reference["proportions"]) == 10
    assert np.isclose(sum(reference["proportions"]), 1.0)


def test_numeric_reference_handles_repeated_values():
    values = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])

    reference = numeric_reference(values, n_bins=5)

    assert reference["type"] == "numeric"
    assert len(reference["edges"]) == len(set(reference["edges"]))
    assert np.isclose(sum(reference["proportions"]), 1.0)


def test_categorical_reference_calculates_shares():
    values = ["Month-to-month", "One year", "Month-to-month", "Two year"]

    reference = categorical_reference(values)

    assert reference["type"] == "categorical"
    assert reference["proportions"]["Month-to-month"] == 0.5
    assert reference["proportions"]["One year"] == 0.25
    assert reference["proportions"]["Two year"] == 0.25


def test_identical_distributions_have_zero_psi():
    proportions = [0.2, 0.3, 0.5]

    psi = psi_from_proportions(proportions, proportions)

    assert psi == pytest.approx(0.0)


def test_different_distributions_have_positive_psi():
    reference = [0.2, 0.3, 0.5]
    current = [0.6, 0.2, 0.2]

    psi = psi_from_proportions(reference, current)

    assert psi > 0.0


def test_numeric_feature_psi_is_zero_for_same_distribution():
    training_values = np.arange(1, 101)

    reference = numeric_reference(training_values, n_bins=10)

    psi = feature_psi(reference, training_values)

    assert psi == pytest.approx(0.0)


def test_numeric_feature_psi_detects_distribution_shift():
    training_values = np.arange(1, 101)
    shifted_values = np.arange(501, 601)

    reference = numeric_reference(training_values, n_bins=10)

    psi = feature_psi(reference, shifted_values)

    assert psi > 0.25


def test_categorical_feature_psi_is_zero_for_same_distribution():
    training_values = ["DSL", "Fiber", "DSL", "No", "Fiber", "DSL"]

    reference = categorical_reference(training_values)

    psi = feature_psi(reference, training_values)

    assert psi == pytest.approx(0.0)


def test_categorical_feature_psi_detects_unseen_categories():
    training_values = ["DSL", "Fiber", "DSL", "No", "Fiber", "DSL"]
    current_values = ["Satellite"] * 100

    reference = categorical_reference(training_values)

    psi = feature_psi(reference, current_values)

    assert psi > 0.25


def test_empty_input_returns_zero_psi():
    reference = numeric_reference(np.arange(1, 101), n_bins=10)

    psi = feature_psi(reference, [])

    assert psi == 0.0
