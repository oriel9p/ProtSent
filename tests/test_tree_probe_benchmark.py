"""Unit tests for probe evaluators integrated into the main benchmark suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_benchmark_suite import (
    DEFAULT_RESULT_PROBE,
    _make_probe_model_for_training_size,
    evaluate_classification_probe,
    evaluate_regression_probe,
    probe_label,
)


def test_probe_label_returns_humanized_name() -> None:
    """Probe labels should remain stable for CLI and result display."""
    assert probe_label(DEFAULT_RESULT_PROBE) == "Linear"
    assert probe_label("histgb") == "HistGradientBoosting"
    assert probe_label("knn") == "K-Nearest Neighbors"


def test_knn_probe_uses_train_size_as_upper_bound() -> None:
    """KNN probes should shrink k automatically on tiny training sets."""
    probe_model = _make_probe_model_for_training_size("knn", "binary", train_size=2)
    assert probe_model.n_neighbors == 2


def test_evaluate_classification_probe_histgb_is_deterministic() -> None:
    """Binary HistGB probe metrics should be deterministic with fixed seed."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(120, 8))
    y_train = (X_train[:, 0] + 0.3 * X_train[:, 1] > 0).astype(int)
    X_test = rng.normal(size=(40, 8))
    y_test = (X_test[:, 0] + 0.3 * X_test[:, 1] > 0).astype(int)

    metrics_1 = evaluate_classification_probe(
        "histgb", "binary", X_train, y_train, X_test, y_test
    )
    metrics_2 = evaluate_classification_probe(
        "histgb", "binary", X_train, y_train, X_test, y_test
    )

    assert set(metrics_1) == {"Accuracy", "F1", "AUC", "AP"}
    assert metrics_1 == pytest.approx(metrics_2)


def test_evaluate_regression_probe_histgb_is_deterministic() -> None:
    """Regression HistGB probe metrics should be deterministic with fixed seed."""
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(160, 10))
    y_train = (
        2.5 * X_train[:, 0] - 1.2 * X_train[:, 1] + rng.normal(scale=0.1, size=160)
    )
    X_test = rng.normal(size=(60, 10))
    y_test = 2.5 * X_test[:, 0] - 1.2 * X_test[:, 1] + rng.normal(scale=0.1, size=60)

    metrics_1 = evaluate_regression_probe("histgb", X_train, y_train, X_test, y_test)
    metrics_2 = evaluate_regression_probe("histgb", X_train, y_train, X_test, y_test)

    assert set(metrics_1) == {"Spearman", "MSE"}
    assert metrics_1 == pytest.approx(metrics_2)
    assert metrics_1["Spearman"] > 0.7


def test_evaluate_classification_probe_knn_multiclass_returns_expected_metrics() -> (
    None
):
    """Multiclass KNN probe should expose the standard multiclass metrics."""
    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(180, 12))
    y_train = np.argmax(X_train[:, :3], axis=1)
    X_test = rng.normal(size=(60, 12))
    y_test = np.argmax(X_test[:, :3], axis=1)

    metrics = evaluate_classification_probe(
        "knn", "multiclass", X_train, y_train, X_test, y_test
    )

    assert set(metrics) == {"Accuracy", "F1_Weighted", "F1_Macro", "AUC"}
    assert metrics["Accuracy"] > 0.5
