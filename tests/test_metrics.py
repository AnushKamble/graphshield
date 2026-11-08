"""Unit tests for GraphShield evaluation metrics and alert fatigue mitigation."""

import numpy as np
import pytest

from graphshield.metrics.evaluation import (
    compute_classification_metrics,
    compute_precision_at_k,
    compute_detection_lead_time,
)
from graphshield.metrics.alert_reduction import CascadeAlertFatigueMitigation


def test_classification_metrics_standard():
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    y_scores = np.array([0.05, 0.1, 0.2, 0.15, 0.85, 0.9, 0.3, 0.1, 0.05, 0.8])

    metrics = compute_classification_metrics(y_true, y_scores)
    assert metrics["pr_auc"] > 0.8
    assert metrics["roc_auc"] > 0.9
    assert metrics["f1_score"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["brier_score"] < 0.1


def test_classification_metrics_edge_case():
    # Only negative instances
    y_true = np.zeros(20, dtype=int)
    y_scores = np.random.uniform(0.01, 0.2, size=20)
    metrics = compute_classification_metrics(y_true, y_scores)
    assert metrics["roc_auc"] == 0.5
    assert metrics["f1_score"] == 0.0


def test_precision_at_k():
    y_true = np.array([1, 1, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.4, 0.3, 0.1])
    p_at_2 = compute_precision_at_k(y_true, y_scores, k=2)
    assert p_at_2 == 1.0

    p_at_5 = compute_precision_at_k(y_true, y_scores, k=5)
    assert p_at_5 == 0.4


def test_detection_lead_time():
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.95, 0.85, 0.1, 0.05])
    lags = np.array([60.0, 90.0, 45.0, 30.0])

    lead_time = compute_detection_lead_time(y_true, y_scores, lags)
    # Expected mean of ~ (60*0.95 + 90*0.85) / 2 = (57 + 76.5) / 2 = 66.75
    assert lead_time > 60.0


def test_alert_fatigue_mitigation():
    y_true = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    y_scores = np.array([0.9, 0.1, 0.2, 0.05, 0.1, 0.85, 0.05, 0.1, 0.2, 0.15])
    # Naive scanner alerted on 6 packages (e.g. any package with unpinned deps)
    naive_mask = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0], dtype=bool)

    res = CascadeAlertFatigueMitigation.compute_alert_reduction(
        y_true, y_scores, naive_mask, threshold=0.5
    )
    assert res["naive_alert_count"] == 6.0
    assert res["model_alert_count"] == 2.0
    # Reduction: (6 - 2)/6 = 66.67%
    assert res["alert_reduction_pct"] > 60.0
    assert res["model_precision"] == 1.0
    assert res["precision_gain_factor"] > 1.0


def test_2d_probability_matrix_support():
    """Verify that (N, 2) probability matrices from predict_proba are handled cleanly."""
    y_true = np.array([0, 1, 0, 1])
    # 2D matrix: col 0 = p0, col 1 = p1
    y_scores_2d = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.85, 0.15],
        [0.1, 0.9],
    ])

    m = compute_classification_metrics(y_true, y_scores_2d)
    assert m["pr_auc"] > 0.8
    assert m["f1_score"] == 1.0

    pk = compute_precision_at_k(y_true, y_scores_2d, k=2)
    assert pk == 1.0

    lead = compute_detection_lead_time(y_true, y_scores_2d, np.array([60, 90, 30, 45]))
    assert lead > 50.0

    red = CascadeAlertFatigueMitigation.compute_alert_reduction(
        y_true, y_scores_2d, np.array([True, True, True, True])
    )
    assert red["model_alert_count"] == 2.0
    assert red["alert_reduction_pct"] == 50.0


def test_metric_length_mismatch_validation():
    """Verify that length mismatches raise ValueError with informative messages."""
    y_true = np.array([0, 1, 0])
    y_scores = np.array([0.1, 0.9])  # length 2 != 3

    with pytest.raises(ValueError, match="Sample count mismatch"):
        compute_classification_metrics(y_true, y_scores)

    with pytest.raises(ValueError, match="Sample count mismatch"):
        compute_precision_at_k(y_true, y_scores)

    with pytest.raises(ValueError, match="Sample count mismatch"):
        compute_detection_lead_time(y_true, y_scores, np.array([60, 60, 60]))

    # Test lag array length mismatch when y_true and y_scores match (length 3 vs 2)
    with pytest.raises(ValueError, match="Sample count mismatch"):
        compute_detection_lead_time(np.array([0, 1, 0]), np.array([0.1, 0.9, 0.2]), np.array([60, 60]))

    with pytest.raises(ValueError, match="Sample count mismatch"):
        CascadeAlertFatigueMitigation.compute_alert_reduction(
            y_true, y_scores, np.array([True, True, True])
        )


def test_empty_metrics_handling():
    """Verify that empty inputs to compute_classification_metrics return default metrics gracefully."""
    res = compute_classification_metrics([], [])
    assert res["pr_auc"] == 0.0
    assert res["roc_auc"] == 0.5
    assert res["f1_score"] == 0.0
    assert res["brier_score"] == 0.0

    lead = compute_detection_lead_time([], [], [])
    assert lead == 0.0

    alert_red = CascadeAlertFatigueMitigation.compute_alert_reduction([], [], [])
    assert alert_red["alert_reduction_pct"] == 0.0
