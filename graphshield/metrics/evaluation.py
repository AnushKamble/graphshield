"""
Comprehensive Scientific Evaluation Metrics for GraphShield.
Computes PR-AUC, ROC-AUC, F1-Score, Brier Score, and Detection Lead Time.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)


def _extract_score_vector(y_scores: Union[np.ndarray, list]) -> np.ndarray:
    """Safely extract 1D positive-class probability vector from 1D or 2D inputs."""
    scores = np.asarray(y_scores, dtype=float)
    if scores.ndim == 2:
        if scores.shape[1] == 2:
            scores = scores[:, 1]
        elif scores.shape[1] == 1:
            scores = scores[:, 0]
        else:
            raise ValueError(f"y_scores 2D array must have 1 or 2 columns, got {scores.shape[1]}")
    return scores.ravel()


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate full evaluation metrics given ground truth binary labels and predicted probabilities.
    Handles edge cases with single class or zero positives gracefully.
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_scores_arr = _extract_score_vector(y_scores)

    if len(y_true_arr) != len(y_scores_arr):
        raise ValueError(
            f"Sample count mismatch: y_true has {len(y_true_arr)} samples, "
            f"y_scores has {len(y_scores_arr)} samples."
        )

    if len(y_true_arr) == 0:
        return {
            "pr_auc": 0.0,
            "roc_auc": 0.5,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "brier_score": 0.0,
        }

    # Binary predictions based on threshold
    y_pred = (y_scores_arr >= threshold).astype(int)

    n_pos = np.sum(y_true_arr == 1)
    n_neg = np.sum(y_true_arr == 0)

    # PR-AUC (Average Precision)
    if n_pos > 0 and n_neg > 0:
        try:
            pr_auc = float(average_precision_score(y_true_arr, y_scores_arr))
        except Exception:
            pr_auc = 0.0
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
        except Exception:
            roc_auc = 0.5
    else:
        pr_auc = float(n_pos / max(len(y_true_arr), 1))
        roc_auc = 0.5

    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true_arr, y_scores_arr))

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "brier_score": round(brier, 4),
    }


def compute_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    """Compute Precision@K for top-K prioritized package triage."""
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_scores_arr = _extract_score_vector(y_scores)

    if len(y_true_arr) != len(y_scores_arr):
        raise ValueError(
            f"Sample count mismatch: y_true has {len(y_true_arr)} samples, "
            f"y_scores has {len(y_scores_arr)} samples."
        )

    k = min(k, len(y_scores_arr))
    if k == 0:
        return 0.0

    top_k_indices = np.argsort(y_scores_arr)[::-1][:k]
    hits = np.sum(y_true_arr[top_k_indices] == 1)
    return float(hits / k)


def compute_detection_lead_time(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    disclosure_lags: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Compute mean detection lead time in days prior to official advisory disclosure
    for correctly identified vulnerable packages (True Positives).
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_scores_arr = _extract_score_vector(y_scores)
    lags_arr = np.asarray(disclosure_lags, dtype=float).ravel()

    if len(y_true_arr) != len(y_scores_arr) or len(y_true_arr) != len(lags_arr):
        raise ValueError(
            f"Sample count mismatch: y_true has {len(y_true_arr)} samples, "
            f"y_scores has {len(y_scores_arr)} samples, "
            f"disclosure_lags has {len(lags_arr)} samples."
        )

    preds = (y_scores_arr >= threshold).astype(int)
    tp_mask = (y_true_arr == 1) & (preds == 1)

    if np.sum(tp_mask) == 0:
        return 0.0

    # For true positives, lead time is proportional to prediction confidence * disclosure lag
    lead_times = lags_arr[tp_mask] * np.clip(y_scores_arr[tp_mask], 0.5, 1.0)
    return round(float(np.mean(lead_times)), 1)
