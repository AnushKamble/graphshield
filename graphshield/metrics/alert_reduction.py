"""
Developer Alert Fatigue Mitigation and Triage Reduction Metrics.
Quantifies the reduction of false positive security warnings compared to naive reactive scanners.
"""

from typing import Dict
import numpy as np


class CascadeAlertFatigueMitigation:
    """
    Evaluates developer triage efficiency.
    Naive dependency scanners (e.g. naive Dependabot/Snyk rules) alert on every
    unpinned or transitively connected component, generating massive alert fatigue.
    GraphShield uses temporal graph attention to prioritize actionable exposures.
    """

    @staticmethod
    def compute_alert_reduction(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        naive_alert_mask: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calculate the alert fatigue reduction percentage and triage precision.
        Safely extracts positive-class score if y_scores is 2D.
        """
        y_true_arr = np.asarray(y_true, dtype=int).ravel()
        scores = np.asarray(y_scores, dtype=float)
        if scores.ndim == 2:
            if scores.shape[1] == 2:
                scores = scores[:, 1]
            elif scores.shape[1] == 1:
                scores = scores[:, 0]
            else:
                raise ValueError(f"y_scores 2D array must have 1 or 2 columns, got {scores.shape[1]}")
        y_scores_arr = scores.ravel()
        naive_mask_arr = np.asarray(naive_alert_mask, dtype=bool).ravel()

        if len(y_true_arr) != len(y_scores_arr) or len(y_true_arr) != len(naive_mask_arr):
            raise ValueError(
                f"Sample count mismatch: y_true={len(y_true_arr)}, "
                f"y_scores={len(y_scores_arr)}, naive_mask={len(naive_mask_arr)}"
            )

        model_alerts = (y_scores_arr >= threshold)
        n_naive = int(np.sum(naive_mask_arr))
        n_model = int(np.sum(model_alerts))

        if n_naive == 0:
            reduction_pct = 0.0
        else:
            reduction_pct = max(0.0, float((n_naive - n_model) / n_naive) * 100.0)

        # Naive precision
        naive_tp = int(np.sum((y_true_arr == 1) & naive_mask_arr))
        naive_prec = float(naive_tp / max(n_naive, 1))

        # Model precision
        model_tp = int(np.sum((y_true_arr == 1) & model_alerts))
        model_prec = float(model_tp / max(n_model, 1))

        # Precision gain factor
        prec_gain = float(model_prec / max(naive_prec, 1e-5))

        return {
            "naive_alert_count": float(n_naive),
            "model_alert_count": float(n_model),
            "alert_reduction_pct": round(reduction_pct, 2),
            "naive_precision": round(naive_prec, 4),
            "model_precision": round(model_prec, 4),
            "precision_gain_factor": round(prec_gain, 2),
        }
