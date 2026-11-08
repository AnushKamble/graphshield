"""
Baseline Models for Cascading Vulnerability Prediction.
Includes:
1. PageRankBaseline: Heuristic ranking via topological centrality & blast radius.
2. RandomForestBaseline: Random Forest on node tabular metadata.
3. XGBoostBaseline: Gradient Boosting with class-imbalance weighting.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class PageRankBaseline:
    """
    Unsupervised topological baseline.
    Scores packages by combining downstream PageRank, in-degree, and blast radius.
    Packages with high downstream reachability have greater exposure and cascade impact.
    """

    def __init__(self, pagerank_idx: int = 6, blast_radius_idx: int = 9, in_degree_idx: int = 4):
        self.pagerank_idx = pagerank_idx
        self.blast_radius_idx = blast_radius_idx
        self.in_degree_idx = in_degree_idx
        self.fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PageRankBaseline":
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Produce a normalized probability score based on topological features.
        Score = 0.5 * normalized_pagerank + 0.3 * normalized_blast_radius + 0.2 * normalized_in_degree
        """
        if len(X) == 0:
            return np.empty((0, 2), dtype=float)

        pr = X[:, self.pagerank_idx]
        blast = X[:, self.blast_radius_idx]
        indeg = X[:, self.in_degree_idx]

        def _minmax(arr: np.ndarray) -> np.ndarray:
            if len(arr) == 0:
                return np.empty((0,), dtype=float)
            denom = arr.max() - arr.min()
            if denom <= 1e-8:
                return np.zeros_like(arr)
            return (arr - arr.min()) / denom

        pr_norm = _minmax(pr)
        blast_norm = _minmax(blast)
        indeg_norm = _minmax(indeg)

        score = 0.5 * pr_norm + 0.3 * blast_norm + 0.2 * indeg_norm
        # Clip to [0.001, 0.999] to prevent degeneracies
        p1 = np.clip(score, 0.001, 0.999)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)


class RandomForestBaseline:
    """Random Forest tabular classifier with balanced subsample weights."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 8, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0, 2), dtype=float)
        probs = self.clf.predict_proba(X)
        if probs.shape[1] == 1:
            # Handle edge case if only one class present in training
            single_class = self.clf.classes_[0]
            if single_class == 0:
                return np.column_stack([probs[:, 0], np.zeros_like(probs[:, 0])])
            else:
                return np.column_stack([np.zeros_like(probs[:, 0]), probs[:, 0]])
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)


class XGBoostBaseline:
    """Extreme Gradient Boosting tabular classifier with positive scale weighting."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self.single_class: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.single_class = int(unique_classes[0]) if len(unique_classes) == 1 else 0
            self.model = None
            return self

        self.single_class = None
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        scale_pos_weight = float(n_neg / max(n_pos, 1))

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=2,
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0, 2), dtype=float)
        if self.single_class is not None:
            p1 = np.ones(len(X), dtype=float) if self.single_class == 1 else np.zeros(len(X), dtype=float)
            return np.column_stack([1.0 - p1, p1])
        if self.model is None:
            raise RuntimeError("XGBoost model not fitted yet.")
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 1:
            single_class = self.model.classes_[0]
            if single_class == 0:
                return np.column_stack([probs[:, 0], np.zeros_like(probs[:, 0])])
            else:
                return np.column_stack([np.zeros_like(probs[:, 0]), probs[:, 0]])
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)
