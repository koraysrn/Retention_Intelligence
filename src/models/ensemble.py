"""Weighted soft-voting ensemble + Platt calibration wrapper.

Combines the ``predict_proba`` output of multiple models into a single
probability via a weighted average. ``CalibratedEnsemble`` passes that
probability through a logistic calibrator trained on OOF (out-of-fold) scores,
producing more realistic probabilities that do not collapse to extreme 0/1.
"""

from __future__ import annotations

import numpy as np


class SoftVotingEnsemble:
    """Ensemble that averages model probabilities.

    Args:
        models: Estimators supporting ``fit`` and ``predict_proba``.
        weights: Weight per model (normalized to sum to 1). Equal weights are
            used when omitted.
    """

    def __init__(self, models: list, weights: list[float] | None = None) -> None:
        if not models:
            raise ValueError("At least one model is required")
        if weights is not None and len(weights) != len(models):
            raise ValueError("Weight count must match model count")
        self.models = models
        self.weights = np.asarray(weights, dtype=float) if weights is not None else None
        self.classes_ = np.asarray([0, 1])

    def _normalized_weights(self) -> np.ndarray:
        if self.weights is None:
            return np.full(len(self.models), 1.0 / len(self.models))
        total = self.weights.sum()
        if total <= 0:
            raise ValueError("Weight sum must be greater than zero")
        return self.weights / total

    def fit(self, x, y):
        for model in self.models:
            model.fit(x, y)
        return self

    def predict_proba(self, x) -> np.ndarray:
        weights = self._normalized_weights()
        positive = np.zeros(len(x))
        for weight, model in zip(weights, self.models, strict=False):
            positive += weight * model.predict_proba(x)[:, 1]
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x) -> np.ndarray:
        proba = self.predict_proba(x)
        return (proba[:, 1] >= 0.5).astype(int)


class CalibratedEnsemble:
    """Platt (sigmoid) calibrated wrapper."""

    def __init__(self, estimator, calibrator) -> None:
        self.estimator = estimator
        self.calibrator = calibrator

    def predict_proba(self, x) -> np.ndarray:
        raw = np.asarray(self.estimator.predict_proba(x)[:, 1], dtype=float)
        cal = np.asarray(self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, x) -> np.ndarray:
        proba = self.predict_proba(x)
        return (proba[:, 1] >= 0.5).astype(int)
