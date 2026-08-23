"""Soft voting ensemble tests."""

import numpy as np
import pytest
from src.models.ensemble import SoftVotingEnsemble


class _FakeModel:
    def __init__(self, p: float) -> None:
        self._p = p

    def fit(self, x, y):
        return self

    def predict_proba(self, x) -> np.ndarray:
        n = len(x)
        p = np.full(n, self._p)
        return np.column_stack([1 - p, p])


def test_predict_proba_averages() -> None:
    ensemble = SoftVotingEnsemble([_FakeModel(0.4), _FakeModel(0.6)])
    x = np.zeros((10, 2))
    proba = ensemble.predict_proba(x)
    assert proba.shape == (10, 2)
    assert np.allclose(proba[:, 1], 0.5)
    assert np.allclose(proba[:, 0] + proba[:, 1], 1.0)


def test_weighted_predict_proba() -> None:
    ensemble = SoftVotingEnsemble([_FakeModel(0.4), _FakeModel(0.6)], weights=[0.75, 0.25])
    proba = ensemble.predict_proba(np.zeros((5, 2)))
    # 0.75 * 0.4 + 0.25 * 0.6 = 0.45
    assert np.allclose(proba[:, 1], 0.45)


def test_predict_returns_binary() -> None:
    ensemble = SoftVotingEnsemble([_FakeModel(0.9)])
    pred = ensemble.predict(np.zeros((4, 2)))
    assert np.array_equal(pred, np.ones(4))


def test_fit_returns_self() -> None:
    ensemble = SoftVotingEnsemble([_FakeModel(0.5)])
    x = np.zeros((5, 1))
    y = np.zeros(5)
    assert ensemble.fit(x, y) is ensemble


def test_empty_models_raises() -> None:
    with pytest.raises(ValueError):
        SoftVotingEnsemble([])


def test_weight_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Weight count"):
        SoftVotingEnsemble([_FakeModel(0.5), _FakeModel(0.5)], weights=[1.0])
