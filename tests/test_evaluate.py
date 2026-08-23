"""Model evaluation metric tests."""

from __future__ import annotations

import pandas as pd
from src.models.evaluate import (
    evaluate_binary,
    find_optimal_threshold,
    full_metrics,
    lift_at_top_decile,
    profit_curve,
    summarize_cv,
)


def test_evaluate_binary_perfect_classifier() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1])
    y_score = pd.Series([0.1, 0.9, 0.2, 0.8, 0.95])
    m = evaluate_binary(y_true, y_score, threshold=0.5)
    assert m["pr_auc"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_evaluate_binary_random_scores() -> None:
    y_true = pd.Series([0, 1, 0, 1, 0, 1])
    y_score = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    m = evaluate_binary(y_true, y_score, threshold=0.5)
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_profit_curve_shape_and_non_empty() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_score = pd.Series([0.1, 0.9, 0.15, 0.85, 0.7, 0.3, 0.6, 0.2, 0.8, 0.95])
    curve = profit_curve(y_true, y_score, ltv=100.0, incentive_cost=10.0)
    assert {"threshold", "profit", "precision", "recall"}.issubset(curve.columns)
    assert len(curve) > 0


def test_find_optimal_threshold_between_zero_and_one() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_score = pd.Series([0.1, 0.9, 0.15, 0.85, 0.7, 0.3, 0.6, 0.2, 0.8, 0.95])
    threshold = find_optimal_threshold(y_true, y_score, ltv=100.0, incentive_cost=10.0)
    assert 0.0 <= threshold <= 1.0


def test_lift_top_decile_greater_than_one_for_good_model() -> None:
    y_true = pd.Series([1] * 10 + [0] * 90)
    y_score = pd.Series([0.9] * 10 + [0.1] * 90)
    lift = lift_at_top_decile(y_true, y_score)
    assert lift > 1.0


def test_full_metrics_keys() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    y_score = pd.Series([0.1, 0.8, 0.3, 0.7])
    m = full_metrics(y_true, y_score)
    assert {"pr_auc", "roc_auc", "lift_top_decile", "n", "pos_rate"}.issubset(m)
    assert m["n"] == 4


def test_summarize_cv() -> None:
    metrics = [
        {"fold": 0, "pr_auc": 0.5, "roc_auc": 0.6},
        {"fold": 1, "pr_auc": 0.7, "roc_auc": 0.8},
    ]
    summary = summarize_cv(metrics)
    assert summary["pr_auc"]["mean"] == 0.6
    assert summary["roc_auc"]["mean"] == 0.7
