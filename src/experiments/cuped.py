"""CUPED — variance reduction using pre-experiment metrics.

Y_adjusted = Y − θ × (X_pre − X̄_pre)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_theta(
    outcome: pd.Series,
    pre_experiment_metric: pd.Series,
    group_assignment: pd.Series,
) -> float:
    """Estimate the CUPED adjustment coefficient (θ).

    θ = Cov(X_pre, Y) / Var(X_pre) — computed over all data (pooled).
    """
    data = pd.DataFrame(
        {"y": outcome, "x": pre_experiment_metric, "group": group_assignment}
    ).dropna()
    if len(data) < 2:
        return 0.0

    var_x = float(data["x"].var(ddof=1))
    if not np.isfinite(var_x) or var_x == 0:
        return 0.0

    cov_xy = float(data["x"].cov(data["y"]))
    return cov_xy / var_x


def apply_cuped(
    outcome: pd.Series,
    pre_experiment_metric: pd.Series,
    group_assignment: pd.Series,
) -> pd.Series:
    """Apply the CUPED adjustment to the outcome metric.

    Args:
        outcome: Post-experiment metric.
        pre_experiment_metric: Pre-experiment value of the same metric.
        group_assignment: 'control' / 'treatment'.

    Returns:
        CUPED-adjusted outcome series. Unusable observations (missing
        pre-metric) remain NaN.
    """
    data = pd.DataFrame(
        {"y": outcome, "x": pre_experiment_metric, "group": group_assignment}
    ).dropna()

    theta = estimate_theta(outcome, pre_experiment_metric, group_assignment)
    x_mean = float(data["x"].mean())
    data["y_adjusted"] = data["y"] - theta * (data["x"] - x_mean)

    result = pd.Series(index=outcome.index, dtype=float)
    result.loc[data.index] = data["y_adjusted"]
    return result


def variance_reduction(
    outcome: pd.Series,
    pre_experiment_metric: pd.Series,
    group_assignment: pd.Series,
) -> float:
    """Return the relative variance reduction (0-1) provided by CUPED."""
    data = pd.DataFrame(
        {"y": outcome, "x": pre_experiment_metric, "group": group_assignment}
    ).dropna()
    if len(data) < 2:
        return 0.0

    adjusted = apply_cuped(outcome, pre_experiment_metric, group_assignment).dropna()
    var_raw = float(data["y"].var(ddof=1))
    var_adj = float(adjusted.var(ddof=1))
    if var_raw <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - var_adj / var_raw))
