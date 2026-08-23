"""Model evaluation metrics.

Metric selection rationale: docs/metrics.md
Priority: PR-AUC > Recall/Precision > ROC-AUC. Threshold optimization is also
performed via the profit curve.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5) -> dict:
    """Compute binary classification metrics."""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def profit_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    ltv: float,
    incentive_cost: float,
) -> pd.DataFrame:
    """Compute expected profit per threshold.

    Profit(τ) = TP(τ)×LTV − FP(τ)×incentive_cost − FN(τ)×LTV
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    n_pos = int(y_true.sum())

    rows = []
    for p, r, t in zip(precision, recall, thresholds, strict=False):
        tp = r * n_pos
        fp = (tp / p) - tp if p > 0 else 0
        fn = n_pos - tp
        profit = tp * ltv - fp * incentive_cost - fn * ltv
        rows.append({"threshold": t, "profit": profit, "precision": p, "recall": r})
    return pd.DataFrame(rows)


def find_optimal_threshold(
    y_true: pd.Series, y_score: pd.Series, ltv: float, incentive_cost: float
) -> float:
    """Find the threshold that maximizes profit."""
    curve = profit_curve(y_true, y_score, ltv, incentive_cost)
    return float(curve.loc[curve["profit"].idxmax(), "threshold"])


def find_best_threshold_f1(y_true: pd.Series, y_score: pd.Series) -> float:
    """Find the decision threshold that maximizes the F1 score."""
    import numpy as np

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns an n+1 length thresholds array
    thresholds = np.append(thresholds, 1.0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def find_best_threshold_youden(y_true: pd.Series, y_score: pd.Series) -> float:
    """Find the threshold maximizing Youden's J (TPR - FPR) on the ROC curve."""
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def lift_at_top_decile(y_true: pd.Series, y_score: pd.Series) -> float:
    """Compute the lift within the top decile."""
    df = pd.DataFrame({"y": y_true, "score": y_score})
    k = max(1, int(len(df) * 0.1))
    top = df.nlargest(k, "score")
    baseline = df["y"].mean()
    if baseline == 0:
        return float("inf")
    return float(top["y"].mean() / baseline)


def full_metrics(y_true: pd.Series, y_score: pd.Series) -> dict:
    """Threshold-independent summary metrics."""
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "lift_top_decile": lift_at_top_decile(y_true, y_score),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }


def summarize_cv(metrics_list: list[dict]) -> dict:
    """Summarize fold metrics as mean/standard deviation."""
    import numpy as np

    keys = {k for m in metrics_list for k in m if k != "fold"}
    summary: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        vals = [m[key] for m in metrics_list if key in m and m[key] is not None]
        if vals:
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary
