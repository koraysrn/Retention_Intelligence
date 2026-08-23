"""Drift detection (data drift, prediction drift, concept drift).

PSI (Population Stability Index) based; retraining is triggered when the
threshold is exceeded. The prototype uses no dependencies beyond scipy/sklearn;
Evidently/whylogs integration is recommended for production.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_PSI_THRESHOLD = 0.2


@dataclass
class DriftReport:
    drift_detected: bool = False
    drifted_features: list[str] = field(default_factory=list)
    psi_scores: dict[str, float] = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


def _numeric_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = expected.dropna()
    a = actual.dropna()
    if len(e) < bins or len(a) == 0:
        return 0.0
    try:
        _, edges = pd.qcut(e, bins, retbins=True, duplicates="drop")
    except ValueError:
        return 0.0

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    e_hist = np.histogram(e, bins=edges)[0] / len(e)
    a_hist = np.histogram(a, bins=edges)[0] / len(a)
    e_hist = np.clip(e_hist, 1e-6, None)
    a_hist = np.clip(a_hist, 1e-6, None)
    return float(np.sum((a_hist - e_hist) * np.log(a_hist / e_hist)))


def _categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    idx = e.index.union(a.index)
    e = e.reindex(idx, fill_value=0.0).to_numpy()
    a = a.reindex(idx, fill_value=0.0).to_numpy()
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute the PSI value between two distributions."""
    if pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(actual):
        return _numeric_psi(expected, actual, bins)
    return _categorical_psi(expected.astype(str), actual.astype(str))


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = DEFAULT_PSI_THRESHOLD,
) -> DriftReport:
    """Measure data drift between feature distributions using PSI."""
    report = DriftReport()
    common = [c for c in reference.columns if c in current.columns]
    for col in common:
        psi = compute_psi(reference[col], current[col])
        report.psi_scores[col] = psi
        if psi > threshold:
            report.drifted_features.append(col)
    report.drift_detected = bool(report.drifted_features)
    return report


def detect_prediction_drift(
    reference_preds: pd.Series | np.ndarray,
    current_preds: pd.Series | np.ndarray,
    threshold: float = DEFAULT_PSI_THRESHOLD,
) -> DriftReport:
    """Measure drift in the score distribution (prediction drift)."""
    psi = compute_psi(pd.Series(reference_preds), pd.Series(current_preds))
    report = DriftReport(psi_scores={"prediction": psi})
    if psi > threshold:
        report.drifted_features.append("prediction")
    report.drift_detected = bool(report.drifted_features)
    return report


def detect_concept_drift(
    reference_metric: float,
    current_metric: float,
    min_drop: float = 0.10,
) -> dict:
    """Measure the drop in a performance metric (concept drift)."""
    drop = float(reference_metric) - float(current_metric)
    return {
        "reference_metric": float(reference_metric),
        "current_metric": float(current_metric),
        "drop": drop,
        "drift_detected": bool(drop > min_drop),
    }


def should_retrain(
    data_report: DriftReport | None = None,
    prediction_report: DriftReport | None = None,
    concept_report: dict | None = None,
) -> bool:
    """Report whether any drift signal requires retraining."""
    flags = [
        bool(data_report and data_report.drift_detected),
        bool(prediction_report and prediction_report.drift_detected),
        bool(concept_report and concept_report.get("drift_detected", False)),
    ]
    return any(flags)
