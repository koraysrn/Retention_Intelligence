"""Drift detection tests."""

import numpy as np
import pandas as pd
from src.monitoring.drift import (
    DriftReport,
    compute_psi,
    detect_concept_drift,
    detect_data_drift,
    detect_prediction_drift,
    should_retrain,
)


def test_psi_identical_distributions_near_zero() -> None:
    rng = np.random.default_rng(0)
    e = pd.Series(rng.normal(0, 1, 5000))
    a = pd.Series(rng.normal(0, 1, 5000))
    assert compute_psi(e, a) < 0.1


def test_psi_different_distributions_large() -> None:
    rng = np.random.default_rng(0)
    e = pd.Series(rng.normal(0, 1, 5000))
    a = pd.Series(rng.normal(3, 1, 5000))
    assert compute_psi(e, a) > 0.2


def test_detect_data_drift_no_drift() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    report = detect_data_drift(ref, cur, threshold=0.2)
    assert not report.drift_detected


def test_detect_data_drift_detected() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"x": rng.normal(3, 1, 5000)})
    report = detect_data_drift(ref, cur, threshold=0.2)
    assert report.drift_detected
    assert "x" in report.drifted_features


def test_detect_prediction_drift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.random(5000)
    cur = np.full(5000, 0.9)
    report = detect_prediction_drift(ref, cur, threshold=0.2)
    assert report.drift_detected
    assert "prediction" in report.drifted_features


def test_prediction_drift_no_drift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.random(5000)
    cur = rng.random(5000)
    assert not detect_prediction_drift(ref, cur, threshold=0.2).drift_detected


def test_concept_drift_detection() -> None:
    r = detect_concept_drift(0.70, 0.50, min_drop=0.10)
    assert r["drift_detected"]
    r2 = detect_concept_drift(0.70, 0.68, min_drop=0.10)
    assert not r2["drift_detected"]


def test_should_retrain_any_signal() -> None:
    assert should_retrain(DriftReport(drift_detected=True)) is True
    assert should_retrain(DriftReport(drift_detected=False)) is False
    assert should_retrain(concept_report={"drift_detected": True}) is True
    assert should_retrain() is False
