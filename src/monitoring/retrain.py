"""Automatic retraining trigger.

Calls the training pipeline when drift is detected. ``dry_run`` provides a safe
mode (default); set it to false and connect it to a scheduler for production.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from src.monitoring.drift import detect_data_drift

logger = logging.getLogger(__name__)


def check_and_retrain(
    reference,
    current,
    threshold: float = 0.2,
    train_command: list[str] | None = None,
    dry_run: bool = True,
) -> dict:
    """Run a drift check; trigger retraining when the threshold is exceeded (unless dry_run).

    Args:
        reference: Reference (training) feature distribution.
        current: Current (serving) feature distribution.
        threshold: PSI threshold.
        train_command: Training command to execute (default: src.models.train).
        dry_run: When True only reports, does not run training.

    Returns:
        Dictionary with the drift report + ``retrained`` information.
    """
    report = detect_data_drift(reference, current, threshold)
    payload = report.to_dict()
    payload["retrained"] = False

    if report.drift_detected and not dry_run:
        cmd = train_command or [
            sys.executable,
            "-m",
            "src.models.train",
            "--no-mlflow",
            "--out",
            "artifacts/model_ecommerce_ensemble",
        ]
        logger.info("Drift detected, starting retraining: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        payload["retrained"] = proc.returncode == 0
        payload["retrain_exit_code"] = proc.returncode
        payload["retrain_tail"] = (proc.stdout or proc.stderr or "")[-500:]
    elif report.drift_detected:
        payload["retrain_skipped"] = True

    return payload
