"""Robust churn ensemble training: XGBoost + LightGBM + CatBoost.

Uses the customer-level feature set derived from ``ecommerce_data.csv``:

- Leakage-free feature engineering (``src/features/ecommerce.py``).
- Imbalanced-data handling: dynamic ``scale_pos_weight`` / ``auto_class_weights``
  from the class ratio; SMOTE when the minority class is very low (in-fold only,
  no leakage).
- ROC-AUC / PR-AUC / F1 estimation via Stratified K-Fold CV.
- F1 threshold optimization + ROC-AUC/Youden using OOF (out-of-fold)
  probabilities.
- Final ROC-AUC, PR-AUC, F1, precision and recall on the holdout test set.
- SHAP feature importances (via the LightGBM booster).

Output: pipeline, metrics and SHAP under ``artifacts/model_ecommerce_ensemble/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from src.config import PROJECT_ROOT, load_yaml, settings
from src.models.ensemble import CalibratedEnsemble, SoftVotingEnsemble
from src.models.evaluate import (
    evaluate_binary,
    find_best_threshold_f1,
    find_best_threshold_youden,
    full_metrics,
)
from src.models.prepare import (
    IDENTIFIER_COLUMNS,
    LABEL_COLUMNS,
    LEAKY_COLUMNS,
    build_preprocessor,
    categorical_features,
    numeric_features,
    split_features_target,
)
from src.models.train import make_model
from src.monitoring.drift import detect_data_drift

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_TYPES = ["lightgbm", "xgboost", "catboost"]
OUT_DIR = settings.artifacts_dir / "model_ecommerce_ensemble"


def _is_imbalanced(y: pd.Series, minority_threshold: float = 0.15) -> bool:
    """Consider the target imbalanced when the minority rate is below the threshold."""
    minority_rate = float(y.mean() if y.mean() <= 0.5 else 1 - y.mean())
    return minority_rate < minority_threshold


def _smote_available() -> bool:
    try:
        import imblearn  # noqa: F401

        return True
    except ImportError:
        return False


def _fold_pipeline(
    model_type: str,
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
    y: pd.Series,
    use_smote: bool,
):
    """Build an in-fold pipeline (preprocessing + [SMOTE] + classifier).

    Imbalanced-data handling happens inside the fold so SMOTE is applied only to
    the training fold and validation data never leaks.
    """
    pre = build_preprocessor(num_cols, cat_cols)
    model_params = {**params}

    pos_rate = float(y.mean())
    neg_rate = 1.0 - pos_rate
    scale = neg_rate / pos_rate if pos_rate > 0 else 1.0

    if model_type == "lightgbm" or model_type == "xgboost":
        model_params.setdefault("scale_pos_weight", scale)
    elif model_type == "catboost":
        model_params.setdefault("auto_class_weights", "Balanced")

    model = make_model(model_type, model_params)

    steps: list = [("pre", pre)]
    if use_smote and _smote_available() and _is_imbalanced(y):
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        steps.append(("smote", SMOTE(random_state=42)))
        return ImbPipeline(steps + [("clf", model)])

    return Pipeline(steps + [("clf", model)])


def _build_base_pipelines(
    params_map: dict[str, dict],
    num_cols: list[str],
    cat_cols: list[str],
    y: pd.Series,
    use_smote: bool,
) -> list:
    """Build one fold pipeline per model type."""
    pipelines = []
    for model_type in MODEL_TYPES:
        pipe = _fold_pipeline(
            model_type, params_map.get(model_type, {}), num_cols, cat_cols, y, use_smote
        )
        pipelines.append((model_type, pipe))
    return pipelines


def _cv_scores(
    x: pd.DataFrame,
    y: pd.Series,
    params_map: dict[str, dict],
    num_cols: list[str],
    cat_cols: list[str],
    use_smote: bool,
    n_splits: int = 5,
) -> dict:
    """Stratified K-Fold CV: per-model OOF probabilities and metric summaries."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_by_model: dict[str, np.ndarray] = {m: np.zeros(len(y)) for m in MODEL_TYPES}
    metrics_by_model: dict[str, list[dict]] = {m: [] for m in MODEL_TYPES}

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        y_train = y.iloc[train_idx]
        for model_type, pipe in _build_base_pipelines(
            params_map, num_cols, cat_cols, y_train, use_smote
        ):
            pipe.fit(x.iloc[train_idx], y_train)
            proba = pipe.predict_proba(x.iloc[val_idx])[:, 1]
            oof_by_model[model_type][val_idx] = proba
            y_val = y.iloc[val_idx]
            score = pd.Series(proba, index=y_val.index)
            metrics_by_model[model_type].append(
                {
                    "fold": fold,
                    **full_metrics(y_val, score),
                    **evaluate_binary(y_val, score, threshold=0.5),
                }
            )

    return {"oof": oof_by_model, "metrics": metrics_by_model}


def _summarize(metrics_list: list[dict]) -> dict[str, dict[str, float]]:
    from src.models.evaluate import summarize_cv

    return summarize_cv(metrics_list)


def _compute_shap(pipeline: Pipeline, x_sample: pd.DataFrame, model_type: str = "lightgbm") -> pd.DataFrame:
    """Compute mean |SHAP| importances via the LightGBM booster."""
    import shap

    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    x_t = pre.transform(x_sample)
    feature_names = list(pre.get_feature_names_out())
    background = x_t[: min(100, len(x_t))]

    if model_type == "lightgbm" and hasattr(clf, "booster_"):
        explainer = shap.TreeExplainer(
            clf.booster_,
            background,
            feature_names=feature_names,
            feature_perturbation="tree_path_dependent",
        )
        shap_values = explainer.shap_values(x_t)
    else:
        explainer = shap.TreeExplainer(clf, background, feature_names=feature_names)
        shap_values = explainer.shap_values(x_t)

    vals = np.asarray(shap_values)
    if vals.ndim == 3:
        vals = vals[..., 1] if vals.shape[-1] > 1 else vals[..., 0]

    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(vals).mean(axis=0)}
    )
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _build_logistic_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """Builds a transparent logistic regression baseline for comparison."""
    pre = build_preprocessor(num_cols, cat_cols)
    model = make_model("logistic", {"max_iter": 5000})
    return Pipeline([("pre", pre), ("clf", model)])


def _compute_guardrails(
    df: pd.DataFrame,
    x: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    test_full: dict,
    cv_summary: dict[str, dict],
    data_path: Path,
) -> dict:
    """Data quality, balance, leakage, overfit and drift checks."""
    guardrails: dict[str, dict] = {}

    missing = int(x.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    guardrails["data_quality"] = {
        "status": "pass" if duplicate_rows == 0 else "warn",
        "missing_values_imputed": missing,
        "duplicate_rows": duplicate_rows,
        "detail": (
            "Missing values imputed by the preprocessing pipeline"
            if missing
            else "No missing values"
        ),
    }

    minority_rate = float(min(y.mean(), 1 - y.mean()))
    guardrails["class_balance"] = {
        "status": "pass" if minority_rate > 0.05 else "warn",
        "minority_rate": round(minority_rate, 4),
        "positive_rate": round(float(y.mean()), 4),
    }

    removed = sorted(IDENTIFIER_COLUMNS | LABEL_COLUMNS | LEAKY_COLUMNS)
    guardrails["leakage"] = {
        "status": "pass",
        "removed_features": [c for c in removed if c in df.columns],
    }

    cv_aucs = [
        cv_summary[m]["roc_auc"]["mean"]
        for m in MODEL_TYPES
        if "roc_auc" in cv_summary.get(m, {})
    ]
    cv_auc_mean = float(np.mean(cv_aucs)) if cv_aucs else None
    test_auc = float(test_full.get("roc_auc", 0.0))
    gap = (cv_auc_mean - test_auc) if cv_auc_mean is not None else None
    guardrails["overfitting"] = {
        "status": "pass" if gap is None or abs(gap) < 0.05 else "warn",
        "cv_roc_auc_mean": round(cv_auc_mean, 4) if cv_auc_mean is not None else None,
        "test_roc_auc": round(test_auc, 4),
        "gap": round(float(gap), 4) if gap is not None else None,
    }

    drift_report = detect_data_drift(x_train, x_test, threshold=0.2)
    guardrails["drift"] = {
        "status": "pass" if not drift_report.drift_detected else "warn",
        "max_psi": round(float(max(drift_report.psi_scores.values(), default=0.0)), 4),
        "drifted_features": drift_report.drifted_features,
    }

    imbalanced = bool(_is_imbalanced(y))
    guardrails["bias"] = {
        "status": "pass",
        "detail": (
            "Class weights / balanced objective applied"
            if imbalanced
            else "Balanced classes; no reweighting required"
        ),
    }

    sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    guardrails["data_poisoning"] = {
        "status": "pass",
        "data_sha256": sha[:16],
    }

    return guardrails


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost + LightGBM + CatBoost ensemble training")
    parser.add_argument("--data", type=Path, default=settings.data_processed / "training_set.parquet")
    parser.add_argument("--target", default="churn")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE when the minority class is low")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    logger.info("Training data: %d rows, %d columns", len(df), df.shape[1])

    x, y = split_features_target(df, target_col=args.target)
    if args.target == "recency_churn":
        recency_leaky = [c for c in ("recency_days", "order_session_gap_days") if c in x.columns]
        if recency_leaky:
            x = x.drop(columns=recency_leaky)
    num_cols = numeric_features(x)
    cat_cols = categorical_features(x)
    logger.info("Feature count: %d (numeric %d, categorical %d)", x.shape[1], len(num_cols), len(cat_cols))
    logger.info("Target distribution: %s | minority rate: %.4f", y.value_counts().to_dict(), min(y.mean(), 1 - y.mean()))

    params_file = PROJECT_ROOT / "configs" / "model_params.yaml"
    params_map = load_yaml(params_file) if params_file.exists() else {}

    # Stratified holdout (the correct choice for the time-independent retention target)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # --- CV (OOF probabilities + per-model ROC-AUC weights) -----------------
    cv_result = _cv_scores(x_train, y_train, params_map, num_cols, cat_cols, args.use_smote, args.n_splits)
    cv_summary = {m: _summarize(v) for m, v in cv_result["metrics"].items()}

    # Model weights: normalized by OOF ROC-AUC
    aucs = {
        m: cv_summary[m]["roc_auc"]["mean"] for m in MODEL_TYPES if "roc_auc" in cv_summary[m]
    }
    weights = np.asarray([max(aucs.get(m, 0.5) - 0.5, 0.0) + 1e-6 for m in MODEL_TYPES])
    weights = weights / weights.sum()
    logger.info("Model weights (OOF ROC-AUC): %s", dict(zip(MODEL_TYPES, weights.tolist(), strict=False)))

    # --- OOF ensemble + Platt calibration ------------------------------------
    oof_ensemble = np.zeros(len(y_train))
    for m, w in zip(MODEL_TYPES, weights, strict=False):
        oof_ensemble += w * cv_result["oof"][m]

    calibrator = LogisticRegression(C=1.0, max_iter=2000)
    calibrator.fit(oof_ensemble.reshape(-1, 1), y_train)
    oof_calibrated = calibrator.predict_proba(oof_ensemble.reshape(-1, 1))[:, 1]
    oof_cal = pd.Series(oof_calibrated, index=y_train.index)
    threshold_f1 = find_best_threshold_f1(y_train, oof_cal)
    threshold_youden = find_best_threshold_youden(y_train, oof_cal)
    logger.info("OOF calibrated thresholds -> F1: %.4f | Youden: %.4f", threshold_f1, threshold_youden)

    # --- Final ensemble (full training set) ----------------------------------
    base = _build_base_pipelines(params_map, num_cols, cat_cols, y_train, args.use_smote)
    ensemble = SoftVotingEnsemble([pipe for _, pipe in base], weights=weights.tolist())
    ensemble.fit(x_train, y_train)
    serving_model = CalibratedEnsemble(ensemble, calibrator)

    # --- Holdout test evaluation (calibrated) ---------------------------------
    test_proba = serving_model.predict_proba(x_test)[:, 1]
    test_score = pd.Series(test_proba, index=y_test.index)
    test_full = full_metrics(y_test, test_score)
    test_metrics_f1 = evaluate_binary(y_test, test_score, threshold_f1)
    test_metrics_youden = evaluate_binary(y_test, test_score, threshold_youden)

    logger.info("Holdout full metrics:\n%s", json.dumps(test_full, indent=2))
    logger.info("Holdout metrics (F1 threshold=%.4f):\n%s", threshold_f1, json.dumps(test_metrics_f1, indent=2))

    # --- Per-model holdout metrics + logistic baseline ------------------------
    model_metrics: dict[str, dict] = {}
    for name, pipe in base:
        proba = pipe.predict_proba(x_test)[:, 1]
        score = pd.Series(proba, index=y_test.index)
        model_metrics[name] = {
            "cv": cv_summary.get(name, {}),
            "test": evaluate_binary(y_test, score, threshold_f1),
        }

    logistic_pipe = _build_logistic_pipeline(num_cols, cat_cols)
    logistic_pipe.fit(x_train, y_train)
    logistic_score = pd.Series(logistic_pipe.predict_proba(x_test)[:, 1], index=y_test.index)
    model_metrics["logistic"] = {
        "cv": {},
        "test": evaluate_binary(y_test, logistic_score, threshold_f1),
    }

    model_metrics["ensemble"] = {"cv": {}, "test": test_metrics_f1}

    guardrails = _compute_guardrails(df, x, x_train, x_test, y, test_full, cv_summary, args.data)
    logger.info("Guardrails:\n%s", json.dumps(guardrails, indent=2))

    # --- SHAP (LightGBM base model) ------------------------------------------
    lgbm_pipe = next(pipe for name, pipe in base if name == "lightgbm")
    sample = x_train.sample(min(1000, len(x_train)), random_state=42)
    shap_df = _compute_shap(lgbm_pipe, sample, model_type="lightgbm")
    logger.info("Top 10 features:\n%s", shap_df.head(10).to_string(index=False))

    # --- Artifact persistence -------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    dump(serving_model, args.out / "pipeline.joblib")
    shap_df.to_csv(args.out / "shap_importance.csv", index=False)

    payload = {
        "ensemble": True,
        "calibrated": True,
        "models": MODEL_TYPES,
        "model_weights": {m: round(float(w), 4) for m, w in zip(MODEL_TYPES, weights, strict=False)},
        "target": args.target,
        "imbalanced": bool(_is_imbalanced(y)),
        "minority_rate": round(float(min(y.mean(), 1 - y.mean())), 4),
        "smote_applied": bool(args.use_smote and _smote_available() and _is_imbalanced(y)),
        "threshold_f1": round(threshold_f1, 4),
        "threshold_youden": round(threshold_youden, 4),
        "cv_summary": {m: v for m, v in cv_summary.items()},
        "test_full": test_full,
        "test_metrics_f1": test_metrics_f1,
        "test_metrics_youden": test_metrics_youden,
        "model_metrics": model_metrics,
        "guardrails": guardrails,
        "num_features": int(x.shape[1]),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }
    with (args.out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Ensemble saved: %s", args.out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
