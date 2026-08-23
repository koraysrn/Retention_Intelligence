"""Churn model training pipeline (Phase 2).

Pipeline:
1. Load the training data and sort it chronologically.
2. Split features/target (leakage prevention).
3. Select the model with time-based walk-forward CV (PR-AUC first).
4. Train the final model; optimize the threshold via the profit curve.
5. Compute final metrics on the holdout test set.
6. Log to MLflow (falls back to local logging when no server is available).
7. Produce feature importances with SHAP.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT, load_yaml, settings
from src.models.evaluate import (
    evaluate_binary,
    find_optimal_threshold,
    full_metrics,
    summarize_cv,
)
from src.models.prepare import (
    build_preprocessor,
    categorical_features,
    numeric_features,
    split_features_target,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "churn_classifier"

DEFAULT_PARAMS: dict[str, dict] = {
    "logistic": {"max_iter": 5000},
    "lightgbm": {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "n_estimators": 300,
        "num_leaves": 31,
        "random_state": 42,
    },
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "n_estimators": 300,
        "max_depth": 6,
        "random_state": 42,
    },
    "catboost": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.05,
        "iterations": 300,
        "depth": 6,
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
    },
}


def make_model(model_type: str, params: dict | None = None):
    """Instantiate a classifier for the given model type."""
    p = {**DEFAULT_PARAMS.get(model_type, {}), **(params or {})}

    if model_type == "logistic":
        return LogisticRegression(max_iter=p.pop("max_iter", 2000))
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**p)
    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(**p)
    if model_type == "catboost":
        import catboost as cat

        return cat.CatBoostClassifier(**p)
    raise ValueError(f"Unknown model type: {model_type}")


def load_training_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the Phase 1 training set."""
    path = Path(path or settings.data_processed / "training_set.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Training set not found: {path}. Run `python -m scripts.build_features` first."
        )
    df = pd.read_parquet(path)
    logger.info("Training data loaded: %d rows, %d columns", len(df), df.shape[1])
    return df


def build_pipeline(
    model_type: str,
    params: dict | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> Pipeline:
    """Build an sklearn Pipeline composed of preprocessing + classifier."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = make_model(model_type, params)
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def walk_forward_cv(
    x: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    n_splits: int = 5,
    params: dict | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> list[dict]:
    """Time-based walk-forward cross-validation.

    The data is expected to be chronologically ordered (sorted inside main()).
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(x)):
        pipe = build_pipeline(model_type, params, numeric_cols, categorical_cols)
        pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
        proba = pipe.predict_proba(x.iloc[val_idx])[:, 1]
        y_val = y.iloc[val_idx]
        y_score = pd.Series(proba, index=y_val.index)

        fold_result = {"fold": fold}
        fold_result.update(full_metrics(y_val, y_score))
        fold_result.update(evaluate_binary(y_val, y_score, threshold=0.5))
        results.append(fold_result)

    return results


def _shap_explainer(
    model: object, background: np.ndarray, feature_names: list[str], model_type: str
):
    """Build the appropriate SHAP explainer for the given model type."""
    import shap

    if model_type == "logistic":
        return shap.LinearExplainer(model, background, feature_names=feature_names)

    # Tree-based models: use the raw booster instead of the wrappers
    if model_type == "lightgbm" and hasattr(model, "booster_"):
        model = model.booster_
    elif model_type == "xgboost" and hasattr(model, "get_booster"):
        model = model.get_booster()

    return shap.TreeExplainer(
        model,
        background,
        feature_names=feature_names,
        feature_perturbation="tree_path_dependent",
    )


def compute_shap_importance(
    pipeline: Pipeline, x: pd.DataFrame, model_type: str = "lightgbm"
) -> pd.DataFrame:
    """Compute feature importances from mean |SHAP| values."""
    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    x_t = pre.transform(x)
    feature_names = list(pre.get_feature_names_out())

    # Keep the background small: sufficient for the expected-value computation
    background = x_t[: min(100, len(x_t))]
    explainer = _shap_explainer(clf, background, feature_names, model_type)
    shap_values = explainer.shap_values(x_t)

    vals = np.asarray(shap_values[-1]) if isinstance(shap_values, list) else np.asarray(shap_values)
    if vals.ndim == 3:
        vals = vals[..., 1]

    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(vals).mean(axis=0)}
    )
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_local_artifacts(
    out_dir: Path,
    pipeline: Pipeline,
    metrics_payload: dict,
    shap_df: pd.DataFrame,
) -> None:
    """Persist local outputs outside MLflow (always runs)."""
    from joblib import dump

    out_dir.mkdir(parents=True, exist_ok=True)
    dump(pipeline, out_dir / "pipeline.joblib")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    shap_df.to_csv(out_dir / "shap_importance.csv", index=False)
    logger.info("Local artifacts saved: %s", out_dir)


def log_to_mlflow(
    model_type: str,
    params: dict,
    cv_summary: dict,
    test_metrics: dict,
    threshold: float,
    shap_df: pd.DataFrame,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
) -> None:
    """Log the model to MLflow; warn and continue when no server is available."""
    import mlflow

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(MODEL_NAME)
        run_name = f"{model_type}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params or {})
            for key, value in cv_summary.items():
                mlflow.log_metric(f"cv_{key}_mean", value["mean"])
                mlflow.log_metric(f"cv_{key}_std", value["std"])
            for key, value in test_metrics.items():
                mlflow.log_metric(f"test_{key}", value)
            mlflow.log_metric("optimal_threshold", threshold)

            shap_path = Path("shap_importance.csv")
            shap_df.to_csv(shap_path, index=False)
            mlflow.log_artifact(str(shap_path))

            mlflow.sklearn.log_model(
                pipeline,
                "model",
                input_example=x_train.head(5),
            )
        logger.info("MLflow logging completed: %s", run_name)
    except Exception as exc:  # noqa: BLE001 — MLflow is optional; training must continue
        logger.warning("MLflow logging skipped (server unreachable): %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Churn model training")
    parser.add_argument(
        "--data",
        type=Path,
        default=settings.data_processed / "training_set.parquet",
    )
    parser.add_argument(
        "--model-type",
        default="lightgbm",
        choices=["lightgbm", "xgboost", "catboost", "logistic"],
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--target", default="churn")
    parser.add_argument("--ltv", type=float, default=500.0)
    parser.add_argument("--incentive-cost", type=float, default=50.0)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--out", type=Path, default=settings.artifacts_dir / "model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Hyperparameters from YAML (if present) become the defaults before the CLI
    params: dict = {}
    params_file = PROJECT_ROOT / "configs" / "model_params.yaml"
    if params_file.exists():
        loaded = load_yaml(params_file)
        params = loaded.get(args.model_type, {})

    df = load_training_data(args.data)
    if "last_order_date" in df.columns:
        df = df.sort_values("last_order_date").reset_index(drop=True)

    x, y = split_features_target(df, target_col=args.target)
    num_cols = numeric_features(x)
    cat_cols = categorical_features(x)
    logger.info("Numeric features: %s", num_cols)
    logger.info("Categorical features: %s", cat_cols)

    # 70 / 15 / 15 chronological split
    n = len(df)
    tr_end = int(n * 0.70)
    val_end = int(n * 0.85)
    x_train, y_train = x.iloc[:tr_end], y.iloc[:tr_end]
    x_val, y_val = x.iloc[tr_end:val_end], y.iloc[tr_end:val_end]
    x_test, y_test = x.iloc[val_end:], y.iloc[val_end:]

    cv_metrics = walk_forward_cv(
        x_train, y_train, args.model_type, args.n_splits, params, num_cols, cat_cols
    )
    cv_summary = summarize_cv(cv_metrics)
    logger.info("CV summary (walk-forward):\n%s", json.dumps(cv_summary, indent=2))

    pipeline = build_pipeline(args.model_type, params, num_cols, cat_cols)
    pipeline.fit(x_train, y_train)

    val_proba = pd.Series(pipeline.predict_proba(x_val)[:, 1], index=y_val.index)
    threshold = find_optimal_threshold(y_val, val_proba, args.ltv, args.incentive_cost)
    logger.info("Optimal threshold (profit curve): %.4f", threshold)

    test_proba = pd.Series(pipeline.predict_proba(x_test)[:, 1], index=y_test.index)
    test_metrics = evaluate_binary(y_test, test_proba, threshold)
    test_full = full_metrics(y_test, test_proba)
    logger.info(
        "Holdout test metrics (threshold=%.4f):\n%s",
        threshold,
        json.dumps(test_metrics, indent=2),
    )
    logger.info("Holdout full metrics:\n%s", json.dumps(test_full, indent=2))

    shap_df = compute_shap_importance(pipeline, x_train, args.model_type)
    logger.info("Top 5 features:\n%s", shap_df.head(5).to_string(index=False))

    metrics_payload = {
        "model_type": args.model_type,
        "cv_summary": cv_summary,
        "threshold": threshold,
        "ltv": args.ltv,
        "incentive_cost": args.incentive_cost,
        "test_metrics": test_metrics,
        "test_full": test_full,
        "num_features": len(num_cols) + len(cat_cols),
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
    }
    save_local_artifacts(args.out, pipeline, metrics_payload, shap_df)

    if not args.no_mlflow:
        log_to_mlflow(
            args.model_type,
            params,
            cv_summary,
            test_metrics,
            threshold,
            shap_df,
            pipeline,
            x_train,
        )


if __name__ == "__main__":
    main()
