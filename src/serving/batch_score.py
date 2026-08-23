"""Daily batch risk scoring pipeline.

Flow:
1. Load the persisted pipeline (joblib).
2. Fetch customer features from the feature store (parquet).
3. Compute churn probability and risk tier.
4. Write the result to parquet (the A/B assignment and agent pipeline consume it).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = settings.artifacts_dir / "model_ecommerce_ensemble" / "pipeline.joblib"


def assign_risk_tier(prob: pd.Series) -> pd.Series:
    """Map probabilities to risk tiers (thresholds come from config).

    Thresholds operate on absolute probabilities; because the model output is
    calibrated, the distribution reflects actual model behaviour.
    """
    return pd.cut(
        prob,
        bins=[-0.01, settings.medium_risk_threshold, settings.high_risk_threshold, 1.01],
        labels=["low", "medium", "high"],
    )


def load_model(model_path: Path | str | None = None):
    """Load the persisted sklearn pipeline."""
    from joblib import load

    path = Path(model_path or DEFAULT_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return load(path)


def score_customers(features: pd.DataFrame, model=None) -> pd.DataFrame:
    """Produce churn probability and risk tier from customer features."""
    pipeline = model or load_model()

    ids = features["customer_id"].reset_index(drop=True)
    x = features.drop(columns=["customer_id"], errors="ignore")
    proba = pipeline.predict_proba(x)[:, 1]

    out = pd.DataFrame({"customer_id": ids, "churn_probability": proba})
    out["risk_tier"] = assign_risk_tier(pd.Series(proba, index=out.index))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch risk scoring")
    parser.add_argument(
        "--features",
        type=Path,
        default=settings.data_processed / "customer_features.parquet",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", type=Path, default=settings.data_processed / "scores.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.features.exists():
        raise FileNotFoundError(f"Feature set not found: {args.features}")

    features = pd.read_parquet(args.features)
    scores = score_customers(features)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.out, index=False)
    logger.info("Scores written: %s (%d customers)", args.out, len(scores))
    logger.info("Risk distribution:\n%s", scores["risk_tier"].value_counts().to_dict())
    logger.info("Mean churn probability: %.4f", scores["churn_probability"].mean())


if __name__ == "__main__":
    main()
