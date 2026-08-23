"""Phase 1 — Feature set generation (ecommerce_data.csv).

Reads the ``ecommerce_data.csv`` source, builds the leakage-free feature set and
the churn label via ``src/features/ecommerce.py``, and writes the parquet files
that the ML phase consumes.

Usage: python -m scripts.build_features
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import settings
from src.data.loader import load_ecommerce_data
from src.features.ecommerce import build_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ecommerce_data.csv -> parquet")
    parser.add_argument("--csv", type=Path, default=settings.ecommerce_data)
    parser.add_argument("--out", type=Path, default=settings.data_processed)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {args.csv}")

    logger.info("Loading ecommerce_data.csv: %s", args.csv)
    df = load_ecommerce_data(args.csv)

    dataset = build_dataset(df)
    features = dataset["features"]
    labels = dataset["labels"]
    training = dataset["training_set"]

    logger.info("Features: %d rows, %d columns", len(features), features.shape[1])
    logger.info("Labels: %d rows, %d columns", len(labels), labels.shape[1])

    args.out.mkdir(parents=True, exist_ok=True)

    features.to_parquet(args.out / "customer_features.parquet", index=False)
    labels.to_parquet(args.out / "churn_labels.parquet", index=False)
    training.to_parquet(args.out / "training_set.parquet", index=False)

    logger.info("Files written:")
    logger.info("  - customer_features.parquet (%d rows)", len(features))
    logger.info("  - churn_labels.parquet (%d rows)", len(labels))
    logger.info("  - training_set.parquet (%d rows)", len(training))

    logger.info("Churn distribution:\n%s", training["churn"].value_counts().to_dict())


if __name__ == "__main__":
    main()
