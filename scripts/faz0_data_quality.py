"""Phase 0 — ecommerce_data.csv data exploration and quality report.

Usage: python -m scripts.faz0_data_quality
Dependency: pandas only (no heavy ML dependencies required).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import settings
from src.data.loader import load_ecommerce_data
from src.data.quality import run_quality_checks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV = settings.ecommerce_data


def main() -> None:
    parser = argparse.ArgumentParser(description="ecommerce_data.csv data quality exploration")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Source CSV path")
    parser.add_argument("--out", type=Path, default=Path("artifacts/faz0_quality_report.json"))
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {args.csv}")

    logger.info("Loading CSV: %s", args.csv)
    df = load_ecommerce_data(args.csv)

    report = run_quality_checks(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("Report saved: %s", args.out)

    print("\n=== Sample rows ===")
    print(df.head(3))
    print("\n=== Quality Report ===")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
