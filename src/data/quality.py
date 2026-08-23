"""Data quality checks for ``ecommerce_data.csv`` (Phase 0 discovery layer).

Known conditions detected:
- ``avg_rating_given`` is largely missing (records without a purchase profile)
- order-profile columns (``preferred_payment`` etc.) are empty for non-buyers
- the ``no_purchase`` segment where ``total_orders == 0``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Data quality report."""

    total_rows: int = 0
    unique_customers: int = 0
    duplicated_customer_ids: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    no_purchase_count: int = 0
    abandoned_cart_count: int = 0
    repeat_customer_count: int = 0
    invalid_dates: int = 0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def run_quality_checks(df: pd.DataFrame) -> QualityReport:
    """Run e-commerce quality checks on a DataFrame."""
    report = QualityReport(total_rows=len(df))

    if "customer_id" in df:
        report.unique_customers = int(df["customer_id"].nunique(dropna=True))
        report.duplicated_customer_ids = int(df["customer_id"].duplicated().sum())

    report.missing_values = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().any()
    }

    if "total_orders" in df:
        report.no_purchase_count = int((df["total_orders"] == 0).sum())
    if "has_abandoned_cart" in df:
        report.abandoned_cart_count = int((df["has_abandoned_cart"] == 1).sum())
    if "is_repeat_customer" in df:
        report.repeat_customer_count = int((df["is_repeat_customer"] == 1).sum())

    if "signup_date" in df:
        report.invalid_dates = int(df["signup_date"].isna().sum())

    if report.duplicated_customer_ids:
        report.issues.append(f"{report.duplicated_customer_ids} duplicate customer_id")
    if report.missing_values:
        report.issues.append(f"Missing values: {report.missing_values}")
    if report.invalid_dates:
        report.issues.append(f"{report.invalid_dates} rows with invalid signup_date")

    logger.info("Quality report: %s", report.to_dict())
    return report
