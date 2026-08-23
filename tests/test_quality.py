"""ecommerce_data.csv data quality check tests."""

import pandas as pd
from src.data.quality import run_quality_checks


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["1", "2", "1"],
            "country": ["TR", "US", "TR"],
            "total_orders": [0, 2, 1],
            "has_abandoned_cart": [0, 0, 1],
            "is_repeat_customer": [0, 1, 0],
            "signup_date": pd.to_datetime(["2022-01-01", "2022-02-01", None]),
            "avg_rating_given": [None, 4.0, 3.0],
        }
    )


def test_quality_detects_ecommerce_issues() -> None:
    report = run_quality_checks(_sample())
    assert report.total_rows == 3
    assert report.unique_customers == 2
    assert report.duplicated_customer_ids == 1
    assert report.no_purchase_count == 1
    assert report.abandoned_cart_count == 1
    assert report.repeat_customer_count == 1
    assert report.invalid_dates == 1
    assert report.issues


def test_quality_clean_data_has_no_issues() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["1", "2"],
            "country": ["TR", "US"],
            "total_orders": [1, 2],
            "has_abandoned_cart": [0, 0],
            "is_repeat_customer": [0, 1],
            "signup_date": pd.to_datetime(["2022-01-01", "2022-02-01"]),
            "avg_rating_given": [3.0, 4.0],
        }
    )
    report = run_quality_checks(df)
    assert report.duplicated_customer_ids == 0
    assert report.invalid_dates == 0
    assert report.issues == []
