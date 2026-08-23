"""ecommerce_data.csv feature engineering tests (heavy).

Verifies leakage prevention, label generation and the determinism of the feature
schema.
"""

from __future__ import annotations

import pandas as pd
import pytest
from src.features.ecommerce import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    build_churn_labels,
    build_customer_features,
    build_dataset,
)


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["1", "2", "3"],
            "name": ["A", "B", "C"],
            "email": ["a@x.com", "b@x.com", "c@x.com"],
            "country": ["TR", "US", "DE"],
            "age": [25, 40, 55],
            "age_group": ["18-24", "35-44", "55+"],
            "signup_date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
            "marketing_opt_in": [True, False, True],
            "total_orders": [0, 1, 3],
            "total_spend_usd": [0.0, 50.0, 300.0],
            "avg_order_value": [None, 50.0, 100.0],
            "avg_discount_pct": [None, 5.0, 10.0],
            "first_order_date": pd.to_datetime([None, "2022-02-10", "2022-03-10"]),
            "last_order_date": pd.to_datetime([None, "2022-02-10", "2022-06-10"]),
            "preferred_payment": [None, "card", "paypal"],
            "preferred_device_ord": [None, "desktop", "mobile"],
            "preferred_source": [None, "email", "organic"],
            "top_category_bought": [None, "Beauty", "Sports"],
            "avg_rating_given": [None, 4.0, None],
            "total_sessions": [2, 5, 8],
            "preferred_device_sess": ["mobile", "desktop", "mobile"],
            "preferred_source_sess": ["direct", "email", "organic"],
            "first_session_date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
            "last_session_date": pd.to_datetime(["2022-06-01", "2022-07-01", "2022-08-01"]),
            "has_abandoned_cart": [0, 1, 0],
            "clv_tier": ["no_purchase", "low", "medium"],
            "is_repeat_customer": [0, 0, 1],
        }
    )


def test_churn_label_is_inverse_of_repeat() -> None:
    labels = build_churn_labels(_sample())
    assert TARGET_COLUMN in labels.columns
    assert ID_COLUMN in labels.columns
    assert list(labels[TARGET_COLUMN]) == [1, 1, 0]


def test_churn_label_raises_on_missing_label_column() -> None:
    df = _sample().drop(columns="is_repeat_customer")
    with pytest.raises(ValueError, match="Label column not found"):
        build_churn_labels(df)


def test_features_exclude_pii_label_and_leaky_columns() -> None:
    features = build_customer_features(_sample())
    # PII, label and directly leaky columns
    for col in ("name", "email", "is_repeat_customer", "total_orders"):
        assert col not in features.columns
    # Derived columns that leak the order count must also be excluded
    for col in (
        "total_spend_usd",
        "avg_order_value",
        "avg_discount_pct",
        "order_span_days",
        "clv_tier",
        "preferred_payment",
        "top_category_bought",
    ):
        assert col not in features.columns
    # Raw date columns never enter the model
    for col in (
        "signup_date",
        "first_order_date",
        "last_order_date",
        "first_session_date",
        "last_session_date",
    ):
        assert col not in features.columns


def test_features_have_deterministic_schema() -> None:
    features = build_customer_features(_sample())
    assert list(features.columns) == [ID_COLUMN] + FEATURE_COLUMNS


def test_customer_id_present_and_unique() -> None:
    features = build_customer_features(_sample())
    assert features[ID_COLUMN].is_unique
    assert set(features[ID_COLUMN]) == {"1", "2", "3"}


def test_has_purchase_indicator_computed_without_total_orders() -> None:
    features = build_customer_features(_sample())
    assert features["has_purchase"].tolist() == [0, 1, 1]


def test_categorical_missing_filled_with_sentinel() -> None:
    features = build_customer_features(_sample())
    # Session device/source columns must be flagged as MISSING when empty
    row = features.loc[features[ID_COLUMN] == "3"].iloc[0]
    assert row["preferred_device_sess"] == "mobile"
    assert row["preferred_source_sess"] == "organic"


def test_recency_and_tenure_computed() -> None:
    features = build_customer_features(_sample())
    assert features["tenure_days"].notna().all()
    assert features["recency_days"].notna().any()
    # Recency can be NaN for non-buyers; session recency must be populated
    assert features["session_recency_days"].notna().all()


def test_build_dataset_merges_on_customer_id() -> None:
    dataset = build_dataset(_sample())
    assert set(dataset) == {"features", "labels", "training_set"}
    training = dataset["training_set"]
    assert TARGET_COLUMN in training.columns
    assert len(training) == len(_sample())
    assert training[ID_COLUMN].is_unique
    # The training set contains all feature + label columns
    assert set(FEATURE_COLUMNS).issubset(training.columns)


def test_build_dataset_churn_distribution() -> None:
    dataset = build_dataset(_sample())
    assert dataset["training_set"][TARGET_COLUMN].value_counts().to_dict() == {1: 2, 0: 1}
