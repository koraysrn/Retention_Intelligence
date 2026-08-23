"""Customer-level feature engineering and churn labelling for ``ecommerce_data.csv``.

This module consumes the customer-level ``ecommerce_data.csv`` source, which
replaces the Online Retail II (transaction-level) dataset. Outputs:

- ``customer_features``: behavioural + RFM-derived features used as model input.
- ``churn_labels``: the ``churn`` target (the inverse of the ``is_repeat_customer``
  flag: 1 = one-time shopper / non-repeat buyer, 0 = repeat buyer).
- ``training_set``: the join of the two.

Leakage prevention (critical for this dataset):
- PII/identifier columns (``name``, ``email``) and raw date columns are dropped.
- ``total_orders`` is never used as a model feature because it determines the
  ``is_repeat_customer`` label exactly (``total_orders >= 2`` ⟺
  ``is_repeat_customer == 1``). Instead, a ``has_purchase`` (>= 1 order)
  indicator is derived, which does not leak class information.
- Total spend, average order value, order-profile categories, order span
  (``order_span_days``) and CLV tier - all of which leak the order count
  indirectly - are **deliberately excluded** from the feature set. The model is
  trained on demographic, session/engagement and cart-behaviour signals that are
  known before the repeat-purchase decision, so ROC-AUC/F1 reflect genuine
  (leakage-free) predictive power.
"""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "customer_id"
LABEL_COLUMN = "is_repeat_customer"
TARGET_COLUMN = "churn"
RECENCY_CHURN_COLUMN = "recency_churn"

# PII / identifier columns - never enter the model input
PII_COLUMNS = ["name", "email"]

# Columns that determine the label deterministically (leaky)
LEAKY_COLUMNS = ["total_orders"]

RAW_DATE_COLUMNS = [
    "signup_date",
    "first_order_date",
    "last_order_date",
    "first_session_date",
    "last_session_date",
]

AGE_GROUP_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Categorical columns encoded with OrdinalEncoder inside the preprocessor
STRING_CATEGORICALS = [
    "country",
    "preferred_device_sess",
    "preferred_source_sess",
]

# Final feature columns (deterministic order - train/serve share the same schema).
# Only signals known BEFORE the repeat-purchase decision are used.
FEATURE_COLUMNS = [
    # Demographics
    "age",
    "age_group_code",
    "country",
    # Acquisition / marketing
    "marketing_opt_in",
    "tenure_days",
    # Session / engagement
    "total_sessions",
    "days_to_first_session",
    "session_span_days",
    "session_recency_days",
    "preferred_device_sess",
    "preferred_source_sess",
    # Cart behaviour
    "has_abandoned_cart",
    # Purchase PRESENCE (not count) + first order timing + order recency
    "has_purchase",
    "days_to_first_order",
    "recency_days",
    "order_session_gap_days",
    # Rating
    "has_rating",
    "avg_rating_given",
]


def _ensure_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast date columns to datetime (invalid values become NaT)."""
    out = df.copy()
    for col in RAW_DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _reference_date(df: pd.DataFrame) -> pd.Timestamp:
    """Return the most recent reference date across all date columns."""
    maxima: list[pd.Timestamp] = []
    for col in RAW_DATE_COLUMNS:
        if col in df.columns and df[col].notna().any():
            maxima.append(df[col].max())
    if not maxima:
        raise ValueError("Reference date could not be computed: no date column found")
    return max(maxima)


def build_recency_churn_labels(
    df: pd.DataFrame,
    horizon_days: int = 365,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate the time-based ``recency_churn`` label.

    ``recency_churn = 1``: the customer has not ordered within the last
    ``horizon_days`` days (or has never ordered); ``0``: ordered recently.
    """
    out = _ensure_datetimes(df)
    if reference_date is None:
        reference_date = _reference_date(out)

    last_order = out["last_order_date"]
    churned = last_order.isna() | ((reference_date - last_order).dt.days > horizon_days)
    return pd.DataFrame(
        {
            ID_COLUMN: out[ID_COLUMN],
            RECENCY_CHURN_COLUMN: churned.astype(int),
        }
    )


def build_churn_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the per-customer binary ``churn`` label.

    ``churn = 1 - is_repeat_customer``: non-repeat buyers (one-time shoppers)
    are flagged as churned.
    """
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column not found: {LABEL_COLUMN}")
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Identifier column not found: {ID_COLUMN}")

    y = df[LABEL_COLUMN]
    if set(y.dropna().unique()) - {0, 1}:
        raise ValueError(f"{LABEL_COLUMN} must only contain 0/1")

    return pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN],
            TARGET_COLUMN: (1 - y).astype(int),
        }
    )


def build_customer_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build the full customer-level feature set (leakage-free).

    Args:
        df: Raw DataFrame read from ``ecommerce_data.csv``.
        reference_date: Reference date for time-derived features. When omitted,
            the most recent date in the data is used.

    Returns:
        DataFrame containing ``customer_id`` + ``FEATURE_COLUMNS``.
    """
    out = _ensure_datetimes(df)
    if reference_date is None:
        reference_date = _reference_date(out)

    # --- Binary indicators --------------------------------------------------
    out["has_purchase"] = (out["total_orders"] >= 1).astype(int)
    out["has_rating"] = out["avg_rating_given"].notna().astype(int)
    out["marketing_opt_in"] = out["marketing_opt_in"].astype(int)
    out["has_abandoned_cart"] = out["has_abandoned_cart"].astype(int)

    # --- Ordinal encoding ---------------------------------------------------
    age_map = {g: i for i, g in enumerate(AGE_GROUP_ORDER)}
    out["age_group_code"] = out["age_group"].map(age_map).fillna(-1).astype(int)

    # --- Time-derived features ---------------------------------------------
    signup = out["signup_date"]
    out["tenure_days"] = (reference_date - signup).dt.days.clip(lower=0)
    out["days_to_first_order"] = (out["first_order_date"] - signup).dt.days
    out["days_to_first_session"] = (out["first_session_date"] - signup).dt.days
    out["session_span_days"] = (out["last_session_date"] - out["first_session_date"]).dt.days
    out["recency_days"] = (reference_date - out["last_order_date"]).dt.days
    out["session_recency_days"] = (reference_date - out["last_session_date"]).dt.days
    out["order_session_gap_days"] = (out["last_order_date"] - out["last_session_date"]).dt.days

    # --- Missing categorical values are flagged as "MISSING" -----------------
    for col in STRING_CATEGORICALS:
        out[col] = out[col].fillna("MISSING").astype(str)

    features = out[[ID_COLUMN] + FEATURE_COLUMNS].copy()
    return features


def build_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the feature, label and training sets in a single pass."""
    features = build_customer_features(df)
    labels = build_churn_labels(df)
    recency_labels = build_recency_churn_labels(df)
    training = features.merge(labels, on=ID_COLUMN, how="inner").merge(
        recency_labels, on=ID_COLUMN, how="inner"
    )
    return {
        "features": features,
        "labels": labels,
        "training_set": training,
    }
