"""dbt mart-layer integration tests (ecommerce_data.csv).

These tests depend on the warehouse file produced after running
`python -m scripts.ingest` and `dbt run`. They are skipped when the warehouse is
missing.
"""

from __future__ import annotations

import duckdb
import pytest
from src.config import settings

WAREHOUSE = settings.data_processed / "warehouse.duckdb"


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(WAREHOUSE), read_only=True)


@pytest.mark.integration
def test_raw_customers_row_count() -> None:
    if not WAREHOUSE.exists():
        pytest.skip("warehouse.duckdb not found; run ingest + dbt run first")

    con = _connect()
    n = con.execute("SELECT count(*) FROM main.customers").fetchone()[0]
    con.close()
    assert n == 20000


@pytest.mark.integration
def test_mart_customer_features_schema() -> None:
    if not WAREHOUSE.exists():
        pytest.skip("warehouse.duckdb not found; run ingest + dbt run first")

    con = _connect()
    df = con.execute("SELECT * FROM main_mart.customer_features").fetchdf()
    con.close()

    expected_cols = {
        "customer_id",
        "total_spend_usd",
        "total_sessions",
        "tenure_days",
        "recency_days",
        "has_abandoned_cart",
        "has_purchase",
        "clv_tier",
        "country",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df["customer_id"].is_unique
    assert not df["tenure_days"].isna().any()
    assert len(df) == 20000


@pytest.mark.integration
def test_mart_churn_labels_binary_and_consistent() -> None:
    if not WAREHOUSE.exists():
        pytest.skip("warehouse.duckdb not found; run ingest + dbt run first")

    con = _connect()
    labels = con.execute("SELECT * FROM main_mart.churn_labels").fetchdf()
    con.close()

    assert labels["customer_id"].is_unique
    assert set(labels["churn"].unique()).issubset({0, 1})
    assert labels["churn"].nunique() == 2
    assert len(labels) == 20000


@pytest.mark.integration
def test_churn_label_matches_python_definition() -> None:
    """The dbt label and the Python feature-module label must be identical."""
    if not WAREHOUSE.exists():
        pytest.skip("warehouse.duckdb not found; run ingest + dbt run first")


    from src.data.loader import load_ecommerce_data
    from src.features.ecommerce import build_churn_labels

    con = _connect()
    dbt_labels = con.execute(
        "SELECT customer_id, churn FROM main_mart.churn_labels ORDER BY customer_id"
    ).fetchdf()
    con.close()

    py_labels = build_churn_labels(load_ecommerce_data(settings.ecommerce_data))
    py_labels["customer_id"] = py_labels["customer_id"].astype("int64")
    py_labels = py_labels.sort_values("customer_id").reset_index(drop=True)

    assert dbt_labels["churn"].tolist() == py_labels["churn"].tolist()


@pytest.mark.integration
def test_parquet_outputs_exist() -> None:
    if not WAREHOUSE.exists():
        pytest.skip("warehouse.duckdb not found; run ingest + dbt run first")

    for name in ("customer_features.parquet", "churn_labels.parquet", "training_set.parquet"):
        assert (settings.data_processed / name).exists()
