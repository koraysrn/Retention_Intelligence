"""ecommerce_data.csv loading and DuckDB ingestion tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.data.loader import ingest_ecommerce_to_duckdb, load_ecommerce_data

CSV_CONTENT = """customer_id,name,email,country,age,age_group,signup_date,marketing_opt_in,total_orders,total_spend_usd,avg_order_value,avg_discount_pct,first_order_date,last_order_date,preferred_payment,preferred_device_ord,preferred_source,top_category_bought,avg_rating_given,total_sessions,preferred_device_sess,preferred_source_sess,first_session_date,last_session_date,has_abandoned_cart,clv_tier,is_repeat_customer
1,Jennifer,nicholas59@example.org,JP,71,55+,2020-09-04,True,2,115.39,57.69,7.5,2022-03-18T04:16:29,2025-06-25T16:02:53,paypal,desktop,email,Beauty,3.0,5,desktop,email,2022-03-18T01:58:29,2025-06-25T14:09:53,0,low,1
2,Phillip,christinarubio@example.com,IN,26,25-34,2020-04-05,False,2,68.52,34.26,7.5,2023-12-16T17:48:30,2025-01-02T02:48:29,card,desktop,email,Sports,,3,mobile,organic,2021-06-09T11:10:13,2025-01-02T01:01:29,0,low,1
"""


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "ecommerce_data.csv"
    p.write_text(CSV_CONTENT, encoding="utf-8")
    return p


def test_load_ecommerce_parses_types(csv_file: Path) -> None:
    df = load_ecommerce_data(csv_file)
    assert df.shape == (2, 27)
    assert pd.api.types.is_datetime64_any_dtype(df["signup_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["last_order_date"])
    assert df["marketing_opt_in"].dtype == bool
    assert pd.api.types.is_string_dtype(df["customer_id"]) or df["customer_id"].dtype == object
    assert df.iloc[0]["customer_id"] == "1"


def test_load_ecommerce_keeps_nan_rating(csv_file: Path) -> None:
    df = load_ecommerce_data(csv_file)
    assert pd.isna(df.iloc[1]["avg_rating_given"])


def test_ingest_writes_to_duckdb(csv_file: Path, tmp_path: Path) -> None:
    import duckdb

    db_path = tmp_path / "warehouse.duckdb"
    ingest_ecommerce_to_duckdb(csv_file, db_path, table="customers")

    con = duckdb.connect(str(db_path), read_only=True)
    cols = con.execute("DESCRIBE customers").fetchdf()
    n = con.execute("SELECT count(*) FROM customers").fetchone()[0]
    con.close()

    assert n == 2
    assert {"customer_id", "country", "is_repeat_customer"}.issubset(set(cols["column_name"]))
