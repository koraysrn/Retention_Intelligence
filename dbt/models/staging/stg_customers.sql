-- Cleaned ecommerce_data.csv customer data (staging)
-- Dialect: DuckDB (prototype). Adapted to Snowflake/BigQuery in enterprise.
-- Source: ecommerce_data.csv (loaded into the raw.customers table by scripts/ingest.py)

with source as (
    select * from {{ source('raw', 'customers') }}
),

cleaned as (
    select
        customer_id,
        name,
        email,
        country,
        age,
        age_group,
        try_cast(signup_date as timestamp)        as signup_date,
        marketing_opt_in,
        total_orders,
        total_spend_usd,
        avg_order_value,
        avg_discount_pct,
        try_cast(first_order_date as timestamp)   as first_order_date,
        try_cast(last_order_date as timestamp)    as last_order_date,
        preferred_payment,
        preferred_device_ord,
        preferred_source,
        top_category_bought,
        avg_rating_given,
        total_sessions,
        preferred_device_sess,
        preferred_source_sess,
        try_cast(first_session_date as timestamp) as first_session_date,
        try_cast(last_session_date as timestamp)  as last_session_date,
        has_abandoned_cart,
        clv_tier,
        is_repeat_customer
    from source
    where customer_id is not null
)

select * from cleaned
