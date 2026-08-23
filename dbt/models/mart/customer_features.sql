-- Customer-level feature set (mart layer) — ecommerce_data.csv
-- Core behavioural features for governance/reporting. The full feature
-- engineering that produces the model input lives in src/features/ecommerce.py
-- (including leakage prevention); this table is its auditable SQL counterpart.

with customers as (
    select * from {{ ref('stg_customers') }}
),

reference as (
    select
        greatest(
            max(last_order_date),
            max(last_session_date),
            max(first_order_date),
            max(first_session_date),
            max(signup_date)
        ) as ref_date
    from customers
),

enriched as (
    select
        c.customer_id,
        c.country,
        c.age,
        c.age_group,
        c.clv_tier,
        c.total_spend_usd,
        c.total_sessions,
        c.avg_order_value,
        c.avg_discount_pct,
        c.avg_rating_given,
        c.has_abandoned_cart,
        c.marketing_opt_in,
        c.preferred_payment,
        c.preferred_device_ord,
        c.preferred_source,
        c.top_category_bought,
        c.preferred_device_sess,
        c.preferred_source_sess,
        case when c.total_orders >= 1 then 1 else 0 end              as has_purchase,
        date_diff('day', c.signup_date, r.ref_date)                  as tenure_days,
        date_diff('day', c.last_order_date, r.ref_date)              as recency_days,
        date_diff('day', c.last_session_date, r.ref_date)            as session_recency_days,
        c.total_spend_usd / (c.total_sessions + 1)                   as spend_per_session
    from customers c
    cross join reference r
)

select * from enriched
