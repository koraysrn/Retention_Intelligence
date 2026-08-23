-- Churn label (mart layer) — ecommerce_data.csv (customer level)
-- Definition: docs/churn_definition.md
--
-- churn = 1: one-time shopper / non-repeat buyer (is_repeat_customer = 0)
-- churn = 0: repeat buyer (is_repeat_customer = 1)
--
-- Note: produces the same definition as src/features/ecommerce.py on the
-- Python side; this model is the auditable SQL counterpart of the label.

with customers as (
    select * from {{ ref('stg_customers') }}
),

labels as (
    select
        customer_id,
        case when is_repeat_customer = 0 then 1 else 0 end as churn
    from customers
)

select * from labels
