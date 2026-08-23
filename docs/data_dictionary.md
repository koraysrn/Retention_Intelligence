# Data Dictionary and Phase 0 Discovery Findings

Source file: [`ecommerce_data.csv`](../ecommerce_data.csv)

## 1. Columns

| Column | Type | Description |
|---|---|---|
| `customer_id` | int | Customer key (unique) |
| `name` | string | Customer name (PII — removed from the model input) |
| `email` | string | Email (PII — removed from the model input) |
| `country` | string | Country |
| `age` | int | Age |
| `age_group` | string | Age group (18-24, 25-34, 35-44, 45-54, 55+) |
| `signup_date` | datetime | Registration date |
| `marketing_opt_in` | bool | Marketing consent |
| `total_orders` | int | Total order count (determines the label exactly — leaky) |
| `total_spend_usd` | float | Total spend |
| `avg_order_value` | float | Average order value |
| `avg_discount_pct` | float | Average discount percentage |
| `first_order_date` / `last_order_date` | datetime | First / last order date |
| `preferred_payment` | string | Preferred payment method |
| `preferred_device_ord` / `preferred_device_sess` | string | Preferred device for order / session |
| `preferred_source` / `preferred_source_sess` | string | Order / session source channel |
| `top_category_bought` | string | Most bought category |
| `avg_rating_given` | float | Average product rating (empty for non-buyers) |
| `total_sessions` | int | Total session count |
| `first_session_date` / `last_session_date` | datetime | First / last session date |
| `has_abandoned_cart` | int | Cart abandonment flag (0/1) |
| `clv_tier` | string | CLV tier (no_purchase, low, medium, high, vip) |
| `is_repeat_customer` | int | Repeat purchase label (1 = rebought, 0 = one-time shopper) |

## 2. Phase 0 Discovery Findings

### 2.1 Size and Granularity

- Total rows (unique customers): **20,000**
- Column count: **27**
- Duplicate `customer_id`: **0**
- Granularity is **customer level**; each row is a customer.

### 2.2 Target Distribution

- `is_repeat_customer = 1` (rebought): **10,045**
- `is_repeat_customer = 0` (one-time shopper / churn): **9,955**
- `has_abandoned_cart = 1`: **3,677** (all in the churn class)
- `total_orders = 0` (never purchased): **3,732**

### 2.3 Missing Values

- `first_order_date`, `last_order_date`, `preferred_payment`, `preferred_device_ord`,
  `preferred_source`, `top_category_bought`: empty in **3,732** rows (non-buyers)
- `avg_rating_given`: empty in **14,741** rows (non-buyers + those who never rated)
- Session profile columns (`preferred_*_sess`, session dates): empty in **55** rows

### 2.4 Label-Leakage Relationship (Critical)

The equality `is_repeat_customer == (total_orders >= 2)` holds 100%. Therefore
`total_orders` and derivatives that leak the order count (total spend, average
order value, `order_span_days`, CLV tier, order profile categories) are removed
from the model input. Details: [`docs/churn_definition.md`](churn_definition.md).

### 2.5 Churn Label

```text
churn = 1 - is_repeat_customer
```

The label is produced by `build_churn_labels()` in
[`src/features/ecommerce.py`](../src/features/ecommerce.py) and can be audited
on the dbt side via
[`dbt/models/mart/churn_labels.sql`](../dbt/models/mart/churn_labels.sql).
