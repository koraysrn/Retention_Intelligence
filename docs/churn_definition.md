# Churn Label Definition and Leakage Prevention

This document explains how the churn target variable is defined and how data
leakage is prevented during model training. An incorrect label definition is
the most common failure reason in enterprise churn projects.

## 1. Structure of the Source Data

The source file [`ecommerce_data.csv`](../ecommerce_data.csv) is at the
**customer level**; each row carries a single customer profile for a
`customer_id` (20,000 customers, 27 columns). The model is also built at the
**customer level**.

Relevant columns:

| Column | Role |
|---|---|
| `customer_id` | Customer key |
| `is_repeat_customer` | Repeat purchase label (1 = rebought, 0 = one-time shopper) |
| `total_orders` | Total order count (determines the label exactly — leaky) |
| `total_spend_usd` | Total spend |
| `total_sessions` | Total session count |
| `has_abandoned_cart` | Cart abandonment flag |
| `last_order_date` / `last_session_date` | Last order / session time |

## 2. Churn Definition

The dataset has no subscription status (`subscription_status`), but it does have
an explicit retention label: `is_repeat_customer`. Churn is defined as the
inverse of this label:

```text
churn = 1 - is_repeat_customer
churn = 1  if  the customer did not rebuy (one-time shopper / churned)
churn = 0  if  the customer rebought (retained)
```

This definition aligns exactly with the goal of the re-engagement flow:
`churn = 1` customers represent the audience to win back.

## 3. Leakage Prevention (Critical)

`is_repeat_customer` is determined **exactly** by `total_orders >= 2` in the
dataset (`total_orders >= 2 ⟺ is_repeat_customer == 1`). Therefore:

1. `total_orders` is never used directly as a feature.
2. Derived features that leak the order count indirectly are also excluded:
   - `total_spend_usd`, `avg_order_value`, `avg_discount_pct` (order total/average)
   - `order_span_days` (`last_order_date - first_order_date`: 0 for a single order, > 0 for repeaters)
   - `clv_tier`, `preferred_payment`, `preferred_device_ord`, `preferred_source`,
     `top_category_bought` (order profile categories)
3. Instead, signals known **before the repeat-purchase decision** are used:
   - Demographics: `age`, `age_group_code`, `country`
   - Acquisition/marketing: `marketing_opt_in`, `tenure_days`
   - Session/engagement: `total_sessions`, session time derivatives, device/source
   - Cart behaviour: `has_abandoned_cart`
   - Purchase **presence** (not count): `has_purchase`
   - First order timing and order recency: `days_to_first_order`, `recency_days`
   - Rating: `has_rating`, `avg_rating_given`

This separation is implemented in
[`src/features/ecommerce.py`](../src/features/ecommerce.py) via `LEAKY_COLUMNS`
and `FEATURE_COLUMNS`.

## 4. Leakage Types to Avoid

| Leakage | Example | Prevention |
|---|---|---|
| Target leakage | Using `total_orders` / `order_span_days` as features | `LEAKY_COLUMNS` + derivative exclusion |
| PII leakage | `name`, `email` in the model input | PII columns are dropped |
| Label column | `is_repeat_customer`/`churn` remaining as features | Label columns are dropped |
| Data duplication | The same customer in both train and test | `customer_id` unique; stratified split |
| Sampling bias | Class imbalance in a random split | Stratified holdout + Stratified K-Fold CV |

## 5. Imbalanced Data Handling

- The `churn` class distribution is measured; the minority class rate is reported.
- A dynamic `scale_pos_weight` (XGBoost/LightGBM) and `auto_class_weights`
  (CatBoost) derived from the class ratio are applied for each base model.
- When the minority class rate falls below the threshold, SMOTE is applied
  **inside the fold** (only to the training fold; validation leakage is prevented).
- CV and holdout splits use `stratify=y`.

## 6. Train/Validation/Test Split

```text
|--------- TRAIN (80%) ---------|---- TEST / holdout (20%) ----|
```

- Because the target is retention-based, a **stratified** split (not
  chronological) is used.
- **Stratified K-Fold CV** (5 folds) produces OOF (out-of-fold) probabilities;
  ROC-AUC/PR-AUC/F1 averages are reported.
- **F1 and Youden thresholds** are optimized with OOF probabilities; final
  metrics are computed on the holdout test set.
