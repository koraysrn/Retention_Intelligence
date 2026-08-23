# Model Metrics and Selection Rationale

This document explains **why** particular metrics are chosen when evaluating the
success of the churn model. Churn is typically an imbalanced classification
problem, so `accuracy` alone is **not used**.

## 1. Metric Table

| Metric | What it measures | Role in this project | Priority |
|---|---|---|---|
| **PR-AUC** | Area under the precision-recall curve | The most informative ranking metric for imbalanced data | ⭐ Highest |
| **Recall** | How many real churns were captured | Missing churn = revenue loss | High |
| **Precision** | How many flagged "churns" are correct | False positives = wasted discount cost | High |
| **ROC-AUC** | Threshold-independent ranking quality | General comparison | Medium |
| **F1-Score** | Harmonic balance of precision and recall | Single-number summary | Medium |
| **Profit curve** | Expected profit per threshold | Threshold optimization | ⭐ Business decision |
| **Lift (top decile)** | Gain within the top 10% | Campaign targeting effectiveness | Supporting |

## 2. Why These Metrics?

### 2.1 Why is PR-AUC prioritized over ROC-AUC?

ROC-AUC can be "optimistic" when the negative class dominates. When churn is a
minority (5–20%), the difference between `TP` and `FP` is drowned out by the
`TN` mass. PR-AUC focuses only on the positive class (churn) and therefore
measures the **small but critical churn segment** more honestly.

### 2.2 Precision vs Recall — business cost trade-off

```text
Precision ↑ : fewer unnecessary discounts (cost control)
Recall    ↑ : fewer missed customers (revenue protection)
```

- **High discount cost, narrow margin** → Precision takes priority.
- **High LTV of a lost customer** → Recall takes priority.

This trade-off is optimized with the **profit curve** rather than a single fixed
threshold.

### 2.3 Expected Profit (Profit Curve)

```text
ExpectedProfit(τ) = (TP(τ) × LTV) − (FP(τ) × DiscountCost) − (FN(τ) × LTV)
```

The optimum `τ* = argmax ExpectedProfit(τ)` is decided together with the
business unit. Since the model output is a probability, the threshold is tuned
with the business team.

## 3. Evaluation Protocol

1. **Split:** Chronological train/validation/test (see
   [`docs/churn_definition.md`](churn_definition.md)).
2. **CV:** Walk-forward — PR-AUC, ROC-AUC, Recall@top-decile in every fold.
3. **Final:** PR-AUC + calibration (Brier score) + SHAP on the holdout set.
4. **Baseline comparison:** Net improvement is reported against Logistic
   Regression and the naive "label everyone as churn" baseline.

## 4. Calibration and Explainability

- **Calibration:** When `churn_probability` should be interpreted as a real
  probability, it is validated with the Brier score; isotonic/Platt calibration
  is applied if needed.
- **SHAP:** Explains why each customer is risky. This output feeds both
  regulation (KVKK/GDPR transparency) and the personalization input of the
  Agentic AI layer.

## 5. Thresholds and Reporting

```text
risk_tier = high    if p ≥ 0.70         → Agentic AI intervention
            medium if 0.40 ≤ p < 0.70   → standard campaign
            low    if p < 0.40          → no action
```

Thresholds are managed from configuration according to the profit curve on the
validation set, campaign budget and operational capacity (see
[`configs/config.yaml`](../configs/config.yaml)).
