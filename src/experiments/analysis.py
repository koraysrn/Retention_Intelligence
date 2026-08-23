"""Statistical analysis for A/B tests.

- Two-proportion z-test for ratio metrics (churn rate).
- CUPED-adjusted Welch t-test for continuous metrics (net revenue).
- Alpha-spending framework for sequential testing (O'Brien-Fleming).
- End-to-end analysis: ``analyze_experiment`` + synthetic experiment simulation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.assignment import assign_groups
from src.experiments.cuped import apply_cuped

logger = logging.getLogger(__name__)


@dataclass
class ProportionsTestResult:
    p1: float
    p2: float
    lift: float
    z_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool

    def to_dict(self) -> dict:
        return asdict(self)


def proportions_z_test(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.05,
) -> ProportionsTestResult:
    """Two-independent-proportion z-test (churn-rate comparison).

    ``lift`` = (p2 − p1) / p1; a negative value indicates improvement for churn.
    """
    n1, n2 = len(control), len(treatment)
    if n1 == 0 or n2 == 0:
        raise ValueError("Both groups must have at least one observation")

    x1, x2 = int(control.sum()), int(treatment.sum())
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    margin = stats.norm.ppf(1 - alpha / 2) * se_diff
    lift = (p2 - p1) / p1 if p1 > 0 else float("inf")

    return ProportionsTestResult(
        p1=float(p1),
        p2=float(p2),
        lift=float(lift),
        z_stat=float(z),
        p_value=p_value,
        ci_low=float((p2 - p1) - margin),
        ci_high=float((p2 - p1) + margin),
        significant=bool(p_value < alpha),
    )


def cuped_t_test(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """Two-sample Welch t-test (for CUPED-adjusted continuous metrics)."""
    n1, n2 = len(control), len(treatment)
    if n1 < 2 or n2 < 2:
        raise ValueError("At least 2 observations are required per group for the t-test")

    c = control.astype(float)
    t = treatment.astype(float)
    mean_diff = float(t.mean() - c.mean())

    var1 = float(c.var(ddof=1))
    var2 = float(t.var(ddof=1))
    se = math.sqrt(var1 / n1 + var2 / n2)

    if se > 0:
        t_stat = mean_diff / se
        df = (var1 / n1 + var2 / n2) ** 2 / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
        margin = stats.t.ppf(1 - alpha / 2, df) * se
    else:
        t_stat, df, p_value, margin = 0.0, float(n1 + n2 - 2), 1.0, 0.0

    return {
        "mean_diff": float(mean_diff),
        "t_stat": float(t_stat),
        "df": float(df),
        "p_value": float(p_value),
        "ci_low": float(mean_diff - margin),
        "ci_high": float(mean_diff + margin),
        "significant": bool(p_value < alpha),
    }


def obrien_fleming_boundary(information_fraction: float, alpha: float = 0.05) -> float:
    """O'Brien-Fleming interim-analysis boundary (z-statistic threshold).

    Controls the type-I error inflation caused by peeking in sequential testing.
    """
    if not 0 < information_fraction <= 1:
        raise ValueError("information_fraction must be in the 0-1 range")
    return float(stats.norm.ppf(1 - alpha / 2) / math.sqrt(information_fraction))


def simulate_outcomes(
    assignment: pd.Series,
    seed: int = 7,
    baseline_churn: float = 0.50,
    treatment_effect: float = 0.20,
    avg_revenue: float = 300.0,
    incentive_cost: float = 50.0,
) -> pd.DataFrame:
    """Generate synthetic experiment outcomes with a deterministic seed.

    Churn decreases relatively by ``treatment_effect`` in the treatment group;
    non-churned customers generate revenue. Pre-experiment revenue is produced
    as the CUPED covariate.
    """
    if not 0 <= treatment_effect <= 1:
        raise ValueError("treatment_effect must be in the 0-1 range")

    rng = np.random.default_rng(seed)
    n = len(assignment)
    is_treatment = (assignment.astype(str) == "treatment").to_numpy()

    p_churn = np.where(is_treatment, baseline_churn * (1 - treatment_effect), baseline_churn)
    p_churn = np.clip(p_churn, 0.0, 1.0)
    churn = rng.random(n) < p_churn

    pre_revenue = np.maximum(rng.normal(avg_revenue, avg_revenue * 0.5, size=n), 0.0)
    post_revenue = np.where(
        churn,
        0.0,
        np.maximum(rng.normal(avg_revenue, avg_revenue * 0.4, size=n), 0.0),
    )
    discount_cost = np.where(is_treatment, float(incentive_cost), 0.0)

    return pd.DataFrame(
        {
            "assignment": assignment.astype(str),
            "churn": churn.astype(int),
            "revenue": post_revenue,
            "pre_revenue": pre_revenue,
            "discount_cost": discount_cost,
        }
    )


def analyze_experiment(
    data: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """Analyze experiment data end-to-end and produce a decision report.

    Expected columns: ``assignment`` ('control'/'treatment'), ``churn`` (0/1).
    Optional: ``revenue``, ``pre_revenue``, ``discount_cost``.
    """
    required = {"assignment", "churn"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    assignment = data["assignment"].astype(str)
    control_mask = assignment == "control"
    treatment_mask = assignment == "treatment"

    churn_test = proportions_z_test(
        data.loc[control_mask, "churn"],
        data.loc[treatment_mask, "churn"],
        alpha,
    )

    report: dict = {
        "n_control": int(control_mask.sum()),
        "n_treatment": int(treatment_mask.sum()),
        "churn_test": churn_test.to_dict(),
    }

    net_revenue_test = None
    if "revenue" in data.columns:
        discount = (
            data["discount_cost"]
            if "discount_cost" in data.columns
            else pd.Series(0.0, index=data.index)
        )
        net_revenue = data["revenue"] - discount

        if "pre_revenue" in data.columns:
            adjusted = apply_cuped(net_revenue, data["pre_revenue"], assignment)
            adjusted = adjusted.fillna(net_revenue)
        else:
            adjusted = net_revenue

        net_revenue_test = cuped_t_test(
            adjusted[control_mask],
            adjusted[treatment_mask],
            alpha,
        )
        mean_control = float(net_revenue[control_mask].mean())
        mean_treatment = float(net_revenue[treatment_mask].mean())
        net_revenue_test["lift"] = (
            (mean_treatment - mean_control) / mean_control if mean_control > 0 else float("inf")
        )
        report["net_revenue_test"] = net_revenue_test

    # Decision rule (see docs/ab_test_design.md)
    churn_wins = churn_test.significant and churn_test.p2 < churn_test.p1
    net_erodes = bool(net_revenue_test and net_revenue_test["mean_diff"] < 0)

    if churn_wins and not net_erodes:
        decision = "LAUNCH"
    elif churn_wins and net_erodes:
        decision = "OPTIMIZE"
    else:
        decision = "HOLD"
    report["decision"] = decision
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B experiment analysis (synthetic demo)")
    parser.add_argument("--n", type=int, default=10000, help="Total number of customers")
    parser.add_argument("--baseline", type=float, default=0.50, help="Baseline churn rate")
    parser.add_argument("--effect", type=float, default=0.20, help="Relative treatment effect")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ab_analysis_report.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    customers = pd.DataFrame({"customer_id": [f"C{i}" for i in range(args.n)]})
    assignment = assign_groups(customers, treatment_ratio=0.5, seed=42)
    outcomes = simulate_outcomes(
        assignment,
        seed=args.seed,
        baseline_churn=args.baseline,
        treatment_effect=args.effect,
    )
    report = analyze_experiment(outcomes, args.alpha)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Analysis report saved: %s", args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
