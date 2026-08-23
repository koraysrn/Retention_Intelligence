"""A/B test sample size and power analysis.

Formula (two-proportion test):
    n = (Zα/2 + Zβ)² × [p1(1−p1) + p2(1−p2)] / (p1 − p2)²
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SampleSizeResult:
    n_per_group: int
    total_required: int
    mde_absolute: float
    mde_relative: float
    power: float
    alpha: float
    baseline_rate: float
    treatment_rate: float

    def to_dict(self) -> dict:
        return self.__dict__


def sample_size_proportions(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
    buffer: float = 0.20,
) -> SampleSizeResult:
    """Compute the sample size for a two-proportion test.

    Args:
        baseline_rate: Expected rate in the control group (e.g. churn rate).
        mde_relative: Relative minimum detectable effect (e.g. 0.20 = 20%).
        alpha: Type-I error probability.
        power: Statistical power (1 − β).
        buffer: Operational safety buffer (%).

    Returns:
        SampleSizeResult
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1")
    if not 0 < mde_relative < 1:
        raise ValueError("mde_relative must be between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1")
    if buffer < 0:
        raise ValueError("buffer cannot be negative")

    p1 = baseline_rate
    p2 = p1 * (1 - mde_relative)

    z_alpha = stats.norm.ppf(1 - alpha / 2)  # two-tailed
    z_beta = stats.norm.ppf(power)

    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p1 - p2) ** 2
    n = numerator / denominator

    n_per_group = int(n * (1 + buffer)) + 1
    return SampleSizeResult(
        n_per_group=n_per_group,
        total_required=n_per_group * 2,
        mde_absolute=p1 - p2,
        mde_relative=mde_relative,
        power=power,
        alpha=alpha,
        baseline_rate=p1,
        treatment_rate=p2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B sample size calculation")
    parser.add_argument("--baseline", type=float, required=True, help="Baseline rate (churn)")
    parser.add_argument("--mde", type=float, default=0.20, help="Relative MDE")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--buffer", type=float, default=0.20)
    parser.add_argument("--out", type=Path, default=Path("artifacts/sample_size_report.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = sample_size_proportions(args.baseline, args.mde, args.alpha, args.power, args.buffer)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("Sample size report saved: %s", args.out)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
