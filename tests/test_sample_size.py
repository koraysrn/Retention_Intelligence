"""Sample size calculation tests."""

import pytest
from src.experiments.sample_size import sample_size_proportions


def test_sample_size_baseline_20pct_mde_20pct() -> None:
    result = sample_size_proportions(
        baseline_rate=0.20,
        mde_relative=0.20,
        alpha=0.05,
        power=0.80,
        buffer=0.20,
    )
    # MDE: 0.20 -> 0.16 (absolute -4pp)
    assert abs(result.mde_absolute - 0.04) < 1e-9
    # Buffered n per group should be around ~1,730
    assert 1500 < result.n_per_group < 2000
    assert result.total_required == result.n_per_group * 2
    assert result.baseline_rate == 0.20
    assert abs(result.treatment_rate - 0.16) < 1e-9


def test_smaller_mde_requires_larger_sample() -> None:
    small_effect = sample_size_proportions(0.20, 0.10)
    large_effect = sample_size_proportions(0.20, 0.30)
    assert small_effect.n_per_group > large_effect.n_per_group


def test_buffer_scales_sample_size() -> None:
    no_buffer = sample_size_proportions(0.20, 0.20, buffer=0.0)
    with_buffer = sample_size_proportions(0.20, 0.20, buffer=0.20)
    assert with_buffer.n_per_group > no_buffer.n_per_group


def test_higher_power_requires_larger_sample() -> None:
    low_power = sample_size_proportions(0.20, 0.20, power=0.80)
    high_power = sample_size_proportions(0.20, 0.20, power=0.95)
    assert high_power.n_per_group > low_power.n_per_group


@pytest.mark.parametrize(
    "baseline,mde,alpha,power,buffer",
    [
        (0.0, 0.20, 0.05, 0.80, 0.20),
        (1.0, 0.20, 0.05, 0.80, 0.20),
        (0.20, 0.0, 0.05, 0.80, 0.20),
        (0.20, 1.0, 0.05, 0.80, 0.20),
        (0.20, 0.20, 0.0, 0.80, 0.20),
        (0.20, 0.20, 0.05, 1.0, 0.20),
        (0.20, 0.20, 0.05, 0.80, -0.1),
    ],
)
def test_invalid_inputs_raise(baseline, mde, alpha, power, buffer) -> None:
    with pytest.raises(ValueError):
        sample_size_proportions(baseline, mde, alpha, power, buffer)


def test_result_to_dict_keys() -> None:
    result = sample_size_proportions(0.20, 0.20)
    d = result.to_dict()
    for key in (
        "n_per_group",
        "total_required",
        "mde_absolute",
        "mde_relative",
        "power",
        "alpha",
        "baseline_rate",
        "treatment_rate",
    ):
        assert key in d
