"""Stratified A/B assignment tests."""

import pandas as pd
import pytest
from src.experiments.assignment import assign_groups, strata_balance_summary


def _customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(200)],
            "country": ["UK"] * 100 + ["DE"] * 100,
            "risk_decile": list(range(10)) * 20,
            "tenure_bucket": ["0-30", "31-90"] * 100,
        }
    )


def test_assignment_is_binary_and_deterministic() -> None:
    customers = _customers()
    a1 = assign_groups(customers, treatment_ratio=0.5, seed=42)
    a2 = assign_groups(customers, treatment_ratio=0.5, seed=42)
    assert set(a1.unique()).issubset({"control", "treatment"})
    assert (a1 == a2).all()


def test_different_seed_changes_assignment() -> None:
    customers = _customers()
    a1 = assign_groups(customers, treatment_ratio=0.5, seed=1)
    a2 = assign_groups(customers, treatment_ratio=0.5, seed=2)
    assert not (a1 == a2).all()


def test_treatment_ratio_approximate() -> None:
    customers = _customers()
    a = assign_groups(customers, treatment_ratio=0.3, seed=42)
    ratio = (a == "treatment").mean()
    assert 0.2 < ratio < 0.4


def test_invalid_treatment_ratio_raises() -> None:
    with pytest.raises(ValueError):
        assign_groups(_customers(), treatment_ratio=1.5)


def test_missing_id_column_raises() -> None:
    with pytest.raises(ValueError, match="customer_id"):
        assign_groups(pd.DataFrame({"x": [1, 2]}), id_column="customer_id")


def test_strata_balance_summary_has_expected_columns() -> None:
    customers = _customers()
    a = assign_groups(customers, 0.5, seed=42)
    summary = strata_balance_summary(customers, a)
    assert {"country", "risk_decile", "tenure_bucket", "n", "treatment_ratio"}.issubset(
        summary.columns
    )
    assert summary["treatment_ratio"].between(0, 1).all()


def test_strata_balance_summary_empty_when_no_strata() -> None:
    customers = pd.DataFrame({"customer_id": ["A", "B", "C"]})
    a = assign_groups(customers, 0.5, seed=1)
    summary = strata_balance_summary(customers, a)
    assert summary.empty
