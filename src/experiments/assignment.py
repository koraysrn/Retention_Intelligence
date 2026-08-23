"""Stratified randomization (A/B assignment) — ecommerce_data.csv.

Customers are deterministically assigned to control/treatment groups using
hash-based assignment. Hash-based assignment preserves the expected ratio within
each stratum (country, risk decile, tenure bucket) and is reproducible with a
seed.
"""

from __future__ import annotations

import hashlib

import pandas as pd

STRATA_COLUMNS = ["country", "risk_decile", "tenure_bucket"]


def assign_groups(
    customers: pd.DataFrame,
    treatment_ratio: float = 0.5,
    seed: int = 42,
    id_column: str = "customer_id",
) -> pd.Series:
    """Deterministically assign customers to control/treatment groups.

    Args:
        customers: Customer DataFrame; must contain ``id_column``.
        treatment_ratio: Probability of assignment to the treatment group (0-1).
        seed: Seed for reproducibility.
        id_column: Unique key column used for hashing.

    Returns:
        A Series with 'control' / 'treatment' values.
    """
    if not 0 < treatment_ratio < 1:
        raise ValueError("treatment_ratio must be between 0 and 1")
    if id_column not in customers.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame")

    keys = customers[id_column].astype(str) + f"::{seed}"
    hashes = keys.map(lambda s: int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16))
    quantile = hashes / 2**128
    return pd.Series(
        ["treatment" if q < treatment_ratio else "control" for q in quantile],
        index=customers.index,
        name="group",
    )


def strata_balance_summary(
    customers: pd.DataFrame,
    assignment: pd.Series,
) -> pd.DataFrame:
    """Report group size and treatment ratio per stratum."""
    strata_cols = [c for c in STRATA_COLUMNS if c in customers.columns]
    if not strata_cols:
        return pd.DataFrame()

    df = customers[strata_cols].copy()
    df["group"] = assignment.astype(str)

    rows: list[dict] = []
    for keys, sub in df.groupby(strata_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        n_treatment = int((sub["group"] == "treatment").sum())
        row = dict(zip(strata_cols, keys, strict=True))
        row["n"] = n
        row["treatment_ratio"] = n_treatment / n if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
