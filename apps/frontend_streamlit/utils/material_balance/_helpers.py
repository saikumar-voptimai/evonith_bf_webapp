"""Package-private helpers shared across material_balance submodules."""

from __future__ import annotations

import pandas as pd


def _get_pct(row: pd.Series, col: str) -> float:
    """Pull a percentage column safely; NaN/missing → 0.0."""
    if col not in row.index:
        return 0.0
    v = row[col]
    if v is None or pd.isna(v):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
