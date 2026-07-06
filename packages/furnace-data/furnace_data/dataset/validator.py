"""Rigorous checks on a cleaned ML dataset DataFrame.

Usage::

    from furnace_data.dataset.validator import validate_dataset
    report = validate_dataset(df_clean)
    # report["passed"]   -> bool
    # report["warnings"] -> [str]
    # report["errors"]   -> [str]
    # report["checks"]   -> dict with detailed results per section
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# KPI columns (app-format names) that must be present and fully non-NaN
_KPI_COLS = [
    "ACT. FUEL RATEKG/THM.",
    "FURNACETOPGASANALYSISCO2ETACO",
    "PRODUCTIONTONNESPERHR",
    "COKE RATE KG/THM",
    "PCI_KG/THM",
]

# Valid ranges for key process parameters (inclusive bounds)
_RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "ACT. FUEL RATEKG/THM.":       (100, 670),
    "FURNACETOPGASANALYSISCO2ETACO": (38, 47),
    "PRODUCTIONTONNESPERHR":        (60, 100),
    "HOT BLAST VOLUMENM3/HR.":      (90_000, float("inf")),
    "PCI_KG/THM":                (70, float("inf")),
    "COKE RATE KG/THM":             (100, 500),
}

# NaN threshold: columns above this fraction are flagged
_NAN_WARN_FRACTION = 0.02

# Monthly minimum rows below which coverage is flagged
_MIN_MONTHLY_ROWS = 20

# Max tolerated gap between consecutive timestamps (hours) before flagging
_MAX_GAP_HOURS = 4.0


def validate_dataset(
    df: pd.DataFrame,
    zero_fill_cols: set[str] | None = None,
) -> dict[str, Any]:
    """Run rigorous checks on a cleaned ML dataset.

    Args:
        df:              Cleaned DataFrame (app-format column names, IST-naive
                         DatetimeIndex, no NaN expected after imputation).
        zero_fill_cols:  Set of app-format column names that are legitimately
                         zero (e.g. flux, pellet, sinter quantities).
                         If None, the function tries to derive it from the
                         default cleaning config.

    Returns:
        ``{
            "passed": bool,
            "warnings": [str],
            "errors": [str],
            "checks": {
                "shape": {...},
                "monthly_coverage": pd.Series,
                "gaps": {...},
                "nan_rates": {col: pct},
                "zero_audit": {col: {"count": int, "first_date": str}},
                "kpi_stats": pd.DataFrame,
                "range_violations": {col: {"count": int, "fraction": float}},
            }
        }``
    """
    warnings: list[str] = []
    errors: list[str] = []
    checks: dict[str, Any] = {}

    if df is None or df.empty:
        errors.append("Dataset is empty.")
        return {"passed": False, "warnings": warnings, "errors": errors, "checks": checks}

    # ── Derive zero_fill_cols from cleaning config if not provided ──────────
    if zero_fill_cols is None:
        try:
            from furnace_data.dataset.cleaning import build_default_config
            cfg = build_default_config()
            zero_fill_cols = set(cfg.zero_fill_columns)
        except Exception:
            zero_fill_cols = set()

    # ── 1. Shape ─────────────────────────────────────────────────────────────
    checks["shape"] = {
        "rows": len(df),
        "cols": len(df.columns),
        "start": str(df.index.min()),
        "end":   str(df.index.max()),
    }
    if len(df) < 100:
        warnings.append(f"Very few rows: {len(df)} (expected ≥ 100).")

    # ── 2. Monthly coverage ──────────────────────────────────────────────────
    monthly = df.resample("ME").size().rename("rows")
    monthly.index = monthly.index.strftime("%Y-%m")
    checks["monthly_coverage"] = monthly
    thin_months = monthly[monthly < _MIN_MONTHLY_ROWS]
    if not thin_months.empty:
        for m, cnt in thin_months.items():
            warnings.append(f"Thin coverage: {m} has only {cnt} rows.")

    # ── 3. Timestamp gaps ────────────────────────────────────────────────────
    if len(df) > 1:
        gaps_h = df.index.to_series().diff().dt.total_seconds().dropna() / 3600
        max_gap = float(gaps_h.max())
        large_gaps = gaps_h[gaps_h > _MAX_GAP_HOURS]
        gap_list = [
            {"start": str(df.index[i - 1]), "end": str(df.index[i]), "hours": round(float(h), 1)}
            for i, h in zip(large_gaps.index.map(df.index.get_loc), large_gaps)
            if isinstance(i, int)
        ]
        # Simpler: just note dates of large gaps
        large_gap_starts = [
            str(df.index[df.index.get_loc(idx) - 1]) if df.index.get_loc(idx) > 0 else "?"
            for idx in large_gaps.index
        ]
        checks["gaps"] = {"max_gap_hours": round(max_gap, 1), "n_large_gaps": len(large_gaps)}
        if max_gap > _MAX_GAP_HOURS:
            warnings.append(
                f"Max timestamp gap: {round(max_gap, 1)}h "
                f"({len(large_gaps)} gap(s) > {_MAX_GAP_HOURS}h)."
            )
    else:
        checks["gaps"] = {"max_gap_hours": None, "n_large_gaps": 0}

    # ── 4. NaN rates ─────────────────────────────────────────────────────────
    nan_frac = df.isna().mean()
    flagged_nan = nan_frac[nan_frac > _NAN_WARN_FRACTION].sort_values(ascending=False)
    checks["nan_rates"] = flagged_nan.to_dict()
    for col, frac in flagged_nan.items():
        warnings.append(f"High NaN: {col} = {frac:.1%}.")

    # ── 5. Zero audit (only columns not in zero_fill_cols) ───────────────────
    zero_audit: dict[str, dict] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in zero_fill_cols:
            continue
        zero_mask = df[col] == 0
        if zero_mask.any():
            zero_audit[col] = {
                "count": int(zero_mask.sum()),
                "first_date": str(df.index[zero_mask].min()),
            }
            errors.append(
                f"Unexpected zeros in {col}: {zero_mask.sum()} rows "
                f"(first: {df.index[zero_mask].min().date()})."
            )
    checks["zero_audit"] = zero_audit

    # ── 6. KPI stats ─────────────────────────────────────────────────────────
    kpi_present = [c for c in _KPI_COLS if c in df.columns]
    if kpi_present:
        stats = df[kpi_present].agg(["mean", "std", "min", "max"]).round(3)
        checks["kpi_stats"] = stats
    else:
        checks["kpi_stats"] = pd.DataFrame()
        errors.append(f"KPI columns missing: {[c for c in _KPI_COLS if c not in df.columns]}")

    # ── 7. Range violations ──────────────────────────────────────────────────
    range_violations: dict[str, dict] = {}
    for col, (lo, hi) in _RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        viol = ((s < lo) | (s > hi)) & s.notna()
        if viol.any():
            range_violations[col] = {
                "count": int(viol.sum()),
                "fraction": round(float(viol.mean()), 4),
            }
            warnings.append(
                f"Range violation in {col} [{lo}, {hi}]: "
                f"{viol.sum()} rows ({viol.mean():.1%})."
            )
    checks["range_violations"] = range_violations

    passed = len(errors) == 0
    return {"passed": passed, "warnings": warnings, "errors": errors, "checks": checks}
