"""Static-dataset defaults for BMO slag basicity bounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

OXIDE_COLUMNS = (
    "SLAG_PCT_CAO",
    "SLAG_PCT_MGO",
    "SLAG_PCT_SIO2",
)


def derive_basicity_bounds_from_static_dataset(
    path: str | Path, *, window_days: int = 30
) -> dict[str, float]:
    """
    Derive initial BMO basicity bounds from the local static ML dataset.

    Bounds use the recent P10/P90 range and calculate plant convention ratios
    directly from slag chemistry:
    - Slag basicity = CaO / SiO2
    - Slag T basicity = (CaO + MgO) / SiO2
    """

    csv_path = Path(path)
    if not csv_path.exists():
        return {}

    header = pd.read_csv(csv_path, nrows=0)
    available = set(header.columns)
    if not set(OXIDE_COLUMNS).issubset(available):
        return {}

    usecols = list(OXIDE_COLUMNS)
    time_col = "time" if "time" in available else None
    if time_col:
        usecols.append(time_col)

    df = pd.read_csv(csv_path, usecols=usecols)
    for column in OXIDE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["SLAG_PCT_SIO2"] > 0].copy()
    if df.empty:
        return {}

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        max_time = df[time_col].max()
        if pd.notna(max_time):
            recent = df[df[time_col] >= max_time - pd.Timedelta(days=window_days)]
            if not recent.empty:
                df = recent

    basicity = df["SLAG_PCT_CAO"] / df["SLAG_PCT_SIO2"]
    t_basicity = (df["SLAG_PCT_CAO"] + df["SLAG_PCT_MGO"]) / df["SLAG_PCT_SIO2"]
    if basicity.dropna().empty or t_basicity.dropna().empty:
        return {}
    return {
        "target_slag_basicity_min": _quantile(basicity, 0.10),
        "target_slag_basicity_max": _quantile(basicity, 0.90),
        "target_slag_t_basicity_min": _quantile(t_basicity, 0.10),
        "target_slag_t_basicity_max": _quantile(t_basicity, 0.90),
    }


def _quantile(series: pd.Series, q: float) -> float:
    value: Any = series.dropna().quantile(q)
    return round(float(value), 3)
