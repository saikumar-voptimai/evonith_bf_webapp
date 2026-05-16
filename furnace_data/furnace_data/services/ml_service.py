"""Pure ML dataset helpers - no Streamlit dependency.

All functions operate on a DataFrame that has already been loaded by the
caller. Loading and session-level caching are intentionally left to the adapter
or UI layers.

Public API
----------
now_ist_naive       Current time as IST tz-naive Timestamp.
parse_ist_naive     Parse an ISO-8601 / YYYY-MM-DD string as IST tz-naive.
ml_column_summary   Grouped column summary string for a ML dataset slice.
shift_window        Compute the (start, end) IST timestamps for an 8-hour shift.
slice_ml_df         Filter + optionally resample a ML DataFrame by date range.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


# IST time helpers

def now_ist_naive() -> pd.Timestamp:
    """Current time as an IST tz-naive :class:`~pandas.Timestamp`."""
    return pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=5, minutes=30)


def parse_ist_naive(s: str) -> pd.Timestamp:
    """Parse an ISO-8601 or YYYY-MM-DD string to a tz-naive IST Timestamp."""
    ts = pd.to_datetime(s)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
    return ts


# Shift window

def shift_window(shift_date: str, shift_label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(start, end)`` IST tz-naive timestamps for an 8-hour shift.

    Shifts:
    - A  06:00 - 14:00 IST
    - B  14:00 - 22:00 IST
    - C  22:00 - 06:00 IST (next day)

    Parameters
    ----------
    shift_date :
        ISO date string ``YYYY-MM-DD``.
    shift_label :
        One of ``"A"``, ``"B"``, ``"C"``.
    """
    date = pd.to_datetime(shift_date).normalize()
    if shift_label == "C":
        start = date + pd.Timedelta(hours=22)
        end = date + pd.Timedelta(hours=30)  # 30 h = next-day 06:00
    elif shift_label == "A":
        start = date + pd.Timedelta(hours=6)
        end = date + pd.Timedelta(hours=14)
    elif shift_label == "B":
        start = date + pd.Timedelta(hours=14)
        end = date + pd.Timedelta(hours=22)
    else:
        raise ValueError(f"Unknown shift_label: {shift_label!r}. Must be 'A', 'B', or 'C'.")
    return start, end


# ML dataset operations

def ml_column_summary(df: pd.DataFrame) -> str:
    """Return a compact grouped column summary for an ML dataset slice."""
    groups: dict[str, list[str]] = {
        "KPIs": [],
        "Process params": [],
        "Temperature": [],
        "Materials": [],
        "Hot metal / Slag": [],
        "Burden": [],
        "Other": [],
    }
    for col in df.columns:
        cu = col.upper()
        if any(k in cu for k in ["FUEL RATE", "ETACO", "PRODUCTIONTONNES", "COKE RATE KG", "UNITCOST"]):
            groups["KPIs"].append(col)
        elif any(k in cu for k in ["HOT BLAST", "TOPPRESSURE", "BOTTOMBAR", "TOPBAR", "STEAM",
                                    "O2 ENRICH", "PERMEABILITY", "DIFFERENTIAL PRESSURE", "RAFT",
                                    "TUYERE", "OXYGEN"]):
            groups["Process params"].append(col)
        elif any(k in cu for k in ["_TEMP_", "HEARTH_TEMP", "BELLY_TEMP", "BOSH_TEMP",
                                    "LOWER_STACK", "UPTAKE_TEMP", "HEAT LOAD"]):
            groups["Temperature"].append(col)
        elif any(k in cu for k in ["COKE_", "NUTCOKE_", "PCI_", "ORE_", "SINTER_", "PELLET_", "FLUX_"]):
            groups["Materials"].append(col)
        elif any(k in cu for k in ["CHEM_PCT", "SLAG_", "HMT_", "GEOMIN"]):
            groups["Hot metal / Slag"].append(col)
        elif any(k in cu for k in ["PORTION", "ANGLE", "DISCHARGE_TIME", "CHARGES", "STOCK"]):
            groups["Burden"].append(col)
        else:
            groups["Other"].append(col)

    lines = []
    for grp, cols in groups.items():
        if cols:
            suffix = " ..." if len(cols) > 6 else ""
            lines.append(f"  {grp} ({len(cols)}): {', '.join(cols[:6])}{suffix}")
    return "\n".join(lines)


def slice_ml_df(
    df: pd.DataFrame,
    req_start: pd.Timestamp,
    req_end: pd.Timestamp,
    *,
    columns: Optional[list[str]] = None,
    resample: Optional[str] = None,
) -> pd.DataFrame:
    """Slice and optionally filter / resample a ML dataset DataFrame.

    Parameters
    ----------
    df :
        Full ML dataset with IST tz-naive DatetimeIndex.
    req_start / req_end :
        IST tz-naive timestamps defining the requested range.
    columns :
        Optional list of keyword substrings for case-insensitive column filtering.
    resample :
        Pandas-compatible frequency alias for downsampling (e.g. ``"8h"``, ``"1d"``).
        ``None`` keeps the native 1-hour resolution.
    """
    sliced = df.loc[(df.index >= req_start) & (df.index <= req_end)].copy()

    if columns:
        matched: list[str] = []
        for kw in columns:
            kw_lower = kw.lower()
            matched += [c for c in df.columns if kw_lower in c.lower() and c not in matched]
        if matched:
            sliced = sliced[matched]

    if resample and resample != "1h":
        sliced = sliced.resample(resample).mean(numeric_only=True).dropna(how="all")

    return sliced
