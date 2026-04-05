# utils/validators.py
# Purpose: Input validation helpers for FurnaceMind

import pandas as pd


def validate_hourly_dataframe(df: pd.DataFrame) -> None:
    """
    Validate long-duration hourly data (used in offline ingestion).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        raise ValueError("All columns must be numeric.")


def validate_shift_dataframe(
    df: pd.DataFrame,
    expected_hours: int = 8,
) -> None:
    """
    Validate a single 8-hour shift dataframe (used in UI).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Shift data must have a DatetimeIndex.")

    if df.empty:
        raise ValueError("Shift DataFrame is empty.")

    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        raise ValueError("All columns must be numeric.")

    # Basic sanity check on size (hourly data → ~8 rows)
    if len(df) < expected_hours - 1 or len(df) > expected_hours + 1:
        raise ValueError(
            f"Expected approximately {expected_hours} rows for an 8-hour shift, "
            f"but received {len(df)} rows."
        )
