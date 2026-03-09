# FurnaceMind/utils/validators.py
# Purpose: Input validation helpers for FurnaceMind
# Fixed: Shift validator now accepts configurable sampling interval,
#        defaults to 15-minute intervals (32 rows per 8-hour shift)

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
    sampling_minutes: int = 15,
    tolerance: float = 0.2,
) -> None:
    """
    Validate a single shift dataframe.

    Args:
        df: The shift DataFrame to validate
        expected_hours: Shift duration in hours (default: 8)
        sampling_minutes: Data sampling interval in minutes (default: 15)
        tolerance: Fraction of expected rows allowed as deviation (default: 0.2 = 20%)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Shift data must have a DatetimeIndex.")

    if df.empty:
        raise ValueError("Shift DataFrame is empty.")

    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        raise ValueError("All columns must be numeric.")

    expected_rows = (expected_hours * 60) // sampling_minutes
    min_rows = int(expected_rows * (1 - tolerance))
    max_rows = int(expected_rows * (1 + tolerance))

    if len(df) < min_rows or len(df) > max_rows:
        raise ValueError(
            f"Expected approximately {expected_rows} rows for a {expected_hours}-hour shift "
            f"at {sampling_minutes}-minute intervals (tolerance ±{tolerance:.0%}), "
            f"but received {len(df)} rows."
        )