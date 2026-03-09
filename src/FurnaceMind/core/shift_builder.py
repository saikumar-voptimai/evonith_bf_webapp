# FurnaceMind/core/shift_builder.py
# Purpose: Create fixed 8-hour shifts from time-series data
# Fixed: Standardized shift ID format to "YYYY-MM-DD_SHIFT_X" (uppercase)
#        to match the main app and window_helpers.

from dataclasses import dataclass
from typing import Dict
import pandas as pd


# Canonical shift ID format — used everywhere
SHIFT_ID_FORMAT = "{date}_SHIFT_{label}"


def make_shift_id(dt: pd.Timestamp | str, label: str) -> str:
    """
    Generate a canonical shift ID.
    Usage: make_shift_id("2025-01-01", "A") → "2025-01-01_SHIFT_A"
    """
    if isinstance(dt, pd.Timestamp):
        date_str = dt.strftime("%Y-%m-%d")
    else:
        date_str = str(dt)[:10]
    return SHIFT_ID_FORMAT.format(date=date_str, label=label)



@dataclass
class ShiftData:
    shift_id: str
    shift_name: str        # A / B / C
    shift_start: pd.Timestamp
    shift_end: pd.Timestamp
    data: pd.DataFrame


class ShiftBuilder:
    """
    Creates fixed 8-hour shifts from time-series data.
    Shift naming follows plant convention: A, B, C.
    """

    def __init__(self, shift_hours: int = 8):
        if shift_hours != 8:
            raise ValueError("Only 8-hour shifts are supported.")
        self.shift_hours = shift_hours

    def build_shifts(self, df: pd.DataFrame) -> Dict[str, ShiftData]:
        self._validate(df)
        df = df.sort_index()

        shifts: Dict[str, ShiftData] = {}

        for date_val, df_day in df.groupby(df.index.date):
            day_start = pd.Timestamp(date_val)

            windows = [
                (day_start, day_start + pd.Timedelta(hours=8), "A"),
                (day_start + pd.Timedelta(hours=8), day_start + pd.Timedelta(hours=16), "B"),
                (day_start + pd.Timedelta(hours=16), day_start + pd.Timedelta(hours=24), "C"),
            ]

            for start, end, label in windows:
                shift_df = df_day[(df_day.index >= start) & (df_day.index < end)]
                if shift_df.empty:
                    continue

                shift_id = make_shift_id(start, label)

                shifts[shift_id] = ShiftData(
                    shift_id=shift_id,
                    shift_name=f"Shift {label}",
                    shift_start=start,
                    shift_end=end,
                    data=shift_df.copy(),
                )

        return shifts

    @staticmethod
    def _validate(df: pd.DataFrame):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")