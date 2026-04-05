"""Shift partitioning utilities for the BF2 blast furnace operations.

This module defines the 8-hour A/B/C shift windows used throughout the
application for anomaly analysis, LLM reporting, and vector-store storage.
"""

# core/shift_builder.py

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class ShiftData:
    """Container for a single 8-hour shift slice.

    Attributes:
        shift_id:    Unique identifier in ``YYYY-MM-DD_Shift_A|B|C`` format.
        shift_name:  Human-readable label (e.g. ``"Shift A"``).
        shift_start: Inclusive start timestamp of the shift window.
        shift_end:   Exclusive end timestamp of the shift window.
        data:        Raw time-series DataFrame restricted to this window.
    """

    shift_id: str
    shift_name: str  # NEW (A / B / C)
    shift_start: pd.Timestamp
    shift_end: pd.Timestamp
    data: pd.DataFrame


class ShiftBuilder:
    """
    Creates fixed 8-hour shifts from time-series data.
    Shift naming follows plant convention: A, B, C.
    """

    def __init__(self, shift_hours: int = 8) -> None:
        """Initialise the builder.

        Args:
            shift_hours: Duration of each shift in hours.  Only ``8`` is
                supported; any other value raises ``ValueError``.
        """
        if shift_hours != 8:
            raise ValueError("Only 8-hour shifts are supported.")
        self.shift_hours = shift_hours

    def build_shifts(self, df: pd.DataFrame) -> Dict[str, ShiftData]:
        """Partition a DatetimeIndex DataFrame into 8-hour A/B/C shift windows.

        For each calendar day present in *df* three windows are produced:
        * **A** – 00:00–08:00
        * **B** – 08:00–16:00
        * **C** – 16:00–24:00

        Windows with no data are skipped silently.

        Args:
            df: Time-series DataFrame with a ``pd.DatetimeIndex``.

        Returns:
            Mapping of ``shift_id → ShiftData`` for every non-empty window found
            in *df*.

        Raises:
            ValueError: If *df* has no ``DatetimeIndex`` or is empty.
        """
        self._validate(df)
        df = df.sort_index()

        shifts: Dict[str, ShiftData] = {}

        for date, df_day in df.groupby(df.index.date):
            day_start = pd.Timestamp(date)

            windows = [
                (day_start, day_start + pd.Timedelta(hours=8), "A"),
                (
                    day_start + pd.Timedelta(hours=8),
                    day_start + pd.Timedelta(hours=16),
                    "B",
                ),
                (
                    day_start + pd.Timedelta(hours=16),
                    day_start + pd.Timedelta(hours=24),
                    "C",
                ),
            ]

            for start, end, label in windows:
                shift_df = df_day[(df_day.index >= start) & (df_day.index < end)]
                if shift_df.empty:
                    continue

                shift_id = f"{start.strftime('%Y-%m-%d')}_Shift_{label}"

                shifts[shift_id] = ShiftData(
                    shift_id=shift_id,
                    shift_name=f"Shift {label}",
                    shift_start=start,
                    shift_end=end,
                    data=shift_df.copy(),
                )

        return shifts

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Assert that *df* is suitable for shift partitioning.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If the index is not a ``DatetimeIndex`` or the frame is
                empty.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
