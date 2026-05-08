"""Fetch all raw data needed for one shift from InfluxDB."""

from __future__ import annotations

from datetime import date, timezone
from typing import Literal

import pandas as pd

from furnace_data.influx.offline import fetch_offline_data as _fetch_offline
from furnace_data.influx.online import fetch_online_df
from reports.base import ReportFetcher
from reports.shift_report.data import ShiftRawData
from utils.shift_windows import shift_window

_ONLINE_GROUPS = [
    "process_params",
    "temperature_profile",
    "delta_t",
    "miscellaneous",
]
_OFFLINE_DB = "bf2_evonith_offline_utc"


def _safe(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Return a non-empty dataframe or an empty dataframe fallback.

    Args:
         - df: pd.DataFrame | None - Optional dataframe returned by a data fetch.

    Returns:
         - return: pd.DataFrame - Original dataframe or empty fallback.
    """
    return df if df is not None and not df.empty else pd.DataFrame()


class ShiftFetcher(ReportFetcher[ShiftRawData]):
    """Fetch raw online and offline data for one configured shift."""

    def fetch(  # type: ignore[override]
        self,
        *,
        shift_date: date,
        shift_label: Literal["A", "B", "C"],
    ) -> ShiftRawData:
        """
        Fetch raw datasets for one configured shift window.

        Args:
             - shift_date: date - Calendar date assigned to the shift.
             - shift_label: Literal["A", "B", "C"] - Shift label to fetch.

        Returns:
             - return: ShiftRawData - Raw online and offline shift data.
        """
        start_ist, end_ist = shift_window(shift_date, shift_label)
        start_utc = start_ist.astimezone(timezone.utc)
        end_utc = end_ist.astimezone(timezone.utc)

        online_df = _safe(
            fetch_online_df(
                selected_measurements=_ONLINE_GROUPS,
                time_range="last 8 hours",
                window_by="15 minutes",
                start_time_override=start_utc,
                end_time_override=end_utc,
            )
        )

        hm_slag_df = _safe(
            _fetch_offline(
                measurement="hotmetal_slag_updated_data",
                time_range=(start_utc, end_utc),
                database=_OFFLINE_DB,
            )
        )

        charge_df = _safe(
            _fetch_offline(
                measurement="latest_charge_data",
                time_range=(start_utc, end_utc),
                database=_OFFLINE_DB,
            )
        )

        return ShiftRawData(
            shift_date=shift_date,
            shift_label=shift_label,
            shift_start_ist=start_ist,
            shift_end_ist=end_ist,
            online_df=online_df,
            hm_slag_df=hm_slag_df,
            charge_df=charge_df,
        )
