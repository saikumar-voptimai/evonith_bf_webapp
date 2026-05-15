"""Fetch all raw data needed for one shift."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Literal

import pandas as pd

from furnace_data.influx.online import fetch_online_df
from furnace_data.neon_db.offline import fetch_offline_report as _fetch_neon_offline
from reports.base import ReportFetcher
from reports.shift_report.data import ShiftRawData

_IST = timezone(timedelta(hours=5, minutes=30))
_SHIFT_START_H: dict[str, int] = {"A": 6, "B": 14, "C": 22}

_ONLINE_GROUPS = [
    "process_params",
    "temperature_profile",
    "delta_t",
    "miscellaneous",
]
def _shift_window(d: date, label: str) -> tuple[datetime, datetime]:
    """Return (start_ist, end_ist) for the given shift."""
    h = _SHIFT_START_H[label]
    start = datetime(d.year, d.month, d.day, h, 0, 0, tzinfo=_IST)
    return start, start + timedelta(hours=8)


def _safe(df: pd.DataFrame | None) -> pd.DataFrame:
    return df if df is not None and not df.empty else pd.DataFrame()


class ShiftFetcher(ReportFetcher[ShiftRawData]):
    def fetch(  # type: ignore[override]
        self,
        *,
        shift_date: date,
        shift_label: Literal["A", "B", "C"],
    ) -> ShiftRawData:
        start_ist, end_ist = _shift_window(shift_date, shift_label)
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
            _fetch_neon_offline(
                report_type="HM_SLAG",
                time_range=(start_utc, end_utc),
            )
        )

        charge_df = _safe(
            _fetch_neon_offline(
                report_type="CHARGE",
                time_range=(start_utc, end_utc),
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
