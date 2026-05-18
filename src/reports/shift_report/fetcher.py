"""Fetch all raw data needed for one shift."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Literal

import pandas as pd
from sqlalchemy import text

from config.config_loader import load_config
from furnace_data.influx.online import fetch_online_df
from furnace_data.neon_db.offline import fetch_offline_data as _fetch_neon_table
from furnace_data.neon_db.offline import fetch_offline_report as _fetch_neon_offline
from furnace_data.relational.engine import build_relational_engine
from reports.base import ReportFetcher
from reports.shift_report.data import ShiftRawData
from utils.shift_windows import shift_window

_SHIFT_REPORT_CONFIG = load_config("shift_report.yml") or {}
_ONLINE_GROUPS = tuple(str(group) for group in _SHIFT_REPORT_CONFIG["online_groups"])
_ANALYSIS_LOOKBACK_DAYS = int(_SHIFT_REPORT_CONFIG["analysis_lookback_days"])
_ANALYSIS_TABLES = {
    str(key): str(value)
    for key, value in _SHIFT_REPORT_CONFIG.get("analysis_tables", {}).items()
}
_MATERIAL_FINES_TABLE = {
    str(key): str(value)
    for key, value in _SHIFT_REPORT_CONFIG.get("material_fines_table", {}).items()
}


def _safe(df: pd.DataFrame | None) -> pd.DataFrame:
    return df if df is not None and not df.empty else pd.DataFrame()


def _safe_table(
    table_name: str, start_utc: datetime, end_utc: datetime
) -> pd.DataFrame:
    try:
        return _safe(_fetch_neon_table(table_name, time_range=(start_utc, end_utc)))
    except Exception:
        return pd.DataFrame()


def _safe_material_fines_table(start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    query = text(
        f'SELECT * FROM "{_MATERIAL_FINES_TABLE["schema"]}".'
        f'"{_MATERIAL_FINES_TABLE["table"]}" '
        'WHERE "date_time" >= :start_time AND "date_time" <= :end_time '
        'ORDER BY "date_time"'
    )
    engine = None
    try:
        engine = build_relational_engine()
        df = pd.read_sql_query(
            query,
            engine,
            params={"start_time": start_utc, "end_time": end_utc},
        )
    except Exception:
        return pd.DataFrame()
    finally:
        if engine is not None:
            engine.dispose()

    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
        df = df.set_index("date_time")
        df.index.name = "time"
    return df


class ShiftFetcher(ReportFetcher[ShiftRawData]):
    def fetch(  # type: ignore[override]
        self,
        *,
        shift_date: date,
        shift_label: Literal["A", "B", "C"],
    ) -> ShiftRawData:
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
        analysis_start_utc = start_utc - timedelta(days=_ANALYSIS_LOOKBACK_DAYS)

        return ShiftRawData(
            shift_date=shift_date,
            shift_label=shift_label,
            shift_start_ist=start_ist,
            shift_end_ist=end_ist,
            online_df=online_df,
            hm_slag_df=hm_slag_df,
            charge_df=charge_df,
            ore_chemistry_df=_safe_table(
                _ANALYSIS_TABLES["ore"],
                analysis_start_utc,
                end_utc,
            ),
            fuel_chemistry_df=_safe_table(
                _ANALYSIS_TABLES["fuel"],
                analysis_start_utc,
                end_utc,
            ),
            flux_chemistry_df=_safe_table(
                _ANALYSIS_TABLES["flux"],
                analysis_start_utc,
                end_utc,
            ),
            material_fines_df=_safe_material_fines_table(analysis_start_utc, end_utc),
        )
