"""Fetch all raw data needed for one report window."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import logging

import pandas as pd
from sqlalchemy import text

from config.config_loader import load_config
from furnace_data.influx.online import fetch_online_df
from furnace_data.neon_db.offline import fetch_offline_data as _fetch_neon_table
from furnace_data.neon_db.offline import fetch_offline_report as _fetch_neon_offline
from furnace_data.relational.engine import build_relational_engine
from reports.base import ReportFetcher
from reports.furnace_report.data import (
    ReportTimeframe,
    ReportType,
    ShiftLabel,
    ShiftRawData,
)
from reports.furnace_report.timeframe import get_report_timeframe

_SHIFT_REPORT_CONFIG = load_config("shift_report.yml") or {}
_ONLINE_GROUPS = tuple(str(group) for group in _SHIFT_REPORT_CONFIG["online_groups"])
_TOTAL_O2_FLOW_COLUMN = "Process Params - BF2_TOTAL OXYGEN FLOW"
_ONLINE_WINDOW_BY_REPORT_TYPE = {
    "Shift": "15 minutes",
    "Day": "1 hour",
    **{
        str(key): str(value)
        for key, value in (_SHIFT_REPORT_CONFIG.get("online_window_by") or {}).items()
    },
}
_ANALYSIS_LOOKBACK_DAYS = int(_SHIFT_REPORT_CONFIG["analysis_lookback_days"])
_ANALYSIS_TABLES = {
    str(key): str(value)
    for key, value in _SHIFT_REPORT_CONFIG.get("analysis_tables", {}).items()
}
_MATERIAL_FINES_TABLE = {
    str(key): str(value)
    for key, value in _SHIFT_REPORT_CONFIG.get("material_fines_table", {}).items()
}
logger = logging.getLogger(__name__)


def _safe(df: pd.DataFrame | None) -> pd.DataFrame:
    return df if df is not None and not df.empty else pd.DataFrame()


def _safe_table(
    table_name: str, start_utc: datetime, end_utc: datetime
) -> pd.DataFrame:
    try:
        return _safe(_fetch_neon_table(table_name, time_range=(start_utc, end_utc)))
    except Exception:
        logger.warning(
            "Failed to fetch furnace report table %s",
            table_name,
            exc_info=True,
        )
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
        logger.warning("Failed to fetch material fines table", exc_info=True)
        return pd.DataFrame()
    finally:
        if engine is not None:
            engine.dispose()

    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
        df = df.set_index("date_time")
        df.index.name = "time"
    return df


@lru_cache(maxsize=1)
def _cached_materials_table() -> pd.DataFrame:
    return _safe(
        _fetch_neon_table(
            "materials",
            time_range="full",
            columns=("material_code", "material_name", "is_active"),
        )
    )


def _safe_materials_table() -> pd.DataFrame:
    try:
        return _cached_materials_table().copy()
    except Exception:
        logger.warning("Failed to fetch materials table", exc_info=True)
        _cached_materials_table.cache_clear()
        return pd.DataFrame()


def _total_o2_flow_window(timeframe: ReportTimeframe) -> tuple[datetime, datetime]:
    return timeframe.start_ist + timedelta(minutes=15), timeframe.end_ist + timedelta(
        minutes=14
    )


def _safe_total_o2_flow_max(timeframe: ReportTimeframe) -> float | None:
    if timeframe.report_type != "Day":
        return None

    start_ist, end_ist = _total_o2_flow_window(timeframe)
    try:
        df = _safe(
            fetch_online_df(
                selected_measurements=["process_params"],
                time_range="last 1 day",
                request_type="ts",
                window_by=None,
                start_time_override=start_ist.astimezone(timezone.utc),
                end_time_override=end_ist.astimezone(timezone.utc),
            )
        )
    except Exception:
        logger.warning("Failed to fetch total oxygen flow window", exc_info=True)
        return None

    if df.empty or _TOTAL_O2_FLOW_COLUMN not in df.columns:
        return None
    values = pd.to_numeric(df[_TOTAL_O2_FLOW_COLUMN], errors="coerce").dropna()
    return round(float(values.max()), 2) if len(values) else None


class ShiftFetcher(ReportFetcher[ShiftRawData]):
    def fetch(  # type: ignore[override]
        self,
        *,
        shift_date: date | None = None,
        shift_label: ShiftLabel | str | None = None,
        report_type: ReportType | str = "Shift",
        timeframe: ReportTimeframe | None = None,
    ) -> ShiftRawData:
        if timeframe is None:
            if shift_date is None:
                raise ValueError(
                    "shift_date is required when timeframe is not provided."
                )
            timeframe = get_report_timeframe(report_type, shift_date, shift_label)

        start_ist, end_ist = timeframe.start_ist, timeframe.end_ist
        start_utc = start_ist.astimezone(timezone.utc)
        end_utc = end_ist.astimezone(timezone.utc)

        online_df = _safe(
            fetch_online_df(
                selected_measurements=_ONLINE_GROUPS,
                time_range="last 8 hours",
                window_by=_ONLINE_WINDOW_BY_REPORT_TYPE.get(
                    timeframe.report_type,
                    _ONLINE_WINDOW_BY_REPORT_TYPE["Shift"],
                ),
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
            shift_date=timeframe.selected_date,
            shift_label=timeframe.shift_label,
            shift_start_ist=start_ist,
            shift_end_ist=end_ist,
            online_df=online_df,
            total_o2_flow_max=_safe_total_o2_flow_max(timeframe),
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
            materials_df=_safe_materials_table(),
            report_type=timeframe.report_type,
        )
