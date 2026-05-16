"""Agent adapter functions for FurnaceMind data tools."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import ValidationError

from agents.furnacemind.tools._utils import log_tool_error, summarize_df
from agents.furnacemind.tools.artifact_store import get_artifact_store
from agents.furnacemind.tools.schemas import (
    ConcatArgs,
    MergeArgs,
    MLDataArgs,
    OfflineFetchArgs,
    OnlineFetchArgs,
    StaticShiftArgs,
)
from data.fetch_presets import OFFLINE_REPORT_LABEL_MAP
from furnace_data.services import data_fetch_service, ml_dataset_service, ml_service

_REPORT_TYPE_ALIASES: Dict[str, str] = {
    "RAW_MATERIAL_COMPOSITION": "RM_COMPOSITION",
}


def fetch_online_data(
    *,
    lookback: Optional[str] = None,
    window: Optional[str] = None,
    measurement_groups: Optional[List[str]] = None,
    start_time_utc: Optional[str] = None,
    end_time_utc: Optional[str] = None,
    lookback_days: Optional[int] = None,
    lookback_hours: Optional[int] = None,
    lookback_minutes: Optional[int] = None,
) -> str:
    """Fetch live telemetry and store the result as the active dataset."""
    if lookback is None and start_time_utc is None:
        if lookback_hours is not None:
            lookback = f"{lookback_hours}h"
        elif lookback_days is not None:
            lookback = f"{lookback_days}d"
        elif lookback_minutes is not None:
            lookback = f"{lookback_minutes}m"

    params: Dict[str, Any] = {
        "lookback": lookback,
        "window": window,
        "measurement_groups": measurement_groups,
        "start_time_utc": start_time_utc or None,
        "end_time_utc": end_time_utc or None,
    }
    try:
        args = OnlineFetchArgs.model_validate(params)
        df, time_range_label = data_fetch_service.fetch_online_df_for_agent(
            lookback=args.lookback,
            window=args.window,
            measurement_groups=list(args.measurement_groups) if args.measurement_groups else None,
            start_time_utc=args.start_time_utc,
            end_time_utc=args.end_time_utc,
        )

        store = get_artifact_store()
        dataset_id = store.new_dataset_id("online")
        meta: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "type": "online",
            "time_range": time_range_label,
            "window": args.window,
            "measurement_groups": list(args.measurement_groups) if args.measurement_groups else None,
        }
        store.save_dataset(dataset_id=dataset_id, df=df, meta=meta)
        return summarize_df(df, dataset_id=dataset_id, title="ONLINE DATA")

    except (ValidationError, Exception) as exc:
        log_tool_error(tool_name="fetch_online_data", params=params, error=str(exc))
        return f"Fetch Error: {exc}"


def fetch_offline_data(
    *,
    report_type: str,
    table_name: Optional[str] = None,
    start_time_utc: Optional[str] = None,
    end_time_utc: Optional[str] = None,
    lookback_days: Optional[int] = 10,
    cadence: Optional[str] = None,
) -> str:
    """Fetch an offline report and store the result as the active dataset."""
    params: Dict[str, Any] = {
        "report_type": report_type,
        "table_name": table_name,
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "lookback_days": lookback_days,
        "cadence": cadence,
    }
    try:
        args = OfflineFetchArgs.model_validate(params)
        resolved_report_type = _REPORT_TYPE_ALIASES.get(args.report_type, args.report_type)
        label = (
            "Bunker Report"
            if resolved_report_type == "RM_COMPOSITION"
            else OFFLINE_REPORT_LABEL_MAP.get(resolved_report_type, resolved_report_type)
        )

        now = datetime.now(timezone.utc)
        end = (
            data_fetch_service.parse_iso8601_utc(args.end_time_utc)
            if args.end_time_utc
            else now
        )
        if args.start_time_utc:
            start = data_fetch_service.parse_iso8601_utc(args.start_time_utc)
        else:
            lookback = max(1, min(int(args.lookback_days or 10), 365))
            start = end - timedelta(days=lookback)

        if start > now:
            return (
                f"Fetch Error: start_time_utc {start.isoformat()} is in the future "
                f"(current UTC: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}). "
                "No offline data exists for future dates."
            )
        if end > now:
            end = now

        df, source_detail = data_fetch_service.fetch_offline_df_for_agent(
            report_type=resolved_report_type,
            table_name=args.table_name,
            start=start,
            end=end,
            cadence=args.cadence,
        )

        if df is not None and not df.empty:
            df = df.rename(columns={c: f"Offline[{label}] - {c}" for c in df.columns})

        store = get_artifact_store()
        dataset_id = store.new_dataset_id("offline")
        meta: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "type": "offline",
            "report_type": args.report_type,
            "source": "offline_database",
            "label": label,
            "source_detail": source_detail,
            "start_time_utc": start.isoformat(),
            "end_time_utc": end.isoformat(),
            "cadence": args.cadence,
        }
        store.save_dataset(dataset_id=dataset_id, df=df, meta=meta)
        return summarize_df(df, dataset_id=dataset_id, title="OFFLINE DATA")

    except (ValidationError, Exception) as exc:
        log_tool_error(tool_name="fetch_offline_data", params=params, error=str(exc))
        return f"Fetch Error: {exc}"


def merge_furnace_data(
    *,
    online_dataset_id: str,
    offline_dataset_ids: List[str],
    fill_method: str = "ffill",
) -> str:
    """Merge offline datasets onto an online dataset's timestamp spine."""
    params: Dict[str, Any] = {
        "online_dataset_id": online_dataset_id,
        "offline_dataset_ids": offline_dataset_ids,
        "fill_method": fill_method,
    }
    try:
        args = MergeArgs.model_validate(params)
        store = get_artifact_store()

        online_entry = store.get_dataset(args.online_dataset_id)
        if not online_entry or "df" not in online_entry:
            raise ValueError(f"Unknown online_dataset_id: {args.online_dataset_id}")
        online_df: pd.DataFrame = online_entry["df"]
        if online_df is None or online_df.empty:
            raise ValueError("Online dataset is empty; cannot merge.")

        offline_parts: List[pd.DataFrame] = []
        for dataset_id in args.offline_dataset_ids:
            entry = store.get_dataset(dataset_id)
            if not entry or "df" not in entry:
                raise ValueError(f"Unknown offline_dataset_id: {dataset_id}")
            if entry["df"] is not None and not entry["df"].empty:
                offline_parts.append(entry["df"])

        if not offline_parts:
            raise ValueError("No non-empty offline datasets provided.")

        merged = data_fetch_service.merge_dfs(online_df, offline_parts, args.fill_method)
        dataset_id = store.new_dataset_id("merged")
        meta: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "type": "merged",
            "online_dataset_id": args.online_dataset_id,
            "offline_dataset_ids": args.offline_dataset_ids,
            "fill_method": args.fill_method,
        }
        store.save_dataset(dataset_id=dataset_id, df=merged, meta=meta)
        return summarize_df(merged, dataset_id=dataset_id, title="MERGED DATA")

    except (ValidationError, Exception) as exc:
        log_tool_error(tool_name="merge_furnace_data", params=params, error=str(exc))
        return f"Merge Error: {exc}"


def fetch_ml_data(
    *,
    start_time: str,
    end_time: Optional[str] = None,
    resample: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> str:
    """Fetch a date-range slice from the static pre-merged ML dataset."""
    params: Dict[str, Any] = {
        "start_time": start_time,
        "end_time": end_time,
        "resample": resample,
        "columns": columns,
    }
    try:
        args = MLDataArgs.model_validate(params)
        req_start = ml_service.parse_ist_naive(args.start_time)
        req_end = (
            ml_service.parse_ist_naive(args.end_time)
            if args.end_time
            else ml_service.now_ist_naive()
        )
        if req_end <= req_start:
            return "Error: end_time must be after start_time."

        store = get_artifact_store()
        df = store.get_ml_cache()
        if df is None:
            df = ml_dataset_service.load_static_dataset()
            if df.empty:
                raise ValueError("Static ML dataset returned no rows.")
            store.set_ml_cache(df)

        csv_start, csv_end = df.index.min(), df.index.max()
        overlap_start = max(req_start, csv_start)
        overlap_end = min(req_end, csv_end)

        if overlap_start > overlap_end + pd.Timedelta(hours=1):
            return (
                f"Requested range ({req_start} - {req_end} IST) has no overlap with the "
                f"static ML dataset ({csv_start} - {csv_end} IST). "
                "Use fetch_online_data directly for this query."
            )

        slice_df = ml_service.slice_ml_df(
            df,
            overlap_start,
            overlap_end,
            columns=args.columns,
            resample=args.resample,
        )

        dataset_id = store.new_dataset_id("ml_static")
        meta: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "type": "ml_static",
            "source": "offline_feed.historical_static_ml_dataset",
            "start": str(overlap_start.date()),
            "end": str(overlap_end.date()),
            "resample": args.resample or "1h (native)",
        }
        store.save_dataset(dataset_id=dataset_id, df=slice_df, meta=meta)

        col_summary = (
            ml_service.ml_column_summary(slice_df)
            if not args.columns
            else f"  Filtered: {list(slice_df.columns)}"
        )
        summary = (
            f"ML STATIC DATA | {overlap_start.date()} -> {overlap_end.date()} IST\n"
            f"dataset_id={dataset_id} | {len(slice_df)} rows x {len(slice_df.columns)} cols\n"
            f"Columns available:\n{col_summary}"
        )

        gap_threshold = pd.Timedelta(hours=2)
        if req_end > csv_end + gap_threshold:
            gap_hours = max(1, int((req_end - csv_end).total_seconds() / 3600) + 1)
            summary += (
                f"\n\nGAP NOTE: Static dataset ends {csv_end} IST; your request goes to "
                f"{req_end.strftime('%Y-%m-%d %H:%M')} IST.\n"
                f"To fill the ~{gap_hours}h gap:\n"
                f"  1. fetch_online_data(lookback_hours={gap_hours})\n"
                f"  2. concat_datasets(dataset_ids=['{dataset_id}', '<online_dataset_id>'])\n"
                "Note: online columns use InfluxDB names (e.g. 'fuel_rate') - ML static uses "
                "ML names (e.g. 'ACT. FUEL RATEKG/THM.'). Plot whichever column is non-null."
            )

        return summary

    except Exception as exc:
        log_tool_error(tool_name="fetch_ml_data", params=params, error=str(exc))
        return f"fetch_ml_data Error: {exc}"


def concat_datasets(*, dataset_ids: List[str]) -> str:
    """Concatenate multiple datasets vertically."""
    params: Dict[str, Any] = {"dataset_ids": dataset_ids}
    try:
        args = ConcatArgs.model_validate(params)
        store = get_artifact_store()

        frames: List[pd.DataFrame] = []
        for dataset_id in args.dataset_ids:
            entry = store.get_dataset(dataset_id)
            if not entry or "df" not in entry:
                raise ValueError(f"Unknown dataset_id: '{dataset_id}'. Fetch it first.")
            frames.append(entry["df"])

        combined = data_fetch_service.concat_dfs(frames)
        dataset_id = store.new_dataset_id("concat")
        meta: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "type": "concat",
            "source_ids": args.dataset_ids,
            "rows": len(combined),
            "start": str(combined.index.min()),
            "end": str(combined.index.max()),
        }
        store.save_dataset(dataset_id=dataset_id, df=combined, meta=meta)

        return (
            f"CONCAT DATA | {combined.index.min()} -> {combined.index.max()}\n"
            f"dataset_id={dataset_id} | {len(combined)} rows x {len(combined.columns)} cols\n"
            f"Sources: {args.dataset_ids}"
        )

    except Exception as exc:
        log_tool_error(tool_name="concat_datasets", params=params, error=str(exc))
        return f"concat_datasets Error: {exc}"


def load_static_shift_data(*, shift_date: str, shift_label: str) -> str:
    """Load one 8-hour shift from the static ML dataset."""
    params: Dict[str, Any] = {"shift_date": shift_date, "shift_label": shift_label}
    try:
        args = StaticShiftArgs.model_validate(params)
        shift_start, shift_end = ml_service.shift_window(args.shift_date, args.shift_label)

        store = get_artifact_store()
        df = store.get_ml_cache()
        if df is None:
            df = ml_dataset_service.load_static_dataset()
            if df.empty:
                raise ValueError("Static ML dataset returned no rows.")
            store.set_ml_cache(df)

        csv_min, csv_max = df.index.min(), df.index.max()
        if shift_start < csv_min or shift_end > csv_max + pd.Timedelta(hours=1):
            return (
                f"Shift {shift_date} Shift {shift_label} ({shift_start} to {shift_end}) "
                f"is outside the static dataset range ({csv_min} to {csv_max}). "
                "Use fetch_online_data and fetch_offline_data to retrieve this shift instead."
            )

        shift_df = df.loc[(df.index >= shift_start) & (df.index < shift_end)]
        if shift_df.empty:
            return f"No data found for {shift_date} Shift {shift_label} in the static dataset."

        dataset_id = store.new_dataset_id("static_shift")
        meta: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "type": "static_shift",
            "shift_date": shift_date,
            "shift_label": shift_label,
            "shift_start": str(shift_start),
            "shift_end": str(shift_end),
            "source": "offline_feed.historical_static_ml_dataset",
            "rows": len(shift_df),
        }
        store.save_dataset(dataset_id=dataset_id, df=shift_df, meta=meta)

        return summarize_df(
            shift_df,
            dataset_id=dataset_id,
            title=f"Static shift data: {shift_date} Shift {shift_label} ({len(shift_df)} hourly rows)",
        )

    except Exception as exc:
        log_tool_error(tool_name="load_static_shift_data", params=params, error=str(exc))
        return f"Error loading static shift data: {exc}"
