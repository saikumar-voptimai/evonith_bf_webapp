import pandas as pd
import streamlit as st
from langchain.tools import tool
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Literal, List

from pydantic import BaseModel, Field, ValidationError

from utils.helper_functions_explorer import data_retrieval as dr
from config.config_loader import load_config

# CONFIG
config = load_config("setting_ds_dv.yml")

OFFLINE_MEASUREMENTS = config.get("offline_measurements", {}) or {}
INFLUX_OFFLINE_DB = (config.get("influx_offline", {}) or {}).get("database", "bf2_evonith_offline_utc")

MEASUREMENT_LABELS = {
    "heatload_delta_t": "Heatload Delta T",
    "process_params": "Process Params",
    "temperature_profile": "Temperature Profile",
    "cooling_water": "Cooling Water",
    "delta_t": "Delta T",
    "miscellaneous": "Miscellaneous",
}

FREQUENCY_TO_TIMEDTA = {
    "None": None,
    "1 minute": "1min",
    "5 minutes": "5min",
    "10 minutes": "10min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "1 hour": "1h",
    "6 hours": "6h",
    "8 hours": "8h",
    "12 hours": "12h",
    "1 day": "1d",
}

FIELD_LABELS = {
    internal_key: human_label
    for mapping in config["data_mapping"].values()
    for human_label, internal_key in mapping.items()
}


_TOOL_ERRORS_PATH = Path(__file__).resolve().parent / "tool_errors.md"


_DATASET_CSV_PATH = Path("current_furnace_data.csv")
_ML_DATASET_PATH = Path("data/ml_dataset_filtered.csv")


def _ensure_dataset_store() -> Dict[str, Any]:
    if "fm_datasets" not in st.session_state or not isinstance(st.session_state.get("fm_datasets"), dict):
        st.session_state["fm_datasets"] = {}
    return st.session_state["fm_datasets"]


def _new_dataset_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    counter = st.session_state.get("fm_dataset_counter", 0) + 1
    st.session_state["fm_dataset_counter"] = counter
    return f"{prefix}_{ts}_{counter}"


def _to_ist_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        # Assume UTC if tz-naive
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    df.index = df.index.tz_convert("Asia/Kolkata")
    df.index.name = "time (IST)"
    return df


def _parse_iso8601_utc(s: str) -> datetime:
    dt = pd.to_datetime(s, utc=True)
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    raise ValueError(f"Invalid datetime: {s}")


def _lookback_to_time_range_key(*, days: int | None, hours: int | None, minutes: int | None) -> str:
    if days is not None:
        if days <= 0:
            raise ValueError("lookback_days must be >= 1")
        return f"last {int(days)} days" if int(days) != 1 else "last 1 day"
    if hours is not None:
        if hours <= 0:
            raise ValueError("lookback_hours must be >= 1")
        return f"last {int(hours)} hours" if int(hours) != 1 else "last 1 hour"
    if minutes is not None:
        if minutes <= 0:
            raise ValueError("lookback_minutes must be >= 1")
        return f"last {int(minutes)} minutes" if int(minutes) != 1 else "last 1 minute"
    return "last 8 hours"


def _resolve_online_window(*, lookback: timedelta, window: Optional[str]) -> str:
    if isinstance(window, str) and window.strip():
        return window.strip()
    # Policy from user:
    # - hourly averaging if more than 1 day
    # - else 15mins averaging
    return "1 hour" if lookback > timedelta(days=1) else "15 minutes"


class OnlineFetchArgs(BaseModel):
    lookback_days: Optional[int] = Field(default=None, description="Look back this many days (max 90). Mutually exclusive with lookback_hours/minutes.")
    lookback_hours: Optional[int] = Field(default=None, description="Look back this many hours. Mutually exclusive with lookback_days/minutes.")
    lookback_minutes: Optional[int] = Field(default=None, description="Look back this many minutes. Mutually exclusive with lookback_days/hours.")
    window: Optional[str] = Field(default=None, description="Averaging window. If omitted, tool applies policy: >1 day => 1 hour, else 15 minutes.")
    measurement_groups: Optional[List[Literal[
        "process_params",
        "cooling_water",
        "heatload_delta_t",
        "delta_t",
        "temperature_profile",
        "miscellaneous",
    ]]] = Field(default=None, description="Which online measurement groups to include. If omitted, fetches all groups.")


class OfflineReportType(str):
    """Canonical offline report types for tool-calling."""


class OfflineFetchArgs(BaseModel):
    report_type: Literal["HM_SLAG", "CHARGE", "RAW_MATERIAL_COMPOSITION", "DPR"] = Field(
        description="Which offline dataset to fetch. RAW_MATERIAL_COMPOSITION maps to Bunker Report in config."
    )
    start_time_utc: Optional[str] = Field(default=None, description="ISO-8601 UTC start time. If omitted, uses lookback_days.")
    end_time_utc: Optional[str] = Field(default=None, description="ISO-8601 UTC end time. Defaults to now.")
    lookback_days: Optional[int] = Field(default=10, description="If start_time_utc is omitted, fetch the last N days (defaults to 10).")
    cadence: Optional[Literal["1h", "8h", "1d"]] = Field(
        default=None,
        description="Resampling cadence. If omitted, defaults by report_type: HM_SLAG/CHARGE=1h, RAW_MATERIAL_COMPOSITION=8h, DPR=1d.",
    )


class MergeArgs(BaseModel):
    online_dataset_id: str = Field(description="Dataset id returned by fetch_online_data.")
    offline_dataset_ids: List[str] = Field(description="One or more dataset ids returned by fetch_offline_data.")
    fill_method: Literal["ffill", "none"] = Field(default="ffill", description="How to align offline rows onto online timestamps.")


class StaticShiftArgs(BaseModel):
    shift_date: str = Field(description="ISO date string YYYY-MM-DD")
    shift_label: Literal["A", "B", "C"] = Field(description="Shift: A (00:00-08:00), B (08:00-16:00), C (16:00-24:00) IST")


def _save_dataset(*, dataset_id: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    store = _ensure_dataset_store()
    store[dataset_id] = {"df": df, "meta": meta}
    # Keep a conventional 'current' dataset for plotting tool
    df.to_csv(_DATASET_CSV_PATH, index=True)
    st.session_state.copilot_df = df
    st.session_state.copilot_df_meta = meta


def _summarize_df(df: pd.DataFrame, *, dataset_id: str, title: str) -> str:
    if df is None or df.empty:
        return f"{title}: No data found."
    preview = df.head(2).to_string() if len(df) else "<empty>"
    return (
        f"{title}: dataset_id={dataset_id}\n"
        f"Saved to '{_DATASET_CSV_PATH.as_posix()}'.\n"
        f"Shape: {df.shape}\n"
        f"Columns ({len(df.columns)}): {list(df.columns)}\n\n"
        f"Preview:\n{preview}"
    )


def load_static_shift_data(*, shift_date: str, shift_label: str) -> str:
    """Load 8-hour shift data from the static ML dataset (hourly, online+offline pre-merged).

    Covers Jan 2024 to Mar 2026. Returns an error if the requested shift is outside
    this range, instructing the LLM to fall back to fetch_online_data + fetch_offline_data.
    """
    params = {"shift_date": shift_date, "shift_label": shift_label}
    try:
        args = StaticShiftArgs.model_validate(params)

        # Compute shift window (IST, tz-naive to match CSV)
        date = pd.to_datetime(args.shift_date).normalize()
        shift_hours = {"A": (0, 8), "B": (8, 16), "C": (16, 24)}
        start_h, end_h = shift_hours[args.shift_label]

        shift_start = date + pd.Timedelta(hours=start_h)
        shift_end = date + pd.Timedelta(hours=end_h)

        if not _ML_DATASET_PATH.exists():
            return f"Error: Static ML dataset not found at {_ML_DATASET_PATH}"

        df = pd.read_csv(_ML_DATASET_PATH, index_col=0, parse_dates=True)

        # Check date range coverage
        csv_min, csv_max = df.index.min(), df.index.max()
        if shift_start < csv_min or shift_end > csv_max + pd.Timedelta(hours=1):
            return (
                f"Shift {shift_date} Shift {shift_label} ({shift_start} to {shift_end}) "
                f"is outside the static dataset range ({csv_min} to {csv_max}). "
                f"Use fetch_online_data and fetch_offline_data to retrieve this shift's data instead."
            )

        shift_df = df.loc[(df.index >= shift_start) & (df.index < shift_end)]

        if shift_df.empty:
            return f"No data rows found for {shift_date} Shift {shift_label} in the static dataset."

        dataset_id = _new_dataset_id("static_shift")
        meta = {
            "type": "static_shift",
            "shift_date": shift_date,
            "shift_label": shift_label,
            "shift_start": str(shift_start),
            "shift_end": str(shift_end),
            "source": str(_ML_DATASET_PATH),
            "rows": len(shift_df),
        }
        _save_dataset(dataset_id=dataset_id, df=shift_df, meta=meta)

        return _summarize_df(
            shift_df,
            dataset_id=dataset_id,
            title=f"Static shift data: {shift_date} Shift {shift_label} ({len(shift_df)} hourly rows)",
        )

    except Exception as e:
        _append_tool_error(tool_name="load_static_shift_data", params=params, error=str(e))
        return f"Error loading static shift data: {e}"


def fetch_online_data(*, lookback_days: int | None = None, lookback_hours: int | None = None, lookback_minutes: int | None = None,
                      window: str | None = None, measurement_groups: list[str] | None = None) -> str:
    """Fetch online (high-frequency) telemetry with policy-constrained lookback and averaging.

    Policy:
    - lookback_days max 90
    - if lookback > 1 day and window omitted => 1 hour averaging
    - else (<=1 day) and window omitted => 15 minutes averaging
    """
    params = {
        "lookback_days": lookback_days,
        "lookback_hours": lookback_hours,
        "lookback_minutes": lookback_minutes,
        "window": window,
        "measurement_groups": measurement_groups,
    }
    try:
        args = OnlineFetchArgs.model_validate(params)
        # Mutually exclusive enforcement
        provided = [v is not None for v in [args.lookback_days, args.lookback_hours, args.lookback_minutes]]
        if sum(provided) > 1:
            raise ValueError("Provide only one of lookback_days, lookback_hours, lookback_minutes")
        if args.lookback_days is not None and args.lookback_days > 90:
            raise ValueError("Online lookback_days exceeds max 90")

        time_range_key = _lookback_to_time_range_key(days=args.lookback_days, hours=args.lookback_hours, minutes=args.lookback_minutes)
        normalized_time_range = _normalize_time_range(time_range_key)

        # Determine actual lookback timedelta for policy
        lookback_td = getattr(dr, "TIMEDELTAS", {}).get(normalized_time_range)
        if not isinstance(lookback_td, timedelta):
            lookback_td = timedelta(hours=8)
        window_final = _resolve_online_window(lookback=lookback_td, window=args.window)

        selected_measurements = list(args.measurement_groups) if args.measurement_groups else [
            "process_params",
            "cooling_water",
            "heatload_delta_t",
            "delta_t",
            "temperature_profile",
            "miscellaneous",
        ]

        df = dr.fetch_online_df(
            selected_measurements=selected_measurements,
            time_range=normalized_time_range,
            average_range=window_final,
            FREQUENCY_TO_TIMEDTA=FREQUENCY_TO_TIMEDTA,
            MEASUREMENT_LABELS=MEASUREMENT_LABELS,
            FIELD_LABELS=FIELD_LABELS,
        )
        df = _to_ist_index(df)

        dataset_id = _new_dataset_id("online")
        meta = {
            "type": "online",
            "time_range": normalized_time_range,
            "window": window_final,
            "measurement_groups": selected_measurements,
        }
        _save_dataset(dataset_id=dataset_id, df=df, meta=meta)
        return _summarize_df(df, dataset_id=dataset_id, title="ONLINE DATA")

    except (ValidationError, Exception) as e:
        _append_tool_error(tool_name="fetch_online_data", params=params, error=str(e))
        return f"Fetch Error: {str(e)}"


def fetch_offline_data(*, report_type: str, start_time_utc: str | None = None, end_time_utc: str | None = None,
                       lookback_days: int | None = 10, cadence: str | None = None) -> str:
    """Fetch offline (report) datasets with type-specific cadence defaults.

    Defaults:
    - HM_SLAG, CHARGE: hourly (1h)
    - RAW_MATERIAL_COMPOSITION: shiftwise (8h)
    - DPR: daily (1d)
    """
    params = {
        "report_type": report_type,
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "lookback_days": lookback_days,
        "cadence": cadence,
    }
    try:
        args = OfflineFetchArgs.model_validate(params)

        # Resolve measurement label and Influx measurement name
        label_map = {
            "HM_SLAG": "HM & Slag",
            "CHARGE": "Charge",
            "RAW_MATERIAL_COMPOSITION": "Bunker Report",
            "DPR": "DPR",
        }
        label = label_map.get(args.report_type)
        if not label:
            raise ValueError(f"Unsupported report_type: {args.report_type}")
        measurement = OFFLINE_MEASUREMENTS.get(label)
        if not measurement:
            raise ValueError(f"Offline measurement not configured for label: {label}")

        now = datetime.now(timezone.utc)
        end = _parse_iso8601_utc(args.end_time_utc) if args.end_time_utc else now
        if args.start_time_utc:
            start = _parse_iso8601_utc(args.start_time_utc)
        else:
            lb = int(args.lookback_days or 10)
            lb = max(1, min(lb, 365))
            start = end - timedelta(days=lb)

        cadence_default = {
            "HM_SLAG": "1h",
            "CHARGE": "1h",
            "RAW_MATERIAL_COMPOSITION": "8h",
            "DPR": "1d",
        }[args.report_type]
        cadence_final = args.cadence or cadence_default

        df = dr.fetch_offline_data(
            measurement=measurement,
            time_range=(start, end),
            database=INFLUX_OFFLINE_DB,
        )

        # Offline fetch returns UTC index (as per helper); convert + resample
        df = _to_ist_index(df)
        if df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex):
            df = df.resample(cadence_final).mean(numeric_only=True)
            df = df.dropna(how="all")

        # Prefix columns to avoid collisions during merge
        df = df.rename(columns={c: f"Offline[{label}] - {c}" for c in df.columns}) if df is not None else df

        dataset_id = _new_dataset_id("offline")
        meta = {
            "type": "offline",
            "report_type": args.report_type,
            "label": label,
            "measurement": measurement,
            "db": INFLUX_OFFLINE_DB,
            "start_time_utc": start.isoformat(),
            "end_time_utc": end.isoformat(),
            "cadence": cadence_final,
        }
        _save_dataset(dataset_id=dataset_id, df=df, meta=meta)
        return _summarize_df(df, dataset_id=dataset_id, title="OFFLINE DATA")

    except (ValidationError, Exception) as e:
        _append_tool_error(tool_name="fetch_offline_data", params=params, error=str(e))
        return f"Fetch Error: {str(e)}"


def merge_furnace_data(*, online_dataset_id: str, offline_dataset_ids: list[str], fill_method: str = "ffill") -> str:
    """Merge offline datasets onto an online dataset, repeating/forward-filling offline to match online frequency."""
    params = {
        "online_dataset_id": online_dataset_id,
        "offline_dataset_ids": offline_dataset_ids,
        "fill_method": fill_method,
    }
    try:
        args = MergeArgs.model_validate(params)
        store = _ensure_dataset_store()

        online_entry = store.get(args.online_dataset_id)
        if not online_entry or "df" not in online_entry:
            raise ValueError(f"Unknown online_dataset_id: {args.online_dataset_id}")
        online_df = online_entry["df"]
        if online_df is None or online_df.empty:
            raise ValueError("Online dataset is empty; cannot merge")

        offline_parts: list[pd.DataFrame] = []
        for did in args.offline_dataset_ids:
            ent = store.get(did)
            if not ent or "df" not in ent:
                raise ValueError(f"Unknown offline_dataset_id: {did}")
            df_part = ent["df"]
            if df_part is None or df_part.empty:
                continue
            offline_parts.append(df_part)

        if not offline_parts:
            raise ValueError("No non-empty offline datasets provided")

        offline_df = offline_parts[0]
        for part in offline_parts[1:]:
            offline_df = offline_df.join(part, how="outer")

        online_df = _to_ist_index(online_df)
        offline_df = _to_ist_index(offline_df)

        if args.fill_method == "ffill":
            offline_aligned = offline_df.sort_index().reindex(online_df.index, method="ffill")
        else:
            offline_aligned = offline_df

        merged = online_df.join(offline_aligned, how="left")

        dataset_id = _new_dataset_id("merged")
        meta = {
            "type": "merged",
            "online_dataset_id": args.online_dataset_id,
            "offline_dataset_ids": args.offline_dataset_ids,
            "fill_method": args.fill_method,
        }
        _save_dataset(dataset_id=dataset_id, df=merged, meta=meta)
        return _summarize_df(merged, dataset_id=dataset_id, title="MERGED DATA")

    except (ValidationError, Exception) as e:
        _append_tool_error(tool_name="merge_furnace_data", params=params, error=str(e))
        return f"Merge Error: {str(e)}"


def get_openai_tool_schemas() -> list[dict]:
    """Return OpenAI/OpenRouter tool schemas for LLM function-calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_online_data",
                "description": "Fetch online telemetry (max lookback 90 days). If window omitted: >1 day => 1 hour avg, else 15 minutes avg. Saves data to current_furnace_data.csv and returns dataset_id + column preview.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lookback_days": {"type": "integer", "minimum": 1, "maximum": 90},
                        "lookback_hours": {"type": "integer", "minimum": 1},
                        "lookback_minutes": {"type": "integer", "minimum": 1},
                        "window": {"type": "string", "description": "Averaging window like '15 minutes' or '1 hour'. Optional."},
                        "measurement_groups": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "process_params",
                                    "cooling_water",
                                    "heatload_delta_t",
                                    "delta_t",
                                    "temperature_profile",
                                    "miscellaneous",
                                ],
                            },
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_offline_data",
                "description": "Fetch offline report datasets (HM/Slag, Charge, Raw material composition/Bunker, DPR). Defaults: HM_SLAG/CHARGE hourly; RAW_MATERIAL_COMPOSITION shiftwise (8h); DPR daily (1d). Saves to current_furnace_data.csv and returns dataset_id + preview.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report_type": {"type": "string", "enum": ["HM_SLAG", "CHARGE", "RAW_MATERIAL_COMPOSITION", "DPR"]},
                        "start_time_utc": {"type": "string", "description": "ISO-8601 UTC start time. Optional."},
                        "end_time_utc": {"type": "string", "description": "ISO-8601 UTC end time. Optional; defaults to now."},
                        "lookback_days": {"type": "integer", "minimum": 1, "maximum": 365},
                        "cadence": {"type": "string", "enum": ["1h", "8h", "1d"], "description": "Optional override for resampling cadence."},
                    },
                    "required": ["report_type"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "merge_furnace_data",
                "description": "Merge offline datasets onto an online dataset by aligning to online timestamps (repeat/forward-fill). Produces merged dataset_id and writes current_furnace_data.csv.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "online_dataset_id": {"type": "string"},
                        "offline_dataset_ids": {"type": "array", "items": {"type": "string"}},
                        "fill_method": {"type": "string", "enum": ["ffill", "none"], "default": "ffill"},
                    },
                    "required": ["online_dataset_id", "offline_dataset_ids"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_shift_history",
                "description": "Search past shift summaries (semantic).",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_docs",
                "description": "Search uploaded knowledge documents (SOPs, manuals, specs).",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python_plot",
                "description": "Execute restricted Python to create a Plotly figure 'fig' using df (loaded from current_furnace_data.csv).",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_static_shift_data",
                "description": (
                    "Load 8-hour shift data from the static ML dataset (hourly, online+offline pre-merged). "
                    "Covers Jan 2024 to Mar 2026. Returns error if the shift is outside this range — "
                    "use fetch_online_data + fetch_offline_data as fallback."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shift_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                        "shift_label": {
                            "type": "string",
                            "enum": ["A", "B", "C"],
                            "description": "Shift: A (00:00-08:00), B (08:00-16:00), C (16:00-24:00) IST",
                        },
                    },
                    "required": ["shift_date", "shift_label"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def execute_openai_tool_call(*, name: str, arguments: Dict[str, Any]) -> str:
    """Dispatcher used by the OpenRouter tool-calling loop."""
    if name == "fetch_online_data":
        return fetch_online_data(**arguments)
    if name == "fetch_offline_data":
        return fetch_offline_data(**arguments)
    if name == "merge_furnace_data":
        return merge_furnace_data(**arguments)
    if name == "search_shift_history":
        return search_shift_history.invoke(arguments)
    if name == "search_knowledge_docs":
        return search_knowledge_docs.invoke(arguments)
    if name == "execute_python_plot":
        return execute_python_plot.invoke(arguments)
    if name == "load_static_shift_data":
        return load_static_shift_data(**arguments)
    return f"Unknown tool: {name}"


def _append_tool_error(*, tool_name: str, params: Dict[str, Any], error: str) -> None:
    """Append tool failure details to tool_errors.md (best-effort, never raises)."""
    try:
        _TOOL_ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        entry = (
            f"\n\n## {ts}\n"
            f"**Tool:** {tool_name}\n\n"
            f"**Params:** `{json.dumps(params, ensure_ascii=False)}`\n\n"
            f"**Error:**\n\n```\n{error}\n```\n"
        )
        if _TOOL_ERRORS_PATH.exists():
            _TOOL_ERRORS_PATH.write_text(_TOOL_ERRORS_PATH.read_text(encoding="utf-8") + entry, encoding="utf-8")
        else:
            _TOOL_ERRORS_PATH.write_text(
                "# FurnaceMind Tool Errors & Learnings\n\n"
                "This file is auto-updated when tool execution fails during AI Co-Operate sessions.\n\n---\n"
                + entry,
                encoding="utf-8",
            )
    except Exception:
        return


def _normalize_time_range(user_time_range: str) -> str:
    """Normalize natural language into dr.TIMEDELTAS-compatible keys, extending TIMEDELTAS as needed."""
    tr = (user_time_range or "").strip().lower()
    if not tr:
        return "last 8 hours"

    # Already supported
    if hasattr(dr, "TIMEDELTAS") and tr in dr.TIMEDELTAS:
        return tr

    m = re.search(r"last\s+(\d+)\s*(minute|minutes|min|mins)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} minutes" if n != 1 else "last 1 minute"
        if hasattr(dr, "TIMEDELTAS"):
            dr.TIMEDELTAS.setdefault(key, timedelta(minutes=n))
        return key

    m = re.search(r"last\s+(\d+)\s*(hour|hours|hr|hrs|h)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} hours" if n != 1 else "last 1 hour"
        if hasattr(dr, "TIMEDELTAS"):
            dr.TIMEDELTAS.setdefault(key, timedelta(hours=n))
        return key

    m = re.search(r"last\s+(\d+)\s*(day|days|d)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} days" if n != 1 else "last 1 day"
        if hasattr(dr, "TIMEDELTAS"):
            dr.TIMEDELTAS.setdefault(key, timedelta(days=n))
        return key

    m = re.search(r"last\s+(\d+)\s*(week|weeks|w)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} weeks" if n != 1 else "last 1 week"
        if hasattr(dr, "TIMEDELTAS"):
            dr.TIMEDELTAS.setdefault(key, timedelta(weeks=n))
        return key

    # Fallback to safe default
    return "last 8 hours"


def _normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _infer_online_time_range_and_window(user_request: str) -> tuple[str, str]:
    """Infer online time_range + averaging window from natural language."""
    q = _normalize_text(user_request)

    # Time range: explicit "last N ..." wins
    m = re.search(r"\blast\s+(\d+)\s*(h|hr|hrs|hour|hours)\b", q)
    if m:
        time_range = f"last {int(m.group(1))} hours"
    else:
        m = re.search(r"\blast\s+(\d+)\s*(m|min|mins|minute|minutes)\b", q)
        if m:
            time_range = f"last {int(m.group(1))} minutes"
        elif "today" in q:
            time_range = "last 1 day"
        elif "yesterday" in q:
            time_range = "last 1 day"
        elif "shift" in q or "last shift" in q:
            time_range = "last 8 hours"
        elif any(k in q for k in ["now", "current", "live"]):
            time_range = "last 1 hour"
        else:
            time_range = "last 8 hours"

    normalized_time_range = _normalize_time_range(time_range)

    # Window: allow explicit "X min avg"
    m = re.search(r"\b(\d+)\s*(m|min|mins|minute|minutes)\s*(avg|average)\b", q)
    if m:
        window = f"{int(m.group(1))} minutes"
        return normalized_time_range, window

    # Explicit "raw" / "no averaging"
    if any(k in q for k in ["no averaging", "no avg", "raw", "unaveraged", "30 sec", "30 second", "30 seconds"]):
        return normalized_time_range, "None"

    # Default based on inferred delta
    delta = None
    if hasattr(dr, "TIMEDELTAS"):
        delta = dr.TIMEDELTAS.get(normalized_time_range)

    if isinstance(delta, timedelta):
        if delta <= timedelta(hours=1):
            window = "1 minute"
        elif delta <= timedelta(hours=6):
            window = "5 minutes"
        elif delta <= timedelta(hours=12):
            window = "10 minutes"
        else:
            window = "15 minutes"
    else:
        window = "15 minutes"

    return normalized_time_range, window


def _infer_offline_time_range(user_request: str) -> tuple[tuple[datetime, datetime], str]:
    """Infer a UTC (start,end) tuple for offline measurements.

    Offline data may be delayed (manual entry), so for requests like "today" we widen the window.
    Returns ((start_utc, end_utc), note)
    """
    q = _normalize_text(user_request)
    now = datetime.now(timezone.utc)

    # ISO dates: 2026-03-10 or 2026-03-10 to 2026-03-12
    iso_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    if iso_dates:
        start_d = pd.to_datetime(iso_dates[0], utc=True).to_pydatetime()
        if len(iso_dates) >= 2:
            end_d = pd.to_datetime(iso_dates[1], utc=True).to_pydatetime()
        else:
            end_d = start_d

        start = datetime(start_d.year, start_d.month, start_d.day, tzinfo=timezone.utc)
        end = datetime(end_d.year, end_d.month, end_d.day, 23, 59, 59, tzinfo=timezone.utc)
        return (start, end), "explicit date range"

    # Relative ranges
    m = re.search(r"\blast\s+(\d+)\s*(day|days|d)\b", q)
    if m:
        days = max(1, int(m.group(1)))
        start = now - timedelta(days=min(days, 120))
        return (start, now), f"last {days} days"

    m = re.search(r"\blast\s+(\d+)\s*(week|weeks|w)\b", q)
    if m:
        weeks = max(1, int(m.group(1)))
        start = now - timedelta(weeks=min(weeks, 20))
        return (start, now), f"last {weeks} weeks"

    if "yesterday" in q or "previous day" in q:
        # Widen to catch delayed entry
        start = now - timedelta(days=3)
        return (start, now), "yesterday (widened for delayed entry)"

    if "today" in q:
        # Widen to catch delayed entry
        start = now - timedelta(days=2)
        return (start, now), "today (widened for delayed entry)"

    if "shift" in q:
        start = now - timedelta(days=3)
        return (start, now), "recent shifts (widened)"

    # Safe default
    start = now - timedelta(days=10)
    return (start, now), "default last 10 days"


def _infer_offline_measurement_labels(user_request: str) -> list[str]:
    """Infer which offline measurement(s) the user is referring to.

    Returns measurement *labels* (keys in OFFLINE_MEASUREMENTS).
    """
    if not OFFLINE_MEASUREMENTS:
        return []

    q = _normalize_text(user_request)
    hits: list[str] = []

    # Keyword/synonym matching first
    synonyms: dict[str, list[str]] = {
        "HM & Slag": ["hot metal", "hm", "h m", "slag", "hmt", "silicon", "si"],
        "Bunker Report": ["bunker", "rm", "raw material", "coke", "sinter", "pellet"],
        "DPR": ["dpr", "daily production", "production", "prod"],
        "Charge": ["charge", "charging"],
    }
    for label, keys in synonyms.items():
        if label in OFFLINE_MEASUREMENTS and any(re.search(rf"\b{re.escape(k)}\b", q) for k in keys if k.isalnum()):
            hits.append(label)
        elif label in OFFLINE_MEASUREMENTS and any(k in q for k in keys if not k.isalnum()):
            hits.append(label)

    # Match by label text itself
    for label in OFFLINE_MEASUREMENTS.keys():
        if not isinstance(label, str) or not label.strip():
            continue
        if _normalize_text(label) in q:
            hits.append(label)

    # Unique preserve order
    uniq: list[str] = []
    seen = set()
    for h in hits:
        if h not in seen:
            uniq.append(h)
            seen.add(h)
    return uniq


def _should_fetch_offline(user_request: str) -> bool:
    q = _normalize_text(user_request)
    # Strong offline intent
    if any(k in q for k in ["bunker", "dpr", "charge", "hot metal", "slag", "lab", "analysis"]):
        return True
    if re.search(r"\bhm\b", q):
        return True
    return False


def _should_fetch_online(user_request: str) -> bool:
    q = _normalize_text(user_request)
    # If user asked for a trend/time window, that's usually online
    if any(k in q for k in ["trend", "plot", "graph", "chart", "live", "now", "current", "last ", "minutes", "hours"]):
        return True
    # Common online process words
    if any(k in q for k in ["pressure", "temp", "temperature", "o2", "oxygen", "pci", "fuel", "coke rate", "heatload", "delta t"]):
        return True
    return False


def _safe_exec(code: str, local_vars: Dict[str, Any]) -> None:
    """Execute plotting code with a restricted builtins set and basic static checks."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Empty code string")

    # Basic static blocks
    banned = [
        r"\bimport\b",
        r"\bopen\s*\(",
        r"__import__",
        r"\bos\b",
        r"\bsubprocess\b",
        r"\bsys\b",
        r"\beval\b",
        r"\bexec\b",
    ]
    for pat in banned:
        if re.search(pat, code):
            raise ValueError(f"Disallowed token in code: {pat}")

    safe_builtins = {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "float": float,
        "int": int,
        "str": str,
        "print": print,
    }

    exec(code, {"__builtins__": safe_builtins}, local_vars)


@tool
def fetch_and_summarize_data(time_range: str, window: str = "15 minutes") -> str:
    """
    Fetch data and save to a temp file. 
    Returns the first 5 rows and column names so the Python tool knows how to code.
    """
    
    try:
        normalized_time_range = _normalize_time_range(time_range)

        df = dr.fetch_online_df(
            selected_measurements=[
                "process_params",
                "cooling_water",
                "heatload_delta_t",
                "delta_t",
                "temperature_profile",
                "miscellaneous",
            ],
            time_range=normalized_time_range,
            average_range=window,
            FREQUENCY_TO_TIMEDTA=FREQUENCY_TO_TIMEDTA,
            MEASUREMENT_LABELS=MEASUREMENT_LABELS,
            FIELD_LABELS=FIELD_LABELS,
        )
    except Exception as e:
        _append_tool_error(
            tool_name="fetch_and_summarize_data",
            params={"time_range": time_range, "window": window},
            error=str(e),
        )
        return f"Fetch Error: {str(e)}"
    
    if df is None or df.empty:
        return "No data found."

    # Save to a fixed path for the Python executor to find
    # Keep index so timestamps remain available
    df.to_csv("current_furnace_data.csv", index=True)
    st.session_state.copilot_df = df
    
    # Give the LLM a 'peek' at the data
    summary = (
        "Data saved to 'current_furnace_data.csv'.\n"
        "Note: Columns are renamed as 'Measurement - Field'.\n"
        f"Time range: {time_range} | Window: {window}\n"
        f"Columns ({len(df.columns)}): {list(df.columns)}\n\n"
        f"Preview:\n{df.head(2).to_string()}"
    )
    return summary


@tool
def fetch_and_summarize_furnace_data(request: str) -> str:
    """Smart Influx fetch for AI Co-Operate.

    Provide a natural-language request. The tool will:
    - Decide whether the request needs ONLINE data (30s sampling) via `fetch_online_df`,
      OFFLINE data (manual/shift/day cadence) via `fetch_offline_data`, or BOTH.
    - Infer start/end time windows and an averaging window for online data.
    - Optionally combine online+offline into a single dataframe (offline forward-filled onto online timestamps).

    Side effects:
    - Saves the dataframe to `current_furnace_data.csv`.
    - Stores it in `st.session_state.copilot_df`.
    """
    params: Dict[str, Any] = {"request": request}
    try:
        offline_intent = _should_fetch_offline(request)
        online_intent = _should_fetch_online(request)

        # If user is clearly talking offline, allow offline-only; otherwise default to online
        if not offline_intent and not online_intent:
            online_intent = True

        offline_labels = _infer_offline_measurement_labels(request) if offline_intent else []
        if offline_intent and not offline_labels and OFFLINE_MEASUREMENTS:
            choices = ", ".join(list(OFFLINE_MEASUREMENTS.keys()))
            return (
                "Offline data requested but the measurement is ambiguous. "
                f"Please specify one of: {choices}."
            )

        online_df: Optional[pd.DataFrame] = None
        offline_df: Optional[pd.DataFrame] = None

        online_time_range = None
        online_window = None
        if online_intent:
            online_time_range, online_window = _infer_online_time_range_and_window(request)
            normalized_time_range = _normalize_time_range(online_time_range)

            online_df = dr.fetch_online_df(
                selected_measurements=[
                    "process_params",
                    "cooling_water",
                    "heatload_delta_t",
                    "delta_t",
                    "temperature_profile",
                    "miscellaneous",
                ],
                time_range=normalized_time_range,
                average_range=online_window,
                FREQUENCY_TO_TIMEDTA=FREQUENCY_TO_TIMEDTA,
                MEASUREMENT_LABELS=MEASUREMENT_LABELS,
                FIELD_LABELS=FIELD_LABELS,
            )

        offline_range_note = None
        if offline_intent and offline_labels:
            (start_utc, end_utc), offline_range_note = _infer_offline_time_range(request)
            parts: list[pd.DataFrame] = []
            for label in offline_labels:
                measurement = OFFLINE_MEASUREMENTS.get(label)
                if not measurement:
                    continue
                df_part = dr.fetch_offline_data(
                    measurement=measurement,
                    time_range=(start_utc, end_utc),
                    database=INFLUX_OFFLINE_DB,
                )
                if df_part is None or df_part.empty:
                    continue
                # Convert to IST for alignment with online df
                if isinstance(df_part.index, pd.DatetimeIndex) and df_part.index.tz is not None:
                    df_part = df_part.sort_index()
                    df_part.index = df_part.index.tz_convert("Asia/Kolkata")
                    df_part.index.name = "time (IST)"
                # Prefix columns to avoid collisions
                df_part = df_part.rename(columns={c: f"Offline[{label}] - {c}" for c in df_part.columns})
                parts.append(df_part)

            if parts:
                offline_df = parts[0]
                for df_part in parts[1:]:
                    offline_df = offline_df.join(df_part, how="outer")

        # Combine
        df_final: Optional[pd.DataFrame]
        if online_df is not None and not online_df.empty and offline_df is not None and not offline_df.empty:
            offline_aligned = offline_df.sort_index().reindex(online_df.index, method="ffill")
            df_final = online_df.join(offline_aligned, how="left")
        elif online_df is not None and not online_df.empty:
            df_final = online_df
        elif offline_df is not None and not offline_df.empty:
            df_final = offline_df
        else:
            return "No data found."

        # Persist
        df_final.to_csv("current_furnace_data.csv", index=True)
        st.session_state.copilot_df = df_final
        st.session_state.copilot_df_meta = {
            "request": request,
            "fetched_online": bool(online_df is not None and not online_df.empty),
            "fetched_offline": bool(offline_df is not None and not offline_df.empty),
            "online_time_range": online_time_range,
            "online_window": online_window,
            "offline_measurements": offline_labels,
            "offline_range_note": offline_range_note,
            "offline_db": INFLUX_OFFLINE_DB,
        }

        summary_lines = [
            "Data saved to 'current_furnace_data.csv'.",
            "Note: Online columns are usually 'Measurement - Field'. Offline columns are prefixed as 'Offline[<label>] - <field>'.",
            f"Request: {request}",
        ]
        if online_df is not None and not online_df.empty:
            summary_lines.append(f"Online: time_range={online_time_range} | window={online_window} | rows={len(online_df)}")
        if offline_df is not None and not offline_df.empty:
            summary_lines.append(
                f"Offline: measurements={offline_labels} | range={offline_range_note} | rows={len(offline_df)} | db={INFLUX_OFFLINE_DB}"
            )
        summary_lines.append(f"Final shape: {df_final.shape}")
        summary_lines.append(f"Columns ({len(df_final.columns)}): {list(df_final.columns)}")
        summary_lines.append("\nPreview:\n" + df_final.head(2).to_string())

        return "\n".join(summary_lines)

    except Exception as e:
        _append_tool_error(tool_name="fetch_and_summarize_furnace_data", params=params, error=str(e))
        return f"Fetch Error: {str(e)}"

@tool
def execute_python_plot(code: str) -> str:
    """
    Execute python code to create a Plotly figure. 
    The code MUST:
    1. Read data from 'current_furnace_data.csv'.
    2. Create a plotly figure named 'fig'.
    3. Not use 'fig.show()'.
    Example: fig = px.scatter(pd.read_csv('current_furnace_data.csv'), x='A', y='B')
    """
    try:
        # Preload the dataframe for convenience (LLM may still choose to read CSV explicitly)
        try:
            df = pd.read_csv("current_furnace_data.csv", index_col=0, parse_dates=True)
        except Exception:
            df = None

        # Create a local environment for execution
        local_vars = {"pd": pd, "px": px, "go": go, "df": df}

        # Execute the LLM-generated code (restricted)
        _safe_exec(code, local_vars)
        
        if "fig" in local_vars:
            # Save the figure object to session state for the UI to pick up
            st.session_state.copilot_fig = local_vars["fig"]
            st.session_state.last_plot_code = code
            return "Successfully generated Plotly figure."
        else:
            return "Code executed but no variable named 'fig' was found."
            
    except Exception as e:
        _append_tool_error(
            tool_name="execute_python_plot",
            params={"code": (code[:2000] + "…") if isinstance(code, str) and len(code) > 2000 else code},
            error=str(e),
        )
        st.session_state.last_plot_error = str(e)
        return f"Python Error: {str(e)}"


@tool
def search_shift_history(query: str) -> str:
    """
    Search past shift summaries using semantic similarity.
    Use for questions about past shifts, stability, anomalies, or shift performance.
    """
    shift_store = st.session_state.get("shift_store")
    if shift_store is None:
        return "Shift store not initialized."

    results = shift_store.search_similar_shifts(query=query, top_k=5)

    if not results:
        return "No shift summaries found for this query."

    parts = []
    for i, r in enumerate(results, 1):
        payload = r.get("payload", {})
        text = payload.get("summary_text", "No summary.")
        window_id = payload.get("window_id", "unknown")
        parts.append(f"[{i}] Shift: {window_id}\n{text}")

    return "\n\n".join(parts)


@tool
def search_knowledge_docs(query: str) -> str:
    """
    Search uploaded knowledge documents (SOPs, manuals, specs, policies).
    Use for questions about procedures, specifications, or reference material.
    """
    knowledge_store = st.session_state.get("knowledge_store")
    if knowledge_store is None:
        return "Knowledge store not initialized."

    results = knowledge_store.search(query, top_k=5)

    if not results:
        return "No knowledge documents found for this query."

    parts = []
    for i, r in enumerate(results, 1):
        payload = r.get("payload", {})
        content = payload.get("content", "No content.")
        source = payload.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{content}")

    return "\n\n".join(parts)