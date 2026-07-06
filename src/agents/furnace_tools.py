"""Tool functions for the FurnaceMind AI Co-Operate agent.

Exposes nine tool functions dispatched by
:func:`execute_openai_tool_call`:

1. ``fetch_online_data`` — fetch InfluxDB telemetry for any measurement group.
2. ``fetch_offline_data`` — fetch shift/daily report data from the offline database.
3. ``merge_furnace_data`` — align and merge online + offline datasets on timestamps.
4. ``fetch_ml_data`` — load a date-range slice from the static pre-merged ML dataset.
5. ``concat_datasets`` — concatenate multiple datasets vertically (temporal union).
6. ``load_static_shift_data`` — load 8-hour shift data from the static ML dataset.
7. ``search_shift_history`` — semantic vector search over Qdrant shift summaries.
8. ``search_knowledge_docs`` — semantic search over uploaded operator documents.
9. ``execute_python_plot`` — sandboxed execution of agent-generated Plotly code.
"""

import base64
import io
import json
import mimetypes
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain.tools import tool
from pydantic import BaseModel, Field, ValidationError

from config.config_loader import load_config
from data.fetch_presets import (
    OFFLINE_REPORT_LABEL_MAP,
)
from data.ml.static_csv import load_static_dataset
from furnace_data.influx.online import fetch_online_df  # noqa: F401
from furnace_data.influx.query import TIMEDELTAS  # noqa: F401
from furnace_data.offline import (
    OFFLINE_TABLES,
)
from furnace_data.offline import fetch_offline_data as _fetch_offline_table_df
from furnace_data.offline import fetch_offline_report as _fetch_offline_report_df
from furnace_data.offline import (
    resolve_offline_table_name,
)
from utils.shift_windows import shift_window_naive

# CONFIG
config = load_config("setting_ds_dv.yml")

_OFFLINE_REPORT_TYPE_ALIASES = {
    "RAW_MATERIAL_COMPOSITION": "RM_COMPOSITION",
}


_TOOL_ERRORS_PATH = Path(__file__).resolve().parent / "tool_errors.md"
_KNOWLEDGE_IMAGE_RESULT_LIMIT = 3
_KNOWLEDGE_IMAGE_MAX_BYTES = 4_000_000
_KNOWLEDGE_RERANK_CANDIDATES = 16
_KNOWLEDGE_RETURN_LIMIT = 8


def _ensure_dataset_store() -> Dict[str, Any]:
    """Return the Streamlit session-state store for temporary datasets.

    FurnaceMind tools pass data between model/tool calls by storing fetched or
    merged DataFrames in ``st.session_state["fm_datasets"]``. This helper lazily
    creates that dictionary and protects against stale non-dictionary values.
    """
    if "fm_datasets" not in st.session_state or not isinstance(
        st.session_state.get("fm_datasets"), dict
    ):
        st.session_state["fm_datasets"] = {}
    return st.session_state["fm_datasets"]


def _new_dataset_id(prefix: str) -> str:
    """Create a unique id for a dataset produced during the current session.

    Args:
        prefix: Short source label such as ``online``, ``offline``, ``merged``,
            or ``static_shift``.

    Returns:
        Stable session-local id containing the prefix, UTC timestamp, and a
        monotonically increasing Streamlit session counter.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    counter = st.session_state.get("fm_dataset_counter", 0) + 1
    st.session_state["fm_dataset_counter"] = counter
    return f"{prefix}_{ts}_{counter}"


def _to_ist_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame datetime index to Asia/Kolkata time.

    Online and offline sources may return UTC-aware, UTC-naive, or already-local
    timestamps. The UI and shift logic expect an IST index named ``time (IST)``.
    Non-datetime and empty frames are returned unchanged.
    """
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
    """Parse a user/tool ISO timestamp into a UTC-aware ``datetime``.

    Args:
        s: ISO-8601 timestamp accepted by pandas, usually ending with ``Z``.

    Returns:
        Python ``datetime`` normalized to UTC.

    Raises:
        ValueError: If pandas cannot produce a timestamp for the input.
    """
    dt = pd.to_datetime(s, utc=True)
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    raise ValueError(f"Invalid datetime: {s}")


def _resolve_online_window(*, lookback: timedelta, window: Optional[str]) -> str:
    """Choose the aggregation window for online telemetry fetches.

    Args:
        lookback: Resolved time span requested by the model/user.
        window: Optional explicit aggregation window from the tool arguments.

    Returns:
        The explicit window when provided; otherwise ``1 hour`` for ranges above
        one day and ``15 minutes`` for shorter ranges.
    """
    if isinstance(window, str) and window.strip():
        return window.strip()
    # Policy from user:
    # - hourly averaging if more than 1 day
    # - else 15mins averaging
    return "1 hour" if lookback > timedelta(days=1) else "15 minutes"


class OnlineFetchArgs(BaseModel):
    """Validated arguments for fetching online Influx telemetry.

    The model may request either a relative lookback window or exact UTC start
    and end timestamps. Measurement groups are constrained to known online
    telemetry groups so generated tool calls remain safe and predictable.
    """

    lookback: Optional[str] = Field(
        default=None,
        description=(
            "Relative window as a compact string: '8h', '2d', '30m', '1 week'. "
            "Use this OR start_time_utc/end_time_utc — never both."
        ),
    )
    window: Optional[str] = Field(
        default=None,
        description="Averaging window. If omitted, tool applies policy: >1 day => 1 hour, else 15 minutes.",
    )
    start_time_utc: Optional[str] = Field(
        default=None,
        description="ISO-8601 UTC start time e.g. '2026-05-01T00:30:00Z'. Use for exact windows instead of lookback.",
    )
    end_time_utc: Optional[str] = Field(
        default=None,
        description="ISO-8601 UTC end time. Defaults to now if omitted.",
    )
    measurement_groups: Optional[
        List[
            Literal[
                "process_params",
                "cooling_water",
                "heatload_delta_t",
                "delta_t",
                "temperature_profile",
                "miscellaneous",
            ]
        ]
    ] = Field(
        default=None,
        description="Which online measurement groups to include. If omitted, fetches all groups.",
    )


class OfflineReportType(str):
    """Canonical offline report type labels accepted by tool-calling.

    The class is intentionally lightweight because the actual validation is
    handled by ``OfflineFetchArgs.report_type``. Keeping these labels in one
    place documents the offline data families exposed to the LLM.
    """


class OfflineFetchArgs(BaseModel):
    """Validated arguments for loading offline report tables.

    Offline data can be fetched by canonical report type, optional explicit
    table name, time window, lookback period, and resampling cadence. The schema
    keeps generated LLM tool calls inside the supported report families.
    """

    report_type: Literal[
        "HM_SLAG",
        "CHARGE",
        "DPR",
        "RAW_MATERIAL_COMPOSITION",
        "RM_COMPOSITION",
        "RAW_MATERIAL_STRENGTH",
        "BURDEN_DISTRIBUTION",
        "HOPPER_MANAGEMENT",
    ] = Field(
        description="Which offline dataset to fetch. Use RAW_MATERIAL_STRENGTH for coke/sinter strength properties; use RM_COMPOSITION for the broader raw material chemistry report."
    )
    table_name: Optional[str] = Field(
        default=None,
        description="Optional explicit table override, e.g. ore_chemistry or charge_data.",
    )
    start_time_utc: Optional[str] = Field(
        default=None,
        description="ISO-8601 UTC start time. If omitted, uses lookback_days.",
    )
    end_time_utc: Optional[str] = Field(
        default=None, description="ISO-8601 UTC end time. Defaults to now."
    )
    lookback_days: Optional[int] = Field(
        default=10,
        description="If start_time_utc is omitted, fetch the last N days (defaults to 10).",
    )
    cadence: Optional[Literal["1h", "8h", "1d"]] = Field(
        default=None,
        description="Resampling cadence. If omitted, defaults by report_type: HM_SLAG/CHARGE=1h, RAW_MATERIAL_COMPOSITION=8h, RAW_MATERIAL_STRENGTH=8h, DPR=1d.",
    )


class MergeArgs(BaseModel):
    """Validated arguments for merging online and offline datasets.

    The merge tool aligns one online dataset with one or more offline datasets
    so downstream analysis or plotting can operate on a single DataFrame.
    """

    online_dataset_id: str = Field(
        description="Dataset id returned by fetch_online_data."
    )
    offline_dataset_ids: List[str] = Field(
        description="One or more dataset ids returned by fetch_offline_data."
    )
    fill_method: Literal["ffill", "none"] = Field(
        default="ffill", description="How to align offline rows onto online timestamps."
    )


class StaticShiftArgs(BaseModel):
    """Validated arguments for loading one static 8-hour shift slice.

    This schema is used when the model needs a known historical shift from the
    pre-merged static dataset instead of live online/offline fetches.
    """

    shift_date: str = Field(description="ISO date string YYYY-MM-DD")
    shift_label: Literal["A", "B", "C"] = Field(
        description="Shift: A (06:00-14:00), B (14:00-22:00), C (22:00-06:00 next day) IST"
    )


class MLDataArgs(BaseModel):
    """Validated arguments for reading the static pre-merged ML dataset.

    The static dataset is indexed in IST and can be sliced by time, optionally
    resampled, and optionally reduced to columns matching user-provided keyword
    filters.
    """

    start_time: str = Field(
        description="Start of range. ISO-8601 or YYYY-MM-DD. Treated as IST (matches static dataset index). E.g. '2026-03-01' or '2026-03-01T06:00:00'."
    )
    end_time: Optional[str] = Field(
        default=None,
        description="End of range. ISO-8601 or YYYY-MM-DD. Defaults to current IST time. Omit for 'up to now'.",
    )
    resample: Optional[Literal["1h", "4h", "8h", "1d"]] = Field(
        default=None,
        description="Downsampling cadence. Native resolution is 1h. Use '8h' for shift-level, '1d' for daily views. Omit to keep native.",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Optional keyword list to filter columns (case-insensitive substring match). E.g. ['fuel rate', 'si', 'etaco']. Omit to return all columns.",
    )


class ConcatArgs(BaseModel):
    """Validated arguments for vertically concatenating temporary datasets.

    The concat tool lets the agent combine multiple compatible time slices into
    one session DataFrame before plotting or comparing trends.
    """

    dataset_ids: List[str] = Field(
        description="Dataset IDs to concatenate vertically (temporal union). Sorted by index; duplicate timestamps keep the last entry (prefer recent data)."
    )


def _current_artifact_turn_id() -> str | None:
    """Return the active chat-turn id used for artifact ownership.

    The chat page sets ``fm_current_artifact_turn_id`` before one agent run.
    Dataset and plot tools copy that id onto UI artifacts so the renderer can
    show only charts/tables created by the current turn and hide stale outputs
    from previous questions.

    Returns:
        Clean turn id from Streamlit session state, or ``None`` when no agent
        turn is active.
    """
    value = st.session_state.get("fm_current_artifact_turn_id")
    clean = str(value or "").strip()
    return clean or None


def _tag_artifact_turn(key: str) -> None:
    """Attach current-turn ownership to a Streamlit artifact key.

    Args:
        key: Session-state key that stores the owner id for a UI artifact, such
            as ``fm_df_turn_id`` or ``fm_fig_turn_id``.

    Behavior:
        If a turn id exists, the key is updated with that id. If no turn is
        active, the key is removed so old artifacts are not treated as current.
    """
    turn_id = _current_artifact_turn_id()
    if turn_id:
        st.session_state[key] = turn_id
    else:
        st.session_state.pop(key, None)


def _save_dataset(*, dataset_id: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    """Persist a tool-produced DataFrame for later tools and UI rendering.

    Args:
        dataset_id: Session-local id returned to the model.
        df: DataFrame produced by a fetch, merge, concat, or static-shift tool.
        meta: Human-readable metadata describing source, time window, and shape.

    Side effects:
        Updates the dataset store, marks the active DataFrame for plotting, and
        tags the DataFrame with the current chat turn id.
    """
    store = _ensure_dataset_store()
    store[dataset_id] = {"df": df, "meta": meta}
    st.session_state.fm_df = df
    st.session_state.fm_df_meta = meta
    _tag_artifact_turn("fm_df_turn_id")


def _summarize_df(df: pd.DataFrame, *, dataset_id: str, title: str) -> str:
    """Build a compact text summary returned after a dataset tool runs.

    The summary gives the model enough context to decide whether it needs a
    follow-up tool call, a plot, or a final natural-language answer without
    sending the full dataset through the chat transcript.
    """
    if df is None or df.empty:
        return f"{title}: No data found."
    preview = df.head(2).to_string() if len(df) else "<empty>"
    return (
        f"{title}: dataset_id={dataset_id}\n"
        f"Shape: {df.shape}\n"
        f"Columns ({len(df.columns)}): {list(df.columns)}\n\n"
        f"Preview:\n{preview}"
    )


def _now_ist_naive() -> pd.Timestamp:
    """Return current time as an IST tz-naive timestamp.

    The static ML dataset uses tz-naive IST timestamps, so this helper avoids
    mixing timezone-aware values with that index during range selection.
    """
    return pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=5, minutes=30)


def _parse_ist_naive(s: str) -> pd.Timestamp:
    """Parse a date/time string into the static dataset's IST-naive format.

    Timezone-aware inputs are converted to Asia/Kolkata and then made naive so
    they can be compared directly with the static ML dataset index.
    """
    ts = pd.to_datetime(s)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
    return ts


def _load_ml_dataset() -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Load the static ML dataset with aggressive session-level caching.
    Returns (df, data_start, data_end). The index is tz-naive IST at hourly resolution.
    """
    cache_key = "fm_ml_df_cache"
    if cache_key not in st.session_state:
        df = load_static_dataset()
        if df.empty:
            raise ValueError("Static ML dataset returned no rows.")
        st.session_state[cache_key] = df
    df: pd.DataFrame = st.session_state[cache_key]
    return df, df.index.min(), df.index.max()


def _ml_column_summary(df: pd.DataFrame) -> str:
    """Return a compact grouped column summary for the ML dataset.

    The text is included in tool responses when column discovery is useful, so
    the model can choose relevant fields without receiving every row of data.
    """
    groups: dict[str, list[str]] = {
        "KPIs": [],
        "Process params": [],
        "Temperature": [],
        "Materials": [],
        "Hot metal / Slag": [],
        "Burden": [],
        "Other": [],
    }
    for col in df.columns:
        cu = col.upper()
        if any(
            k in cu
            for k in [
                "FUEL RATE",
                "ETACO",
                "PRODUCTIONTONNES",
                "COKE RATE KG",
                "UNITCOST",
            ]
        ):
            groups["KPIs"].append(col)
        elif any(
            k in cu
            for k in [
                "HOT BLAST",
                "TOPPRESSURE",
                "BOTTOMBAR",
                "TOPBAR",
                "STEAM",
                "O2 ENRICH",
                "PERMEABILITY",
                "DIFFERENTIAL PRESSURE",
                "RAFT",
                "TUYERE",
                "OXYGEN",
            ]
        ):
            groups["Process params"].append(col)
        elif any(
            k in cu
            for k in [
                "_TEMP_",
                "HEARTH_TEMP",
                "BELLY_TEMP",
                "BOSH_TEMP",
                "LOWER_STACK",
                "UPTAKE_TEMP",
                "HEAT LOAD",
            ]
        ):
            groups["Temperature"].append(col)
        elif any(
            k in cu
            for k in [
                "COKE_",
                "NUTCOKE_",
                "PCI_",
                "ORE_",
                "SINTER_",
                "PELLET_",
                "FLUX_",
            ]
        ):
            groups["Materials"].append(col)
        elif any(k in cu for k in ["CHEM_PCT", "SLAG_", "HMT_", "GEOMIN"]):
            groups["Hot metal / Slag"].append(col)
        elif any(
            k in cu for k in ["PORTION", "ANGLE", "DISCHARGE_TIME", "CHARGES", "STOCK"]
        ):
            groups["Burden"].append(col)
        else:
            groups["Other"].append(col)
    lines = []
    for grp, cols in groups.items():
        if cols:
            lines.append(
                f"  {grp} ({len(cols)}): {', '.join(cols[:6])}"
                + (" …" if len(cols) > 6 else "")
            )
    return "\n".join(lines)


def fetch_ml_data(
    *,
    start_time: str,
    end_time: str | None = None,
    resample: str | None = None,
    columns: list[str] | None = None,
) -> str:
    """
    Fetch a date-range slice from the static pre-merged ML dataset (hourly, IST-naive index).
    Covers 2024-01-01 to ~current month. Fast: reads the static dataset cached in session.

    If the requested range extends beyond the static dataset end (recent gap):
    - Returns the covered static portion as a dataset.
    - Includes a GAP NOTE instructing the caller to also run fetch_online_data for the gap,
      then concat_datasets to stitch them together.
    """
    params = {
        "start_time": start_time,
        "end_time": end_time,
        "resample": resample,
        "columns": columns,
    }
    try:
        args = MLDataArgs.model_validate(params)

        req_start = _parse_ist_naive(args.start_time)
        req_end = _parse_ist_naive(args.end_time) if args.end_time else _now_ist_naive()

        if req_end <= req_start:
            return "Error: end_time must be after start_time."

        df, csv_start, csv_end = _load_ml_dataset()

        overlap_start = max(req_start, csv_start)
        overlap_end = min(req_end, csv_end)

        if overlap_start > overlap_end + pd.Timedelta(hours=1):
            return (
                f"Requested range ({req_start} – {req_end} IST) has no overlap with the "
                f"static ML dataset ({csv_start} – {csv_end} IST). "
                f"Use fetch_online_data directly for this query."
            )

        slice_df = df.loc[
            (df.index >= overlap_start) & (df.index <= overlap_end)
        ].copy()

        # Optional column filter (fuzzy substring)
        if args.columns:
            matched: list[str] = []
            for kw in args.columns:
                kw_lower = kw.lower()
                matched += [
                    c for c in df.columns if kw_lower in c.lower() and c not in matched
                ]
            if matched:
                slice_df = slice_df[matched]

        # Optional resample (native is 1h)
        if args.resample and args.resample != "1h":
            slice_df = (
                slice_df.resample(args.resample)
                .mean(numeric_only=True)
                .dropna(how="all")
            )

        dataset_id = _new_dataset_id("ml_static")
        meta = {
            "dataset_id": dataset_id,
            "type": "ml_static",
            "source": "offline_feed.historical_static_ml_dataset",
            "start": str(overlap_start.date()),
            "end": str(overlap_end.date()),
            "resample": args.resample or "1h (native)",
        }
        _save_dataset(dataset_id=dataset_id, df=slice_df, meta=meta)

        col_summary = (
            _ml_column_summary(slice_df)
            if not args.columns
            else f"  Filtered: {list(slice_df.columns)}"
        )
        summary = (
            f"ML STATIC DATA | {overlap_start.date()} -> {overlap_end.date()} IST\n"
            f"dataset_id={dataset_id} | {len(slice_df)} rows × {len(slice_df.columns)} cols\n"
            f"Columns available:\n{col_summary}"
        )

        # Gap note: if request extends beyond static dataset end by more than 2 hours
        gap_threshold = pd.Timedelta(hours=2)
        if req_end > csv_end + gap_threshold:
            gap_hours = max(1, int((req_end - csv_end).total_seconds() / 3600) + 1)
            summary += (
                f"\n\nGAP NOTE: Static dataset ends {csv_end} IST; your request goes to {req_end.strftime('%Y-%m-%d %H:%M')} IST.\n"
                f"To fill the ~{gap_hours}h gap:\n"
                f"  1. fetch_online_data(lookback_hours={gap_hours})\n"
                f"  2. concat_datasets(dataset_ids=['{dataset_id}', '<online_dataset_id>'])\n"
                f"Note: online columns use InfluxDB names (e.g. 'fuel_rate') — ML static uses ML names (e.g. 'ACT. FUEL RATEKG/THM.'). "
                f"Plot whichever column is non-null in each time region."
            )

        return summary

    except Exception as e:
        _append_tool_error(tool_name="fetch_ml_data", params=params, error=str(e))
        return f"fetch_ml_data Error: {e}"


def concat_datasets(*, dataset_ids: list[str]) -> str:
    """
    Concatenate multiple datasets vertically (temporal union).
    Useful for stitching static ML data with a recent online fetch.
    Duplicate timestamps keep the last entry (later dataset wins — prefer online for recent rows).
    Column mismatches are handled with outer join (NaN where a column doesn't exist in a given frame).
    """
    params = {"dataset_ids": dataset_ids}
    try:
        args = ConcatArgs.model_validate(params)
        store = _ensure_dataset_store()

        frames: list[pd.DataFrame] = []
        for did in args.dataset_ids:
            entry = store.get(did)
            if not entry or "df" not in entry:
                raise ValueError(f"Unknown dataset_id: '{did}'. Fetch it first.")
            frames.append(entry["df"])

        if not frames:
            raise ValueError("No datasets provided to concat.")

        # Normalise timezones: ML static data is IST-naive; online data is UTC-aware.
        # Mixed tz-naive + tz-aware causes a TypeError during sort_index.
        has_tz_aware = any(
            isinstance(f.index, pd.DatetimeIndex) and f.index.tz is not None
            for f in frames
        )
        has_tz_naive = any(
            isinstance(f.index, pd.DatetimeIndex) and f.index.tz is None for f in frames
        )
        if has_tz_aware and has_tz_naive:
            normalized_frames: list[pd.DataFrame] = []
            for f in frames:
                if isinstance(f.index, pd.DatetimeIndex):
                    if f.index.tz is None:
                        # IST-naive → localise to IST then convert to UTC
                        f = f.copy()
                        f.index = f.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
                    else:
                        f = f.copy()
                        f.index = f.index.tz_convert("UTC")
                normalized_frames.append(f)
            frames = normalized_frames

        combined = pd.concat(frames, axis=0, join="outer")
        combined = combined.sort_index()
        combined = combined[
            ~combined.index.duplicated(keep="last")
        ]  # later dataset wins on overlap

        dataset_id = _new_dataset_id("concat")
        meta = {
            "dataset_id": dataset_id,
            "type": "concat",
            "source_ids": args.dataset_ids,
            "rows": len(combined),
            "start": str(combined.index.min()),
            "end": str(combined.index.max()),
        }
        _save_dataset(dataset_id=dataset_id, df=combined, meta=meta)

        return (
            f"CONCAT DATA | {combined.index.min()} -> {combined.index.max()}\n"
            f"dataset_id={dataset_id} | {len(combined)} rows × {len(combined.columns)} cols\n"
            f"Sources: {args.dataset_ids}"
        )

    except Exception as e:
        _append_tool_error(tool_name="concat_datasets", params=params, error=str(e))
        return f"concat_datasets Error: {e}"


def load_static_shift_data(*, shift_date: str, shift_label: str) -> str:
    """Load 8-hour shift data from the static ML dataset (hourly, online+offline pre-merged).

    Covers Jan 2024 to Mar 2026. Returns an error if the requested shift is outside
    this range, instructing the LLM to fall back to fetch_online_data + fetch_offline_data.
    """
    params = {"shift_date": shift_date, "shift_label": shift_label}
    try:
        args = StaticShiftArgs.model_validate(params)

        shift_day = pd.to_datetime(args.shift_date).date()
        shift_start_dt, shift_end_dt = shift_window_naive(
            shift_day,
            args.shift_label,
        )
        shift_start = pd.Timestamp(shift_start_dt)
        shift_end = pd.Timestamp(shift_end_dt)

        df, csv_min, csv_max = _load_ml_dataset()
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
            "source": "offline_feed.historical_static_ml_dataset",
            "rows": len(shift_df),
        }
        _save_dataset(dataset_id=dataset_id, df=shift_df, meta=meta)

        return _summarize_df(
            shift_df,
            dataset_id=dataset_id,
            title=f"Static shift data: {shift_date} Shift {shift_label} ({len(shift_df)} hourly rows)",
        )

    except Exception as e:
        _append_tool_error(
            tool_name="load_static_shift_data", params=params, error=str(e)
        )
        return f"Error loading static shift data: {e}"


def fetch_online_data(
    *,
    lookback: str | None = None,
    window: str | None = None,
    measurement_groups: list[str] | None = None,
    start_time_utc: str | None = None,
    end_time_utc: str | None = None,
    # Legacy params — accepted but ignored to avoid errors if LLM still sends them
    lookback_days: int | None = None,
    lookback_hours: int | None = None,
    lookback_minutes: int | None = None,
) -> str:
    """Fetch online (high-frequency) telemetry.

    Pass either ``lookback`` (e.g. ``"8h"``, ``"2d"``, ``"30m"``) OR
    ``start_time_utc`` + ``end_time_utc`` for an exact window — never both.
    If ``window`` is omitted the tool applies: >1 day => 1 hour avg, else 15 min avg.
    """
    # Coerce empty strings to None (LLMs sometimes send "" for omitted fields)
    start_time_utc = start_time_utc or None
    end_time_utc = end_time_utc or None

    # Legacy int params: if the LLM still sends the old style, convert to string lookback
    if lookback is None and start_time_utc is None:
        if lookback_hours is not None:
            lookback = f"{lookback_hours}h"
        elif lookback_days is not None:
            lookback = f"{lookback_days}d"
        elif lookback_minutes is not None:
            lookback = f"{lookback_minutes}m"

    params = {
        "lookback": lookback,
        "window": window,
        "measurement_groups": measurement_groups,
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
    }
    try:
        args = OnlineFetchArgs.model_validate(params)

        selected_measurements = (
            list(args.measurement_groups)
            if args.measurement_groups
            else [
                "process_params",
                "cooling_water",
                "heatload_delta_t",
                "delta_t",
                "temperature_profile",
                "miscellaneous",
            ]
        )

        if args.start_time_utc:
            # Absolute-time window path
            start_dt = _parse_iso8601_utc(args.start_time_utc)
            _now = datetime.now(timezone.utc)
            end_dt = (
                _parse_iso8601_utc(args.end_time_utc) if args.end_time_utc else _now
            )
            # Guard: reject future windows
            if start_dt > _now:
                return (
                    f"Fetch Error: start_time_utc {start_dt.isoformat()} is in the future "
                    f"(current UTC time is {_now.strftime('%Y-%m-%dT%H:%M:%SZ')}). "
                    "No online data exists for future dates. Please use the current or a past date."
                )
            if end_dt > _now:
                end_dt = _now
            duration = end_dt - start_dt
            window_final = _resolve_online_window(lookback=duration, window=args.window)
            df = fetch_online_df(
                selected_measurements=selected_measurements,
                time_range="last 8 hours",  # unused when overrides are set
                window_by=window_final,
                start_time_override=start_dt,
                end_time_override=end_dt,
                column_naming="field",
            )
            time_range_label = f"{args.start_time_utc} → {args.end_time_utc or 'now'}"
        else:
            # Relative lookback path — normalise the string to a TIMEDELTAS key
            normalized_time_range = _normalize_time_range(
                args.lookback or "last 8 hours"
            )

            lookback_td = TIMEDELTAS.get(normalized_time_range)
            if not isinstance(lookback_td, timedelta):
                lookback_td = timedelta(hours=8)
            window_final = _resolve_online_window(
                lookback=lookback_td, window=args.window
            )

            df = fetch_online_df(
                selected_measurements=selected_measurements,
                time_range=normalized_time_range,
                window_by=window_final,
                column_naming="field",
            )
            time_range_label = normalized_time_range

        df = _to_ist_index(df)

        dataset_id = _new_dataset_id("online")
        meta = {
            "type": "online",
            "time_range": time_range_label,
            "window": window_final,
            "measurement_groups": selected_measurements,
        }
        _save_dataset(dataset_id=dataset_id, df=df, meta=meta)
        return _summarize_df(df, dataset_id=dataset_id, title="ONLINE DATA")

    except (ValidationError, Exception) as e:
        _append_tool_error(tool_name="fetch_online_data", params=params, error=str(e))
        return f"Fetch Error: {str(e)}"


def fetch_offline_data(
    *,
    report_type: str,
    table_name: str | None = None,
    start_time_utc: str | None = None,
    end_time_utc: str | None = None,
    lookback_days: int | None = 10,
    cadence: str | None = None,
) -> str:
    """Fetch offline (report) datasets with type-specific cadence defaults.

    Defaults:
    - HM_SLAG, CHARGE: hourly (1h)
    - RAW_MATERIAL_COMPOSITION / RAW_MATERIAL_STRENGTH: shiftwise (8h)
    - DPR: daily (1d)
    """
    params = {
        "report_type": report_type,
        "table_name": table_name,
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "lookback_days": lookback_days,
        "cadence": cadence,
    }
    try:
        args = OfflineFetchArgs.model_validate(params)

        offline_report_type = _OFFLINE_REPORT_TYPE_ALIASES.get(
            args.report_type, args.report_type
        )
        label = (
            "Bunker Report"
            if offline_report_type == "RM_COMPOSITION"
            else OFFLINE_REPORT_LABEL_MAP.get(offline_report_type, offline_report_type)
        )
        now = datetime.now(timezone.utc)
        end = _parse_iso8601_utc(args.end_time_utc) if args.end_time_utc else now
        if args.start_time_utc:
            start = _parse_iso8601_utc(args.start_time_utc)
        else:
            lb = int(args.lookback_days or 10)
            lb = max(1, min(lb, 365))
            start = end - timedelta(days=lb)

        # Guard: reject future windows; offline reports have no future rows.
        if start > now:
            return (
                f"Fetch Error: start_time_utc {start.isoformat()} is in the future "
                f"(current UTC time is {now.strftime('%Y-%m-%dT%H:%M:%SZ')}). "
                "No offline data exists for future dates. Please use the current or a past date."
            )
        # Cap end at now to avoid empty half-future windows
        if end > now:
            end = now

        cadence_default = {
            "HM_SLAG": "1h",
            "CHARGE": "1h",
            "RAW_MATERIAL_COMPOSITION": "8h",
            "RM_COMPOSITION": "8h",
            "RAW_MATERIAL_STRENGTH": "8h",
            "DPR": "1d",
            "BURDEN_DISTRIBUTION": "1d",
            "HOPPER_MANAGEMENT": "1d",
        }[args.report_type]
        cadence_final = args.cadence or cadence_default

        if args.table_name:
            try:
                resolved_table = resolve_offline_table_name(args.table_name)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            df = _fetch_offline_table_df(
                table_name=resolved_table,
                time_range=(start, end),
            )
            source_detail = resolved_table
        else:
            df = _fetch_offline_report_df(
                report_type=offline_report_type,
                time_range=(start, end),
            )
            source_detail = offline_report_type

        # Offline fetch returns UTC index (as per helper); convert + resample
        df = _to_ist_index(df)
        skip_resample = offline_report_type in {
            "RM_COMPOSITION",
            "RAW_MATERIAL_STRENGTH",
            "BURDEN_DISTRIBUTION",
            "HOPPER_MANAGEMENT",
        } or bool(args.table_name)
        if (
            not skip_resample
            and df is not None
            and not df.empty
            and isinstance(df.index, pd.DatetimeIndex)
        ):
            df = df.resample(cadence_final).mean(numeric_only=True)
            df = df.dropna(how="all")

        # Prefix columns to avoid collisions during merge
        df = (
            df.rename(columns={c: f"Offline[{label}] - {c}" for c in df.columns})
            if df is not None
            else df
        )

        dataset_id = _new_dataset_id("offline")
        meta = {
            "type": "offline",
            "report_type": args.report_type,
            "source": "offline_database",
            "label": label,
            "source_detail": source_detail,
            "start_time_utc": start.isoformat(),
            "end_time_utc": end.isoformat(),
            "cadence": cadence_final,
        }
        _save_dataset(dataset_id=dataset_id, df=df, meta=meta)
        return _summarize_df(df, dataset_id=dataset_id, title="OFFLINE DATA")

    except (ValidationError, Exception) as e:
        _append_tool_error(tool_name="fetch_offline_data", params=params, error=str(e))
        return f"Fetch Error: {str(e)}"


def merge_furnace_data(
    *,
    online_dataset_id: str,
    offline_dataset_ids: list[str],
    fill_method: str = "ffill",
) -> str:
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
            offline_aligned = offline_df.sort_index().reindex(
                online_df.index, method="ffill"
            )
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
    """Return OpenAI/OpenRouter tool schemas for FurnaceMind function-calling.

    The schema list is the contract between the LLM and the local dispatcher.
    Keep names, descriptions, required arguments, and enum values aligned with
    the concrete tool functions below.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_online_data",
                "description": (
                    "Fetch live InfluxDB telemetry (max 90 days). "
                    "Use EITHER lookback (e.g. '8h', '2d', '30m') "
                    "OR start_time_utc + end_time_utc for exact windows — never both. "
                    "If window omitted: >1 day => 1 hour avg, else 15 min avg. "
                    "Stores data in session (fm_df) and returns dataset_id + column preview."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lookback": {
                            "type": "string",
                            "description": (
                                "Relative window as a compact string: '8h', '2d', '30m', '1 week'. "
                                "Omit if using start_time_utc/end_time_utc."
                            ),
                        },
                        "window": {
                            "type": "string",
                            "description": "Averaging window like '15 minutes' or '1 hour'. Optional.",
                        },
                        "start_time_utc": {
                            "type": "string",
                            "description": "ISO-8601 UTC start e.g. '2026-05-01T00:30:00Z'. Omit if using lookback.",
                        },
                        "end_time_utc": {
                            "type": "string",
                            "description": "ISO-8601 UTC end. Defaults to now if omitted.",
                        },
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
                            "description": "Which measurement groups to fetch. Omit to fetch all.",
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
                "description": "Fetch offline report datasets. Covers HM/Slag, Charge, raw material composition, DPR, burden distribution, and hopper management. Stores the active dataframe in session (fm_df) and returns dataset_id + preview.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report_type": {
                            "type": "string",
                            "enum": [
                                "HM_SLAG",
                                "CHARGE",
                                "RAW_MATERIAL_COMPOSITION",
                                "RM_COMPOSITION",
                                "RAW_MATERIAL_STRENGTH",
                                "DPR",
                                "BURDEN_DISTRIBUTION",
                                "HOPPER_MANAGEMENT",
                            ],
                        },
                        "table_name": {
                            "type": "string",
                            "enum": sorted(OFFLINE_TABLES.keys()),
                            "description": "Optional explicit table override.",
                        },
                        "start_time_utc": {
                            "type": "string",
                            "description": "ISO-8601 UTC start time. Optional.",
                        },
                        "end_time_utc": {
                            "type": "string",
                            "description": "ISO-8601 UTC end time. Optional; defaults to now.",
                        },
                        "lookback_days": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 365,
                        },
                        "cadence": {
                            "type": "string",
                            "enum": ["1h", "8h", "1d"],
                            "description": "Optional override for resampling cadence.",
                        },
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
                "description": "Merge offline datasets onto an online dataset by aligning to online timestamps (repeat/forward-fill). Produces merged dataset_id and stores it as the active session dataframe (fm_df).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "online_dataset_id": {"type": "string"},
                        "offline_dataset_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "fill_method": {
                            "type": "string",
                            "enum": ["ffill", "none"],
                            "default": "ffill",
                        },
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
                "description": "Search uploaded multimodal knowledge documents (SOPs, manuals, specs, images, slides, tables, scanned pages).",
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
                "description": "Execute restricted Python to create a Plotly figure 'fig' using df (loaded from active session dataframe fm_df).",
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
                "name": "fetch_ml_data",
                "description": (
                    "PRIMARY tool for any historical data query spanning more than 2 days. "
                    "Reads from the local pre-merged ML dataset (hourly, IST-naive, 2024-01-01 -> present). "
                    "Fast — no InfluxDB call. Covers process params, material quality, burden, KPIs, hot metal chemistry. "
                    "If the requested range extends beyond the dataset end, returns a GAP NOTE with exact instructions "
                    "to call fetch_online_data + concat_datasets for the recent gap. "
                    "Use fetch_online_data directly only for: last ≤2 days, sub-hourly resolution, or when this tool reports no coverage."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "string",
                            "description": "ISO-8601 or YYYY-MM-DD. Treated as IST. E.g. '2026-03-01'.",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "ISO-8601 or YYYY-MM-DD. Defaults to now IST if omitted.",
                        },
                        "resample": {
                            "type": "string",
                            "enum": ["1h", "4h", "8h", "1d"],
                            "description": "Downsampling. Native is 1h. Use '8h' for shift views, '1d' for daily.",
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional keyword substrings to filter columns, e.g. ['si', 'fuel rate', 'etaco']. Omit for all.",
                        },
                    },
                    "required": ["start_time"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "concat_datasets",
                "description": (
                    "Concatenate datasets vertically (temporal union). "
                    "Use after fetching static + online portions to stitch them into one continuous dataset. "
                    "Sorts by timestamp; duplicate rows keep the last dataset's value (online wins over static on overlap). "
                    "Column mismatches handled with outer join (NaN where a column doesn't exist in a frame)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of dataset IDs to concatenate, in chronological order.",
                        },
                    },
                    "required": ["dataset_ids"],
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
                        "shift_date": {
                            "type": "string",
                            "description": "ISO date YYYY-MM-DD",
                        },
                        "shift_label": {
                            "type": "string",
                            "enum": ["A", "B", "C"],
                            "description": "Shift: A (06:00-14:00), B (14:00-22:00), C (22:00-06:00 next day) IST",
                        },
                    },
                    "required": ["shift_date", "shift_label"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def execute_openai_tool_call(*, name: str, arguments: Dict[str, Any]) -> str:
    """Dispatch one validated OpenAI/OpenRouter tool call to local code.

    Args:
        name: Function name selected by the model.
        arguments: JSON-decoded arguments emitted by the model.

    Returns:
        String result from the matching tool, or a clear error string for
        unknown tools and malformed plotting requests.
    """
    if name == "fetch_ml_data":
        return fetch_ml_data(**arguments)
    if name == "concat_datasets":
        return concat_datasets(**arguments)
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
        if not arguments.get("code"):
            return "Error: execute_python_plot requires a non-empty 'code' argument containing valid Python that creates a Plotly figure named 'fig'."
        return execute_python_plot.invoke(arguments)
    if name == "load_static_shift_data":
        return load_static_shift_data(**arguments)
    return f"Unknown tool: {name}"


def _knowledge_location(payload: dict[str, Any]) -> str:
    """Build the source label shown in MRAG retrieval output.

    Args:
        payload: Qdrant payload for one retrieved knowledge part.

    Returns:
        A compact human-readable location such as
        ``"BMO_Analysis.pptx, slide 16, slide_image"``. The label is used in
        tool output, visual-evidence prompts, and retrieval traces so users can
        connect an answer back to the uploaded document.
    """
    bits = [str(payload.get("source") or "unknown")]
    if payload.get("page_number") is not None:
        bits.append(f"page {payload['page_number']}")
    if payload.get("slide_number") is not None:
        bits.append(f"slide {payload['slide_number']}")
    if payload.get("sheet_name"):
        bits.append(f"sheet {payload['sheet_name']}")
    if payload.get("modality"):
        bits.append(str(payload["modality"]))
    return ", ".join(bits)


_VISUAL_KNOWLEDGE_MODALITIES = {"image", "page_image", "slide_image", "slide_render"}
_IMAGE_FILE_TYPES = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}


def _knowledge_payload_has_visual(payload: dict[str, Any]) -> bool:
    """Return whether a retrieved knowledge payload can produce an image input.

    The current MRAG flow stores original uploaded files in PostgreSQL and
    stores chunk payloads in Qdrant. New payloads advertise visual evidence via
    ``has_visual``, visual modalities, or image file types. Older chunks may
    still carry ``image_path`` from the previous disk-backed implementation, so
    that field remains supported during rollout.
    """
    modality = str(payload.get("modality") or "").strip().lower()
    file_type = str(payload.get("file_type") or payload.get("type") or "").lower()
    return bool(
        payload.get("image_path")
        or payload.get("has_visual")
        or modality in _VISUAL_KNOWLEDGE_MODALITIES
        or file_type in _IMAGE_FILE_TYPES
    )


def _stored_document_file_for_payload(payload: dict[str, Any]) -> Any | None:
    """Load the PostgreSQL-stored original file for a Qdrant payload.

    Qdrant payloads carry the stable MRAG ``document_id`` derived from the file
    hash. The SQL table has its own primary key, so the repository bridges the
    two ids through ``memory_documents.metadata`` and returns a row with
    ``file_bytes`` loaded.

    Args:
        payload: Qdrant payload for one retrieved knowledge chunk.

    Returns:
        Repository file record with original upload bytes, or ``None`` when the
        repository/user/document mapping is unavailable.
    """
    repository = st.session_state.get("knowledge_document_repository")
    if repository is None or not hasattr(repository, "get_document_file_by_mrag_id"):
        return None
    mrag_document_id = str(payload.get("document_id") or "").strip()
    if not mrag_document_id:
        return None
    try:
        return repository.get_document_file_by_mrag_id(
            user_id=st.session_state.get("fm_user_id"),
            mrag_document_id=mrag_document_id,
        )
    except Exception as exc:
        st.session_state["fm_last_knowledge_visual_error"] = str(exc)
        return None


def _image_mime_type(
    image_bytes: bytes,
    *,
    fallback_name: str = "visual.png",
    fallback_type: str | None = None,
) -> str:
    """Infer a MIME type for image bytes sent to the vision model.

    The resolver first trusts filename/type hints from the payload or SQL row.
    If those are missing, it opens the bytes with PIL and falls back to PNG when
    the format cannot be identified.
    """
    guessed = mimetypes.guess_type(fallback_name)[0]
    if guessed:
        return guessed
    if fallback_type and fallback_type in _IMAGE_FILE_TYPES:
        normalized = "jpeg" if fallback_type in {"jpg", "jpeg"} else fallback_type
        return f"image/{normalized}"
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        fmt = (image.format or "png").lower()
        fmt = "jpeg" if fmt in {"jpg", "jpeg"} else fmt
        return f"image/{fmt}"
    except Exception:
        return "image/png"


def _render_pdf_page_visual(file_bytes: bytes, page_number: int | None) -> bytes | None:
    """Render one PDF page from stored upload bytes for visual evidence.

    This is used for ``page_image`` chunks after removing persistent rendered
    page images from the codebase. The original PDF is read from PostgreSQL and
    the requested page is rendered on demand.

    Returns:
        PNG bytes for the requested page, or ``None`` when rendering fails or
        the page number is missing.
    """
    if page_number is None:
        return None
    try:
        from agents.multimodal.parsers import extract_pdf_pages

        for page in extract_pdf_pages(file_bytes, render_pages=True):
            if int(page.get("page_number") or 0) == int(page_number):
                image_bytes = page.get("image_bytes")
                return bytes(image_bytes) if image_bytes else None
    except Exception as exc:
        st.session_state["fm_last_knowledge_visual_error"] = str(exc)
    return None


def _render_pptx_slide_visual(
    file_bytes: bytes, slide_number: int | None
) -> bytes | None:
    """Render one PPTX slide from stored upload bytes for visual evidence.

    The previous implementation could attach a saved slide image path. The new
    implementation keeps the original PPTX in PostgreSQL and recreates the slide
    image only when a retrieved ``slide_render`` chunk is needed by the model.

    Returns:
        PNG bytes for the requested slide, or ``None`` when rendering is not
        available or the slide cannot be found.
    """
    if slide_number is None:
        return None
    try:
        from agents.multimodal.parsers import render_pptx_slides

        for slide in render_pptx_slides(file_bytes):
            if int(slide.get("slide_number") or 0) == int(slide_number):
                image_bytes = slide.get("image_bytes")
                return bytes(image_bytes) if image_bytes else None
    except Exception as exc:
        st.session_state["fm_last_knowledge_visual_error"] = str(exc)
    return None


def _extract_pptx_embedded_visual(
    file_bytes: bytes,
    *,
    slide_number: int | None,
    image_index: int | None,
) -> bytes | None:
    """Extract one embedded PPTX image from stored upload bytes.

    This resolves ``slide_image`` chunks. Instead of rendering the whole slide,
    it opens the original PPTX stored in PostgreSQL and returns the specific
    embedded image referenced by ``slide_number`` and ``slide_image_index``.
    """
    if slide_number is None:
        return None
    try:
        from agents.multimodal.parsers import extract_pptx_slides

        for slide in extract_pptx_slides(file_bytes):
            if int(slide.get("slide_number") or 0) != int(slide_number):
                continue
            image_blobs = list(slide.get("image_blobs") or [])
            if not image_blobs:
                return None
            index = int(image_index or 0)
            if index < 0 or index >= len(image_blobs):
                return None
            return bytes(image_blobs[index])
    except Exception as exc:
        st.session_state["fm_last_knowledge_visual_error"] = str(exc)
    return None


def _visual_bytes_from_stored_document(
    payload: dict[str, Any],
    stored_document: Any,
) -> tuple[bytes, str] | None:
    """Reconstruct visual bytes for a retrieved MRAG chunk from PostgreSQL.

    The resolver routes by payload modality: image uploads return original file
    bytes, PDF page-image chunks render the requested page, PPTX slide-render
    chunks render the requested slide, and PPTX slide-image chunks extract the
    referenced embedded image.

    Returns:
        ``(image_bytes, mime_type)`` when the visual can be reconstructed, else
        ``None``.
    """
    file_bytes = getattr(stored_document, "file_bytes", None)
    if not file_bytes:
        return None
    file_bytes = bytes(file_bytes)
    filename = str(
        payload.get("source") or getattr(stored_document, "filename", "visual.png")
    )
    file_type = str(
        payload.get("file_type")
        or payload.get("type")
        or getattr(stored_document, "file_type", "")
    ).lower()
    modality = str(payload.get("modality") or "").lower()

    if modality == "image" or file_type in _IMAGE_FILE_TYPES:
        return file_bytes, _image_mime_type(
            file_bytes, fallback_name=filename, fallback_type=file_type
        )
    if modality == "page_image" and file_type == "pdf":
        image_bytes = _render_pdf_page_visual(file_bytes, payload.get("page_number"))
        return (image_bytes, "image/png") if image_bytes else None
    if modality == "slide_render" and file_type == "pptx":
        image_bytes = _render_pptx_slide_visual(file_bytes, payload.get("slide_number"))
        return (image_bytes, "image/png") if image_bytes else None
    if modality == "slide_image" and file_type == "pptx":
        image_bytes = _extract_pptx_embedded_visual(
            file_bytes,
            slide_number=payload.get("slide_number"),
            image_index=payload.get("slide_image_index"),
        )
        if image_bytes:
            return image_bytes, _image_mime_type(
                image_bytes,
                fallback_name=filename,
                fallback_type=str(payload.get("image_format") or ""),
            )
    return None


def _visual_bytes_from_legacy_path(image_path: str) -> tuple[bytes, str] | None:
    """Read visual evidence from old path-backed payloads.

    This compatibility path supports chunks indexed before original files were
    stored in PostgreSQL. New uploads should not rely on this path, but keeping
    it avoids breaking older Qdrant payloads during rollout.
    """
    path = Path(str(image_path or ""))
    if not path.exists() or not path.is_file():
        return None
    if path.stat().st_size > _KNOWLEDGE_IMAGE_MAX_BYTES:
        return None
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return path.read_bytes(), mime_type


def _resolve_visual_attachment(
    attachment: dict[str, Any],
) -> tuple[bytes, str, str] | None:
    """Resolve a queued MRAG visual attachment for model input.

    The function supports both storage modes: legacy ``image_path`` payloads and
    the new PostgreSQL-backed original-file flow. It also enforces the maximum
    image size before the bytes are base64-encoded for the next multimodal LLM
    call.

    Returns:
        ``(image_bytes, mime_type, source_label)`` when a usable visual exists,
        otherwise ``None``.
    """
    payload = attachment.get("payload") or {}
    label = str(attachment.get("label") or _knowledge_location(payload) or "visual")
    image_path = str(payload.get("image_path") or attachment.get("image_path") or "")
    if image_path:
        legacy = _visual_bytes_from_legacy_path(image_path)
        if legacy:
            return legacy[0], legacy[1], label

    stored_document = _stored_document_file_for_payload(payload)
    if stored_document is None:
        return None
    resolved = _visual_bytes_from_stored_document(payload, stored_document)
    if not resolved:
        return None
    image_bytes, mime_type = resolved
    if len(image_bytes) > _KNOWLEDGE_IMAGE_MAX_BYTES:
        return None
    return image_bytes, mime_type, label


def _store_knowledge_image_results(results: list[dict[str, Any]]) -> None:
    """Queue retrieved visual chunks for the next multimodal LLM call.

    Qdrant stores vector payloads, not image bytes. New MRAG payloads carry
    enough location metadata to reconstruct visual evidence from the original
    uploaded document stored in PostgreSQL. Older payloads with ``image_path``
    are still supported during rollout.

    Args:
        results: Reranked knowledge-search results. Only payloads that can
            provide visual evidence are retained, up to
            ``_KNOWLEDGE_IMAGE_RESULT_LIMIT``.

    Side effects:
        Writes ``fm_mrag_image_results`` in Streamlit session state for
        ``consume_pending_mrag_image_message`` to consume after the tool call.
    """
    attachments: list[dict[str, Any]] = []
    for result in results:
        payload = result.get("payload") or {}
        if not _knowledge_payload_has_visual(payload):
            continue
        attachments.append(
            {
                "payload": payload,
                "label": _knowledge_location(payload),
                "score": result.get("score"),
            }
        )
        if len(attachments) >= _KNOWLEDGE_IMAGE_RESULT_LIMIT:
            break

    if attachments:
        st.session_state["fm_mrag_image_results"] = attachments
    else:
        st.session_state.pop("fm_mrag_image_results", None)


def _store_knowledge_document_refs(results: list[dict[str, Any]]) -> None:
    """Store document refs used by the latest MRAG retrieval.

    The chat page uses these refs to mark the current turn as document-backed.
    That provenance prevents uploaded-document facts from being compressed into
    durable long-term memory and lets document removal revoke related context.
    """
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        payload = result.get("payload") or {}
        document_id = str(payload.get("document_id") or "").strip()
        filename = str(payload.get("source") or "").strip()
        if not document_id and not filename:
            continue
        key = (document_id, filename)
        if key in seen:
            continue
        seen.add(key)
        ref: dict[str, str] = {}
        if document_id:
            ref["document_id"] = document_id
        if filename:
            ref["filename"] = filename
        refs.append(ref)

    if refs:
        st.session_state["fm_last_knowledge_document_refs"] = refs
    else:
        st.session_state.pop("fm_last_knowledge_document_refs", None)


def _active_knowledge_document_ids(*, user_id: str | None) -> set[str] | None:
    """Read active SQL document ids used to filter Qdrant retrieval.

    The SQL document table is the source of truth for whether a user has kept a
    knowledge file active. Qdrant still stores the vectors, but this filter stops
    deactivated documents from participating in answer generation.

    Args:
        user_id: Current FurnaceMind user id. When missing, the caller cannot
            perform user-scoped SQL filtering.

    Returns:
        A set of active MRAG document ids. An empty set means the user has no
        active documents. ``None`` means the repository or user context is not
        available, so the caller should continue without this SQL filter.
    """
    repository = st.session_state.get("knowledge_document_repository")
    if repository is None or not user_id:
        return None

    try:
        documents = repository.list_documents(user_id=user_id, active_only=True)
    except Exception:
        return None

    active_ids: set[str] = set()
    for document in documents:
        metadata = getattr(document, "metadata_json", None)
        if not isinstance(metadata, dict):
            continue
        document_id = str(metadata.get("document_id") or "").strip()
        if document_id:
            active_ids.add(document_id)
    return active_ids


def _knowledge_tokens(text: str) -> set[str]:
    """Tokenize text for the local MRAG reranker.

    Args:
        text: User query or candidate payload text.

    Returns:
        Lowercase alphanumeric/underscore tokens with length at least three.
        The tokenizer is intentionally simple and does not remove stop words;
        vector search remains the primary retrieval signal.
    """
    return set(re.findall(r"[a-zA-Z0-9_]{3,}", text.lower()))


def _rerank_knowledge_results(
    query: str,
    results: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Rerank Qdrant candidates with a small lexical boost.

    Qdrant's vector score stays dominant. The local score adds overlap with the
    query tokens, a small exact-query bonus, and an image-modality bonus when the
    user appears to ask for visual evidence. This improves ordering while keeping
    semantic retrieval as the main signal.

    Args:
        query: Original user question or focused retrieval query.
        results: Raw Qdrant result dictionaries to rerank.
        limit: Maximum number of results returned to the LLM.

    Returns:
        The top reranked results, each with ``rerank_score`` added.
    """
    query_tokens = _knowledge_tokens(query)
    if not query_tokens:
        return results[:limit]

    query_text = query.lower().strip()
    reranked: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        payload = result.get("payload") or {}
        searchable = " ".join(
            str(payload.get(key) or "")
            for key in (
                "content",
                "source",
                "file_type",
                "modality",
                "sheet_name",
            )
        )
        doc_tokens = _knowledge_tokens(searchable)
        overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
        vector_score = result.get("score")
        if not isinstance(vector_score, (int, float)):
            vector_score = 0.0
        exact_bonus = 0.05 if query_text and query_text in searchable.lower() else 0.0
        modality_bonus = (
            0.03
            if _knowledge_payload_has_visual(payload) and "image" in query_tokens
            else 0.0
        )
        rerank_score = (0.82 * float(vector_score)) + (0.18 * overlap)
        rerank_score += exact_bonus + modality_bonus
        copy = {**result, "rerank_score": rerank_score, "_source_rank": index}
        reranked.append(copy)

    reranked.sort(
        key=lambda item: (
            item.get("rerank_score", 0.0),
            item.get("score") if isinstance(item.get("score"), (int, float)) else 0.0,
            -item.get("_source_rank", 0),
        ),
        reverse=True,
    )
    for item in reranked:
        item.pop("_source_rank", None)
    return reranked[:limit]


def _log_knowledge_retrieval_trace(
    *,
    query: str,
    user_id: str | None,
    active_document_ids: set[str] | None,
    knowledge_store: Any,
    results: list[dict[str, Any]],
) -> None:
    """Persist a best-effort audit record for an MRAG retrieval.

    Retrieval traces are diagnostic. A trace write failure is stored in session
    state for the UI or developer to inspect, but it must never block the chat
    response.

    Args:
        query: Query sent to the knowledge store.
        user_id: Current FurnaceMind user id, if available.
        active_document_ids: SQL-active document ids used for filtering, or
            ``None`` when no SQL filter was available.
        knowledge_store: Store object used for the vector search. Its collection
            name is copied into the trace.
        results: Final reranked results returned by ``search_knowledge_docs``.

    Returns:
        None.
    """
    repository = st.session_state.get("knowledge_retrieval_trace_repository")
    if repository is None:
        return

    try:
        repository.create_trace(
            user_id=user_id,
            conversation_id=st.session_state.get("fm_conversation_id"),
            query=query,
            qdrant_collection=getattr(knowledge_store, "collection_name", None),
            results=results,
            active_document_ids=(
                sorted(active_document_ids) if active_document_ids is not None else None
            ),
            metadata={
                "tool": "search_knowledge_docs",
                "candidate_limit": _KNOWLEDGE_RERANK_CANDIDATES,
                "returned_limit": len(results),
            },
        )
        st.session_state.pop("fm_last_knowledge_trace_error", None)
    except Exception as exc:
        st.session_state["fm_last_knowledge_trace_error"] = str(exc)


def consume_pending_mrag_image_message() -> dict[str, Any] | None:
    """Build the visual-evidence message consumed after knowledge search.

    ``search_knowledge_docs`` queues visual payload references in session state.
    This function removes that queue, resolves each visual from the original
    PostgreSQL-stored upload or a legacy local path, base64-encodes it, and
    returns a model message compatible with OpenAI/OpenRouter multimodal chat
    inputs. Missing files and oversized images are skipped.

    Returns:
        A user message containing source labels and ``image_url`` parts, or
        ``None`` when no queued image can be attached.
    """
    attachments = st.session_state.pop("fm_mrag_image_results", []) or []
    if not attachments:
        return None

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Retrieved visual evidence from the FurnaceMind MRAG knowledge "
                "store. Use these images only as source context for the user's "
                "question."
            ),
        }
    ]
    attached_count = 0
    for attachment in attachments[:_KNOWLEDGE_IMAGE_RESULT_LIMIT]:
        resolved = _resolve_visual_attachment(attachment)
        if not resolved:
            continue
        image_bytes, mime_type, label = resolved
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content.append(
            {
                "type": "text",
                "text": f"Visual source: {label}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )
        attached_count += 1

    if attached_count == 0:
        return None
    return {"role": "user", "content": content}


def _append_tool_error(*, tool_name: str, params: Dict[str, Any], error: str) -> None:
    """Append tool failure details to ``tool_errors.md`` without raising.

    Tool execution errors should be visible to developers for later diagnosis,
    but they must not crash the Streamlit chat session. This helper records the
    tool name, sanitized parameters, timestamp, and error text on a best-effort
    basis and silently returns if logging itself fails.
    """
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
            _TOOL_ERRORS_PATH.write_text(
                _TOOL_ERRORS_PATH.read_text(encoding="utf-8") + entry, encoding="utf-8"
            )
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
    """Normalize a lookback string into a TIMEDELTAS-compatible key.

    Accepts both compact forms ("8h", "2d", "30m") and natural language
    ("last 8 hours", "last 2 days") and extends TIMEDELTAS on the fly.
    """
    tr = (user_time_range or "").strip().lower()
    if not tr:
        return "last 8 hours"

    # Already a known key
    if tr in TIMEDELTAS:
        return tr

    # Compact bare forms: "8h", "30m", "2d", "1w"
    m = re.match(r"^(\d+)\s*(m|min|mins|minute|minutes)$", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} minutes" if n != 1 else "last 1 minute"
        TIMEDELTAS.setdefault(key, timedelta(minutes=n))
        return key

    m = re.match(r"^(\d+)\s*(h|hr|hrs|hour|hours)$", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} hours" if n != 1 else "last 1 hour"
        TIMEDELTAS.setdefault(key, timedelta(hours=n))
        return key

    m = re.match(r"^(\d+)\s*(d|day|days)$", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} days" if n != 1 else "last 1 day"
        TIMEDELTAS.setdefault(key, timedelta(days=n))
        return key

    m = re.match(r"^(\d+)\s*(w|week|weeks)$", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} weeks" if n != 1 else "last 1 week"
        TIMEDELTAS.setdefault(key, timedelta(weeks=n))
        return key

    # "last N <unit>" natural language forms
    m = re.search(r"last\s+(\d+)\s*(minute|minutes|min|mins)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} minutes" if n != 1 else "last 1 minute"
        TIMEDELTAS.setdefault(key, timedelta(minutes=n))
        return key

    m = re.search(r"last\s+(\d+)\s*(hour|hours|hr|hrs|h)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} hours" if n != 1 else "last 1 hour"
        TIMEDELTAS.setdefault(key, timedelta(hours=n))
        return key

    m = re.search(r"last\s+(\d+)\s*(day|days|d)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} days" if n != 1 else "last 1 day"
        TIMEDELTAS.setdefault(key, timedelta(days=n))
        return key

    m = re.search(r"last\s+(\d+)\s*(week|weeks|w)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} weeks" if n != 1 else "last 1 week"
        TIMEDELTAS.setdefault(key, timedelta(weeks=n))
        return key

    return "last 8 hours"


def _safe_exec(
    code: str,
    local_vars: Dict[str, Any],
    stdout_buf: "io.StringIO | None" = None,
) -> None:
    """Execute plotting code with a restricted builtins set and basic static checks.

    Args:
        code: Python code string to execute.
        local_vars: Namespace dict (mutated in-place; contains execution results).
        stdout_buf: Optional :class:`io.StringIO` buffer.  When provided, all
            ``print()`` calls inside the executed code write to this buffer
            instead of real stdout so callers can capture diagnostic output.
    """
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

    # Route print() to the buffer when one is provided so callers can capture output.
    if stdout_buf is not None:

        def _buffered_print(*args, **kwargs):  # noqa: ANN202
            kwargs.setdefault("file", stdout_buf)
            print(*args, **kwargs)  # noqa: T201

        captured_print = _buffered_print
    else:
        captured_print = print

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
        "bool": bool,
        "round": round,
        "print": captured_print,
        "isinstance": isinstance,
        "hasattr": hasattr,
        "getattr": getattr,
        "None": None,
        "True": True,
        "False": False,
    }

    # Merge builtins into local_vars so list comprehensions can see all names.
    # (Python 3 list comprehensions resolve names via globals, not locals, in exec.)
    local_vars["__builtins__"] = safe_builtins
    exec(code, local_vars)  # noqa: S102


@tool
def execute_python_plot(code: str) -> str:
    """
    Execute restricted Python code to create a Plotly figure.

    PRE-LOADED — do NOT import these, they are already available:
      pd            — pandas
      px            — plotly.express
      go            — plotly.graph_objects
      make_subplots — plotly.subplots.make_subplots
      np            — numpy
      df            — current DataFrame (most recently fetched dataset)

    RULES — the sandbox will reject code that violates these:
      - DO NOT use 'import', '__import__', 'open(', 'os', 'subprocess', 'sys', 'eval', 'exec'
      - DO NOT call fig.show() — the figure is rendered automatically by the UI
      - The code MUST assign a Plotly figure to a variable named 'fig'

    DIAGNOSTIC USE: Code that only calls print() (no fig) is allowed — the output
    will be returned so you can inspect column names, index ranges, etc., then plot
    on the next call.

    Example:
        fig = px.line(df.reset_index(), x='index', y='fuel_rate', title='Fuel Rate')
    """
    try:
        # Load the active DataFrame directly from session state (no disk I/O needed)
        df = st.session_state.get("fm_df")

        import numpy as np  # noqa: PLC0415 — local import intentional for sandbox context
        from plotly.subplots import make_subplots  # noqa: PLC0415

        # Capture stdout so diagnostic print() calls are returned to the LLM
        stdout_buf = io.StringIO()

        # Create a local environment for execution
        local_vars = {
            "pd": pd,
            "px": px,
            "go": go,
            "df": df,
            "np": np,
            "make_subplots": make_subplots,
            "_stdout_buf": stdout_buf,
        }

        # Execute the LLM-generated code (restricted), routing print() to buffer
        _safe_exec(code, local_vars, stdout_buf=stdout_buf)

        captured_output = stdout_buf.getvalue().strip()

        if "fig" in local_vars:
            # Save the figure object to session state for the UI to pick up
            st.session_state.fm_fig = local_vars["fig"]
            st.session_state.last_plot_code = code
            _tag_artifact_turn("fm_fig_turn_id")
            return "Successfully generated Plotly figure."
        elif captured_output:
            # Diagnostic code (e.g. print(df.columns)) — return output so the LLM
            # can use the information to write a proper plot on the next call.
            return f"Diagnostic output (no figure created):\n{captured_output}"
        else:
            return "Code executed but no variable named 'fig' was found."

    except Exception as e:
        _append_tool_error(
            tool_name="execute_python_plot",
            params={
                "code": (
                    (code[:2000] + "…")
                    if isinstance(code, str) and len(code) > 2000
                    else code
                )
            },
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

    results = shift_store.search_similar_windows(query_text=query, top_k=5)

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
    """Search active uploaded MRAG documents for the current chat question.

    This is the LLM tool for uploaded PDFs, PPTX files, DOCX files, tables,
    images, SOPs, manuals, specifications, and scanned pages. It performs
    user-scoped vector search, active-document filtering, local reranking,
    retrieval trace logging, and visual-evidence queuing. Text evidence is
    returned directly as tool output; image evidence is queued for the next
    multimodal model call.

    Args:
        query: User question or focused retrieval query generated by the LLM.

    Returns:
        Formatted evidence chunks with source labels and scores, or a clear
        empty-state message when no active or relevant knowledge is available.
    """
    knowledge_store = st.session_state.get("knowledge_store")
    if knowledge_store is None:
        return "Knowledge store not initialized."

    st.session_state.pop("fm_mrag_image_results", None)
    st.session_state.pop("fm_last_knowledge_document_refs", None)
    user_id = st.session_state.get("fm_user_id")
    active_document_ids = _active_knowledge_document_ids(user_id=user_id)
    if active_document_ids == set():
        return "No active knowledge documents found for this user."

    candidate_results = knowledge_store.search(
        query,
        top_k=_KNOWLEDGE_RERANK_CANDIDATES,
        user_id=user_id,
        active_document_ids=active_document_ids,
    )
    results = _rerank_knowledge_results(
        query, candidate_results, limit=_KNOWLEDGE_RETURN_LIMIT
    )
    _log_knowledge_retrieval_trace(
        query=query,
        user_id=user_id,
        active_document_ids=active_document_ids,
        knowledge_store=knowledge_store,
        results=results,
    )

    if not results:
        return "No knowledge documents found for this query."

    _store_knowledge_image_results(results)
    _store_knowledge_document_refs(results)

    parts = []
    for i, r in enumerate(results, 1):
        payload = r.get("payload", {})
        content = str(payload.get("content") or "").strip()
        location = _knowledge_location(payload)
        score = r.get("score")
        score_text = f" | score={score:.3f}" if isinstance(score, (int, float)) else ""
        rerank_score = r.get("rerank_score")
        if isinstance(rerank_score, (int, float)):
            score_text = f"{score_text} | rerank={rerank_score:.3f}"
        if _knowledge_payload_has_visual(payload):
            visual_note = (
                "Visual attachment retrieved and provided to the model for inspection."
            )
            content = "\n".join(part for part in (content, visual_note) if part)
        if not content:
            content = "No text extracted for this result."
        parts.append(f"[{i}] {location}{score_text}\n{content}")

    return "\n\n".join(parts)
