"""API-first Data Explorer.

This page deliberately has no InfluxDB, PostgreSQL, local CSV, dataset-builder,
or validation imports.  Both API and temporary direct compatibility mode are
accessed exclusively through the Data Explorer gateways.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.services.api_errors import FrontendApiError
from apps.frontend_streamlit.services.data_explorer_gateway import (
    DataQueryGateway,
    DatasetGateway,
    get_data_explorer_gateways,
    new_idempotency_key,
)


DISPLAY_TIMEZONE = "Asia/Kolkata"
LOCAL_TZ = ZoneInfo(DISPLAY_TIMEZONE)
_GATEWAY_ERRORS = (FrontendApiError, ValueError, TypeError, RuntimeError, OSError, KeyError)


def _fingerprint(request: Mapping[str, Any]) -> str:
    value = json.dumps(request, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(key: str) -> dict[str, Any] | None:
    record = st.session_state.get(key)
    return record if isinstance(record, dict) else None


def _store_result(key: str, request: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    st.session_state[key] = {
        "fingerprint": _fingerprint(request),
        "request": dict(request),
        "data": dict(result),
    }


def _result_is_stale(record_key: str, request: Mapping[str, Any]) -> bool:
    """Return whether a saved result belongs to different current form inputs."""

    record = _record(record_key)
    saved_fingerprint = record.get("fingerprint") if record is not None else None
    return isinstance(saved_fingerprint, str) and saved_fingerprint != _fingerprint(request)


def _show_stale_notice(record_key: str, request: Mapping[str, Any], title: str) -> bool:
    stale = _result_is_stale(record_key, request)
    if stale:
        st.info(f"Inputs changed since the displayed {title}. Run the form again to refresh it.")
    return stale


def _show_error(section: str, exc: Exception) -> None:
    if isinstance(exc, FrontendApiError):
        suffix = f" ({exc.error_code})" if exc.error_code else ""
        st.error(f"{section}: {exc.message}{suffix}")
        if exc.request_id:
            st.caption(f"Backend request ID: {exc.request_id}")
    else:
        st.error(f"{section}: {exc}")


def _call(section: str, callback: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return callback()
    except _GATEWAY_ERRORS as exc:
        _show_error(section, exc)
        return None


def _warnings(result: Mapping[str, Any] | None) -> None:
    if result is not None:
        for warning in result.get("warnings") or []:
            st.warning(str(warning))


def _labels(items: list[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(item["id"]): str(item.get("label") or item["id"])
        for item in items
        if item.get("id") is not None
    }


def _find(items: list[Mapping[str, Any]], item_id: str) -> Mapping[str, Any]:
    return next((item for item in items if str(item.get("id")) == str(item_id)), {})


def _format(labels: Mapping[str, str]) -> Callable[[str], str]:
    return lambda item: labels.get(item, item)


def _iso(day: date, value: time) -> str:
    return datetime.combine(day, value, tzinfo=LOCAL_TZ).isoformat()


def _local_date(value: Any, fallback: date) -> date:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            return parsed.astimezone(LOCAL_TZ).date()
    except ValueError:
        pass
    return fallback


def _catalog_limit(catalog: Mapping[str, Any] | None, key: str, fallback: int) -> int:
    value = ((catalog or {}).get("limits") or {}).get(key, fallback)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return fallback


def _range_selector(
    key: str,
    presets: list[Mapping[str, Any]],
    default_start: date,
    default_end: date,
) -> dict[str, str]:
    labels = _labels(presets)
    mode = st.radio(
        "Range",
        ["Preset", "Custom range"] if labels else ["Custom range"],
        horizontal=True,
        key=f"{key}_range_mode",
    )
    if mode == "Preset":
        preset_id = st.selectbox(
            "Time preset", list(labels), format_func=_format(labels), key=f"{key}_preset"
        )
        return {"kind": "preset", "preset_id": preset_id}
    first, second = st.columns(2)
    with first:
        start = st.date_input("Start date", default_start, key=f"{key}_start")
    with second:
        end = st.date_input("End date", default_end, key=f"{key}_end")
    return {"kind": "absolute", "start": _iso(start, time.min), "end": _iso(end, time.max)}


def _preview(record_key: str, title: str, *, stale: bool = False) -> None:
    record = _record(record_key)
    if record is None or not isinstance(record.get("data"), Mapping):
        return
    result = record["data"]
    if stale:
        st.caption(f"Showing the previous {title.lower()} result.")
    _warnings(result)
    rows = result.get("rows") or []
    frame = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
    if result.get("truncated"):
        st.info(
            f"{title} is a bounded preview: {result.get('returned_rows', 0):,} of "
            f"{result.get('total_rows', result.get('row_count', 'more'))} rows."
        )
    if frame.empty:
        st.info(f"No rows were returned for {title.lower()}.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def _export_control(
    gateway: DataQueryGateway | None,
    *,
    preview_key: str,
    export_key: str,
    label: str,
    hot_metal_slag: bool = False,
    stale: bool = False,
) -> None:
    record = _record(preview_key)
    if gateway is None or record is None or not isinstance(record.get("request"), Mapping):
        return
    if st.button(f"Prepare {label}", key=f"{export_key}_prepare", disabled=stale):
        key = new_idempotency_key()
        st.session_state[f"{export_key}.idempotency_key"] = key
        try:
            if hot_metal_slag:
                export = gateway.export_hot_metal_slag(dict(record["request"]), idempotency_key=key)
            else:
                export = gateway.create_export(
                    {"query": dict(record["request"]), "format": "csv"},
                    idempotency_key=key,
                )
            artifact_id = str(export.get("artifact_id") or "")
            if not artifact_id:
                raise ValueError("The export response did not include an artifact id.")
            st.session_state[export_key] = {
                "bytes": gateway.download_artifact(artifact_id),
                "filename": str(export.get("filename") or "data_export.csv"),
            }
        except _GATEWAY_ERRORS as exc:
            _show_error(label, exc)
    download = _record(export_key)
    if download is not None and isinstance(download.get("bytes"), bytes):
        st.download_button(
            f"Download {label}",
            data=download["bytes"],
            file_name=str(download.get("filename") or "data_export.csv"),
            mime="text/csv",
            key=f"{export_key}_download",
        )


def _render_scatter(result: Mapping[str, Any]) -> None:
    x, y = result.get("x") or [], result.get("y") or []
    if not x or not y:
        st.info("No valid scatter points were returned.")
        return
    figure = go.Figure(go.Scatter(x=x, y=y, mode="markers", name="Data", marker={"size": 5}))
    regression = result.get("regression")
    if isinstance(regression, Mapping) and regression.get("line_x") and regression.get("line_y"):
        figure.add_trace(
            go.Scatter(
                x=regression["line_x"],
                y=regression["line_y"],
                mode="lines",
                name=f"Polynomial degree {regression.get('degree')}",
            )
        )
    figure.update_layout(height=450, template="plotly_white")
    st.plotly_chart(figure, use_container_width=True, key="data_explorer_scatter_plot")
    dropped = result.get("dropped_rows") or {}
    st.caption(
        f"{result.get('returned_points', 0):,} shown of {result.get('total_matching_rows', 0):,} matching "
        f"rows · dropped null/non-numeric/non-finite: {dropped.get('null', 0)}/"
        f"{dropped.get('non_numeric', 0)}/{dropped.get('non_finite', 0)}"
    )
    if result.get("downsampled"):
        st.info("Display points were deterministically downsampled; regression used all valid rows.")


def _render_timeseries(result: Mapping[str, Any]) -> None:
    series = result.get("series") or []
    if not series:
        st.info("No time-series points were returned.")
        return
    figure = go.Figure()
    for entry in series:
        if isinstance(entry, Mapping):
            points = entry.get("points") or []
            figure.add_trace(
                go.Scatter(
                    x=[point.get("timestamp") for point in points],
                    y=[point.get("value") for point in points],
                    mode="lines",
                    name=str(entry.get("label") or entry.get("field")),
                )
            )
    figure.update_layout(height=500, template="plotly_white", hovermode="x unified")
    st.plotly_chart(figure, use_container_width=True, key="data_explorer_timeseries_plot")
    if result.get("downsampled"):
        st.info("One or more series were deterministically downsampled for display.")


def _poll_active_job(gateway: DatasetGateway | None) -> Mapping[str, Any] | None:
    active = _record("data_explorer.active_job")
    if gateway is None or active is None:
        return None
    job_id = str(active.get("job_id") or "")
    if not job_id:
        return None
    status = str(active.get("status") or "queued")
    refreshed_metadata: Mapping[str, Any] | None = None
    if status in {"queued", "pending", "running"}:
        try:
            latest = gateway.get_job(job_id)
            event_response = gateway.get_job_events(
                job_id, after=int(active.get("last_event_sequence") or 0)
            )
            events = [event for event in (event_response.get("events") or []) if isinstance(event, Mapping)]
            if events:
                active["events"] = [*(active.get("events") or []), *events]
                active["last_event_sequence"] = max(
                    int(event.get("sequence") or 0) for event in events
                )
            active.update(latest)
            st.session_state["data_explorer.active_job"] = active
            status = str(active.get("status") or status)
        except _GATEWAY_ERRORS as exc:
            _show_error("Dataset job status", exc)
            return None
    progress = active.get("progress")
    if isinstance(progress, (int, float)):
        st.progress(min(max(int(progress), 0), 100), text=f"Dataset job {job_id}: {status}")
    else:
        st.caption(f"Dataset job {job_id}: {status}")
    for event in (active.get("events") or [])[-3:]:
        if isinstance(event, Mapping):
            st.caption(f"{event.get('stage', 'job')}: {event.get('message', '')}")
    if status in {"queued", "pending", "running"} and st.button("Cancel active dataset job"):
        try:
            active.update(gateway.cancel_job(job_id))
            st.session_state["data_explorer.active_job"] = active
            st.rerun()
        except _GATEWAY_ERRORS as exc:
            _show_error("Cancel dataset job", exc)
    elif status == "completed":
        if not active.get("post_completion_refreshed"):
            try:
                refreshed_metadata = gateway.get_static_metadata()
                st.session_state["data_explorer.validation_result"] = gateway.get_validation()
                active["post_completion_refreshed"] = True
                st.session_state["data_explorer.active_job"] = active
            except _GATEWAY_ERRORS as exc:
                _show_error("Dataset completion refresh", exc)
        st.success("Dataset job completed.")
        if st.button("Prepare dataset job download"):
            try:
                st.session_state["data_explorer.job_download"] = {
                    "bytes": gateway.download_job_result(job_id),
                    "filename": f"static_ml_dataset_{job_id}.csv",
                }
            except _GATEWAY_ERRORS as exc:
                _show_error("Dataset job download", exc)
        download = _record("data_explorer.job_download")
        if download is not None and isinstance(download.get("bytes"), bytes):
            st.download_button(
                "Download dataset job result",
                data=download["bytes"],
                file_name=str(download.get("filename") or "static_ml_dataset.csv"),
                mime="text/csv",
            )
    elif status in {"failed", "cancelled", "canceled"}:
        st.error(str(active.get("error_message") or f"Dataset job {status}."))
    return refreshed_metadata

def _presets(catalog: Mapping[str, Any] | None, source: str) -> list[Mapping[str, Any]]:
    return [
        preset
        for preset in ((catalog or {}).get("time_presets") or [])
        if source in (preset.get("sources") or [])
    ]


def _permissions() -> set[str]:
    return {str(value) for value in st.session_state.get("permissions") or []}


st.title("Visualisation tool")
api_mode = is_backend_api_enabled("data_explorer")
st.caption(f"Data mode: **{'Backend API' if api_mode else 'Direct compatibility'}**")

data_gateway: DataQueryGateway | None = None
dataset_gateway: DatasetGateway | None = None
try:
    data_gateway, dataset_gateway = get_data_explorer_gateways(
        access_token=str(st.session_state.get("auth_access_token") or "").strip()
    )
except FrontendApiError as exc:
    # API mode is all-or-nothing. Do not silently fall back to direct access.
    _show_error("Data Explorer gateway", exc)

catalog = _call("Data catalog", data_gateway.get_catalog) if data_gateway is not None else None
metadata = (
    _call("Static dataset metadata", dataset_gateway.get_static_metadata)
    if dataset_gateway is not None
    else None
)
_warnings(catalog)
_warnings(metadata)

# Distribution ---------------------------------------------------------------
st.subheader("Distribution plots")
static_columns = [
    column
    for column in ((metadata or {}).get("columns") or [])
    if isinstance(column, Mapping) and column.get("id") and column.get("plottable", True) and str(column.get("dtype") or "").lower() in {"number", "numeric", "float", "integer", "int"}
]
static_labels = _labels(static_columns)
if not static_labels:
    st.info("Static metadata is unavailable; this does not hide the remaining sections.")
else:
    ids = list(static_labels)
    with st.form("data_explorer_scatter_form"):
        first, second = st.columns(2)
        with first:
            x_field = st.selectbox("Select X feature", ids, format_func=_format(static_labels))
        with second:
            y_field = st.selectbox(
                "Select Y feature",
                ids,
                index=1 if len(ids) > 1 else 0,
                format_func=_format(static_labels),
            )
        filter_on = st.checkbox("Filter results")
        scatter_filter: dict[str, Any] | None = None
        if filter_on:
            one, two, three, four = st.columns(4)
            with one:
                filter_field = st.selectbox("Filter field", ids, format_func=_format(static_labels))
            with two:
                filter_mode = st.selectbox("Filter mode", ["inside", "outside"])
            with three:
                minimum = st.number_input("Minimum", value=0.0)
            with four:
                maximum = st.number_input("Maximum", value=0.0)
            scatter_filter = {
                "field": filter_field,
                "mode": filter_mode,
                "minimum": minimum,
                "maximum": maximum,
            }
        regression_enabled = st.checkbox("Fit polynomial regression", value=True)
        degree = st.selectbox("Polynomial degree", [1, 2, 3, 4, 5], index=1)
        run_scatter = st.form_submit_button("Run analysis")
    scatter_request = {
        "dataset_version": str((metadata or {}).get("version") or ""),
        "x_field": x_field,
        "y_field": y_field,
        "filter": scatter_filter,
        "regression": {"enabled": regression_enabled, "degree": degree},
        "max_points": _catalog_limit(catalog, "max_scatter_points", 5000),
    }
    if run_scatter:
        if x_field == y_field:
            st.warning("Select different X and Y fields.")
        elif scatter_filter is not None and scatter_filter["minimum"] > scatter_filter["maximum"]:
            st.warning("Filter minimum cannot exceed maximum.")
        elif dataset_gateway is not None and metadata is not None:
            result = _call("Distribution analysis", lambda: dataset_gateway.get_scatter_analysis(scatter_request))
            if result is not None:
                _store_result("data_explorer.scatter_result", scatter_request, result)
    _show_stale_notice("data_explorer.scatter_result", scatter_request, "scatter analysis")
    scatter = _record("data_explorer.scatter_result")
    if scatter is not None and isinstance(scatter.get("data"), Mapping):
        _warnings(scatter["data"])
        _render_scatter(scatter["data"])

# Time series ----------------------------------------------------------------
st.subheader("Timeseries plot")
if not static_labels:
    st.info("Static metadata is unavailable; this does not hide the remaining sections.")
else:
    today = datetime.now(LOCAL_TZ).date()
    range_meta = (metadata or {}).get("range") or {}
    default_start = _local_date(range_meta.get("start"), today - timedelta(days=7))
    default_end = _local_date(range_meta.get("end"), today)
    with st.form("data_explorer_timeseries_form"):
        fields = st.multiselect(
            "Select features",
            list(static_labels),
            default=list(static_labels)[:1],
            format_func=_format(static_labels),
            max_selections=_catalog_limit(catalog, "max_selected_fields", len(static_labels)),
        )
        left, right = st.columns(2)
        with left:
            from_date = st.date_input("From date", default_start, key="data_explorer_timeseries_from")
        with right:
            to_date = st.date_input("To date", default_end, key="data_explorer_timeseries_to")
        ts_filter_on = st.checkbox("Filter values", key="data_explorer_ts_filter_on")
        ts_filter: dict[str, Any] | None = None
        if ts_filter_on:
            one, two, three, four = st.columns(4)
            with one:
                ts_field = st.selectbox(
                    "Filter field", list(static_labels), format_func=_format(static_labels)
                )
            with two:
                ts_mode = st.selectbox("Filter mode", ["inside", "outside"], key="data_explorer_ts_mode")
            with three:
                ts_minimum = st.number_input("Minimum", value=0.0, key="data_explorer_ts_min")
            with four:
                ts_maximum = st.number_input("Maximum", value=0.0, key="data_explorer_ts_max")
            ts_filter = {
                "field": ts_field,
                "mode": ts_mode,
                "minimum": ts_minimum,
                "maximum": ts_maximum,
            }
        resample = st.selectbox("Resample", ["none", "5min", "15min", "1h", "6h", "1d"])
        run_timeseries = st.form_submit_button("Run time-series")
    timeseries_request = {
        "dataset_version": str((metadata or {}).get("version") or ""),
        "fields": fields,
        "time_range": {
            "start": _iso(from_date, time.min),
            "end": _iso(to_date, time.max),
        },
        "filter": ts_filter,
        "resample": None if resample == "none" else {"mode": "mean", "window": resample},
        "max_points_per_field": _catalog_limit(catalog, "max_timeseries_points_per_field", 5000),
    }
    if run_timeseries:
        if not fields:
            st.warning("Select at least one feature.")
        elif from_date > to_date:
            st.warning("From date cannot be after To date.")
        elif ts_filter is not None and ts_filter["minimum"] > ts_filter["maximum"]:
            st.warning("Filter minimum cannot exceed maximum.")
        elif dataset_gateway is not None and metadata is not None:
            result = _call("Time-series analysis", lambda: dataset_gateway.get_timeseries(timeseries_request))
            if result is not None:
                _store_result("data_explorer.timeseries_result", timeseries_request, result)
    _show_stale_notice("data_explorer.timeseries_result", timeseries_request, "time-series analysis")
    timeseries = _record("data_explorer.timeseries_result")
    if timeseries is not None and isinstance(timeseries.get("data"), Mapping):
        _warnings(timeseries["data"])
        _render_timeseries(timeseries["data"])

# Online data ----------------------------------------------------------------
st.header("📊 Online Data Downloader")
online_measurements = list((catalog or {}).get("online_measurements") or [])
online_labels = _labels(online_measurements)
if not online_labels:
    st.info("Online catalog metadata is unavailable; other sections remain usable.")
else:
    aggregation_labels = _labels(list((catalog or {}).get("aggregation_windows") or []))
    preview_limit = int(((catalog or {}).get("limits") or {}).get("max_preview_rows") or 500)
    with st.form("data_explorer_online_form"):
        selected_measurements = st.multiselect(
            "Measurements",
            list(online_labels),
            default=list(online_labels),
            format_func=_format(online_labels),
        )
        scoped_online_fields = [
            field
            for measurement in online_measurements
            if str(measurement.get("id")) in selected_measurements
            for field in (measurement.get("fields") or [])
        ]
        field_labels = _labels(scoped_online_fields)
        online_range = _range_selector(
            "data_explorer_online",
            _presets(catalog, "online"),
            datetime.now(LOCAL_TZ).date() - timedelta(days=1),
            datetime.now(LOCAL_TZ).date(),
        )
        aggregation_id = st.selectbox(
            "Aggregation window", list(aggregation_labels), format_func=_format(aggregation_labels)
        )
        selected_fields = st.multiselect(
            "Fields (optional)", list(field_labels), format_func=_format(field_labels),
            max_selections=_catalog_limit(catalog, "max_selected_fields", len(field_labels)),
        )
        fetch_online = st.form_submit_button("Fetch online preview")
    online_request = {
        "source": "online",
        "measurements": selected_measurements,
        "time_range": online_range,
        "aggregation": (
            None if aggregation_id == "none" else {"mode": "mean", "window_id": aggregation_id}
        ),
        "fields": selected_fields or None,
        "limit": preview_limit,
        "offset": 0,
    }
    if fetch_online:
        if not selected_measurements:
            st.warning("Select at least one measurement.")
        elif data_gateway is not None:
            result = _call("Online data preview", lambda: data_gateway.preview(online_request))
            if result is not None:
                _store_result("data_explorer.online_preview", online_request, result)
    online_stale = _show_stale_notice("data_explorer.online_preview", online_request, "online preview")
    _preview("data_explorer.online_preview", "Online data", stale=online_stale)
    _export_control(
        data_gateway,
        preview_key="data_explorer.online_preview",
        export_key="data_explorer.online_export",
        label="full online CSV",
        stale=online_stale,
    )

# Offline data ---------------------------------------------------------------
st.header("📁 Offline Data Downloader")
reports = list((catalog or {}).get("offline_reports") or [])
tables = list((catalog or {}).get("offline_tables") or [])
report_labels = _labels(reports)
table_labels = _labels(tables)
if not report_labels and not table_labels:
    st.info("Offline catalog metadata is unavailable; other sections remain usable.")
else:
    preview_limit = int(((catalog or {}).get("limits") or {}).get("max_preview_rows") or 500)
    with st.form("data_explorer_offline_form"):
        source_kind = st.radio(
            "Offline source",
            (["Report"] if report_labels else []) + (["Table"] if table_labels else []),
            horizontal=True,
        )
        selected_fields: list[str] = []
        if source_kind == "Report":
            report_id = st.selectbox("Report", list(report_labels), format_func=_format(report_labels))
            report_fields = _labels(list(_find(reports, report_id).get("fields") or []))
            selected_fields = st.multiselect(
                "Fields (optional)", list(report_fields), format_func=_format(report_fields),
                max_selections=_catalog_limit(catalog, "max_selected_fields", len(report_fields)),
            )
            selection = {"kind": "report", "report_id": report_id}
        else:
            table_id = st.selectbox("Table", list(table_labels), format_func=_format(table_labels))
            table_fields = _labels(list(_find(tables, table_id).get("fields") or []))
            selected_fields = st.multiselect(
                "Fields (optional)", list(table_fields), format_func=_format(table_fields),
                max_selections=_catalog_limit(catalog, "max_selected_fields", len(table_fields)),
            )
            selection = {"kind": "table", "table_id": table_id}
        offline_range = _range_selector(
            "data_explorer_offline",
            _presets(catalog, "offline"),
            datetime.now(LOCAL_TZ).date() - timedelta(days=30),
            datetime.now(LOCAL_TZ).date(),
        )
        fetch_offline = st.form_submit_button("Fetch offline preview")
    offline_request = {
        "source": "offline",
        "selection": selection,
        "time_range": offline_range,
        "aggregation": None,
        "fields": selected_fields or None,
        "limit": preview_limit,
        "offset": 0,
    }
    if fetch_offline and data_gateway is not None:
        result = _call("Offline data preview", lambda: data_gateway.preview(offline_request))
        if result is not None:
            _store_result("data_explorer.offline_preview", offline_request, result)
    offline_stale = _show_stale_notice("data_explorer.offline_preview", offline_request, "offline preview")
    _preview("data_explorer.offline_preview", "Offline data", stale=offline_stale)
    _export_control(
        data_gateway,
        preview_key="data_explorer.offline_preview",
        export_key="data_explorer.offline_export",
        label="full offline CSV",
        stale=offline_stale,
    )

# Dataset jobs / validation --------------------------------------------------
st.header("📄 ML Dataset")
if metadata is None:
    st.info(
        "No current canonical static dataset is available. You can still build a selected-range "
        "candidate; extend, override, current download, and canonical validation require a dataset."
    )
    today = datetime.now(LOCAL_TZ).date()
    permissions = _permissions()
    can_build = not permissions or "datasets:build" in permissions
    if dataset_gateway is None:
        st.error("Dataset build is unavailable until the Data Explorer gateway can be reached.")
    with st.form("data_explorer_build_form"):
        one, two = st.columns(2)
        with one:
            build_start = st.date_input("Build start", today - timedelta(days=7))
        with two:
            build_end = st.date_input("Build end", today)
        validate = st.checkbox("Validate candidate", value=True)
        produce_download = st.checkbox("Produce download", value=True)
        create_build = st.form_submit_button(
            "Fetch Dataset",
            disabled=not can_build or dataset_gateway is None,
        )
    if create_build:
        if build_start > build_end:
            st.warning("Build start cannot be after build end.")
        elif dataset_gateway is not None:
            request = {
                "operation": "build_range",
                "start": _iso(build_start, time.min),
                "end": _iso(build_end, time.max),
                "options": {"validate": validate, "produce_download": produce_download},
            }
            key = new_idempotency_key()
            st.session_state["data_explorer.build_idempotency_key"] = key
            result = _call(
                "Create dataset build job",
                lambda: dataset_gateway.create_job(request, idempotency_key=key),
            )
            if result is not None:
                st.session_state["data_explorer.active_job"] = {
                    **result,
                    "last_event_sequence": 0,
                    "events": [],
                }
    unavailable_tabs = st.tabs(["Extend to date", "Override range"])
    with unavailable_tabs[0]:
        st.caption("Extend requires a current canonical dataset version.")
        st.button(
            "Extend & Rebuild Dataset",
            key="data_explorer_extend_unavailable",
            disabled=True,
        )
    with unavailable_tabs[1]:
        st.caption("Override requires a current canonical dataset version.")
        st.button(
            "Override & Rebuild",
            key="data_explorer_override_unavailable",
            disabled=True,
        )
    _poll_active_job(dataset_gateway)
else:
    dataset_version = str(metadata.get("version") or "")
    today = datetime.now(LOCAL_TZ).date()
    range_meta = metadata.get("range") or {}
    start_default = _local_date(range_meta.get("start"), today - timedelta(days=7))
    end_default = _local_date(range_meta.get("end"), today)
    permissions = _permissions()
    can_build = not permissions or "datasets:build" in permissions
    can_extend = not permissions or "datasets:refresh" in permissions
    can_override = not permissions or "datasets:override" in permissions
    with st.form("data_explorer_build_form"):
        one, two = st.columns(2)
        with one:
            build_start = st.date_input("Build start", start_default)
        with two:
            build_end = st.date_input("Build end", end_default)
        validate = st.checkbox("Validate candidate", value=True)
        produce_download = st.checkbox("Produce download", value=True)
        create_build = st.form_submit_button("Fetch Dataset", disabled=not can_build)
    if create_build:
        if build_start > build_end:
            st.warning("Build start cannot be after build end.")
        elif dataset_gateway is not None:
            request = {
                "operation": "build_range",
                "start": _iso(build_start, time.min),
                "end": _iso(build_end, time.max),
                "options": {"validate": validate, "produce_download": produce_download},
            }
            key = new_idempotency_key()
            st.session_state["data_explorer.build_idempotency_key"] = key
            result = _call("Create dataset build job", lambda: dataset_gateway.create_job(request, idempotency_key=key))
            if result is not None:
                st.session_state["data_explorer.active_job"] = {
                    **result,
                    "last_event_sequence": 0,
                    "events": [],
                }
    tabs = st.tabs(["Extend to date", "Override range"])
    with tabs[0]:
        with st.form("data_explorer_extend_form"):
            extend_end = st.date_input("Extend through", max(today, end_default))
            create_extend = st.form_submit_button("Extend & Rebuild Dataset", disabled=not can_extend)
        if create_extend and dataset_gateway is not None:
            request = {
                "operation": "extend",
                "end": _iso(extend_end, time.max),
                "expected_dataset_version": dataset_version,
                "options": {"validate": True},
            }
            key = new_idempotency_key()
            st.session_state["data_explorer.extend_idempotency_key"] = key
            result = _call("Create dataset extend job", lambda: dataset_gateway.create_job(request, idempotency_key=key))
            if result is not None:
                st.session_state["data_explorer.active_job"] = {
                    **result,
                    "last_event_sequence": 0,
                    "events": [],
                }
    with tabs[1]:
        st.warning("Override replaces the selected canonical range using backend staging and validation.")
        with st.form("data_explorer_override_form"):
            one, two = st.columns(2)
            with one:
                override_start = st.date_input("Override start", end_default - timedelta(days=30))
            with two:
                override_end = st.date_input("Override end", end_default)
            confirmed = st.checkbox("I confirm this override operation")
            create_override = st.form_submit_button("Override & Rebuild", disabled=not can_override)
        if create_override:
            if not confirmed:
                st.warning("Confirm the override before submitting it.")
            elif override_start > override_end:
                st.warning("Override start cannot be after override end.")
            elif dataset_gateway is not None:
                request = {
                    "operation": "override",
                    "start": _iso(override_start, time.min),
                    "end": _iso(override_end, time.max),
                    "expected_dataset_version": dataset_version,
                    "options": {"validate": True},
                }
                key = new_idempotency_key()
                st.session_state["data_explorer.override_idempotency_key"] = key
                result = _call("Create dataset override job", lambda: dataset_gateway.create_job(request, idempotency_key=key))
                if result is not None:
                    st.session_state["data_explorer.active_job"] = {
                        **result,
                        "last_event_sequence": 0,
                        "events": [],
                    }
    refreshed_metadata = _poll_active_job(dataset_gateway)
    if refreshed_metadata is not None:
        metadata = refreshed_metadata
        dataset_version = str(metadata.get("version") or dataset_version)
    if dataset_gateway is not None and st.button("Prepare current dataset download"):
        try:
            st.session_state["data_explorer.current_dataset_download"] = {
                "bytes": dataset_gateway.download_current_dataset(),
                "filename": f"{metadata.get('dataset_id', 'static_ml_dataset')}_{dataset_version}.csv",
            }
        except _GATEWAY_ERRORS as exc:
            _show_error("Current dataset download", exc)
    current_download = _record("data_explorer.current_dataset_download")
    if current_download is not None and isinstance(current_download.get("bytes"), bytes):
        st.download_button(
            "Download current dataset",
            data=current_download["bytes"],
            file_name=str(current_download.get("filename") or "static_ml_dataset.csv"),
            mime="text/csv",
        )
    st.subheader("Dataset validation")
    st.caption("Validation is explicitly loaded; opening/rerunning this page does not validate or mutate a dataset.")
    if dataset_gateway is not None and st.button("Load validation report"):
        result = _call("Dataset validation", dataset_gateway.get_validation)
        if result is not None:
            st.session_state["data_explorer.validation_result"] = result
    validation = st.session_state.get("data_explorer.validation_result")
    if isinstance(validation, Mapping):
        _warnings(validation)
        summary = validation.get("summary") or {}
        st.caption(
            f"Status: {validation.get('status')} · errors: {summary.get('errors', 0)} · "
            f"warnings: {summary.get('warnings', 0)}"
        )
        if validation.get("checks"):
            st.dataframe(pd.DataFrame(validation["checks"]), use_container_width=True, hide_index=True)

# Hot Metal & Slag -----------------------------------------------------------
st.header("📄 Hot Metal & Slag")
st.caption("The gateway performs hourly/interpolation work; synthetic and interpolated data are disclosed below.")
if data_gateway is None:
    st.info("HM & Slag access is unavailable until the Data Explorer gateway is configured.")
else:
    today = datetime.now(LOCAL_TZ).date()
    hm_max_interval = int(((catalog or {}).get("limits") or {}).get("max_hm_slag_interval_minutes") or 600)
    with st.form("data_explorer_hot_metal_slag_form"):
        left, right = st.columns(2)
        with left:
            hm_start = st.date_input("From date", today - timedelta(days=7), key="data_explorer_hm_start")
        with right:
            hm_end = st.date_input("To date", today, key="data_explorer_hm_end")
        interval = st.number_input("Grid interval (minutes)", min_value=1, max_value=hm_max_interval, value=min(60, hm_max_interval))
        fetch_hm = st.form_submit_button("Fetch HM & Slag Data")
    hm_request = {
        "start": _iso(hm_start, time.min),
        "end": _iso(hm_end, time.max),
        "interval_minutes": int(interval),
        "interpolation": {
            "numeric": "time",
            "metadata": "forward_backward_fill",
        },
        "limit": int(((catalog or {}).get("limits") or {}).get("max_preview_rows") or 500),
        "offset": 0,
    }
    if fetch_hm:
        if hm_start > hm_end:
            st.warning("From date cannot be after To date.")
        else:
            result = _call("HM & Slag preview", lambda: data_gateway.preview_hot_metal_slag(hm_request))
            if result is not None:
                _store_result("data_explorer.hm_slag_preview", hm_request, result)
    hm_stale = _show_stale_notice("data_explorer.hm_slag_preview", hm_request, "HM & Slag preview")
    hm = _record("data_explorer.hm_slag_preview")
    if hm is not None and isinstance(hm.get("data"), Mapping):
        detail = hm["data"]
        _warnings(detail)
        st.caption(
            f"Interval: {detail.get('interval_minutes')} min · synthetic rows: "
            f"{detail.get('synthetic_row_count', 0)} · interpolated columns: "
            f"{len(detail.get('interpolated_columns') or [])}"
        )
    _preview("data_explorer.hm_slag_preview", "HM & Slag data", stale=hm_stale)
    _export_control(
        data_gateway,
        preview_key="data_explorer.hm_slag_preview",
        export_key="data_explorer.hm_slag_export",
        label="full HM & Slag CSV",
        hot_metal_slag=True,
        stale=hm_stale,
    )
