from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import plotly.graph_objs as go
import streamlit as st

from apps.frontend_streamlit.plotters.circumferential_contour import CircumferentialPlotter
from apps.frontend_streamlit.plotters.longitudinal_temp_contour import LongitudinalTemperaturePlotter
from apps.frontend_streamlit.services.api_errors import FrontendApiError
from apps.frontend_streamlit.services.vboard_adapters import (
    circumferential_heatload_rows,
    circumferential_temperature_groups,
    longitudinal_temperature_arrays,
)
from apps.frontend_streamlit.services.vboard_gateway import get_vboard_gateway
from apps.frontend_streamlit.services.vboard_page_helpers import (
    IST,
    absolute_time_range_from_inputs,
    ist_range_caption,
    request_fingerprint,
    utc_range_caption,
)


@st.cache_data(ttl=300, show_spinner=False)
def _circ_fig(field_values, titles, colorbar_title, unit, catalog_version, policy_id, resolution=36):
    plotter = CircumferentialPlotter(mask_file="mask_circular.pkl")
    return plotter.plot_circumferential_quadrants(
        field_values,
        titles=titles,
        colorbar_title=colorbar_title,
        unit=unit or "",
        resolution=resolution,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _long_fig(temperatures, temperatures_max, temperatures_min, catalog_version, policy_id):
    plotter = LongitudinalTemperaturePlotter(mask_file="mask_longitudinal.pkl")
    return plotter.plot_plotly(temperatures, temperatures_max, temperatures_min)


def _render_contour_form(catalog: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    presets = _supported_presets(catalog, "contours")
    default_preset = "last_1_hour" if any(p["id"] == "last_1_hour" for p in presets) else presets[0]["id"]
    options = [p["id"] for p in presets] + ["__absolute__"]
    labels = {p["id"]: p["label"] for p in presets}
    labels["__absolute__"] = "Over selected range"
    now_ist = datetime.now(IST)

    with st.sidebar.form("vboard_contour_form"):
        st.markdown("### Contour - Options")
        selected = st.selectbox(
            "Select Averaging/Display Interval:",
            options=options,
            index=options.index(default_preset),
            format_func=lambda value: labels[value],
            key="vboard.contours.range_kind",
        )
        start_date = now_ist.date() - timedelta(days=1)
        end_date = now_ist.date()
        if selected == "__absolute__":
            left, right = st.columns(2)
            with left:
                from_date = st.date_input("From Date:", value=start_date, key="vboard.contours.from_date")
                from_time = st.time_input("From Time:", value=now_ist.time(), key="vboard.contours.from_time")
            with right:
                to_date = st.date_input("To Date:", value=end_date, key="vboard.contours.to_date")
                to_time = st.time_input("To Time:", value=now_ist.time(), key="vboard.contours.to_time")
        submitted = st.form_submit_button("Refresh Contours")

    try:
        if selected == "__absolute__":
            time_range = absolute_time_range_from_inputs(from_date, from_time, to_date, to_time)
        else:
            time_range = {"kind": "preset", "preset_id": selected}
        return {"time_range": time_range}, submitted, True
    except ValueError as exc:
        st.error(f"Invalid contour range: {exc}")
        return {"time_range": {"kind": "preset", "preset_id": default_preset}}, submitted, False


def _render_timeseries_form(catalog: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    presets = _supported_presets(catalog, "heatload_timeseries")
    rows = catalog.get("rows", [])
    row_ids = [row["id"] for row in rows]
    default_preset = "last_6_hours" if any(p["id"] == "last_6_hours" for p in presets) else presets[0]["id"]
    range_options = [p["id"] for p in presets] + ["__absolute__"]
    range_labels = {p["id"]: p["label"] for p in presets}
    range_labels["__absolute__"] = "Over selected range"
    windows = catalog.get("resolution_windows", [])
    resolution_options = ["auto"] + [window["id"] for window in windows]
    resolution_labels = {"auto": "Auto"}
    resolution_labels.update({window["id"]: window["label"] for window in windows})
    now_ist = datetime.now(IST)

    with st.sidebar.form("vboard_timeseries_form"):
        st.markdown("### Time Series - Options")
        row_id = st.selectbox("Select Row", row_ids, key="vboard.timeseries.row_id")
        selected_range = st.selectbox(
            "TimeSeries - Select Interval:",
            options=range_options,
            index=range_options.index(default_preset),
            format_func=lambda value: range_labels[value],
            key="vboard.timeseries.range_kind",
        )
        selected_resolution = st.selectbox(
            "Resolution",
            options=resolution_options,
            format_func=lambda value: resolution_labels[value],
            key="vboard.timeseries.resolution",
        )
        if selected_range == "__absolute__":
            left, right = st.columns(2)
            with left:
                from_date = st.date_input("From Date:", value=now_ist.date() - timedelta(days=1), key="vboard.timeseries.from_date")
                from_time = st.time_input("From Time:", value=now_ist.time(), key="vboard.timeseries.from_time")
            with right:
                to_date = st.date_input("To Date:", value=now_ist.date(), key="vboard.timeseries.to_date")
                to_time = st.time_input("To Time:", value=now_ist.time(), key="vboard.timeseries.to_time")
        submitted = st.form_submit_button("Load Time Series")

    try:
        if selected_range == "__absolute__":
            time_range = absolute_time_range_from_inputs(from_date, from_time, to_date, to_time)
        else:
            time_range = {"kind": "preset", "preset_id": selected_range}
        resolution = {"mode": "auto"} if selected_resolution == "auto" else {"mode": "fixed", "window_id": selected_resolution}
        return {"row_id": row_id, "time_range": time_range, "resolution": resolution}, submitted, True
    except ValueError as exc:
        st.error(f"Invalid time-series range: {exc}")
        return {"row_id": row_ids[0], "time_range": {"kind": "preset", "preset_id": default_preset}, "resolution": {"mode": "auto"}}, submitted, False


def _load_contours(gateway, request: dict[str, Any]) -> None:
    fingerprint = request_fingerprint(request)
    st.session_state["vboard.contours.request"] = request
    st.session_state["vboard.contours.fingerprint"] = fingerprint
    try:
        with st.spinner("Loading V-Board contours..."):
            st.session_state["vboard.contours.result"] = gateway.get_contours(request)
            st.session_state["vboard.contours.stale"] = False
            st.session_state["vboard.contours.default_loaded"] = True
    except (FrontendApiError, ValueError, OSError, RuntimeError) as exc:
        st.session_state["vboard.contours.default_loaded"] = True
        if st.session_state.get("vboard.contours.result"):
            st.session_state["vboard.contours.stale"] = True
            st.warning("Contour refresh failed; showing the last successful result.")
        st.error(_api_error_message(exc, "Unable to load V-Board contours."))


def _load_timeseries(gateway, request: dict[str, Any]) -> None:
    fingerprint = request_fingerprint(request)
    st.session_state["vboard.timeseries.request"] = request
    st.session_state["vboard.timeseries.fingerprint"] = fingerprint
    try:
        with st.spinner("Loading heat-load time series..."):
            st.session_state["vboard.timeseries.result"] = gateway.get_heatload_timeseries(request)
            st.session_state["vboard.timeseries.stale"] = False
    except (FrontendApiError, ValueError, OSError, RuntimeError) as exc:
        if st.session_state.get("vboard.timeseries.result"):
            st.session_state["vboard.timeseries.stale"] = True
            st.warning("Time-series refresh failed; showing the last successful result.")
        st.error(_api_error_message(exc, "Unable to load heat-load time series."))


def _render_contour_result(catalog: dict[str, Any], result: dict[str, Any]) -> None:
    if st.session_state.get("vboard.contours.stale"):
        st.warning("The contour result is stale because the latest refresh failed.")
    _render_range(result)

    temperature = result.get("temperature", {})
    heatload = result.get("heatload", {})
    policy_id = result.get("processing_policy_id", catalog.get("processing_policy", {}).get("id"))
    catalog_version = result.get("catalog_version", catalog.get("catalog_version"))

    _render_warnings(temperature.get("warnings", []))
    if temperature.get("status") == "unavailable":
        st.error("Temperature contour data is unavailable.")
    elif _section_has_values(temperature, "levels"):
        means, maxima, minima = longitudinal_temperature_arrays(result)
        try:
            fig = _long_fig(_freeze(means), _freeze(maxima), _freeze(minima), catalog_version, policy_id)
            st.plotly_chart(fig, width="stretch", key="data_vis_longitudinal_temp")
        except ValueError as exc:
            st.info(f"Temperature contour is empty for this range: {exc}")
    else:
        st.info("Temperature contour is empty for this range.")

    st.title("Circumferential HeatLoad")
    _render_warnings(heatload.get("warnings", []))
    if heatload.get("status") == "unavailable":
        st.error("Heat-load contour data is unavailable.")
    elif _section_has_values(heatload, "rows"):
        values, titles = circumferential_heatload_rows(catalog, result)
        colorbar_title = heatload.get("display_label") or "Heat-load index"
        fig = _circ_fig(_freeze(values), tuple(titles), colorbar_title, heatload.get("unit"), catalog_version, policy_id)
        st.plotly_chart(fig, width="stretch", key="data_vis_circ_heatload")
    else:
        st.info("Heat-load contour is empty for this range.")

    st.title("Circumferential Temperature")
    if temperature.get("status") == "unavailable":
        st.error("Circumferential temperature data is unavailable.")
    elif _section_has_values(temperature, "levels"):
        for group in circumferential_temperature_groups(catalog, result):
            if len(group["field_values"]) != len(group["titles"]):
                st.error(f"{group['title']} has mismatched title and data counts.")
                continue
            fig = _circ_fig(
                _freeze(group["field_values"]),
                tuple(group["titles"]),
                f"Temperature ({temperature.get('unit') or ''})".strip(),
                temperature.get("unit"),
                catalog_version,
                policy_id,
            )
            st.plotly_chart(fig, width="stretch", key=f"data_vis_circ_temp_{group['id']}")


def _render_timeseries_result(result: dict[str, Any]) -> None:
    st.title("Heat Load Data - Timeseries")
    if st.session_state.get("vboard.timeseries.stale"):
        st.warning("The time-series result is stale because the latest load failed.")
    _render_range(result)
    _render_warnings(result.get("warnings", []))
    processing = result.get("processing", {})
    st.caption(
        " | ".join(
            [
                f"Policy: {processing.get('policy_id')}",
                f"Aggregation: {result.get('resolved_window_seconds')}s",
                f"Smoothing: {processing.get('smoothing_window_seconds')}s",
                f"Downsampled: {result.get('downsampled')}",
            ]
        )
    )

    traces = []
    row_id = result.get("row", {}).get("id", "")
    for series in result.get("series", []):
        points = series.get("points", [])
        traces.append(
            go.Scatter(
                x=[point.get("timestamp") for point in points],
                y=[point.get("value") for point in points],
                name=f"{row_id} {series.get('quadrant_id')}",
                mode="lines",
            )
        )
    if not any(trace.x for trace in traces):
        st.info("Heat-load time series is empty for this range.")
        return
    fig = build_heatload_timeseries_figure(result)
    st.plotly_chart(fig, width="stretch", key="data_vis_heatload_ts")


def _render_range(result: dict[str, Any]) -> None:
    resolved = result.get("resolved_range") or {}
    if resolved:
        st.caption(f"{utc_range_caption(resolved)} | {ist_range_caption(resolved)}")
    request_id = result.get("request_id")
    if request_id:
        st.caption(f"Backend request ID: {request_id}")


def _render_warnings(warnings: list[str]) -> None:
    for warning in warnings or []:
        st.warning(str(warning))


def _supported_presets(catalog: dict[str, Any], surface: str) -> list[dict[str, Any]]:
    return [
        preset
        for preset in catalog.get("presets", [])
        if surface in set(preset.get("supported_for", []))
    ]


def _should_load_default_contours(request: dict[str, Any]) -> bool:
    return (
        not st.session_state.get("vboard.contours.default_loaded")
        and not st.session_state.get("vboard.contours.result")
        and request["time_range"]["kind"] == "preset"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def _section_has_values(section: dict[str, Any], collection_key: str) -> bool:
    for item in section.get(collection_key, []):
        quadrants = item.get("quadrants", [])
        for quadrant in quadrants:
            for key in ("mean", "minimum", "maximum"):
                value = quadrant.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    return True
    return False


def _api_error_message(exc: Exception, fallback: str) -> str:
    if isinstance(exc, FrontendApiError):
        parts = [fallback, exc.message]
        if exc.error_code:
            parts.append(f"code={exc.error_code}")
        if exc.request_id:
            parts.append(f"request_id={exc.request_id}")
        return " ".join(parts)
    return f"{fallback} {exc}"


def main() -> None:
    st.markdown(
        """
<style>
div[data-testid="stVerticalBlock"] { gap: 0rem !important; }
div[data-testid="element-container"] { margin: 0 !important; padding: 0 !important; }
div[data-testid="stPlotlyChart"] { margin: 0 !important; padding: 0 !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 0rem !important; }
.block-container h1:first-of-type { margin-bottom: 1rem !important; }
</style>
""",
        unsafe_allow_html=True,
    )

    st.title("Furnace Temperature Data Visualization")

    try:
        gateway = get_vboard_gateway(access_token=st.session_state.get("auth_access_token"))
        catalog = gateway.get_catalog()
    except FrontendApiError as exc:
        st.error(_api_error_message(exc, "Unable to load V-Board catalog."))
        st.stop()

    contour_request, contour_submitted, contour_valid = _render_contour_form(catalog)
    if _should_load_default_contours(contour_request):
        _load_contours(gateway, contour_request)
    elif contour_submitted and contour_valid:
        _load_contours(gateway, contour_request)

    contour_result = st.session_state.get("vboard.contours.result")
    if contour_result:
        _render_contour_result(catalog, contour_result)
        current_fingerprint = request_fingerprint(contour_request) if contour_valid else None
        rendered_fingerprint = st.session_state.get("vboard.contours.fingerprint")
        if current_fingerprint and rendered_fingerprint and current_fingerprint != rendered_fingerprint:
            st.info("Contour controls changed; refresh to render the new range.")
    else:
        st.info("Choose a contour range and refresh.")

    timeseries_request, timeseries_submitted, timeseries_valid = _render_timeseries_form(catalog)
    if timeseries_submitted and timeseries_valid:
        _load_timeseries(gateway, timeseries_request)

    timeseries_result = st.session_state.get("vboard.timeseries.result")
    if timeseries_result:
        _render_timeseries_result(timeseries_result)
        current_ts_fingerprint = request_fingerprint(timeseries_request) if timeseries_valid else None
        rendered_ts_fingerprint = st.session_state.get("vboard.timeseries.fingerprint")
        if current_ts_fingerprint and rendered_ts_fingerprint and current_ts_fingerprint != rendered_ts_fingerprint:
            st.info("Time-series controls changed; load to render the new selection.")


main()
