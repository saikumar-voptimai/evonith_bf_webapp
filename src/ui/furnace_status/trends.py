"""Trend controls, Plotly construction and trend-view rendering."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone

import plotly.graph_objects as go
import streamlit as st

from data import furnace_status as fs
from ui.furnace_status.components import go_status, h, render_refresh_control
from ui.furnace_status.layout import (
    ACCENT,
    BORDER,
    GRID,
    SURFACE,
    TEXT,
    TEXT_MUTED,
    WARM_TEXT,
)
from ui.furnace_status.responsive import render_fullscreen_control
from ui.streamlit_fragments import fragment
from utils.logger import get_logger

log = get_logger(__name__)
_TREND_LOADED_PREFIX = "fs-trend-loaded:"
_TREND_GOOD_PREFIX = "fs-trend-good:"


@dataclass(frozen=True)
class _TrendDisplay:
    view_data: fs.TrendViewData
    start_utc: datetime
    end_utc: datetime
    updated_at: datetime


PLOT_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "zoomIn2d",
        "zoomOut2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
}


def naive_ist(moment: datetime) -> datetime:
    return moment.astimezone(fs.IST).replace(tzinfo=None)


def build_trend_figure(
    trend: fs.TrendData,
    x_range: tuple[datetime, datetime],
    revision: str,
) -> go.Figure:
    spec = trend.spec
    unit = f" {spec.unit}" if spec.unit else ""
    y_title = f"{spec.title} ({spec.unit})" if spec.unit else spec.title
    lines = [
        (trend.series, "Actual" if trend.setpoint is not None else spec.title, None)
    ]
    if trend.setpoint is not None:
        lines.append((trend.setpoint, "Setpoint", "dash"))
    figure = go.Figure(
        [
            go.Scattergl(
                x=series.index.tz_localize(None),
                y=series.to_numpy(),
                mode="lines",
                name=name,
                connectgaps=False,
                line={
                    "color": WARM_TEXT if dash is None else ACCENT,
                    "width": 2,
                    "dash": dash,
                },
                hovertemplate=f"%{{y:,.{spec.decimals}f}}{h(unit)}<extra>{h(name)}</extra>",
            )
            for series, name, dash in lines
        ]
    )
    axis_style = {
        "gridcolor": GRID,
        "linecolor": BORDER,
        "tickcolor": BORDER,
        "tickfont": {"color": TEXT_MUTED},
        "title": {"font": {"color": TEXT_MUTED}},
    }
    figure.update_layout(
        template="plotly_white",
        height=None,
        autosize=True,
        margin={"l": 64, "r": 16, "t": 12, "b": 48},
        showlegend=trend.setpoint is not None,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "x": 0,
            "font": {"color": TEXT},
        },
        hovermode="x unified",
        hoverlabel={
            "bgcolor": SURFACE,
            "bordercolor": BORDER,
            "font": {"color": TEXT},
        },
        uirevision=revision,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": TEXT_MUTED},
        modebar={
            "bgcolor": "rgba(0,0,0,0)",
            "color": TEXT_MUTED,
            "activecolor": ACCENT,
        },
        xaxis={
            **axis_style,
            "title": {"text": "Time (IST)", "font": {"color": TEXT_MUTED}},
            "type": "date",
            "hoverformat": "%d %b %Y, %H:%M:%S",
            "zeroline": False,
        },
        yaxis={
            **axis_style,
            "title": {"text": y_title, "font": {"color": TEXT_MUTED}},
            "tickformat": f",.{spec.decimals}f" if spec.decimals <= 2 else None,
            "zeroline": False,
        },
    )
    return figure


def _stat_tile(label: str, value: str, unit: str) -> str:
    suffix = f' <span class="fs-trend-current__unit">{h(unit)}</span>' if unit else ""
    return (
        '<div class="fs-stat">'
        f'<div class="fs-stat__label">{h(label)}</div>'
        f'<div class="fs-stat__value">{h(value)}{suffix}</div></div>'
    )


def _source_line(spec: fs.ParameterSpec, last_data: str | None) -> str:
    if spec.components:
        parts = [f"Source: {spec.measurement}"]
    elif spec.has_source:
        parts = [f"Source: {spec.measurement} · {', '.join(spec.source_fields)}"]
    else:
        parts = ["Source: not configured"]
    if spec.source_note:
        parts.append(spec.source_note)
    if last_data is not None:
        parts.append(f"Last data: {last_data}")
    return " · ".join(parts)


def _unavailable_html(why: str) -> str:
    return (
        '<div class="fs-unavailable">'
        f'<div class="fs-unavailable__title">{fs.NOT_AVAILABLE}</div>'
        f'<div class="fs-unavailable__why">{h(why)}</div></div>'
    )


def _head_html(
    spec: fs.ParameterSpec,
    *,
    last_data: str | None = None,
    current_text: str | None = None,
) -> str:
    current = ""
    if current_text is not None:
        current = (
            '<div class="fs-trend-current">'
            f'<span class="fs-trend-current__value">{h(current_text)}</span>'
            f'<span class="fs-trend-current__unit">{h(spec.unit)}</span></div>'
        )
    return (
        '<div class="fs-trend-head"><div>'
        f'<div class="fs-trend-title" role="heading" aria-level="1">{h(spec.title)}</div>'
        f'<div class="fs-trend-source">{h(_source_line(spec, last_data))}</div>'
        f"</div>{current}</div>"
    )


def _render_interval_selector() -> tuple[str, tuple[datetime, datetime] | None]:
    with st.container(key="fs-interval"):
        choice = st.segmented_control(
            "Interval",
            fs.INTERVAL_OPTIONS,
            default=fs.DEFAULT_INTERVAL,
            key="fs-interval-choice",
            label_visibility="collapsed",
        )
    interval = choice or fs.DEFAULT_INTERVAL
    if interval != fs.CUSTOM_INTERVAL:
        return interval, None
    now = datetime.now(timezone.utc)
    default_start, default_end = fs.default_custom_range(now)
    with st.container(key="fs-controls"):
        c1, c2, c3, c4 = st.columns(4)
        start_date = c1.date_input(
            "Start date",
            default_start.date(),
            key="fs-c-start-date",
            format="DD/MM/YYYY",
        )
        start_time = c2.time_input(
            "Start time (IST)", default_start.time(), key="fs-c-start-time", step=60
        )
        end_date = c3.date_input(
            "End date",
            default_end.date(),
            key="fs-c-end-date",
            format="DD/MM/YYYY",
        )
        end_time = c4.time_input(
            "End time (IST)", default_end.time(), key="fs-c-end-time", step=60
        )
    try:
        return interval, fs.resolve_custom_range(
            datetime.combine(start_date, start_time),
            datetime.combine(end_date, end_time),
            now,
        )
    except fs.RangeError as error:
        st.error(str(error))
        return interval, None


def _trend_state_token(
    spec: fs.ParameterSpec,
    interval: str,
    custom_window: tuple[datetime, datetime] | None,
) -> str:
    token = f"{spec.key}|{interval}"
    if custom_window is not None:
        start, end = custom_window
        token += f"|{start.isoformat()}|{end.isoformat()}"
    return token


def _tick_time(moment: datetime) -> str:
    return moment.astimezone(fs.IST).strftime("%H:%M:%S IST")


def _render_trend_body(
    display: _TrendDisplay,
    chart,
    *,
    interval: str,
    refresh_label: str,
    failure_note: str | None,
) -> None:
    view_data = display.view_data
    start_utc, end_utc = display.start_utc, display.end_utc
    current, trend, stats = view_data.current, view_data.trend, view_data.stats
    if trend.status == "ok" and stats is not None:
        revision = (
            f"{trend.spec.key}|{interval}|{start_utc:%Y%m%d%H%M}|{end_utc:%Y%m%d%H%M}"
            if interval == fs.CUSTOM_INTERVAL
            else f"{trend.spec.key}|{interval}"
        )
        with chart.container(key="fs-chart"):
            st.plotly_chart(
                build_trend_figure(
                    trend, (naive_ist(start_utc), naive_ist(end_utc)), revision
                ),
                width="stretch",
                theme=None,
                key=f"fs-trend-{trend.spec.key}",
                config=PLOT_CONFIG,
            )
    else:
        with chart.container(key="fs-chart"):
            st.html(_unavailable_html(trend.message or "No data was returned."))
    minimum = (
        fs.format_value(stats.minimum, trend.spec.decimals)
        if stats is not None
        else fs.NOT_AVAILABLE
    )
    maximum = (
        fs.format_value(stats.maximum, trend.spec.decimals)
        if stats is not None
        else fs.NOT_AVAILABLE
    )
    average = (
        fs.format_value(stats.mean, trend.spec.decimals)
        if stats is not None
        else fs.NOT_AVAILABLE
    )
    current_unit = trend.spec.unit if fs.has_display_value(current) else ""
    range_unit = trend.spec.unit if stats is not None else ""
    st.html(
        '<div class="fs-stats">'
        + _stat_tile("Current", fs.format_reading(current), current_unit)
        + _stat_tile("Minimum", minimum, range_unit)
        + _stat_tile("Maximum", maximum, range_unit)
        + _stat_tile("Average", average, range_unit)
        + "</div>"
    )
    if trend.status == "ok" and stats is not None:
        sample_count = int(trend.series.notna().sum())
        suffix = (
            " · Auto-refresh off for custom range"
            if interval == fs.CUSTOM_INTERVAL
            else f" · auto-refresh {refresh_label} · updated {_tick_time(display.updated_at)}"
        )
        st.caption(
            f"Raw data · {sample_count:,} samples · "
            f"{fs.format_ist(start_utc)} → {fs.format_ist(end_utc)}{suffix}"
        )
    if failure_note:
        st.caption(failure_note)


def _live_trend(
    title,
    chart,
    spec: fs.ParameterSpec,
    interval: str,
    custom_window: tuple[datetime, datetime] | None,
    refresh_label: str,
) -> None:
    state = fs.parse_view_state(st.query_params)
    if (
        state.view != fs.VIEW_TREND
        or state.spec is None
        or state.spec.key != spec.key
    ):
        st.rerun()
        return

    now = datetime.now(timezone.utc)
    start_utc, end_utc = (
        custom_window
        if custom_window is not None
        else fs.resolve_fixed_range(interval, now)
    )
    token = _trend_state_token(spec, interval, custom_window)
    loaded_key = _TREND_LOADED_PREFIX + token
    good_key = _TREND_GOOD_PREFIX + token
    first_load = not bool(st.session_state.get(loaded_key))
    st.session_state[loaded_key] = True
    failure_note = None
    view_data = None
    try:
        context = st.spinner("Loading trend…") if first_load else nullcontext()
        with context:
            view_data = fs.load_trend_view(
                spec,
                start_utc,
                end_utc,
                use_live_current=interval != fs.CUSTOM_INTERVAL,
            )
    except Exception:  # noqa: BLE001 - details stay in server logs
        log.exception("Furnace Status: trend refresh failed for %s", spec.key)

    failed = view_data is None or view_data.trend.status == "error"
    previous = st.session_state.get(good_key)
    if failed and isinstance(previous, _TrendDisplay):
        display = previous
        failure_note = f"Refresh failed at {_tick_time(now)} — showing last data"
    elif view_data is not None:
        display = _TrendDisplay(view_data, start_utc, end_utc, now)
        if not failed and view_data.trend.status == "ok" and view_data.stats is not None:
            st.session_state[good_key] = display
    else:
        title.html(_head_html(spec, current_text=fs.NOT_AVAILABLE), width="stretch")
        with chart.container(key="fs-chart"):
            st.html(_unavailable_html("The data source could not be reached. Try again."))
        return

    current = display.view_data.current
    timestamp = (
        current.timestamp
        if current.timestamp is not None
        else current.setpoint_timestamp
    )
    title.html(
        _head_html(
            spec,
            last_data=fs.format_ist(timestamp) if timestamp is not None else None,
            current_text=fs.format_reading(current),
        ),
        width="stretch",
    )
    _render_trend_body(
        display,
        chart,
        interval=interval,
        refresh_label=refresh_label,
        failure_note=failure_note,
    )


def render_trend_view(spec: fs.ParameterSpec) -> None:
    interval_state = st.session_state.get("fs-interval-choice", fs.DEFAULT_INTERVAL)
    if interval_state not in fs.INTERVAL_OPTIONS:
        interval_state = fs.DEFAULT_INTERVAL
    with st.container(key="fs-toolbar"):
        back, head, refresh, fullscreen = st.columns(
            [3, 6, 2.4, 1.2], vertical_alignment="center"
        )
        with back:
            st.button(
                "← Back to Furnace Status",
                key="fs-back",
                on_click=go_status,
                width="stretch",
            )
        with head:
            title = st.empty()
        with refresh:
            refresh_label, seconds = render_refresh_control()
        with fullscreen:
            render_fullscreen_control(f"{spec.key}|{interval_state}")
    if not spec.has_source:
        title.html(_head_html(spec), width="stretch")
        st.html(_unavailable_html(spec.unavailable_reason))
        return
    interval, custom_window = _render_interval_selector()
    if interval == fs.CUSTOM_INTERVAL and custom_window is None:
        title.html(_head_html(spec), width="stretch")
        return
    chart = st.empty()
    run_every = None if interval == fs.CUSTOM_INTERVAL else seconds
    fragment(_live_trend, run_every=run_every)(
        title,
        chart,
        spec,
        interval,
        custom_window,
        refresh_label,
    )
