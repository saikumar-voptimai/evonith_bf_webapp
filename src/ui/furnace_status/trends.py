"""Trend controls, Plotly construction and trend-view rendering."""

from __future__ import annotations

from datetime import datetime, timezone

import plotly.graph_objects as go
import streamlit as st

from data import furnace_status as fs
from ui.furnace_status.components import go_status, h
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
            go.Scatter(
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
            "range": [x_range[0], x_range[1]],
            "hoverformat": "%d %b %Y, %H:%M",
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
    now = datetime.now(timezone.utc)
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
        return interval, fs.resolve_fixed_range(interval, now)
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


def render_trend_view(spec: fs.ParameterSpec) -> None:
    with st.container(key="fs-toolbar"):
        back, head, fullscreen = st.columns([3, 6, 2], vertical_alignment="center")
        with back:
            st.button(
                "← Back to Furnace Status",
                key="fs-back",
                on_click=go_status,
                width="stretch",
            )
        with fullscreen:
            render_fullscreen_control()
    if not spec.has_source:
        with head:
            st.html(_head_html(spec))
        st.html(_unavailable_html(spec.unavailable_reason))
        return
    interval, window = _render_interval_selector()
    if window is None:
        with head:
            st.html(_head_html(spec))
        return
    start_utc, end_utc = window
    bucket = fs.choose_window(end_utc - start_utc)
    with st.spinner("Loading trend…"):
        view_data = fs.load_trend_view(
            spec,
            start_utc,
            end_utc,
            bucket,
            use_live_current=interval != fs.CUSTOM_INTERVAL,
        )
    current, trend, stats = view_data.current, view_data.trend, view_data.stats
    timestamp = current.timestamp
    if timestamp is None:
        timestamp = current.setpoint_timestamp
    with head:
        st.html(
            _head_html(
                spec,
                last_data=fs.format_ist(timestamp) if timestamp is not None else None,
                current_text=fs.format_reading(current),
            )
        )
    if trend.status == "ok" and stats is not None:
        revision = (
            f"{spec.key}|{interval}|{start_utc:%Y%m%d%H%M}|{end_utc:%Y%m%d%H%M}"
            if interval == fs.CUSTOM_INTERVAL
            else f"{spec.key}|{interval}"
        )
        with st.container(key="fs-chart"):
            st.plotly_chart(
                build_trend_figure(
                    trend, (naive_ist(start_utc), naive_ist(end_utc)), revision
                ),
                width="stretch",
                theme=None,
                key=f"fs-trend-{spec.key}",
                config=PLOT_CONFIG,
            )
    else:
        st.html(_unavailable_html(trend.message or "No data was returned."))
    minimum = (
        fs.format_value(stats.minimum, spec.decimals)
        if stats is not None
        else fs.NOT_AVAILABLE
    )
    maximum = (
        fs.format_value(stats.maximum, spec.decimals)
        if stats is not None
        else fs.NOT_AVAILABLE
    )
    average = (
        fs.format_value(stats.mean, spec.decimals)
        if stats is not None
        else fs.NOT_AVAILABLE
    )
    current_unit = spec.unit if fs.has_display_value(current) else ""
    range_unit = spec.unit if stats is not None else ""
    st.html(
        '<div class="fs-stats">'
        + _stat_tile("Current", fs.format_reading(current), current_unit)
        + _stat_tile("Minimum", minimum, range_unit)
        + _stat_tile("Maximum", maximum, range_unit)
        + _stat_tile("Average", average, range_unit)
        + "</div>"
    )
    if trend.status == "ok" and stats is not None:
        st.caption(
            f"{bucket} average · {fs.format_ist(start_utc)} → {fs.format_ist(end_utc)}"
        )
