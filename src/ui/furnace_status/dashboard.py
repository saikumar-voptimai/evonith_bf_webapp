"""Status-dashboard orchestration and header rendering."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from data import furnace_status as fs
from ui.furnace_status.components import (
    h,
    refresh,
    render_heat_load_panel,
    render_panel,
    render_strip,
)
from ui.furnace_status.layout import (
    INSTRUCTIONS,
    PANEL_COLUMNS,
    ZONE_READINGS,
    ZONE_WIDGET_KEY,
)
from ui.furnace_status.responsive import render_fullscreen_control
from ui.furnace_status.schematic import render_furnace


def _selected_zone() -> str | None:
    value = st.session_state.get(ZONE_WIDGET_KEY)
    return value if isinstance(value, str) and value in ZONE_READINGS else None


def _status_html(snapshot: fs.StatusSnapshot) -> str:
    status = snapshot.status
    return (
        f'<div class="fs-status fs-status--{status.level}" role="status">'
        '<span class="fs-status__dot" aria-hidden="true"></span>'
        '<div class="fs-status__text"><div class="fs-status__line">'
        f'<span class="fs-status__label">{h(status.label)}</span>'
        '<span class="fs-status__kind" title="Live, Partial and Offline describe '
        'telemetry availability, not furnace safety or operating health.">'
        f'<div class="fs-status__time">Last updated: {h(fs.format_ist(snapshot.last_updated))}</div>'
        "</div></div>"
    )


def _render_header(snapshot: fs.StatusSnapshot) -> None:
    with st.container(
        key="fs-header", horizontal=True, vertical_alignment="center", gap="medium"
    ):
        st.html(
            '<div class="fs-head"><div class="fs-title" role="heading" aria-level="1">'
            "BF2 blast furnace status</div>"
            f'<div class="fs-subtitle">{h(INSTRUCTIONS)}</div></div>'
        )
        st.html(_status_html(snapshot), width="content")
        render_fullscreen_control()
        st.button(
            "Refresh",
            key="fs-refresh",
            icon=":material/refresh:",
            help="Reload live values",
            on_click=refresh,
        )


def render_status_view() -> None:
    snapshot = fs.load_status_snapshot(datetime.now(timezone.utc))
    readings = {reading.spec.key: reading for reading in snapshot.readings}
    zone = _selected_zone()
    _render_header(snapshot)
    if snapshot.failed_measurements:
        names = ", ".join(snapshot.failed_measurements)
        st.html(
            f'<div class="fs-warning" role="alert">Data source unavailable: {h(names)}. '
            "Affected parameters show Not available.</div>"
        )
    left_sections, right_sections = PANEL_COLUMNS
    with st.container(key="fs-dashboard"):
        with st.container(key="fs-col-left"):
            for section in left_sections:
                render_panel(section, readings, zone)
        with st.container(key="fs-col-centre"):
            render_strip(fs.SECTION_UPTAKE, readings, zone)
            render_furnace(readings, zone)
            render_strip(fs.SECTION_HEARTH, readings, zone)
        with st.container(key="fs-col-right"):
            for section in right_sections:
                if section == fs.SECTION_HEAT_LOAD:
                    render_heat_load_panel(readings, zone)
                else:
                    render_panel(section, readings, zone)
