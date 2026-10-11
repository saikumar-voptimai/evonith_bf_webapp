"""Small Furnace Status page entry point."""

from __future__ import annotations

import streamlit as st

from data import furnace_status as fs
from ui.furnace_status.components import inject_css, set_view_query
from ui.furnace_status.dashboard import render_status_view
from ui.furnace_status.responsive import render_rotate_hint
from ui.furnace_status.trends import render_trend_view


def render_furnace_status() -> None:
    inject_css()
    state = fs.parse_view_state(st.query_params)
    if state.needs_reset:
        set_view_query(fs.VIEW_STATUS)
    with st.container(key="fs-page"):
        render_rotate_hint()
        if state.view == fs.VIEW_TREND and state.spec is not None:
            render_trend_view(state.spec)
        else:
            render_status_view()
