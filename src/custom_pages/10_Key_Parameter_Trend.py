# src/custom_pages/10_Key_Parameter_Trend.py
"""Trend workspace for Key Parameters.

The whole workspace is one fragment: switching parameter or interval reruns only
the workspace (no page navigation), and both selections live in session state.
"""
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from ui.key_parameters import (
    IST,
    INTERVAL_OPTIONS,
    KEY_PARAMETERS,
    PANEL_CSS,
    QUERY_PARAM,
    SELECTED_KEY,
    WELCOME_PAGE,
    build_trend_figure,
    choose_window,
    custom_range_to_utc,
    fetch_current_values,
    fetch_trend_frame,
    format_value,
    get_parameter,
    render_parameter_list,
    resolve_preset_range,
    window_label,
)

if "auth_user" not in st.session_state:
    st.warning("Please login to access this page.")
    st.stop()

# ── Page-local CSS (only injected on this page) ──────────────────────────────
st.markdown(PANEL_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: #f4f6f9 !important; }
    .block-container { max-width: 1500px !important; padding: 3.2rem 1.2rem .8rem !important; }
    /* toolbar: one row that wraps; graph/rail row never stacks */
    .st-key-tk_top [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; align-items: center;
        gap: .35rem .8rem !important; }
    .st-key-tk_top [data-testid="stColumn"] { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
    .st-key-tk_top [data-testid="stColumn"]:nth-child(2) { flex: 1 1 150px !important; }
    .st-key-tk_body [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
    .st-key-tk_body [data-testid="stColumn"] { min-width: 0 !important; }
    .st-key-tk_body [data-testid="stColumn"]:has(.st-key-kp_panel_rail) { flex: 0 0 270px !important; width: 270px !important; }
    .st-key-tk_top [data-testid="stBaseButton-secondary"] {
        background: #0f1b2e !important; color: #e6f1fb !important;
        border: 1px solid #0f1b2e !important; border-radius: 8px !important; padding: .45rem .9rem !important;
        font-weight: 700 !important; letter-spacing: .06em; text-decoration: none !important; white-space: nowrap; }
    .st-key-tk_top [data-testid="stBaseButton-secondary"]:hover { background: #1f3a52 !important; }
    [data-testid="stBaseButton-segmented_control"] { background: #fff !important; color: #0f172a !important;
        border-color: #cbd5e1 !important; }
    [data-testid="stBaseButton-segmented_controlActive"] { background: #0f1b2e !important; color: #fff !important;
        border-color: #22d3ee !important; }
    .st-key-tk_pop_wrap [data-testid="stPopover"] > button, .st-key-tk_pop_wrap [data-testid="stPopover"] button {
        background: #0f1b2e !important; color: #e6f1fb !important; border: 1px solid #0f1b2e !important;
        font-weight: 600 !important; letter-spacing: .04em; }
    /* app theme is dark: make the Custom card readable on the light workspace */
    [data-testid="stForm"] { background: #fff; border: 1px solid #cbd5e1 !important; border-radius: 10px; }
    [data-testid="stForm"] label, [data-testid="stForm"] label p { color: #334155 !important; }
    [data-testid="stForm"] [data-baseweb="input"], [data-testid="stForm"] [data-baseweb="select"] > div,
    [data-testid="stForm"] [data-baseweb="base-input"] { background: #fff !important; border-color: #cbd5e1 !important; }
    [data-testid="stForm"] input, [data-testid="stForm"] [data-baseweb="select"] div { color: #0f172a !important; }
    [data-testid="stForm"] button { background: #0f1b2e !important; color: #fff !important; border-color: #0f1b2e !important; }
    [data-testid="stBaseButton-tertiary"] { color: #0e7490 !important; padding: 0 !important; min-height: 0 !important; }
    .tk-name { color: #0f172a; font-size: 1.5rem; font-weight: 800; letter-spacing: .02em;
        text-transform: uppercase; line-height: 1.15; }
    .tk-value { color: #0e7490; font-size: 1.35rem; font-weight: 700; font-variant-numeric: tabular-nums; }
    .tk-value small { color: #64748b; font-size: .85rem; font-weight: 500; margin-left: 4px; }
    .tk-meta { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 22px; margin: .1rem 0 .4rem; }
    .tk-sub { color: #475569; font-size: .82rem; }
    .tk-stats { display: flex; flex-wrap: wrap; gap: 2px 18px; color: #64748b; font-size: .72rem; letter-spacing: .06em; }
    .tk-stats b { color: #0f172a; font-size: .9rem; margin-left: 4px; font-variant-numeric: tabular-nums; }
    .tk-empty { border: 1px dashed #cbd5e1; border-radius: 10px; padding: 2rem 1rem; text-align: center;
        color: #64748b; background: #fff; }
    .tk-rotate { display: none; color: #92400e; font-size: .75rem; text-align: center; margin: 2px 0 6px; }
    .st-key-kp_panel_rail { max-height: calc(100vh - 8rem); }
    [data-testid="stColumn"]:has(.st-key-tk_pop_wrap) { display: none !important; }
    /* dedicated workspace: the Streamlit sidebar stays open after navigating from the Welcome drawer,
       so hide it here; parameters live in the rail/popover and EXIT returns to the dashboard */
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] { display: none !important; }
    [data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] > div,
    [data-testid="stPlotlyChart"] .js-plotly-plot, [data-testid="stPlotlyChart"] .plot-container,
    [data-testid="stPlotlyChart"] .svg-container { height: clamp(320px, calc(100vh - 290px), 760px) !important; }
    /* narrow screens: rail becomes a "Parameters" popover; graph gets the full width */
    @media (max-width: 699.98px) {
        [data-testid="stColumn"]:has(.st-key-kp_panel_rail) { display: none !important; }
        [data-testid="stColumn"]:has(.st-key-tk_pop_wrap) { display: block !important; }
        [data-testid="stPopoverBody"] { width: min(92vw, 340px) !important; max-height: 70vh; padding: 0 !important; }
        .tk-name { font-size: 1.15rem; } .tk-value { font-size: 1.05rem; }
        .block-container { padding: 3.2rem .6rem .6rem !important; }
    }
    @media (orientation: portrait) and (max-width: 699.98px) { .tk-rotate { display: block; } }
    /* short landscape (phones): graph takes the viewport */
    @media (orientation: landscape) and (max-height: 500px) {
        [data-testid="stHeader"], [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        .block-container { padding: .35rem .6rem .2rem !important; }
        .tk-name { font-size: 1rem; } .tk-value { font-size: .95rem; }
        .st-key-tk_top [data-testid="stBaseButton-secondary"] { padding: .3rem .7rem !important; }
        .tk-meta { margin: 0 0 .2rem; }
        [data-testid="stSegmentedControl"] button { padding: .15rem .5rem !important; min-height: 2rem !important; font-size: .78rem !important; }
        .st-key-tk_pop_wrap button { padding: .3rem .7rem !important; min-height: 2rem !important; }
        .st-key-kp_panel_rail { max-height: calc(100vh - 1.2rem); }
        [data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot, [data-testid="stPlotlyChart"] .plot-container,
        [data-testid="stPlotlyChart"] .svg-container { height: calc(100vh - 140px) !important; }
    }
    @media (orientation: landscape) and (max-height: 500px) and (max-width: 699.98px) {
        [data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot, [data-testid="stPlotlyChart"] .plot-container,
        [data-testid="stPlotlyChart"] .svg-container { height: calc(100vh - 185px) !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Canonical selection: ?kp= (deep link / arrival from Welcome) -> session ──
_qp = st.query_params.get(QUERY_PARAM)
if get_parameter(_qp) and _qp != st.session_state.get(SELECTED_KEY):
    st.session_state[SELECTED_KEY] = _qp
if not get_parameter(st.session_state.get(SELECTED_KEY)):
    st.session_state[SELECTED_KEY] = KEY_PARAMETERS[0].key
    st.query_params[QUERY_PARAM] = KEY_PARAMETERS[0].key


if st.session_state.pop("tk_exit_req", False):
    st.switch_page(WELCOME_PAGE)


def _range_text(interval: str, start_utc: datetime, end_utc: datetime) -> str:
    if interval == "Custom":
        fmt = "%d %b %H:%M"
        return f"Custom: {start_utc.astimezone(IST):{fmt}} → {end_utc.astimezone(IST):{fmt}} IST"
    return f"Last {interval[:-1]} Hour{'s' if interval != '1h' else ''}"


def _interval_control() -> str:
    interval = st.segmented_control(
        "Interval", INTERVAL_OPTIONS,
        default=st.session_state.get("selected_trend_range", "8h"),
        required=True, key="tk_interval", label_visibility="collapsed",
    )
    st.session_state["selected_trend_range"] = interval
    return interval


def _resolve_range(interval: str) -> tuple[datetime, datetime] | None:
    """Preset range, or the compact Custom card (applied range persists across parameter switches)."""
    if interval != "Custom":
        return resolve_preset_range(interval)

    applied = st.session_state.get("kp_custom_range")
    if applied and not st.session_state.get("tk_custom_edit"):
        st.button("Edit custom range", key="tk_custom_edit_btn", type="tertiary",
                  on_click=st.session_state.__setitem__, args=("tk_custom_edit", True))
        return applied

    now = datetime.now(IST)
    with st.form("tk_custom_form", border=True):
        c1, c2 = st.columns(2)
        f_date = c1.date_input("From date", (now - timedelta(hours=12)).date())
        f_time = c2.time_input("From time", (now - timedelta(hours=12)).time().replace(second=0, microsecond=0))
        c3, c4 = st.columns(2)
        t_date = c3.date_input("To date", now.date())
        t_time = c4.time_input("To time", now.time().replace(second=0, microsecond=0))
        submitted = st.form_submit_button("Apply", width="stretch")
    if submitted:
        try:
            st.session_state["kp_custom_range"] = custom_range_to_utc(f_date, f_time, t_date, t_time)
            st.session_state["tk_custom_edit"] = False
            st.rerun(scope="fragment")
        except ValueError as exc:
            st.error(str(exc))
    return applied if applied else None


@st.fragment
def workspace() -> None:
    param = get_parameter(st.session_state[SELECTED_KEY])
    snapshot = fetch_current_values()  # cached; shared with the Welcome panel
    current = snapshot["values"].get(param.field) if param.field else None

    with st.container(key="tk_top"):
        c_exit, c_title, c_interval, c_pop = st.columns([1, 4, 4, 1], vertical_alignment="center")
        if c_exit.button("← EXIT", key="tk_exit_btn"):
            st.session_state["tk_exit_req"] = True
            st.rerun()  # full-app rerun: switch_page must not run inside the fragment
        unit = f"<small>{param.unit}</small>" if param.unit and current is not None else ""
        c_title.markdown(
            f'<div class="tk-name">{param.label}</div>'
            f'<div class="tk-value">{format_value(param, current)}{unit}</div>',
            unsafe_allow_html=True,
        )
        with c_interval:
            interval = _interval_control()
        with c_pop, st.container(key="tk_pop_wrap"):
            with st.popover("Parameters ☰"):
                with st.container(key="kp_panel_pop"):
                    render_parameter_list("p", snapshot, selectable=True, selected_key=param.key)

    rng = _resolve_range(interval)

    with st.container(key="tk_body"):
        col_graph, col_rail = st.columns([5, 1])

        with col_rail:
            with st.container(key="kp_panel_rail"):
                render_parameter_list("r", snapshot, selectable=True, selected_key=param.key)

        with col_graph:
            if not (param.measurement and param.field):
                st.markdown('<div class="tk-empty">No data source configured for this parameter.</div>',
                            unsafe_allow_html=True)
                return
            if rng is None:
                st.info("Pick a start and end time (IST), then press Apply.")
                return

            start_utc, end_utc = rng
            window = choose_window(end_utc - start_utc)
            sub = f'<span class="tk-sub">{_range_text(interval, start_utc, end_utc)} · {window_label(window)}</span>'
            meta_slot = st.empty()
            meta_slot.markdown(f'<div class="tk-meta">{sub}</div>', unsafe_allow_html=True)
            chart_box = st.container()
            with chart_box:
                try:
                    with st.spinner(f"Loading {param.label} trend…"):
                        frame = fetch_trend_frame(param.measurement, start_utc, end_utc, window)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not load trend data: {exc}")
                    return
                series = (
                    pd.to_numeric(frame[param.field], errors="coerce").dropna()
                    if frame is not None and not frame.empty and param.field in frame.columns
                    else pd.Series(dtype=float)
                )
                if series.empty:
                    st.markdown('<div class="tk-empty">No data for the selected range.</div>',
                                unsafe_allow_html=True)
                    return
                series.index = series.index.tz_convert(IST).tz_localize(None)  # naive IST for Plotly
                meta_slot.markdown(
                    f'<div class="tk-meta">{sub}<div class="tk-stats">'
                    f'<span>LATEST<b>{format_value(param, series.iloc[-1])}</b></span>'
                    f'<span>AVG<b>{format_value(param, series.mean())}</b></span>'
                    f'<span>MIN<b>{format_value(param, series.min())}</b></span>'
                    f'<span>MAX<b>{format_value(param, series.max())}</b></span></div></div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    build_trend_figure(series, param),
                    width="stretch",
                    config={"displaylogo": False, "responsive": True},
                    key="tk_chart",
                )
                st.markdown('<div class="tk-rotate">Rotate device for expanded trend view</div>',
                            unsafe_allow_html=True)


workspace()
