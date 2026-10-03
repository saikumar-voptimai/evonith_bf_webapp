"""Key Parameters: config-driven definitions, helpers and renderers.

Used by the Welcome page (live panel) and the Key Parameter Trend page.
All data comes from :func:`furnace_data.influx.online.fetch_online_df`.
Parameters without a verified Influx source stay in the list with
``field=None`` and are shown honestly as unavailable.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

IST = ZoneInfo("Asia/Kolkata")
TREND_PAGE = "custom_pages/10_Key_Parameter_Trend.py"
WELCOME_PAGE = "custom_pages/1_Welcome.py"
SELECTED_KEY = "selected_key_parameter"
QUERY_PARAM = "kp"
STALE_AFTER = timedelta(minutes=10)


@dataclass(frozen=True)
class KeyParameter:
    """One monitored parameter. ``field=None`` means no verified source."""

    key: str
    label: str
    measurement: str | None = None
    field: str | None = None
    unit: str = ""
    decimals: int = 1
    setpoint_placeholder: bool = False  # show "— / <actual>" until an SP field exists


_PP = "process_params"

KEY_PARAMETERS: tuple[KeyParameter, ...] = (
    KeyParameter("prod_theor", "Production (Theor.)", _PP, "theoretical_production_per_day", "t/day", 0),
    KeyParameter("prod_proj", "Proj. Production"),
    KeyParameter("prod_rate", "Production Rate", _PP, "production_per_hour", "t/hr", 1),
    KeyParameter("cold_blast_vol", "Cold Blast Volume"),
    KeyParameter("hot_blast_vol", "Hot Blast Volume", _PP, "hot_blast_vol_nm3h", "Nm³/hr", 0),
    KeyParameter("blast_press", "Blast Pressure", _PP, "hot_blast_press", "bar", 2),
    KeyParameter("top_press", "Top Pressure", _PP, "top_press_avg", "bar", 2),
    KeyParameter("eta_co", "CO Utilization (ETA CO)", _PP, "body_etaco", "%", 1),
    KeyParameter("top_gas_h2", "Top Gas H₂", _PP, "h2_pct", "%", 1),
    KeyParameter("hbt", "HBT", _PP, "hot_blast_temp", "°C", 0),
    KeyParameter("raft", "RAFT", _PP, "body_raft", "°C", 0),
    KeyParameter("steam", "Steam Injection", _PP, "steam_injection", "kg/hr", 0),
    KeyParameter("steam_bypass", "Steam Bypass Flow"),
    KeyParameter("humidity", "Humidity"),
    KeyParameter("flare_flow", "Flare Gas Flow"),
    KeyParameter("o2_inj", "O₂ Injection", _PP, "oxygen_enrichment_pct", "%", 2),
    KeyParameter("o2_flow", "Oxygen Flow", _PP, "oxygen_flow", "Nm³/hr", 0),
    KeyParameter("fuel_rate", "Fuel Rate", _PP, "fuel_rate", "kg/tHM", 0),
    KeyParameter(
        "pci_rate", "PCI Rate (SP/ACT.)", _PP, "coal_rate_actual_value", "kg/tHM", 1,
        setpoint_placeholder=True,
    ),
    KeyParameter("slag_rate", "Slag Rate"),
    KeyParameter("perm", "Permeability", _PP, "body_perm", "", 2),
    KeyParameter("tuyere_vel", "Tuyere Velocity", _PP, "tuyere_velocity", "", 1),
    KeyParameter("heat_flux", "Heat Flow Flux"),
    KeyParameter("furnace_level", "Furnace Level", "miscellaneous", "stock_rod_radar_level", "", 2),
)

_BY_KEY = {p.key: p for p in KEY_PARAMETERS}


def get_parameter(key: str | None) -> KeyParameter | None:
    return _BY_KEY.get(key) if key else None


def window_label(window: str) -> str:
    """'5 minutes' -> '5 minute average'."""
    n, _, unit = window.partition(" ")
    return f"{n} {unit.rstrip('s')} average"


def format_value(param: KeyParameter, value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{param.decimals}f}"


# ── Pure time-range helpers ───────────────────────────────────────────────────

PRESET_HOURS: dict[str, int] = {"1h": 1, "4h": 4, "8h": 8, "16h": 16, "24h": 24}
INTERVAL_OPTIONS: tuple[str, ...] = (*PRESET_HOURS, "Custom")

# (max span, Influx window) — keeps every range at roughly 50-300 points.
_WINDOW_STEPS: tuple[tuple[timedelta, str], ...] = (
    (timedelta(hours=1), "1 minute"),
    (timedelta(hours=8), "5 minutes"),
    (timedelta(hours=16), "10 minutes"),
    (timedelta(hours=24), "15 minutes"),
    (timedelta(hours=48), "30 minutes"),
    (timedelta(days=12), "1 hour"),
    (timedelta(days=75), "6 hours"),
    (timedelta(days=150), "12 hours"),
)


def choose_window(span: timedelta) -> str:
    """Adaptive aggregation window (a key of ``WINDOWING``) for a time span."""
    for max_span, window in _WINDOW_STEPS:
        if span <= max_span:
            return window
    return "1 day"


def resolve_preset_range(
    label: str, now: datetime | None = None
) -> tuple[datetime, datetime]:
    """UTC ``(start, end)`` for '1h'..'24h'; ``end`` is floored to the minute."""
    hours = PRESET_HOURS[label]
    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    end = end.replace(second=0, microsecond=0)
    return end - timedelta(hours=hours), end


def custom_range_to_utc(
    start_date: date, start_time: time, end_date: date, end_time: time
) -> tuple[datetime, datetime]:
    """Treat operator-entered times as IST and return aware UTC datetimes."""
    start = datetime.combine(start_date, start_time, tzinfo=IST)
    end = datetime.combine(end_date, end_time, tzinfo=IST)
    if start >= end:
        raise ValueError("Start must be earlier than end.")
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


# ── Cached fetchers (single code path: fetch_online_df) ───────────────────────

def _measurements_in_use() -> list[str]:
    return sorted({p.measurement for p in KEY_PARAMETERS if p.measurement})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_current_values() -> dict:
    """Latest non-null value per field, from one 1-minute-window fetch.

    Returns ``{"values": {field: float}, "latest": datetime | None,
    "fetched_at": datetime, "error": str}``; never raises.
    """
    fetched_at = datetime.now(IST)
    out: dict = {"values": {}, "latest": None, "fetched_at": fetched_at, "error": ""}
    try:
        from furnace_data.influx.online import fetch_online_df

        end = datetime.now(timezone.utc)
        df = fetch_online_df(
            selected_measurements=_measurements_in_use(),
            time_range="last 15 minutes",  # required arg; overrides take precedence
            request_type="windowed-average",
            window_by="1 minute",
            start_time_override=end - timedelta(minutes=15),
            end_time_override=end,
            column_naming="field",
        )
        if df is None or df.empty:
            out["error"] = "No data returned."
            return out
        latest = None
        for p in KEY_PARAMETERS:
            if p.field and p.field in df.columns:
                series = pd.to_numeric(df[p.field], errors="coerce").dropna()
                if not series.empty:
                    out["values"][p.field] = float(series.iloc[-1])
                    ts = series.index[-1]
                    latest = ts if latest is None or ts > latest else latest
        out["latest"] = latest.to_pydatetime() if latest is not None else None
    except Exception as exc:  # noqa: BLE001 - the Welcome page must never crash
        out["error"] = str(exc)
    return out


@st.cache_data(ttl=60, show_spinner=False)
def fetch_trend_frame(
    measurement: str, start_utc: datetime, end_utc: datetime, window: str
) -> pd.DataFrame:
    """One windowed fetch per measurement; reusable across its fields."""
    from furnace_data.influx.online import fetch_online_df

    return fetch_online_df(
        selected_measurements=[measurement],
        time_range="last 1 hour",  # required arg; overrides take precedence
        request_type="windowed-average",
        window_by=window,
        start_time_override=start_utc,
        end_time_override=end_utc,
        column_naming="field",
    )


# ── Parameter list (shared by Welcome panel, Trend rail and mobile popover) ──

PANEL_CSS = """
<style>
[class*="st-key-kp_panel_"], [class*="st-key-kprow_"], [class*="st-key-kphdr_"] { gap: 0 !important; }
[class*="st-key-kp_panel_"] > *, [class*="st-key-kprow_"] > *, [class*="st-key-kphdr_"] > * { flex-shrink: 0 !important; }
/* Streamlit auto-height collapses these nested blocks to ~2px: size rows explicitly */
[class*="st-key-kprow_"], [class*="st-key-kphdr_"] { flex: 0 0 auto !important; height: 31px !important; }
[class*="st-key-kp_panel_"] {
    background: #0f1b2e; border: 1px solid #1f3a52; border-radius: 10px; padding: 0 0 4px;
    overflow-y: auto; scrollbar-width: thin; scrollbar-color: #2a4660 transparent;
}
.kp-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: nowrap;
    height: 31px; box-sizing: border-box; padding: 0 12px; border-bottom: 1px solid #1f3a52; }
.kp-title { white-space: nowrap; color: #e6f1fb; font-size: .72rem; font-weight: 700; letter-spacing: .14em; }
.kp-status { color: #7f93a8; font-size: .66rem; letter-spacing: .05em; font-variant-numeric: tabular-nums; }
.kp-live { color: #2dd4bf; font-weight: 700; margin-right: 6px; }
.kp-status { white-space: nowrap; }
.kp-live.warn { color: #fbbf24; }
.kp-live::before { content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; margin-right: 5px; }
[class*="st-key-kprow_"], [class*="st-key-kphdr_"] { position: relative; }
[class*="st-key-kprow_"] > [data-testid="stElementContainer"],
[class*="st-key-kphdr_"] > [data-testid="stElementContainer"] { position: static !important; }
.kp-row { display: flex; justify-content: space-between; align-items: center; gap: 10px;
    height: 31px; box-sizing: border-box; padding: 0 12px; border-bottom: 1px solid rgba(31, 58, 82, .5);
    transition: background .12s; }
[class*="st-key-kprow_"]:hover .kp-row { background: rgba(148, 197, 255, .07); }
.kp-row.sel { background: rgba(34, 211, 238, .11); box-shadow: inset 3px 0 0 #22d3ee; }
.kp-name { flex: 1 1 auto; min-width: 0; color: #aebdd0; font-size: .78rem; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
.kp-row.sel .kp-name { color: #fff; font-weight: 600; }
.kp-val { flex: 0 0 auto; color: #eaf6ff; font-size: .84rem; font-weight: 600; white-space: nowrap;
    font-variant-numeric: tabular-nums; }
.kp-val.na { color: #5b6b7e; font-weight: 400; }
.kp-val small { color: #7f93a8; font-size: .66rem; font-weight: 400; margin-left: 3px; }
.kp-val .sp { color: #5b6b7e; font-weight: 400; margin-right: 3px; }
/* "Updated" word only where there is room (popover) */
.kp-upd { display: none; }
[class*="st-key-kp_panel_pop"] .kp-upd { display: inline; }
/* small refresh icon in the header */
.kp-head { padding-right: 38px; }
[class*="st-key-kphdr_"] [data-testid="stButton"] { position: absolute !important; right: 6px; top: 0; height: 31px;
    display: flex; align-items: center; margin: 0 !important; width: auto !important; }
[class*="st-key-kphdr_"] [data-testid="stButton"] button {
    min-height: 0 !important; height: 25px; width: 28px; padding: 0 !important; border: 1px solid transparent !important;
    display: flex; align-items: center; justify-content: center;
    background: transparent !important; color: #8ea3b8 !important; font-size: 1.05rem; line-height: 1; border-radius: 6px; }
[class*="st-key-kphdr_"] [data-testid="stButton"] button:hover { color: #22d3ee !important; background: rgba(34, 211, 238, .12) !important; }
[class*="st-key-kphdr_"] [data-testid="stButton"] button p { font-size: 1.05rem; line-height: 1; }
/* invisible full-row hit target: page_link (Welcome) or button (Trend) */
[class*="st-key-kprow_"] [data-testid="stPageLink"],
[class*="st-key-kprow_"] [data-testid="stButton"] {
    position: absolute !important; inset: 0 !important; margin: 0 !important; width: 100% !important; }
[class*="st-key-kprow_"] [data-testid="stPageLink"] a,
[class*="st-key-kprow_"] [data-testid="stButton"] button {
    display: block !important; width: 100% !important; height: 100% !important; min-height: 0 !important;
    opacity: 0 !important; background: transparent !important; border: none !important;
    padding: 0 !important; text-decoration: none !important; cursor: pointer; }
</style>
"""

WELCOME_PANEL_CSS = """
<style>
.st-key-kp_panel_desktop { display: none; }
@media (min-width: 768px) {
    .block-container { max-width: none !important; padding-right: 252px !important; }
    .st-key-kp_panel_desktop { display: flex !important; position: fixed; top: 4.3rem; right: 10px;
        width: 232px; max-height: calc(100vh - 5.3rem); z-index: 99; }
    .st-key-top_logout button { white-space: nowrap; padding-inline: .5rem; }
}
@media (min-width: 1024px) {
    .block-container { max-width: 1460px !important; padding-right: 312px !important; }
    .st-key-kp_panel_desktop { width: 284px; }
}
/* phones / small portrait: panel lives only in the sidebar drawer */
@media (max-width: 767.98px) {
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="stSidebar"] { width: min(88vw, 320px) !important; min-width: 0 !important; }
    [data-testid="stSidebarContent"] { overflow-y: auto; }
    .st-key-kp_panel_mobile { border-radius: 8px; max-height: none; }
}
</style>
"""


def refresh_key_parameters(include_trend: bool = False) -> None:
    """Callback for the header refresh icon: drop cached values so the next run refetches."""
    fetch_current_values.clear()
    if include_trend:
        fetch_trend_frame.clear()


def select_parameter(key: str) -> None:
    """Callback: canonical selection in session state, mirrored to ?kp= for deep links."""
    if key in _BY_KEY:
        st.session_state[SELECTED_KEY] = key
        st.query_params[QUERY_PARAM] = key


def _row_html(param: KeyParameter, values: dict[str, float], selected: bool) -> str:
    raw = values.get(param.field) if param.field else None
    unit = f"<small>{html.escape(param.unit)}</small>" if param.unit and raw is not None else ""
    sp = '<span class="sp">— /</span>' if param.setpoint_placeholder else ""
    return (
        f'<div class="kp-row{" sel" if selected else ""}">'
        f'<span class="kp-name" title="{html.escape(param.label)}">{html.escape(param.label)}</span>'
        f'<span class="kp-val{" na" if raw is None else ""}">{sp}{format_value(param, raw)}{unit}</span></div>'
    )


def render_parameter_list(
    prefix: str, snapshot: dict, *, selectable: bool = False, selected_key: str | None = None
) -> None:
    """One renderer for every parameter list.

    ``selectable=False`` (Welcome): rows are page links to the Trend page.
    ``selectable=True`` (Trend): rows are buttons that call :func:`select_parameter`.
    """
    latest = snapshot["latest"]
    live = bool(snapshot["values"]) and latest is not None and (
        datetime.now(timezone.utc) - latest.astimezone(timezone.utc) <= STALE_AFTER
    )
    status = "LIVE" if live else ("STALE" if snapshot["values"] else "NO DATA")
    with st.container(key=f"kphdr_{prefix}"):
        st.markdown(
            f'<div class="kp-head"><span class="kp-title">KEY PARAMETERS</span>'
            f'<span class="kp-status"><span class="kp-live{"" if live else " warn"}">{status}</span>'
            f'<span class="kp-upd">Updated </span>{snapshot["fetched_at"]:%H:%M:%S}</span></div>',
            unsafe_allow_html=True,
        )
        st.button(
            "↻", key=f"kprefresh_{prefix}", help="Refresh values now",
            on_click=refresh_key_parameters, args=(selectable,),
        )
    for param in KEY_PARAMETERS:
        with st.container(key=f"kprow_{prefix}_{param.key}"):
            st.markdown(
                _row_html(param, snapshot["values"], selectable and param.key == selected_key),
                unsafe_allow_html=True,
            )
            if selectable:
                st.button(
                    param.label, key=f"kpbtn_{prefix}_{param.key}",
                    on_click=select_parameter, args=(param.key,),
                )
            else:
                st.page_link(
                    TREND_PAGE, label=f"Open {param.label} trend",
                    query_params={QUERY_PARAM: param.key},
                )


def render_key_parameters() -> None:
    """Welcome page: fixed right rail (>=768px) and sidebar drawer (<768px)."""
    snapshot = fetch_current_values()
    st.markdown(PANEL_CSS + WELCOME_PANEL_CSS, unsafe_allow_html=True)
    with st.container(key="kp_panel_desktop"):
        render_parameter_list("d", snapshot)
    with st.sidebar:
        with st.container(key="kp_panel_mobile"):
            render_parameter_list("s", snapshot)


# ── Trend helpers ─────────────────────────────────────────────────────────────

def build_trend_figure(series: pd.Series, param: KeyParameter):
    """Clean white single-line Plotly figure (``series`` index = naive IST)."""
    import plotly.graph_objects as go

    unit = f" {param.unit}" if param.unit else ""
    fig = go.Figure(
        go.Scatter(
            x=series.index, y=series.values, mode="lines", name=param.label,
            line=dict(color="#0891b2", width=2),
            hovertemplate=f"%{{y:,.{param.decimals}f}}{unit}<extra></extra>",
        )
    )
    axis = dict(gridcolor="#e5e9ef", linecolor="#cbd5e1", zeroline=False, tickfont=dict(color="#334155"))
    fig.update_layout(
        height=520, margin=dict(l=52, r=14, t=8, b=36), hovermode="x unified",
        paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
        font=dict(color="#334155"),
        hoverlabel=dict(bgcolor="white", bordercolor="#cbd5e1", font=dict(color="#0f172a")),
        xaxis=dict(**axis, title=dict(text="Time (IST)", font=dict(size=11, color="#475569"))),
        yaxis=dict(**axis, title=dict(text=param.unit or None, font=dict(size=11, color="#475569"))),
    )
    return fig
