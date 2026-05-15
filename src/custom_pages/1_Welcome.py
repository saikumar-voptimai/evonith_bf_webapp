# src/custom_pages/1_Welcome.py
import base64
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.session import has_permission, logout_user

# ── Auth ──────────────────────────────────────────────────────────────────────
if "auth_user" not in st.session_state:
    st.warning("Please login to access this page.")
    st.stop()

# ── Global CSS (must use st.markdown — st.html is sandboxed) ─────────────────
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: #f1f5f9 !important; }
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    .block-container {
        max-width: 1120px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
    }
    [data-testid="stPageLink"] { margin-top: -4px !important; }
    [data-testid="stPageLink"] a {
        display: block !important;
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
        padding: 0.65rem 1rem !important;
        text-align: center !important;
        color: #334155 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        text-decoration: none !important;
    }
    [data-testid="stPageLink"] a:hover { background: #e2e8f0 !important; color: #0f172a !important; }
    hr { border-color: #e2e8f0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Admin state ───────────────────────────────────────────────────────────────
if "admin_tool_selection" not in st.session_state:
    st.session_state["admin_tool_selection"] = None

selection = st.session_state.get("admin_tool_selection")


def _back_btn():
    if st.button("⬅  Back to Dashboard"):
        st.session_state["admin_tool_selection"] = None
        st.rerun()


if selection == "hopper" and has_permission("hopper:write"):
    _back_btn()
    from ui.hopper_admin_page import hopper_admin_page
    hopper_admin_page(st.session_state.get("auth_user"))
    st.stop()

if selection == "burden" and has_permission("burden:write"):
    _back_btn()
    from ui.burden_admin_page import burden_admin_page
    burden_admin_page(st.session_state.get("auth_user"))
    st.stop()

if selection == "register" and has_permission("users:write"):
    _back_btn()
    from ui.user_management import register_page
    register_page()
    st.stop()

# ── Top-right logout ──────────────────────────────────────────────────────────
_, _top_btn = st.columns([6, 1])
with _top_btn:
    if st.button("🚪 Logout", key="top_logout"):
        logout_user()
        st.stop()

# ── Hero banner ───────────────────────────────────────────────────────────────
def _b64_file(path: str) -> str:
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


_logo_b64 = _b64_file("src/assets/data/newlogo.png")
_hero_b64 = _b64_file("src/assets/data/bf_hero.png")

# ── Logo — sits above the hero, rounded corners ──────────────────────────────
if _logo_b64:
    st.html(f"""
    <div style="text-align:center; margin-bottom:1rem;">
        <img src="data:image/png;base64,{_logo_b64}"
             style="width:440px; max-width:88%; height:auto;
                    border-radius:20px;
                    box-shadow:0 4px 18px rgba(0,0,0,0.12);" />
    </div>
    """)

# ── Hero banner — mild warm/amber overlay so logo stays readable above ────────
if _hero_b64:
    _hero_bg = (
        f"background:"
        f"  linear-gradient(to bottom,"
        f"    rgba(35,20,5,0.58) 0%,"
        f"    rgba(110,70,10,0.22) 52%,"
        f"    rgba(20,10,3,0.68) 100%),"
        f"  url('data:image/png;base64,{_hero_b64}');"
        f"background-size:cover; background-position:center 38%;"
    )
else:
    _hero_bg = (
        "background: linear-gradient(135deg,"
        " #1a0f04 0%, #3d2008 25%, #92400e 60%, #c2410c 100%);"
    )

st.html(f"""
<div style="{_hero_bg}
    border-radius:16px; padding:2.8rem 2rem 2.5rem; text-align:center;
    margin-bottom:1.5rem; overflow:hidden;
    box-shadow:0 8px 32px rgba(0,0,0,0.28);
    min-height:200px; display:flex; flex-direction:column;
    align-items:center; justify-content:center;">
    <h1 style="color:white; font-size:2.8rem; font-weight:800;
               margin:0 0 0.6rem; letter-spacing:-0.5px;
               text-shadow:0 2px 16px rgba(0,0,0,0.7); line-height:1.15;">
        Blast Furnace
        <span style="color:#fb923c;">&nbsp;Intelligence&nbsp;</span>
        Platform
    </h1>
    <p style="color:rgba(255,255,255,0.80); font-size:1.05rem;
              margin:0; max-width:540px; line-height:1.65;
              text-shadow:0 1px 6px rgba(0,0,0,0.6);">
        Real-time AI co-pilot for BF2 operations &#8212; monitor, optimise, and act.
    </p>
</div>
""")

# ── Live KPI bar ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def _live_kpis() -> tuple[dict, str]:
    """Returns (values_dict, error_msg). error_msg is empty string on success."""
    try:
        from furnace_data.influx.online import fetch_online_df
        df = fetch_online_df(
            selected_measurements=["process_params"],
            time_range="last 1 hour",
            window_by="15 minutes",
        )
        if df is None or df.empty:
            return {}, "No data returned for the last hour."
        def m(col):
            return round(float(df[col].dropna().mean()), 1) if col in df.columns else None
        values = {
            "prod":  m("Process Params - BF2_PRODUCTION TONNES PER HR"),
            "fuel":  m("Process Params - BF2_FUEL RATE PER THM"),
            "etaco": m("Process Params - BF2_BODY_ETACO"),
            "wind":  m("Process Params - BF2_PROC Hot Blast Volume"),
        }
        return values, ""
    except Exception as e:
        return {}, str(e)


kpis, _kpi_err = _live_kpis()
if _kpi_err:
    st.caption(f"⚠ Live KPIs unavailable: {_kpi_err}")


def _fmt(v, d=1):
    return f"{v:,.{d}f}" if v is not None else "&#8212;"


def _kpi_card(value: str, label: str, unit: str, grad: str) -> str:
    return f"""
    <div style="background:{grad}; border-radius:12px; padding:1.5rem;
                text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.15);">
        <div style="font-size:2.2rem; font-weight:800; color:white;
                    letter-spacing:-1px; line-height:1;">{value}</div>
        <div style="color:rgba(255,255,255,0.85); font-size:0.75rem;
                    font-weight:700; letter-spacing:0.08em; margin-top:0.5rem;">
            {label}
        </div>
        <div style="color:rgba(255,255,255,0.5); font-size:0.7rem; margin-top:0.2rem;">
            {unit}
        </div>
    </div>"""


_kpi_specs = [
    (_fmt(kpis.get("prod")),   "PRODUCTION RATE", "t / hr",   "linear-gradient(135deg,#166534,#16a34a)"),
    (_fmt(kpis.get("fuel"),0), "FUEL RATE",       "kg / tHM", "linear-gradient(135deg,#92400e,#ea580c)"),
    (_fmt(kpis.get("etaco")),  "ETA CO",          "%",        "linear-gradient(135deg,#1e3a8a,#2563eb)"),
    (_fmt(kpis.get("wind"),0), "BLAST VOLUME",    "Nm³ / hr", "linear-gradient(135deg,#0c4a6e,#0284c7)"),
]

# st.columns ensures the grid reflows properly on narrow viewports
_kc1, _kc2, _kc3, _kc4 = st.columns(4)
for _col, (val, label, unit, grad) in zip([_kc1, _kc2, _kc3, _kc4], _kpi_specs):
    with _col:
        st.html(_kpi_card(val, label, unit, grad))

st.html("<div style='margin-bottom:2rem'></div>")

# ── Module tile grid ──────────────────────────────────────────────────────────
st.html("<p style='color:#64748b; font-size:0.75rem; font-weight:700;"
        "text-transform:uppercase; letter-spacing:0.1em; margin:0 0 1rem;'>"
        "Platform Modules</p>")

_TILES = [
    ("custom_pages/2_Data_Explorer.py",      "📓", "Data Explorer",
     "Browse InfluxDB telemetry, download datasets, and build ML training sets.",     "#2563eb"),
    ("custom_pages/3_Data_Visualisation.py", "📈", "V-Board",
     "Real-time 2D heat load contours and furnace body temperature profiles.",        "#7c3aed"),
    ("custom_pages/4_Recommendations.py",    "💡", "V-Sense",
     "Physics-informed AI recommendations for blast parameters and cost optimisation.","#059669"),
    ("custom_pages/5_AI_Copilot.py",         "🤖", "AI CoPilot",
     "Channeling propensity detection, anomaly scoring, and unit-cost benchmarking.", "#dc2626"),
    ("custom_pages/6_Material_Balance.py",   "⚖️", "Material Balance",
     "Daily 12-element mass balance — Sankey diagram, closure table, per-element bars.","#0891b2"),
    ("custom_pages/7_FurnaceMind.py",        "🧠", "FurnaceMind",
     "AI co-pilot: natural-language queries, trend plots, and live shift reports.",   "#7c2d12"),
]


def _tile(col, page_path: str, icon: str, title: str, desc: str, color: str):
    with col:
        st.html(f"""
        <div style="background:{color}; border-radius:10px 10px 0 0;
                    padding:1.2rem 1rem 1rem;
                    box-shadow:0 2px 0 rgba(0,0,0,0.08);">
            <span style="font-size:2.2rem;">{icon}</span>
            <div style="color:white; font-size:1rem; font-weight:700;
                        margin-top:0.4rem; letter-spacing:-0.2px;">{title}</div>
        </div>
        <div style="background:white; border:1px solid #e2e8f0; border-top:none;
                    padding:0.9rem 1rem 0.7rem;">
            <p style="color:#475569; font-size:0.82rem; margin:0; line-height:1.55;">
                {desc}
            </p>
        </div>
        """)
        st.page_link(page_path, label=f"Open {title}  →", use_container_width=True)


for i in range(0, len(_TILES), 3):
    c1, c2, c3 = st.columns(3)
    for col, tile in zip([c1, c2, c3], _TILES[i : i + 3]):
        _tile(col, *tile)
    st.html("<div style='height:0.6rem'></div>")

# Feedback — centred
_, _fc, _ = st.columns([1, 1, 1])
_tile(_fc, "custom_pages/8_Feedback.py", "📮", "Feedback",
      "Submit feature requests, bug reports, and operational feedback.", "#b45309")

# ── Admin tools ───────────────────────────────────────────────────────────────
if (
    has_permission("hopper:write")
    or has_permission("burden:write")
    or has_permission("users:write")
):
    st.markdown("---")
    st.html("<p style='color:#94a3b8; font-size:0.75rem; font-weight:700;"
            "text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem;'>"
            "Administration</p>")
    _a1, _a2, _a3 = st.columns(3)
    with _a1:
        if st.button("🛠  Hopper Mapping", use_container_width=True):
            st.session_state["admin_tool_selection"] = "hopper"
            st.rerun()
    if has_permission("burden:write"):
        with _a2:
            if st.button("📊  Burden Distribution", use_container_width=True):
                st.session_state["admin_tool_selection"] = "burden"
                st.rerun()
        with _a3:
            if st.button("📝  User Management", use_container_width=True):
                st.session_state["admin_tool_selection"] = "register"
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.html("<p style='text-align:center; color:#94a3b8; font-size:0.75rem;'>"
        "Powered by <strong style='color:#f97316;'>V-OptimAIse</strong>"
        " &middot; Blast Furnace Intelligence Platform</p>")
