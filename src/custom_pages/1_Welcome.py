# src/custom_pages/1_Welcome.py
import base64
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.session import is_admin, is_supervisor, logout_user

# ── Auth ──────────────────────────────────────────────────────────────────────
if "auth_user" not in st.session_state:
    st.warning("Please login to access this page.")
    st.stop()

# ── Global CSS (must use st.markdown — st.html is sandboxed) ─────────────────
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: #f1f5f9 !important; }

    /* Hide default Streamlit page-nav links in sidebar */
    [data-testid="stSidebarNav"] { display: none !important; }

    /* ── Sidebar base ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1e 0%, #0f172a 40%, #111827 100%) !important;
        border-right: 1px solid rgba(251,146,60,0.15) !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.4) !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
    [data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100vh !important;
    }

    /* ── Sidebar buttons — base ── */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.03) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        font-size: 0.83rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 1rem !important;
        text-align: left !important;
        letter-spacing: 0.01em !important;
        transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
        margin-bottom: 6px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        left: 0; top: 0; bottom: 0 !important;
        width: 3px !important;
        background: #fb923c !important;
        border-radius: 0 2px 2px 0 !important;
        opacity: 0 !important;
        transition: opacity 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(251,146,60,0.10) !important;
        border-color: rgba(251,146,60,0.35) !important;
        color: #fed7aa !important;
        transform: translateX(4px) !important;
        box-shadow: 0 2px 12px rgba(251,146,60,0.12) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover::before { opacity: 1 !important; }

    /* ── Logout button ── */
    [data-testid="stSidebar"] .logout-btn button,
    [data-testid="stSidebar"] .stButton:last-of-type > button {
        background: rgba(239,68,68,0.08) !important;
        border-color: rgba(239,68,68,0.2) !important;
        color: #fca5a5 !important;
        margin-top: 0 !important;
    }
    [data-testid="stSidebar"] .stButton:last-of-type > button:hover {
        background: rgba(239,68,68,0.18) !important;
        border-color: rgba(239,68,68,0.5) !important;
        color: #fecaca !important;
        box-shadow: 0 2px 12px rgba(239,68,68,0.15) !important;
    }
    [data-testid="stSidebar"] .stButton:last-of-type > button::before {
        background: #ef4444 !important;
    }

    /* ── Sidebar text ── */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label { color: #64748b !important; }
    [data-testid="stSidebar"] strong { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] hr {
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.06) !important;
        margin: 0.6rem 0 !important;
    }

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


if selection == "hopper" and (is_admin() or is_supervisor()):
    _back_btn()
    from ui.hopper_admin_page import hopper_admin_page
    hopper_admin_page(st.session_state.get("auth_user"))
    st.stop()

if is_admin():
    if selection == "burden":
        _back_btn()
        from ui.burden_admin_page import burden_admin_page
        burden_admin_page(st.session_state.get("auth_user"))
        st.stop()
    elif selection == "register":
        _back_btn()
        from ui.user_management import register_page
        register_page()
        st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
_role_label = "Admin" if is_admin() else "Supervisor" if is_supervisor() else "User"
_role_color = "#fb923c" if is_admin() else "#38bdf8" if is_supervisor() else "#4ade80"
_role_bg    = "rgba(251,146,60,0.15)" if is_admin() else "rgba(56,189,248,0.12)" if is_supervisor() else "rgba(74,222,128,0.12)"
_role_border= "rgba(251,146,60,0.3)"  if is_admin() else "rgba(56,189,248,0.25)"  if is_supervisor() else "rgba(74,222,128,0.25)"
_user_initial = str(st.session_state['auth_user'])[0].upper()
_active = st.session_state.get("admin_tool_selection")

import datetime as _dt
_now = _dt.datetime.now().strftime("%d %b %Y, %H:%M")

with st.sidebar:

    # ── Brand strip ──────────────────────────────────────────────────────────
    st.html(f"""
    <div style="background:linear-gradient(135deg,#7c2d12 0%,#c2410c 50%,#ea580c 100%);
                padding:1.1rem 1.2rem 1rem; margin:-1rem -1rem 0;
                position:relative; overflow:hidden;">
        <div style="position:absolute; top:-18px; right:-18px;
                    width:80px; height:80px; border-radius:50%;
                    background:rgba(255,255,255,0.06);"></div>
        <div style="position:absolute; bottom:-25px; left:10px;
                    width:60px; height:60px; border-radius:50%;
                    background:rgba(255,255,255,0.04);"></div>
        <div style="font-size:1.05rem; font-weight:800; color:white;
                    letter-spacing:0.3px; position:relative;">
            🔥 &nbsp;BF Intelligence
        </div>
        <div style="font-size:0.62rem; color:rgba(255,255,255,0.55);
                    letter-spacing:0.18em; margin-top:3px;
                    font-weight:600; position:relative;">
            BLAST FURNACE PLATFORM
        </div>
    </div>
    """)

    st.html("<div style='height:1rem'></div>")

    # ── Profile card ─────────────────────────────────────────────────────────
    st.html(f"""
    <div style="background:linear-gradient(135deg,rgba(255,255,255,0.05) 0%,rgba(255,255,255,0.02) 100%);
                border:1px solid rgba(255,255,255,0.08);
                border-radius:14px; padding:1rem 1.1rem;
                margin-bottom:0.2rem;
                box-shadow:0 4px 16px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.06);">
        <div style="display:flex; align-items:center; gap:0.85rem;">
            <div style="width:46px; height:46px; border-radius:50%;
                        background:linear-gradient(135deg,#92400e,#fb923c);
                        display:flex; align-items:center; justify-content:center;
                        font-size:1.25rem; font-weight:900; color:white;
                        flex-shrink:0;
                        box-shadow:0 0 0 3px rgba(251,146,60,0.2), 0 4px 12px rgba(251,146,60,0.3);">
                {_user_initial}
            </div>
            <div style="min-width:0; flex:1;">
                <div style="color:#f8fafc; font-weight:700; font-size:0.95rem;
                            white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    {st.session_state['auth_user']}
                </div>
                <div style="margin-top:5px; display:flex; align-items:center; gap:6px;">
                    <span style="background:{_role_bg};
                                 color:{_role_color};
                                 font-size:0.65rem; font-weight:800;
                                 letter-spacing:0.1em;
                                 padding:2px 9px; border-radius:20px;
                                 border:1px solid {_role_border};">
                        {_role_label.upper()}
                    </span>
                </div>
            </div>
        </div>
        <div style="margin-top:0.75rem; padding-top:0.65rem;
                    border-top:1px solid rgba(255,255,255,0.06);
                    display:flex; align-items:center; gap:5px;">
            <span style="width:7px; height:7px; border-radius:50%;
                         background:#4ade80; display:inline-block;
                         box-shadow:0 0 6px #4ade80; flex-shrink:0;"></span>
            <span style="color:#475569; font-size:0.68rem; letter-spacing:0.02em;">
                Session started &nbsp;·&nbsp; {_now}
            </span>
        </div>
    </div>
    """)

    st.html("<div style='height:0.5rem'></div>")

    # ── Admin Tools ──────────────────────────────────────────────────────────
    if is_admin() or is_supervisor():
        st.html("""
        <div style="display:flex; align-items:center; gap:8px; margin:0.4rem 0 0.6rem; padding-left:2px;">
            <div style="height:1px; flex:1; background:rgba(255,255,255,0.06);"></div>
            <span style="color:#475569; font-size:0.63rem; font-weight:700;
                         letter-spacing:0.14em; white-space:nowrap;">
                ⚙ &nbsp;ADMIN TOOLS
            </span>
            <div style="height:1px; flex:1; background:rgba(255,255,255,0.06);"></div>
        </div>
        """)

        _btn_style = lambda active: (
            "background:rgba(251,146,60,0.15) !important; border-color:rgba(251,146,60,0.4) !important; color:#fed7aa !important;"
            if active else ""
        )

        if st.button("🛠 &nbsp; Hopper Mapping", use_container_width=True,
                     key="sb_hopper"):
            st.session_state["admin_tool_selection"] = "hopper"
            st.rerun()

        if is_admin():
            if st.button("📊 &nbsp; Burden Distribution", use_container_width=True,
                         key="sb_burden"):
                st.session_state["admin_tool_selection"] = "burden"
                st.rerun()
            if st.button("📝 &nbsp; User Management", use_container_width=True,
                         key="sb_register"):
                st.session_state["admin_tool_selection"] = "register"
                st.rerun()

    # ── Spacer pushes logout to bottom ───────────────────────────────────────
    st.html("<div style='flex:1; min-height:1.5rem'></div>")

    # ── Divider ──────────────────────────────────────────────────────────────
    st.html("""
    <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(251,146,60,0.2),transparent);
                margin:0 0 0.8rem;"></div>
    """)

    # ── Logout ───────────────────────────────────────────────────────────────
    if st.button("🚪 &nbsp; Logout", use_container_width=True, key="sb_logout"):
        logout_user()
        st.stop()

    # ── Version tag ──────────────────────────────────────────────────────────
    st.html("""
    <div style="text-align:center; padding:0.6rem 0 0.4rem;">
        <span style="color:#1e293b; font-size:0.62rem; letter-spacing:0.1em;
                     font-weight:600; background:rgba(255,255,255,0.03);
                     padding:2px 10px; border-radius:20px;
                     border:1px solid rgba(255,255,255,0.05);">
            v2.1.0 &nbsp;·&nbsp; V-OptimAIse
        </span>
    </div>
    """)

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


_kpi_row = "".join([
    _kpi_card(_fmt(kpis.get("prod")),  "PRODUCTION RATE", "t / hr",   "linear-gradient(135deg,#92400e,#ea580c)"),
    _kpi_card(_fmt(kpis.get("fuel"),0),"FUEL RATE",       "kg / tHM", "linear-gradient(135deg,#92400e,#ea580c)"),
    _kpi_card(_fmt(kpis.get("etaco")), "ETA CO",          "%",        "linear-gradient(135deg,#92400e,#ea580c)"),
    _kpi_card(_fmt(kpis.get("wind"),0),"BLAST VOLUME",    "Nm³ / hr","linear-gradient(135deg,#92400e,#ea580c)"),
])

st.html(f"""
<div style="display:grid; grid-template-columns:repeat(4,1fr);
            gap:1rem; margin-bottom:2rem;">
    {_kpi_row}
</div>
""")

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


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.html("<p style='text-align:center; color:#94a3b8; font-size:0.75rem;'>"
        "Powered by <strong style='color:#f97316;'>V-OptimAIse</strong>"
        " &middot; Blast Furnace Intelligence Platform</p>")