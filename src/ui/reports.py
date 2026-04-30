from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import streamlit as st

from ui.components import show_report
from utils.window_helpers import build_day_window_id, fetch_from_qdrant

# IST offset
_IST = timezone(timedelta(hours=5, minutes=30))

# Shift start hour in IST; all shifts are 8 hours long
_SHIFT_START_IST: dict[str, int] = {"A": 6, "B": 14, "C": 22}


def _shift_utc_window(d: date, label: str) -> tuple[str, str]:
    """Return (start_utc_iso, end_utc_iso) for the given shift date + label."""
    start_h = _SHIFT_START_IST[label]
    start_ist = datetime(d.year, d.month, d.day, start_h, 0, 0, tzinfo=_IST)
    end_ist = start_ist + timedelta(hours=8)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (
        start_ist.astimezone(timezone.utc).strftime(fmt),
        end_ist.astimezone(timezone.utc).strftime(fmt),
    )


def _run_live_shift_report(d: date, label: str) -> str:
    """Run the agent loop to generate a live shift report and return the text."""
    # Deferred imports — these pull in heavy ML deps; only needed in Live mode.
    from agents.furnacemind.agent import run_agent_loop
    from agents.furnacemind.context import SystemPromptContext
    from agents.furnacemind.prompts import TOOL_POLICY
    from agents.furnacemind.skills import SkillEngine
    from agents.furnace_tools import get_openai_tool_schemas
    from agents.llm.llm_client import OpenRouterClient

    start_utc, end_utc = _shift_utc_window(d, label)
    engine = SkillEngine()
    prompt = engine.shift_report_prompt(str(d), label, start_utc, end_utc)

    ctx = SystemPromptContext()
    messages = [
        {
            "role": "system",
            "content": ctx.build(extra=TOOL_POLICY, skill_id="shift_report"),
        },
        {"role": "user", "content": prompt},
    ]

    llm = OpenRouterClient()
    tools = get_openai_tool_schemas()
    status_box = st.empty()
    response_box = st.empty()

    return run_agent_loop(
        llm=llm,
        messages=messages,
        tools=tools,
        status_box=status_box,
        response_box=response_box,
    )


def render_reports(*, vector_store) -> None:
    """Renders the Reports page with Qdrant (historical) or Live (on-the-fly LLM) modes."""

    st.header("📊 Reports")
    st.markdown(
        """
    <style>
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background-color: transparent !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.text("Report Type (Shift)")

    # ── Source selector ──────────────────────────────────────────────────────
    source = st.sidebar.radio(
        "Source",
        ["Qdrant", "Live"],
        horizontal=True,
        help=(
            "**Qdrant** — fetch a pre-stored shift summary.\n\n"
            "**Live** — generate the report on-the-fly from raw InfluxDB data."
        ),
    )

    # ── Report form ──────────────────────────────────────────────────────────
    with st.sidebar.form(key="report_form"):
        selected_date = st.date_input("Select date", date.today())
        shift_label = st.selectbox("Select shift", ["A", "B", "C"])
        btn_label = "Generate Report" if source == "Live" else "Fetch Report"
        fetch_report = st.form_submit_button(btn_label)

    if not fetch_report:
        return

    window_id = f"{selected_date:%Y-%m-%d}_SHIFT_{shift_label}"

    # ── Qdrant mode ──────────────────────────────────────────────────────────
    if source == "Qdrant":
        payload = fetch_from_qdrant(vector_store, window_id)
        if payload:
            show_report(f"📄 Report ({window_id})", payload["summary_text"])
        else:
            st.warning(
                f"No report found in Qdrant for **{window_id}**. "
                "Try the **Live** source to generate one on-the-fly."
            )
        return

    # ── Live mode ────────────────────────────────────────────────────────────
    cache_key = f"live_report_{window_id}"

    # Show cached result immediately (avoid re-generating on widget interaction)
    if cache_key in st.session_state:
        st.info(f"Showing cached live report for **{window_id}**. Re-submit to refresh.")
        show_report(f"⚡ Live Report ({window_id})", st.session_state[cache_key])
        return

    with st.spinner(
        f"Generating live report for **{window_id}** — fetching InfluxDB data and calling LLM…"
    ):
        report_text = _run_live_shift_report(selected_date, shift_label)

    st.session_state[cache_key] = report_text
    show_report(f"⚡ Live Report ({window_id})", report_text)
