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


_SKILL_DOC_PATH = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "storage" / "furnacemind" / "SKILLS_SHIFTREPORT.md"
)

_SHIFT_REPORT_SYSTEM = (
    "You are a BF2 Evonith Steel shift handover report generator.\n"
    "Your ONLY job: call fetch_online_data then fetch_offline_data (HM_SLAG then CHARGE) "
    "as instructed in the user message, then write the structured text report.\n"
    "CRITICAL RULES:\n"
    "- Do NOT call execute_python_plot. No charts, no plots, no code execution.\n"
    "- Do NOT output code, markdown code blocks, planning text, or commentary.\n"
    "- Output the shift report only.\n"
    "- Use start_time_utc / end_time_utc parameters exactly as provided in the steps.\n"
)


def _build_shift_report_system() -> str:
    """Minimal system prompt: just the persona rules + SKILLS_SHIFTREPORT.md content."""
    skill_doc = (
        _SKILL_DOC_PATH.read_text(encoding="utf-8").strip()
        if _SKILL_DOC_PATH.exists()
        else ""
    )
    if skill_doc:
        return _SHIFT_REPORT_SYSTEM + "\nSKILL REFERENCE:\n" + skill_doc
    return _SHIFT_REPORT_SYSTEM


def _run_live_shift_report(
    d: date, label: str, *, status_box, response_box
) -> str:
    """Run the agent loop to generate a live shift report and return the text."""
    # Deferred imports — these pull in heavy ML deps; only needed in Live mode.
    from agents.furnacemind.agent import run_agent_loop
    from agents.furnacemind.skills import SkillEngine
    from agents.furnace_tools import get_openai_tool_schemas
    from agents.llm.llm_client import OpenRouterClient

    start_utc, end_utc = _shift_utc_window(d, label)
    engine = SkillEngine()
    prompt = engine.shift_report_prompt(str(d), label, start_utc, end_utc)

    system_prompt = _build_shift_report_system()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    all_tools = get_openai_tool_schemas()
    fetch_tools = [
        t for t in all_tools
        if t["function"]["name"] in ("fetch_online_data", "fetch_offline_data")
    ]

    llm = OpenRouterClient()
    return run_agent_loop(
        llm=llm,
        messages=messages,
        tools=fetch_tools,
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
        ["Saved", "Live"],
        horizontal=True,
        help=(
            "**Saved** — fetch a pre-stored shift summary.\n\n"
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

    # ── DB-Saved mode ──────────────────────────────────────────────────────────
    if source == "Saved":
        payload = fetch_from_qdrant(vector_store, window_id)
        if payload:
            show_report(f"📄 Report ({window_id})", payload["summary_text"])
        else:
            st.warning(
                f"No report found in DB-Saved for **{window_id}**. "
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

    # Create placeholders here so we can clear them after generation.
    # run_agent_loop streams status badges into status_box and the raw LLM
    # response into response_box.  We clear both before show_report renders
    # the formatted version — otherwise the report appears twice.
    status_box = st.empty()
    response_box = st.empty()

    with st.spinner(
        f"Generating live report for **{window_id}** — fetching InfluxDB data and calling LLM…"
    ):
        report_text = _run_live_shift_report(
            selected_date, shift_label,
            status_box=status_box,
            response_box=response_box,
        )

    # Clear the streaming intermediates before rendering the formatted report.
    response_box.empty()

    st.session_state[cache_key] = report_text
    show_report(f"⚡ Live Report ({window_id})", report_text)
