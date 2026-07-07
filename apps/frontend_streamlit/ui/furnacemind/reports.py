from __future__ import annotations

from datetime import date

import streamlit as st

from apps.frontend_streamlit.reports.rendering import ReportDocument
from apps.frontend_streamlit.reports.furnace_report.timeframe import REPORT_TYPES, get_report_timeframe
from apps.frontend_streamlit.ui.components import show_report_document

_LAST_REPORT_KEY = "furnacemind_last_report"


def _remember_report(document: ReportDocument) -> None:
    st.session_state[_LAST_REPORT_KEY] = document


def _show_last_report() -> bool:
    report = st.session_state.get(_LAST_REPORT_KEY)
    if not report:
        return False

    document = report.get("document") if isinstance(report, dict) else report
    if isinstance(document, ReportDocument):
        show_report_document(document)
        return True
    return False


def _run_live_report(
    report_type: str,
    shift_date: date,
    shift_label: str | None,
    *,
    include_analysis: bool,
    status_box,
) -> ReportDocument:
    """Fetch live report data and return the structured document."""
    from apps.frontend_streamlit.agents.llm.llm_client import OpenRouterClient
    from apps.frontend_streamlit.reports.furnace_report import ShiftReportService

    status_box.status(f"Fetching {report_type.lower()} data...", expanded=False)
    _, document = ShiftReportService(llm_client=OpenRouterClient()).generate_document(
        shift_date,
        shift_label,
        report_type=report_type,
        include_analysis=include_analysis,
    )
    return document


def render_reports() -> None:
    """Render the Reports tab with live shift and day report options."""
    st.header("Reports")

    controls_left, controls_right = st.columns(2)

    with controls_left:
        report_type = st.selectbox(
            "Report Type",
            list(REPORT_TYPES),
            index=0,
            key="report_type",
        )
        selected_date = st.date_input("Date", date.today(), key="report_date")

    with controls_right:
        shift_label = (
            st.selectbox("Shift", ["A", "B", "C"], key="report_shift")
            if report_type == "Shift"
            else None
        )
        agentic_analysis = st.toggle(
            "Agentic analysis",
            value=False,
            help="When enabled, compare current report against previous window.",
            key="report_agentic_analysis",
        )

        selection = (
            f"{selected_date:%Y-%m-%d} Shift {shift_label}"
            if report_type == "Shift"
            else f"{selected_date:%Y-%m-%d}"
        )
        st.caption(f"Selected: {report_type} report | {selection}")

    generate_report = st.button(
        "Generate Report",
        width="stretch",
        key="generate_report",
    )

    if not generate_report:
        _show_last_report()
        return

    timeframe = get_report_timeframe(report_type, selected_date, shift_label)

    status_box = st.empty()
    spinner_message = (
        f"Generating live report for {timeframe.display_name} from source data"
    )
    spinner_message += " and LLM analysis..." if agentic_analysis else "..."

    with st.spinner(spinner_message):
        document = _run_live_report(
            report_type,
            selected_date,
            shift_label,
            include_analysis=agentic_analysis,
            status_box=status_box,
        )

    status_box.empty()
    _remember_report(document)
    show_report_document(document)
