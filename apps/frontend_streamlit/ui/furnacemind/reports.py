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

def _api_token() -> str | None:
    value = str(st.session_state.get("auth_access_token") or "").strip()
    return value or None


def _report_document_from_payload(payload: dict) -> ReportDocument:
    from datetime import datetime

    from apps.frontend_streamlit.reports.rendering import ReportNote, ReportSection

    generated_at = payload.get("generated_at_ist")
    parsed_generated_at = datetime.fromisoformat(generated_at) if generated_at else None
    return ReportDocument(
        title=str(payload.get("title") or "FurnaceMind Report"),
        pre_blocks=tuple(payload.get("pre_blocks") or ()),
        sections=tuple(
            ReportSection(
                title=str(section.get("title") or "Report Table"),
                headers=tuple(section.get("headers") or ()),
                rows=tuple(tuple(row) for row in (section.get("rows") or ())),
                placement=section.get("placement") or "right",
            )
            for section in payload.get("sections") or ()
        ),
        notes=tuple(
            ReportNote(
                text=str(note.get("text") or ""),
                placement=note.get("placement") or "right",
                kind=note.get("kind") or "note",
            )
            for note in payload.get("notes") or ()
        ),
        generated_at_ist=parsed_generated_at,
    )


def render_reports_api() -> None:
    """Render Reports through the FurnaceMind backend API."""
    from apps.frontend_streamlit.services.api_errors import FrontendApiError
    from apps.frontend_streamlit.services import furnacemind_api as api

    st.header("Reports")
    token = _api_token()
    try:
        config = api.get_reports_config(token=token)
    except FrontendApiError as exc:
        st.error(f"Could not load report configuration: {exc.message}")
        return

    controls_left, controls_right = st.columns(2)
    with controls_left:
        report_type = st.selectbox("Report Type", config.get("report_types") or list(REPORT_TYPES), index=0, key="report_type")
        selected_date = st.date_input("Date", date.today(), key="report_date")
    with controls_right:
        shift_label = st.selectbox("Shift", config.get("shift_labels") or ["A", "B", "C"], key="report_shift") if report_type == "Shift" else None
        agentic_analysis = st.toggle("Agentic analysis", value=bool(config.get("default_include_analysis", False)), key="report_agentic_analysis")
        selection = f"{selected_date:%Y-%m-%d} Shift {shift_label}" if report_type == "Shift" else f"{selected_date:%Y-%m-%d}"
        st.caption(f"Selected: {report_type} report | {selection}")

    if not st.button("Generate Report", width="stretch", key="generate_report"):
        _show_last_report()
        return

    try:
        created = api.create_report(
            {
                "report_type": report_type,
                "selected_date": selected_date.isoformat(),
                "shift_label": shift_label,
                "include_analysis": agentic_analysis,
            },
            token=token,
        )
        report = api.get_report(created["id"], token=token)
    except FrontendApiError as exc:
        st.error(f"Report generation failed: {exc.message}")
        return

    for warning in report.get("warnings", []):
        st.warning(warning.get("message", warning))
    document_payload = report.get("document")
    if not document_payload:
        st.info(f"Report status: {report.get('status')}")
        return
    document = _report_document_from_payload(document_payload)
    _remember_report(document)
    show_report_document(document)
