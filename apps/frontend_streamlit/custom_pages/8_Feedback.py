"""Feedback and shared ticket board page."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.config.page_registry import get_feedback_page_options
from apps.frontend_streamlit.data.tickets import TicketCreateRequest, TicketCriticality, TicketService, TicketStatus
from apps.frontend_streamlit.services import feedback_api
from apps.frontend_streamlit.services.api_errors import FrontendApiError
from apps.frontend_streamlit.utils.feedback_page import (
    build_attachment_payloads,
    load_feedback_css,
    render_board,
    render_management_panel,
    render_overview_kpis,
)
from apps.frontend_streamlit.utils.session import is_admin, is_supervisor

MAX_UPLOAD_FILES = 5
MAX_UPLOAD_SIZE_MB = 5

CRITICALITY_OPTIONS = [
    TicketCriticality.LOW.value,
    TicketCriticality.MEDIUM.value,
    TicketCriticality.HIGH.value,
    TicketCriticality.CRITICAL.value,
]
STATUS_OPTIONS = [
    TicketStatus.OPEN.value,
    TicketStatus.IN_PROGRESS.value,
    TicketStatus.RESOLVED.value,
    TicketStatus.DEPENDENCY_CONFLICT.value,
    TicketStatus.CLOSED.value,
]
API_DEFAULT_STATUSES = ["open", "in_progress", "resolved", "closed", "rejected"]
API_DEFAULT_PRIORITIES = ["low", "medium", "high", "critical"]


@st.cache_resource(show_spinner=False)
def get_ticket_service() -> TicketService:
    """Return a cached ticket service instance for the page session."""
    return TicketService()


def _render_form(ticket_service: TicketService) -> None:
    """Render the feedback submission form."""
    st.markdown(
        "<div class='feedback-section-title'>New Feedback</div>",
        unsafe_allow_html=True,
    )

    page_options = get_feedback_page_options()
    auth_user = str(st.session_state.get("auth_user", "")).strip()
    if "feedback_upload_nonce" not in st.session_state:
        st.session_state["feedback_upload_nonce"] = 0
    uploader_key = f"feedback_upload_{st.session_state['feedback_upload_nonce']}"

    with st.form("feedback_create_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            page_name = st.selectbox("Page with issue", options=page_options)
        with col2:
            st.text_input("User name", value=auth_user, disabled=True)

        col3, col4 = st.columns([1, 2])
        with col3:
            criticality = st.selectbox(
                "Issue criticality",
                options=CRITICALITY_OPTIONS,
                format_func=TicketService.criticality_label,
                index=1,
            )
        with col4:
            ideal_closure = st.text_input(
                "Ideal closure",
                placeholder="Example: Need this fixed before next shift handover.",
                max_chars=240,
            )

        description = st.text_area(
            "Issue description",
            placeholder="Describe what went wrong, where, and how to reproduce it.",
            height=140,
            max_chars=2000,
        )
        uploaded_files = st.file_uploader(
            "Screenshots (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help=f"Up to {MAX_UPLOAD_FILES} files, max {MAX_UPLOAD_SIZE_MB} MB each.",
            key=uploader_key,
        )
        if uploaded_files:
            st.caption("Preview")
            preview_cols = st.columns(3)
            for idx, uploaded_file in enumerate(uploaded_files):
                with preview_cols[idx % 3]:
                    st.image(
                        uploaded_file,
                        caption=uploaded_file.name,
                        width="stretch",
                    )

        submitted = st.form_submit_button("Submit Ticket", type="primary")

    if not submitted:
        return

    try:
        created_ticket = ticket_service.create_ticket(
            TicketCreateRequest(
                page_name=page_name,
                reported_by=auth_user,
                criticality=criticality,
                description=description,
                ideal_closure_text=ideal_closure,
                created_by=auth_user,
            ),
            attachments=build_attachment_payloads(uploaded_files or []),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.success(f"Ticket {created_ticket.ticket_code} created successfully.")
    st.session_state["feedback_upload_nonce"] += 1
    st.rerun()


def _api_token() -> str | None:
    """Return the current backend access token, when login used API auth."""
    token = str(st.session_state.get("auth_access_token") or "").strip()
    return token or None


def _api_label(value: str | None) -> str:
    """Render backend snake_case values consistently with direct mode."""
    return str(value or "").replace("_", "-")


def _api_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _api_created_label(ticket: dict[str, Any]) -> str:
    created = _api_datetime(ticket.get("created_at"))
    return created.astimezone().strftime("%Y-%m-%d %H:%M") if created else "-"


def _api_ticket_code(ticket: dict[str, Any]) -> str:
    return str(ticket.get("ticket_number") or ticket.get("id") or "")


def _api_ticket_id(ticket: dict[str, Any]) -> str:
    return str(ticket.get("id") or ticket.get("ticket_number") or "")


def _show_api_error(prefix: str, exc: FrontendApiError) -> None:
    request_id = f" Request ID: {exc.request_id}." if exc.request_id else ""
    st.error(f"{prefix}: {exc.message}.{request_id}")


def _load_api_config() -> dict[str, Any]:
    try:
        config = feedback_api.get_feedback_config(token=_api_token())
    except FrontendApiError as exc:
        _show_api_error("Could not load feedback configuration", exc)
        return {}
    return config if isinstance(config, dict) else {}


def _api_status_options(config: dict[str, Any]) -> list[str]:
    return list(config.get("statuses") or API_DEFAULT_STATUSES)


def _api_priority_options(config: dict[str, Any]) -> list[str]:
    return list(config.get("priorities") or API_DEFAULT_PRIORITIES)


def _api_upload_extensions(config: dict[str, Any]) -> list[str]:
    extensions = config.get("allowed_attachment_extensions") or []
    if extensions:
        return [str(item).lstrip(".") for item in extensions]
    content_types = set(config.get("allowed_attachment_types") or [])
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "application/pdf": "pdf",
        "text/plain": "txt",
        "text/csv": "csv",
    }
    return [ext for content_type, ext in mapping.items() if content_type in content_types]


def _list_api_tickets(filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = feedback_api.list_tickets(filters or {"limit": 200}, token=_api_token())
    if isinstance(data, dict):
        return list(data.get("items") or [])
    return []


def _render_api_overview(tickets: list[dict[str, Any]]) -> None:
    st.markdown(
        "<div class='feedback-section-title'>Case Overview</div>",
        unsafe_allow_html=True,
    )
    open_count = sum(ticket.get("status") == "open" for ticket in tickets)
    progress_count = sum(ticket.get("status") == "in_progress" for ticket in tickets)
    resolved_closed_count = sum(
        ticket.get("status") in {"resolved", "closed"} for ticket in tickets
    )
    rejected_count = sum(ticket.get("status") == "rejected" for ticket in tickets)
    high_critical_count = sum(
        ticket.get("priority") in {"high", "critical"} for ticket in tickets
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Open", open_count)
    k2.metric("In Progress", progress_count)
    k3.metric("Resolved/Closed", resolved_closed_count)
    k4.metric("Rejected", rejected_count)
    k5.metric("High/Critical", high_critical_count)


def _upload_api_attachments(
    *,
    ticket_id: str,
    uploaded_files: list[Any],
    token: str | None,
) -> None:
    for uploaded_file in uploaded_files:
        feedback_api.upload_attachment(
            ticket_id,
            filename=uploaded_file.name,
            content=uploaded_file.getvalue(),
            content_type=uploaded_file.type or "application/octet-stream",
            token=token,
        )


def _render_api_form(config: dict[str, Any]) -> None:
    st.markdown(
        "<div class='feedback-section-title'>New Feedback</div>",
        unsafe_allow_html=True,
    )
    page_options = get_feedback_page_options()
    auth_user = str(st.session_state.get("auth_user", "")).strip()
    if "feedback_api_upload_nonce" not in st.session_state:
        st.session_state["feedback_api_upload_nonce"] = 0
    uploader_key = f"feedback_api_upload_{st.session_state['feedback_api_upload_nonce']}"
    upload_extensions = _api_upload_extensions(config)
    max_files = int(config.get("max_attachments_per_ticket") or 10)
    max_mb = int(config.get("max_attachment_mb") or 10)

    with st.form("feedback_api_create_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            page_name = st.selectbox("Page with issue", options=page_options)
        with col2:
            st.text_input("User name", value=auth_user, disabled=True)

        col3, col4 = st.columns([1, 2])
        with col3:
            priority = st.selectbox(
                "Issue criticality",
                options=_api_priority_options(config),
                format_func=TicketService.criticality_label,
                index=1,
            )
        with col4:
            ideal_closure = st.text_input(
                "Ideal closure",
                placeholder="Example: Need this fixed before next shift handover.",
                max_chars=240,
            )

        description = st.text_area(
            "Issue description",
            placeholder="Describe what went wrong, where, and how to reproduce it.",
            height=140,
            max_chars=2000,
        )
        uploaded_files = st.file_uploader(
            "Attachments (optional)",
            type=upload_extensions or None,
            accept_multiple_files=True,
            help=f"Up to {max_files} files, max {max_mb} MB each.",
            key=uploader_key,
        )
        if uploaded_files:
            st.caption("Preview")
            preview_cols = st.columns(3)
            for idx, uploaded_file in enumerate(uploaded_files[:max_files]):
                with preview_cols[idx % 3]:
                    if str(uploaded_file.type or "").startswith("image/"):
                        st.image(uploaded_file, caption=uploaded_file.name, width="stretch")
                    else:
                        st.caption(uploaded_file.name)

        submitted = st.form_submit_button("Submit Ticket", type="primary")

    if not submitted:
        return
    if not description.strip():
        st.error("Issue description is required.")
        return
    if not ideal_closure.strip():
        st.error("Ideal closure text is required.")
        return
    if len(uploaded_files or []) > max_files:
        st.error(f"Maximum {max_files} attachments are allowed per ticket.")
        return

    token = _api_token()
    try:
        created_ticket = feedback_api.create_ticket(
            {
                "title": f"{page_name} feedback",
                "description": description,
                "category": page_name,
                "priority": priority,
                "page": page_name,
                "metadata": {
                    "reported_by": auth_user,
                    "ideal_closure_text": ideal_closure,
                },
            },
            token=token,
        )
        if uploaded_files:
            _upload_api_attachments(
                ticket_id=_api_ticket_id(created_ticket),
                uploaded_files=list(uploaded_files),
                token=token,
            )
    except FrontendApiError as exc:
        _show_api_error("Could not create ticket", exc)
        return

    st.success(f"Ticket {_api_ticket_code(created_ticket)} created successfully.")
    st.session_state["feedback_api_upload_nonce"] += 1
    st.rerun()


def _render_api_management_panel(config: dict[str, Any], tickets: list[dict[str, Any]]) -> None:
    can_manage = is_admin() or is_supervisor()
    if not can_manage:
        st.info("Status controls are available for admin/supervisor roles.")
        return
    if not tickets:
        st.info("No tickets available for status management yet.")
        return

    st.markdown(
        "<div class='feedback-section-title'>Case Manager</div>",
        unsafe_allow_html=True,
    )
    ticket_lookup = {
        (
            f"{_api_ticket_code(ticket)} | {ticket.get('page') or ticket.get('category') or '-'} | "
            f"{_api_label(ticket.get('status'))} | {_api_label(ticket.get('priority'))}"
        ): ticket
        for ticket in tickets
    }
    selected_label = st.selectbox("Case", list(ticket_lookup.keys()), key="fb_api_manager_case")
    selected_ticket = ticket_lookup[selected_label]
    status_options = _api_status_options(config)
    current_status = str(selected_ticket.get("status") or status_options[0])
    current_index = status_options.index(current_status) if current_status in status_options else 0

    with st.form("feedback_api_status_manager_form"):
        next_status = st.selectbox(
            "New status",
            options=status_options,
            index=current_index,
            format_func=_api_label,
        )
        update_comment = st.text_area(
            "Status note",
            placeholder="Optional progress note...",
            max_chars=800,
            height=90,
        )
        status_submit = st.form_submit_button("Update Case", type="primary")

    if status_submit:
        try:
            feedback_api.update_ticket(
                _api_ticket_id(selected_ticket),
                {
                    "status": next_status,
                    "resolution_notes": update_comment or None,
                },
                token=_api_token(),
            )
            st.success(f"{_api_ticket_code(selected_ticket)} updated to {_api_label(next_status)}.")
            st.rerun()
        except FrontendApiError as exc:
            _show_api_error("Could not update ticket", exc)


def _api_ticket_table(tickets: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Code": _api_ticket_code(ticket),
                "Page": ticket.get("page") or ticket.get("category") or "-",
                "Criticality": TicketService.criticality_label(str(ticket.get("priority") or "")),
                "Status": _api_label(ticket.get("status")),
                "Reporter": ticket.get("created_by_username") or ticket.get("created_by") or "-",
                "Created": _api_created_label(ticket),
            }
            for ticket in tickets
        ]
    )


def _render_api_attachments(ticket: dict[str, Any], config: dict[str, Any]) -> None:
    ticket_id = _api_ticket_id(ticket)
    token = _api_token()
    try:
        attachments = feedback_api.list_attachments(ticket_id, token=token)
    except FrontendApiError as exc:
        _show_api_error("Could not load attachments", exc)
        return

    if attachments:
        st.markdown("**Attachments**")
        for attachment in attachments:
            cols = st.columns([0.55, 0.25, 0.2])
            filename = str(attachment.get("original_filename") or attachment.get("filename") or "attachment")
            size_kb = int(attachment.get("size_bytes") or 0) / 1024
            with cols[0]:
                st.caption(f"{filename} ({size_kb:.1f} KB)")
            with cols[1]:
                try:
                    data = feedback_api.download_attachment(str(attachment["id"]), token=token)
                    st.download_button(
                        "Download",
                        data=data,
                        file_name=filename,
                        mime=str(attachment.get("content_type") or "application/octet-stream"),
                        key=f"fb_api_download_{attachment['id']}",
                    )
                except FrontendApiError as exc:
                    _show_api_error("Could not download attachment", exc)
            with cols[2]:
                if is_admin() or is_supervisor():
                    if st.button("Delete", key=f"fb_api_delete_attachment_{attachment['id']}"):
                        try:
                            feedback_api.delete_attachment(str(attachment["id"]), token=token)
                            st.success(f"{filename} deleted.")
                            st.rerun()
                        except FrontendApiError as exc:
                            _show_api_error("Could not delete attachment", exc)

    upload_extensions = _api_upload_extensions(config)
    with st.form(f"fb_api_add_attachment_{ticket_id}"):
        new_files = st.file_uploader(
            "Add attachment",
            type=upload_extensions or None,
            accept_multiple_files=True,
            key=f"fb_api_detail_upload_{ticket_id}",
        )
        if st.form_submit_button("Upload Attachment"):
            try:
                _upload_api_attachments(
                    ticket_id=ticket_id,
                    uploaded_files=list(new_files or []),
                    token=token,
                )
                st.success("Attachment uploaded.")
                st.rerun()
            except FrontendApiError as exc:
                _show_api_error("Could not upload attachment", exc)


def _render_api_comments(ticket: dict[str, Any]) -> None:
    ticket_id = _api_ticket_id(ticket)
    token = _api_token()
    try:
        comments = feedback_api.list_comments(ticket_id, token=token)
    except FrontendApiError as exc:
        _show_api_error("Could not load comments", exc)
        comments = []

    if comments:
        with st.expander("Comments", expanded=False, key=f"fb_api_comments_{ticket_id}"):
            for comment in comments:
                created = _api_datetime(comment.get("created_at"))
                created_label = created.astimezone().strftime("%Y-%m-%d %H:%M") if created else "-"
                author = comment.get("created_by_username") or comment.get("created_by") or "-"
                st.markdown(f"**{author}** · {created_label}")
                st.write(comment.get("body") or "")

    with st.form(f"fb_api_comment_form_{ticket_id}"):
        body = st.text_area("Add comment", max_chars=1000, height=90)
        if st.form_submit_button("Post Comment"):
            try:
                feedback_api.add_comment(ticket_id, body, token=token)
                st.success("Comment added.")
                st.rerun()
            except FrontendApiError as exc:
                _show_api_error("Could not add comment", exc)


def _render_api_board(config: dict[str, Any]) -> None:
    st.markdown(
        "<div class='feedback-section-title'>All Cases</div>",
        unsafe_allow_html=True,
    )
    try:
        all_tickets = _list_api_tickets({"limit": 200})
    except FrontendApiError as exc:
        _show_api_error("Could not load tickets", exc)
        return

    page_options = sorted({str(ticket.get("page") or ticket.get("category") or "") for ticket in all_tickets if ticket.get("page") or ticket.get("category")})
    with st.expander("Filters", expanded=True, key="fb_api_filters"):
        f1, f2, f3 = st.columns(3)
        with f1:
            selected_status = st.selectbox(
                "Status",
                options=["", *_api_status_options(config)],
                format_func=lambda value: "All" if not value else _api_label(value),
            )
        with f2:
            selected_priority = st.selectbox(
                "Criticality",
                options=["", *_api_priority_options(config)],
                format_func=lambda value: "All" if not value else TicketService.criticality_label(value),
            )
            selected_page = st.selectbox("Page", options=["", *page_options], format_func=lambda value: "All" if not value else value)
        with f3:
            keyword = st.text_input("Keyword", placeholder="code, issue text, user...")

    filters = {
        "status": selected_status or None,
        "priority": selected_priority or None,
        "category": selected_page or None,
        "search": keyword or None,
        "limit": 200,
    }
    try:
        tickets = _list_api_tickets(filters)
    except FrontendApiError as exc:
        _show_api_error("Could not filter tickets", exc)
        return

    if not tickets:
        st.session_state.pop("fb_api_selected_ticket_code", None)
        st.info("No tickets found for the current filters.")
        return

    ticket_map = {_api_ticket_code(ticket): ticket for ticket in tickets}
    table_df = _api_ticket_table(tickets)
    table_height = min(420, max(180, 62 + (len(table_df) * 35)))
    left_col, right_col = st.columns([0.44, 0.56], gap="large")

    with left_col:
        st.caption(f"{len(tickets)} ticket(s) - click a row to view details")
        selection = st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="fb_api_ticket_table",
            height=table_height,
        )

    selected_rows = (selection.selection.get("rows") or []) if selection else []
    selected_code: str | None = None
    if selected_rows:
        selected_row_idx = selected_rows[0]
        if 0 <= selected_row_idx < len(table_df):
            selected_code = str(table_df.iloc[selected_row_idx]["Code"])

    if selected_code:
        st.session_state["fb_api_selected_ticket_code"] = selected_code
    else:
        remembered_code = st.session_state.get("fb_api_selected_ticket_code")
        if isinstance(remembered_code, str) and remembered_code in ticket_map:
            selected_code = remembered_code
        elif len(tickets) == 1:
            selected_code = _api_ticket_code(tickets[0])
            st.session_state["fb_api_selected_ticket_code"] = selected_code
        else:
            st.session_state.pop("fb_api_selected_ticket_code", None)

    selected_ticket = ticket_map.get(selected_code) if selected_code else None

    with right_col:
        if selected_ticket is None:
            st.markdown(
                "<div class='feedback-board-empty'>Select a ticket from the left table to view details.</div>",
                unsafe_allow_html=True,
            )
            return

        ticket = selected_ticket
        st.markdown(f"### `{_api_ticket_code(ticket)}` - {ticket.get('page') or ticket.get('category') or '-'}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Status:** {_api_label(ticket.get('status'))}")
        with c2:
            st.markdown(
                f"**Criticality:** {TicketService.criticality_label(str(ticket.get('priority') or ''))}"
            )
        reporter = ticket.get("created_by_username") or ticket.get("created_by") or "-"
        updated = _api_datetime(ticket.get("updated_at"))
        updated_label = updated.astimezone().strftime("%Y-%m-%d %H:%M") if updated else "-"
        st.markdown(
            f"<div class='feedback-meta'><b>Reporter:</b> {reporter} | "
            f"<b>Created:</b> {_api_created_label(ticket)} | "
            f"<b>Updated:</b> {updated_label}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Issue description**")
        st.write(ticket.get("description") or "")
        metadata = ticket.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("ideal_closure_text"):
            st.markdown("**Ideal closure**")
            st.write(metadata["ideal_closure_text"])

        if is_admin() or is_supervisor():
            with st.form(f"fb_api_inline_status_{_api_ticket_id(ticket)}"):
                status_options = _api_status_options(config)
                current_status = str(ticket.get("status") or status_options[0])
                current_index = status_options.index(current_status) if current_status in status_options else 0
                next_status = st.selectbox(
                    "New status",
                    options=status_options,
                    index=current_index,
                    format_func=_api_label,
                    key=f"fb_api_status_sel_{_api_ticket_id(ticket)}",
                )
                if st.form_submit_button("Update Status", type="primary"):
                    try:
                        feedback_api.update_ticket(
                            _api_ticket_id(ticket),
                            {"status": next_status},
                            token=_api_token(),
                        )
                        st.success(f"Updated to {_api_label(next_status)}.")
                        st.rerun()
                    except FrontendApiError as exc:
                        _show_api_error("Could not update ticket", exc)

        _render_api_attachments(ticket, config)
        _render_api_comments(ticket)


def _render_api_feedback_page() -> None:
    config = _load_api_config()
    try:
        all_tickets = _list_api_tickets({"limit": 200})
    except FrontendApiError as exc:
        _show_api_error("Could not load tickets", exc)
        all_tickets = []

    st.markdown("<div class='feedback-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='feedback-hero'>
            <h2>Feedback & Support Desk</h2>
            <p>
                Raise platform issues, track progress, and collaborate through a shared,
                role-aware ticket board.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.52, 0.48], gap="large")
    with left:
        _render_api_form(config)
    with right:
        _render_api_overview(all_tickets)
        _render_api_management_panel(config, all_tickets)

    st.divider()
    _render_api_board(config)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    """Render the Feedback page."""
    if "auth_user" not in st.session_state:
        st.warning("Please login to access this page.")
        st.stop()

    load_feedback_css(Path(__file__))
    if is_backend_api_enabled("feedback"):
        _render_api_feedback_page()
        return

    ticket_service = get_ticket_service()
    all_tickets = ticket_service.list_tickets()

    st.markdown("<div class='feedback-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='feedback-hero'>
            <h2>Feedback & Support Desk</h2>
            <p>
                Raise platform issues, track progress, and collaborate through a shared,
                role-aware ticket board.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.52, 0.48], gap="large")
    with left:
        _render_form(ticket_service=ticket_service)
    with right:
        render_overview_kpis(all_tickets)
        render_management_panel(
            ticket_service=ticket_service,
            tickets=all_tickets,
            status_options=STATUS_OPTIONS,
        )

    st.divider()
    render_board(
        ticket_service=ticket_service,
        status_options=STATUS_OPTIONS,
        criticality_options=CRITICALITY_OPTIONS,
    )
    st.markdown("</div>", unsafe_allow_html=True)


main()
