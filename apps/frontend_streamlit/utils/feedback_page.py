"""Shared helpers for the Feedback page UI."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import streamlit as st

from furnace_data.runtime_paths import get_repo_root
from apps.frontend_streamlit.data.tickets import (
    TicketCriticality,
    TicketDeleteRequest,
    TicketEventView,
    TicketImageUpload,
    TicketImageView,
    TicketQueryFilter,
    TicketService,
    TicketStatus,
    TicketStatusUpdateRequest,
    TicketView,
)
from apps.frontend_streamlit.utils.session import is_admin, is_supervisor

PROJECT_ROOT = get_repo_root()


def load_feedback_css(page_file: Path) -> None:
    """Load scoped CSS for Feedback page widgets."""
    css_path = page_file.resolve().parents[1] / "assets" / "css" / "feedback_style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def to_utc_datetime_floor(selected: date | datetime | None) -> datetime | None:
    """Convert date-like value to inclusive UTC start timestamp."""
    if selected is None:
        return None
    if isinstance(selected, datetime):
        return selected if selected.tzinfo else selected.replace(tzinfo=timezone.utc)
    return datetime.combine(selected, datetime.min.time(), tzinfo=timezone.utc)


def to_utc_datetime_ceiling(selected: date | datetime | None) -> datetime | None:
    """Convert date-like value to inclusive UTC end timestamp."""
    if selected is None:
        return None
    if isinstance(selected, datetime):
        return selected if selected.tzinfo else selected.replace(tzinfo=timezone.utc)
    return datetime.combine(selected, datetime.max.time(), tzinfo=timezone.utc)


def render_chip(label: str, css_class: str) -> str:
    """Build HTML for one status/criticality chip."""
    return f"<span class='feedback-chip {css_class}'>{label}</span>"


def build_attachment_payloads(uploaded_files: Sequence[Any]) -> list[TicketImageUpload]:
    """Convert uploaded Streamlit files into service upload DTOs."""
    payloads: list[TicketImageUpload] = []
    for uploaded_file in uploaded_files:
        payloads.append(
            TicketImageUpload(
                filename=uploaded_file.name,
                content=uploaded_file.getvalue(),
            )
        )
    return payloads


def render_overview_kpis(tickets: Sequence[TicketView]) -> None:
    """Render compact KPI cards for feedback case overview."""
    open_count = sum(ticket.status == TicketStatus.OPEN.value for ticket in tickets)
    progress_count = sum(
        ticket.status == TicketStatus.IN_PROGRESS.value for ticket in tickets
    )
    resolved_closed_count = sum(
        ticket.status in {TicketStatus.RESOLVED.value, TicketStatus.CLOSED.value}
        for ticket in tickets
    )
    dependency_count = sum(
        ticket.status == TicketStatus.DEPENDENCY_CONFLICT.value for ticket in tickets
    )
    high_critical_count = sum(
        ticket.criticality
        in {TicketCriticality.HIGH.value, TicketCriticality.CRITICAL.value}
        for ticket in tickets
    )

    st.markdown(
        "<div class='feedback-section-title'>Case Overview</div>",
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Open", open_count)
    k2.metric("In Progress", progress_count)
    k3.metric("Resolved/Closed", resolved_closed_count)
    k4.metric("Dependency", dependency_count)
    k5.metric("High/Critical", high_critical_count)


def build_events_dataframe(events: Sequence[TicketEventView]) -> pd.DataFrame:
    """Build event-history dataframe for table display."""
    return pd.DataFrame(
        {
            "Time": [
                event.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
                for event in events
            ],
            "Type": [event.event_type for event in events],
            "From": [
                TicketService.status_label(event.old_status)
                if event.old_status
                else "-"
                for event in events
            ],
            "To": [
                TicketService.status_label(event.new_status)
                if event.new_status
                else "-"
                for event in events
            ],
            "Actor": [event.actor for event in events],
            "Comment": [event.comment or "-" for event in events],
        }
    )


def render_ticket_images_gallery(images: Sequence[TicketImageView]) -> None:
    """Render thumbnail gallery for ticket screenshots."""
    if not images:
        return

    st.markdown("**Screenshots**")
    cols = st.columns(3)
    for idx, image in enumerate(images):
        image_path = Path(image.image_path)
        if image_path.is_absolute():
            full_path = image_path
        else:
            project_candidate = PROJECT_ROOT / image_path
            full_path = (
                project_candidate
                if project_candidate.exists()
                else Path.cwd() / image_path
            )

        with cols[idx % 3]:
            if not full_path.exists():
                st.caption(f"{image.original_filename} (missing file)")
                continue
            st.image(
                str(full_path),
                caption=image.original_filename,
                width="stretch",
            )


def render_management_panel(
    ticket_service: TicketService,
    tickets: Sequence[TicketView],
    status_options: Sequence[str],
) -> None:
    """Render status update and delete controls for admin/supervisor roles."""
    can_manage = is_admin() or is_supervisor()
    if not can_manage:
        st.info("Status and delete controls are available for admin/supervisor roles.")
        return
    if not tickets:
        st.info("No tickets available for status management yet.")
        return

    actor = str(st.session_state.get("auth_user", "")).strip()
    actor_role = str(st.session_state.get("role", "")).strip().lower()
    st.markdown(
        "<div class='feedback-section-title'>Case Manager</div>",
        unsafe_allow_html=True,
    )

    ticket_lookup = {
        (
            f"{ticket.ticket_code} | {ticket.page_name} | "
            f"{TicketService.status_label(ticket.status)} | "
            f"{TicketService.criticality_label(ticket.criticality)}"
        ): ticket
        for ticket in tickets
    }
    selected_label = st.selectbox("Case", list(ticket_lookup.keys()))
    selected_ticket = ticket_lookup[selected_label]

    with st.form("feedback_status_manager_form"):
        next_status = st.selectbox(
            "New status",
            options=status_options,
            index=list(status_options).index(selected_ticket.status),
            format_func=TicketService.status_label,
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
            ticket_service.update_status(
                TicketStatusUpdateRequest(
                    ticket_id=selected_ticket.id,
                    new_status=next_status,
                    actor=actor,
                    actor_role=actor_role,
                    comment=update_comment or None,
                )
            )
            st.success(
                f"{selected_ticket.ticket_code} updated to "
                f"{TicketService.status_label(next_status)}."
            )
            st.rerun()
        except (PermissionError, ValueError) as exc:
            st.error(str(exc))

    with st.form("feedback_delete_ticket_form"):
        confirm_delete = st.checkbox(
            f"Confirm delete for {selected_ticket.ticket_code}",
            value=False,
        )
        delete_submit = st.form_submit_button("Delete Ticket")

    if delete_submit:
        if not confirm_delete:
            st.error("Please confirm deletion before submitting.")
        else:
            try:
                ticket_service.delete_ticket(
                    TicketDeleteRequest(
                        ticket_id=selected_ticket.id,
                        actor=actor,
                        actor_role=actor_role,
                    )
                )
                st.success(f"{selected_ticket.ticket_code} deleted successfully.")
                st.rerun()
            except (PermissionError, ValueError) as exc:
                st.error(str(exc))

    with st.expander("Selected Case Event History", expanded=False, key="fb_event_history"):
        events = ticket_service.list_events(selected_ticket.id)
        if not events:
            st.info("No events available for this ticket.")
        else:
            st.dataframe(
                build_events_dataframe(events),
                width="stretch",
                hide_index=True,
            )


def _build_ticket_table(tickets: Sequence[TicketView]) -> pd.DataFrame:
    """Build dataframe for the selectable ticket table."""
    return pd.DataFrame(
        [
            {
                "Code": ticket.ticket_code,
                "Page": ticket.page_name,
                "Criticality": TicketService.criticality_label(ticket.criticality),
                "Status": TicketService.status_label(ticket.status),
                "Reporter": ticket.reported_by,
                "Created": ticket.created_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            }
            for ticket in tickets
        ]
    )


def render_board(
    ticket_service: TicketService,
    status_options: Sequence[str],
    criticality_options: Sequence[str],
) -> None:
    """Render ticket board with selectable table and detail panel."""
    st.markdown(
        "<div class='feedback-section-title'>All Cases</div>",
        unsafe_allow_html=True,
    )

    all_tickets = ticket_service.list_tickets()
    reporter_options = sorted({ticket.reported_by for ticket in all_tickets})
    page_options = sorted({ticket.page_name for ticket in all_tickets})
    today = date.today()
    default_from = today - timedelta(days=30)

    with st.expander("Filters", expanded=True, key="fb_filters"):
        f1, f2, f3 = st.columns(3)
        with f1:
            selected_statuses = st.multiselect(
                "Status",
                options=status_options,
                format_func=TicketService.status_label,
            )
            selected_pages = st.multiselect("Page", options=page_options)
        with f2:
            selected_criticalities = st.multiselect(
                "Criticality",
                options=criticality_options,
                format_func=TicketService.criticality_label,
            )
            selected_reporters = st.multiselect("Reporter", options=reporter_options)
        with f3:
            keyword = st.text_input("Keyword", placeholder="code, issue text, user...")
            date_range = st.date_input("Created date range", value=(default_from, today))

    date_from: date | None = None
    date_to: date | None = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_from, date_to = date_range
    elif isinstance(date_range, date):
        date_from = date_range
        date_to = date_range

    tickets = ticket_service.list_tickets(
        TicketQueryFilter(
            statuses=selected_statuses or None,
            criticalities=selected_criticalities or None,
            page_names=selected_pages or None,
            reported_bys=selected_reporters or None,
            date_from=to_utc_datetime_floor(date_from),
            date_to=to_utc_datetime_ceiling(date_to),
            keyword=keyword or None,
        )
    )

    if not tickets:
        st.session_state.pop("fb_selected_ticket_code", None)
        st.info("No tickets found for the current filters.")
        return

    ticket_map = {ticket.ticket_code: ticket for ticket in tickets}
    table_df = _build_ticket_table(tickets)
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
            key="fb_ticket_table",
            height=table_height,
        )

    selected_rows = (selection.selection.get("rows") or []) if selection else []
    selected_code: str | None = None
    if selected_rows:
        selected_row_idx = selected_rows[0]
        if 0 <= selected_row_idx < len(table_df):
            selected_code = str(table_df.iloc[selected_row_idx]["Code"])

    if selected_code:
        st.session_state["fb_selected_ticket_code"] = selected_code
    else:
        remembered_code = st.session_state.get("fb_selected_ticket_code")
        if isinstance(remembered_code, str) and remembered_code in ticket_map:
            selected_code = remembered_code
        elif len(tickets) == 1:
            selected_code = tickets[0].ticket_code
            st.session_state["fb_selected_ticket_code"] = selected_code
        else:
            st.session_state.pop("fb_selected_ticket_code", None)

    selected_ticket = ticket_map.get(selected_code) if selected_code else None

    with right_col:
        if selected_ticket is None:
            st.markdown(
                "<div class='feedback-board-empty'>Select a ticket from the left table to view details.</div>",
                unsafe_allow_html=True,
            )
            return

        ticket = selected_ticket
        can_manage = is_admin() or is_supervisor()
        actor = str(st.session_state.get("auth_user", "")).strip()
        actor_role = str(st.session_state.get("role", "")).strip().lower()

        status_css_class = f"feedback-chip-status-{ticket.status.replace('_', '-')}"
        criticality_css_class = f"feedback-chip-criticality-{ticket.criticality}"

        st.markdown(f"### `{ticket.ticket_code}` - {ticket.page_name}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                render_chip(
                    TicketService.status_label(ticket.status),
                    status_css_class,
                ),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_chip(
                    TicketService.criticality_label(ticket.criticality).lower(),
                    criticality_css_class,
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div class='feedback-meta'><b>Reporter:</b> {ticket.reported_by} | "
            f"<b>Created:</b> {ticket.created_at.astimezone().strftime('%Y-%m-%d %H:%M')} | "
            f"<b>Updated by:</b> {ticket.updated_by} | "
            f"<b>Updated:</b> {ticket.updated_at.astimezone().strftime('%Y-%m-%d %H:%M')}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Issue description**")
        st.write(ticket.description)
        if ticket.ideal_closure_text:
            st.markdown("**Ideal closure**")
            st.write(ticket.ideal_closure_text)

        images = ticket_service.list_ticket_images(ticket.id)
        if images:
            render_ticket_images_gallery(images)

        if can_manage:
            st.markdown("**Update status**")
            with st.form(f"status_update_form_{ticket.id}"):
                next_status = st.selectbox(
                    "New status",
                    options=status_options,
                    index=list(status_options).index(ticket.status),
                    format_func=TicketService.status_label,
                    key=f"status_sel_{ticket.id}",
                )
                update_comment = st.text_area(
                    "Comment",
                    placeholder="Optional note...",
                    max_chars=600,
                )
                if st.form_submit_button("Update Status", type="primary"):
                    try:
                        ticket_service.update_status(
                            TicketStatusUpdateRequest(
                                ticket_id=ticket.id,
                                new_status=next_status,
                                actor=actor,
                                actor_role=actor_role,
                                comment=update_comment or None,
                            )
                        )
                        st.success(
                            f"Updated to {TicketService.status_label(next_status)}."
                        )
                        st.rerun()
                    except (PermissionError, ValueError) as exc:
                        st.error(str(exc))

        events = ticket_service.list_events(ticket.id)
        if events:
            with st.expander("Event history", expanded=False, key=f"fb_events_{ticket.id}"):
                st.dataframe(
                    build_events_dataframe(events),
                    width="stretch",
                    hide_index=True,
                )
