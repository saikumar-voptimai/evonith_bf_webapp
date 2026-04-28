"""Feedback and case management page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from domain.ticket_service import (
    CRITICALITY_OPTIONS,
    STATUS_OPTIONS,
    TicketService,
    can_manage_tickets,
)
from ui.page_catalog import get_feedback_page_options


def _get_client_ip() -> str:
    try:
        forwarded_for = st.context.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return st.context.headers.get("REMOTE_ADDR", "unknown")
    except Exception:
        return "unknown"


def _apply_feedback_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.35rem;
        }
        div[data-testid="stForm"] {
            border: 1px solid #d7e0e8;
            border-radius: 8px;
            padding: 1.1rem 1.15rem 1.25rem;
            background: #ffffff;
            box-shadow: 0 10px 24px rgba(16, 24, 40, 0.05);
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d7e0e8;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            background: #f8fafc;
        }
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_dt(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def _filter_tickets(
    tickets: list[dict],
    *,
    page: str,
    statuses: list[str],
    criticalities: list[str],
    search: str,
) -> list[dict]:
    filtered = tickets

    if page != "All":
        filtered = [ticket for ticket in filtered if ticket["page"] == page]

    if statuses:
        status_set = set(statuses)
        filtered = [ticket for ticket in filtered if ticket["status"] in status_set]

    if criticalities:
        criticality_set = set(criticalities)
        filtered = [
            ticket
            for ticket in filtered
            if ticket["criticality"] in criticality_set
        ]

    needle = search.strip().lower()
    if needle:
        fields = (
            "page",
            "reporter_name",
            "submitted_by",
            "criticality",
            "description",
            "ideal_closure",
            "status",
        )
        filtered = [
            ticket
            for ticket in filtered
            if any(needle in str(ticket.get(field, "")).lower() for field in fields)
        ]

    return filtered


def _tickets_to_dataframe(tickets: list[dict]) -> pd.DataFrame:
    rows = []
    for ticket in tickets:
        rows.append(
            {
                "Case": ticket["id"],
                "Status": ticket["status"],
                "Criticality": ticket["criticality"],
                "Page": ticket["page"],
                "Reporter": ticket["reporter_name"],
                "Submitted by": ticket["submitted_by"],
                "Description": ticket["description"],
                "Ideal closure": ticket["ideal_closure"],
                "Created": _format_dt(ticket["created_at"]),
                "Updated": _format_dt(ticket["updated_at"]),
                "Updated by": ticket.get("updated_by") or "",
            }
        )
    return pd.DataFrame(rows)


def _events_to_dataframe(events: list[dict]) -> pd.DataFrame:
    rows = []
    for event in events:
        rows.append(
            {
                "When": _format_dt(event["created_at"]),
                "Actor": event["actor"],
                "From": event.get("old_status") or "",
                "To": event["new_status"],
                "Note": event.get("comment") or "",
            }
        )
    return pd.DataFrame(rows)


def _render_overview(tickets: list[dict]) -> None:
    open_count = sum(ticket["status"] == "open" for ticket in tickets)
    progress_count = sum(ticket["status"] == "in-progress" for ticket in tickets)
    terminal_count = sum(
        ticket["status"] in {"resolved", "closed"} for ticket in tickets
    )
    conflict_count = sum(
        ticket["status"] == "dependency-conflict" for ticket in tickets
    )
    high_count = sum(
        ticket["criticality"] in {"high", "critical"} for ticket in tickets
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Open", open_count)
    k2.metric("In progress", progress_count)
    k3.metric("Resolved/Closed", terminal_count)
    k4.metric("Dependency", conflict_count)
    k5.metric("High/Critical", high_count)


def _render_ticket_form(
    service: TicketService,
    *,
    page_options: list[str],
    username: str,
    ip_address: str,
) -> None:
    st.markdown("### New feedback")

    with st.form("feedback_ticket_form", clear_on_submit=True):
        page = st.selectbox("Page with issue", page_options)
        reporter_name = st.text_input("Your name", value=username)
        criticality = st.selectbox(
            "Issue criticality",
            CRITICALITY_OPTIONS,
            index=1,
            format_func=lambda item: item.replace("-", " ").title(),
        )
        description = st.text_area(
            "Issue description",
            height=145,
            max_chars=4000,
            placeholder="Describe what happened, what you expected, and the impact.",
        )
        ideal_closure = st.text_area(
            "Ideal closure",
            height=95,
            max_chars=2000,
            placeholder="Describe the outcome that would close this case.",
        )

        submitted = st.form_submit_button("Create ticket", type="primary")

    if not submitted:
        return

    try:
        ticket_id = service.create_ticket(
            page=page,
            reporter_name=reporter_name,
            submitted_by=username,
            criticality=criticality,
            description=description,
            ideal_closure=ideal_closure,
            ip_address=ip_address,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not create ticket: {exc}")
        return

    st.session_state["feedback_success_message"] = f"Ticket #{ticket_id} created."
    st.rerun()


def _render_filters(page_options: list[str]) -> tuple[str, list[str], list[str], str]:
    f1, f2, f3, f4 = st.columns([0.24, 0.26, 0.24, 0.26])
    with f1:
        page = st.selectbox("Page", ["All", *page_options], key="ticket_filter_page")
    with f2:
        statuses = st.multiselect(
            "Status",
            STATUS_OPTIONS,
            key="ticket_filter_status",
            format_func=lambda item: item.replace("-", " ").title(),
        )
    with f3:
        criticalities = st.multiselect(
            "Criticality",
            CRITICALITY_OPTIONS,
            key="ticket_filter_criticality",
            format_func=lambda item: item.title(),
        )
    with f4:
        search = st.text_input("Search", key="ticket_filter_search")
    return page, statuses, criticalities, search


def _render_ticket_table(tickets: list[dict]) -> None:
    if not tickets:
        st.info("No tickets match the current filters.")
        return

    df = _tickets_to_dataframe(tickets)
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "Case": st.column_config.NumberColumn("Case", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Criticality": st.column_config.TextColumn("Criticality", width="small"),
            "Page": st.column_config.TextColumn("Page", width="medium"),
            "Description": st.column_config.TextColumn(
                "Description",
                width="large",
            ),
            "Ideal closure": st.column_config.TextColumn(
                "Ideal closure",
                width="large",
            ),
        },
    )


def _render_status_manager(
    service: TicketService,
    *,
    tickets: list[dict],
    username: str,
    role: str | None,
    ip_address: str,
) -> None:
    if not can_manage_tickets(role):
        return

    st.markdown("### Case update")

    if not tickets:
        st.info("No cases are available to update.")
        return

    ticket_by_label = {
        (
            f"#{ticket['id']} | {ticket['page']} | "
            f"{ticket['status']} | {ticket['criticality']}"
        ): ticket
        for ticket in tickets
    }

    c1, c2 = st.columns([0.58, 0.42])
    with c1:
        selected_label = st.selectbox(
            "Case",
            list(ticket_by_label.keys()),
            key="ticket_status_case",
        )
    selected_ticket = ticket_by_label[selected_label]

    with c2:
        current_index = STATUS_OPTIONS.index(selected_ticket["status"])
        new_status = st.selectbox(
            "New status",
            STATUS_OPTIONS,
            index=current_index,
            key="ticket_status_new",
            format_func=lambda item: item.replace("-", " ").title(),
        )

    note = st.text_area(
        "Status note",
        key="ticket_status_note",
        max_chars=1000,
        height=90,
    )

    if st.button("Update case", type="primary", key="ticket_status_submit"):
        try:
            changed = service.update_status(
                ticket_id=selected_ticket["id"],
                status=new_status,
                actor=username,
                actor_role=role,
                comment=note,
                ip_address=ip_address,
            )
        except (ValueError, PermissionError) as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Could not update ticket: {exc}")
            return

        if changed:
            st.session_state["feedback_success_message"] = (
                f"Ticket #{selected_ticket['id']} updated."
            )
            st.rerun()
        else:
            st.info("This ticket already has that status.")

    with st.expander("Audit trail", expanded=False):
        events = service.get_events(selected_ticket["id"])
        if events:
            st.dataframe(
                _events_to_dataframe(events),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No audit events recorded for this ticket.")


if "auth_user" not in st.session_state:
    st.warning("Please login to access this page.")
    st.stop()

_apply_feedback_styles()

service = TicketService()
page_options = get_feedback_page_options()
username = st.session_state.get("auth_user", "")
role = st.session_state.get("role")
ip_address = _get_client_ip()

st.title("Feedback & Cases")

if st.session_state.get("feedback_success_message"):
    st.success(st.session_state.pop("feedback_success_message"))

all_tickets = service.list_tickets()

left, right = st.columns([0.48, 0.52], gap="large")
with left:
    _render_ticket_form(
        service,
        page_options=page_options,
        username=username,
        ip_address=ip_address,
    )
with right:
    st.markdown("### Case overview")
    _render_overview(all_tickets)
    _render_status_manager(
        service,
        tickets=all_tickets,
        username=username,
        role=role,
        ip_address=ip_address,
    )

st.divider()
st.markdown("### All cases")
filter_page, filter_statuses, filter_criticalities, filter_search = _render_filters(
    page_options
)
filtered_tickets = _filter_tickets(
    all_tickets,
    page=filter_page,
    statuses=filter_statuses,
    criticalities=filter_criticalities,
    search=filter_search,
)
_render_ticket_table(filtered_tickets)
