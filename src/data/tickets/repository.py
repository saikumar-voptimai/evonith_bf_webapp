"""Repository layer for ticket CRUD and query operations."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, sessionmaker

from .models import Ticket, TicketCriticality, TicketEvent, TicketStatus, utc_now


def format_ticket_code(ticket_id: int) -> str:
    """Format a numeric ticket ID into a fixed-width public code."""
    return f"TKT-{ticket_id:06d}"


class TicketRepository:
    """Persistence operations for tickets and ticket events."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create repository with an injected SQLAlchemy session factory."""
        self._session_factory = session_factory

    def create_ticket(
        self,
        *,
        page_name: str,
        reported_by: str,
        criticality: TicketCriticality,
        description: str,
        ideal_closure_text: str,
        created_by: str,
        initial_status: TicketStatus = TicketStatus.OPEN,
    ) -> Ticket:
        """Insert a ticket row plus its initial creation event."""
        with self._session_factory() as session:
            ticket = Ticket(
                page_name=page_name,
                reported_by=reported_by,
                criticality=criticality,
                description=description,
                ideal_closure_text=ideal_closure_text,
                status=initial_status,
                created_by=created_by,
                updated_by=created_by,
            )
            session.add(ticket)
            session.flush()

            ticket.ticket_code = format_ticket_code(ticket.id)
            self._add_event(
                session=session,
                ticket_id=ticket.id,
                event_type="created",
                old_status=None,
                new_status=ticket.status,
                comment="Ticket created",
                actor=created_by,
            )

            session.commit()
            session.refresh(ticket)
            session.expunge(ticket)
            return ticket

    def get_ticket(self, ticket_id: int) -> Ticket | None:
        """Fetch one ticket by primary key."""
        with self._session_factory() as session:
            ticket = session.get(Ticket, ticket_id)
            if ticket is None:
                return None
            session.expunge(ticket)
            return ticket

    def list_tickets(
        self,
        *,
        statuses: list[TicketStatus] | None = None,
        criticalities: list[TicketCriticality] | None = None,
        page_names: list[str] | None = None,
        reported_bys: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        keyword: str | None = None,
    ) -> list[Ticket]:
        """Return filtered ticket rows sorted by latest update."""
        with self._session_factory() as session:
            query = select(Ticket)

            if statuses:
                query = query.where(Ticket.status.in_(statuses))
            if criticalities:
                query = query.where(Ticket.criticality.in_(criticalities))
            if page_names:
                query = query.where(Ticket.page_name.in_(page_names))
            if reported_bys:
                query = query.where(Ticket.reported_by.in_(reported_bys))
            if date_from:
                query = query.where(Ticket.created_at >= date_from)
            if date_to:
                query = query.where(Ticket.created_at <= date_to)
            if keyword:
                search_token = f"%{keyword.strip().lower()}%"
                query = query.where(
                    or_(
                        Ticket.ticket_code.ilike(search_token),
                        Ticket.page_name.ilike(search_token),
                        Ticket.reported_by.ilike(search_token),
                        Ticket.description.ilike(search_token),
                        Ticket.ideal_closure_text.ilike(search_token),
                    )
                )

            query = query.order_by(Ticket.updated_at.desc(), Ticket.id.desc())
            tickets = list(session.execute(query).scalars().all())
            for ticket in tickets:
                session.expunge(ticket)
            return tickets

    def update_status(
        self,
        *,
        ticket_id: int,
        new_status: TicketStatus,
        actor: str,
        comment: str | None = None,
    ) -> Ticket:
        """Update ticket status and append a status event."""
        with self._session_factory() as session:
            ticket = session.get(Ticket, ticket_id)
            if ticket is None:
                raise ValueError(f"Ticket {ticket_id} not found.")

            old_status = ticket.status
            ticket.status = new_status
            ticket.updated_by = actor
            ticket.updated_at = utc_now()

            self._add_event(
                session=session,
                ticket_id=ticket.id,
                event_type="status_update",
                old_status=old_status,
                new_status=new_status,
                comment=comment,
                actor=actor,
            )

            session.commit()
            session.refresh(ticket)
            session.expunge(ticket)
            return ticket

    def list_events(self, ticket_id: int) -> list[TicketEvent]:
        """List ticket events newest-first."""
        with self._session_factory() as session:
            query = (
                select(TicketEvent)
                .where(TicketEvent.ticket_id == ticket_id)
                .order_by(TicketEvent.created_at.desc(), TicketEvent.id.desc())
            )
            events = list(session.execute(query).scalars().all())
            for event in events:
                session.expunge(event)
            return events

    @staticmethod
    def _add_event(
        *,
        session: Session,
        ticket_id: int,
        event_type: str,
        old_status: TicketStatus | None,
        new_status: TicketStatus | None,
        comment: str | None,
        actor: str,
    ) -> None:
        """Insert one ticket event row inside an active transaction."""
        session.add(
            TicketEvent(
                ticket_id=ticket_id,
                event_type=event_type,
                old_status=old_status,
                new_status=new_status,
                comment=comment,
                actor=actor,
            )
        )
