"""Repository layer for ticket CRUD and query operations."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence
from uuid import UUID

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from .models import (
    Ticket,
    TicketCriticality,
    TicketEvent,
    TicketImage,
    TicketStatus,
    utc_now,
)


def format_ticket_code(ticket_id: int) -> str:
    """Format a numeric ticket ID into a fixed-width public code."""
    return f"TKT-{ticket_id:06d}"


def _coerce_uuid(value: UUID | str | None) -> UUID | None:
    """Return UUID object for DB writes, or None when unavailable."""
    if value is None or isinstance(value, UUID):
        return value
    value = str(value).strip()
    if not value:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


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
        reported_by_user_id: UUID | str | None = None,
        created_by_user_id: UUID | str | None = None,
        initial_status: TicketStatus = TicketStatus.OPEN,
    ) -> Ticket:
        """Insert a ticket row plus its initial creation event."""
        with self._session_factory() as session:
            reporter_uuid = _coerce_uuid(reported_by_user_id)
            creator_uuid = _coerce_uuid(created_by_user_id) or reporter_uuid
            ticket = Ticket(
                page_name=page_name,
                reported_by=reported_by,
                reported_by_user_id=reporter_uuid,
                criticality=criticality.value,
                description=description,
                ideal_closure_text=ideal_closure_text,
                status=initial_status.value,
                updated_by_user_id=creator_uuid,
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
                actor_user_id=creator_uuid,
            )

            session.commit()
            session.refresh(ticket)
            ticket._updated_by_name = created_by
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
            query = select(Ticket).options(selectinload(Ticket.events))

            if statuses:
                query = query.where(Ticket.status.in_([status.value for status in statuses]))
            if criticalities:
                query = query.where(Ticket.criticality.in_([item.value for item in criticalities]))
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
        actor_user_id: UUID | str | None = None,
        comment: str | None = None,
    ) -> Ticket:
        """Update ticket status and append a status event."""
        with self._session_factory() as session:
            ticket = session.get(Ticket, ticket_id)
            if ticket is None:
                raise ValueError(f"Ticket {ticket_id} not found.")

            old_status = ticket.status
            ticket.status = new_status.value
            ticket.updated_by_user_id = _coerce_uuid(actor_user_id)
            ticket.updated_at = utc_now()

            self._add_event(
                session=session,
                ticket_id=ticket.id,
                event_type="status_update",
                old_status=old_status,
                new_status=new_status,
                comment=comment,
                actor=actor,
                actor_user_id=actor_user_id,
            )

            session.commit()
            session.refresh(ticket)
            ticket._updated_by_name = actor
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

    def add_images(
        self,
        *,
        ticket_id: int,
        image_rows: Sequence[dict[str, object]],
    ) -> list[TicketImage]:
        """Insert image metadata rows for one ticket.

        Args:
            ticket_id: Primary key of the ticket.
            image_rows: Attachment metadata and byte payload dictionaries.
        """
        if not image_rows:
            return []

        with self._session_factory() as session:
            ticket = session.get(Ticket, ticket_id)
            if ticket is None:
                raise ValueError(f"Ticket {ticket_id} not found.")

            images: list[TicketImage] = []
            for row in image_rows:
                image = TicketImage(
                    ticket_id=ticket_id,
                    image_path=str(row["image_path"]),
                    original_filename=str(row["original_filename"]),
                    uploaded_by_user_id=_coerce_uuid(row.get("uploaded_by_user_id")),
                    uploaded_by=str(row["uploaded_by"]),
                    mime_type=str(row["mime_type"]),
                    file_extension=str(row["file_extension"]),
                    size_bytes=int(row["size_bytes"]),
                    content_sha256=str(row["content_sha256"]),
                    image_bytes=row["image_bytes"],
                    metadata_json=row.get("metadata") or {},
                )
                session.add(image)
                images.append(image)

            self._add_event(
                session=session,
                ticket_id=ticket_id,
                event_type="attachment_added",
                old_status=None,
                new_status=ticket.status,
                comment=f"{len(images)} screenshot(s) attached",
                actor=ticket.reported_by,
                actor_user_id=ticket.reported_by_user_id,
            )
            session.commit()

            for image in images:
                session.refresh(image)
                session.expunge(image)
            return images

    def get_image_content(self, image_id: int) -> tuple[bytes, str, str] | None:
        """Return one attachment's bytes, MIME type, and filename."""
        with self._session_factory() as session:
            row = session.execute(
                select(
                    TicketImage.image_bytes,
                    TicketImage.mime_type,
                    TicketImage.original_filename,
                ).where(TicketImage.id == image_id)
            ).one_or_none()
            if row is None or row[0] is None:
                return None
            return bytes(row[0]), str(row[1] or "application/octet-stream"), str(row[2])

    def list_images(self, ticket_id: int) -> list[TicketImage]:
        """List ticket image metadata newest-first."""
        with self._session_factory() as session:
            query = (
                select(TicketImage)
                .where(TicketImage.ticket_id == ticket_id)
                .order_by(TicketImage.created_at.desc(), TicketImage.id.desc())
            )
            images = list(session.execute(query).scalars().all())
            for image in images:
                session.expunge(image)
            return images

    def delete_ticket(self, ticket_id: int) -> None:
        """Delete a ticket and rely on database cascade/transaction cleanup."""
        with self._session_factory() as session:
            ticket = session.get(Ticket, ticket_id)
            if ticket is None:
                raise ValueError(f"Ticket {ticket_id} not found.")

            session.execute(delete(TicketImage).where(TicketImage.ticket_id == ticket_id))
            session.execute(delete(TicketEvent).where(TicketEvent.ticket_id == ticket_id))
            session.delete(ticket)
            session.commit()

    @staticmethod
    def _add_event(
        *,
        session: Session,
        ticket_id: int,
        event_type: str,
        old_status: TicketStatus | str | None,
        new_status: TicketStatus | str | None,
        comment: str | None,
        actor: str,
        actor_user_id: UUID | str | None = None,
    ) -> None:
        """Insert one ticket event row inside an active transaction."""
        session.add(
            TicketEvent(
                ticket_id=ticket_id,
                event_type=event_type,
                old_status=old_status.value if isinstance(old_status, TicketStatus) else old_status,
                new_status=new_status.value if isinstance(new_status, TicketStatus) else new_status,
                comment=comment,
                actor=actor,
                actor_user_id=_coerce_uuid(actor_user_id),
            )
        )
