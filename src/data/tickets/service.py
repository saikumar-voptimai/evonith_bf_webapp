"""Business service layer for feedback tickets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone

from sqlalchemy.engine import Engine

from .engine import build_tickets_engine, build_tickets_session_factory
from .models import Base, Ticket, TicketCriticality, TicketEvent, TicketStatus
from .repository import TicketRepository

ALLOWED_STATUS_EDIT_ROLES = frozenset({"admin", "supervisor"})


@dataclass(frozen=True)
class TicketCreateRequest:
    """Input payload for creating a new ticket."""

    page_name: str
    reported_by: str
    criticality: str | TicketCriticality
    description: str
    ideal_closure_text: str
    created_by: str | None = None


@dataclass(frozen=True)
class TicketStatusUpdateRequest:
    """Input payload for updating a ticket status."""

    ticket_id: int
    new_status: str | TicketStatus
    actor: str
    actor_role: str
    comment: str | None = None


@dataclass(frozen=True)
class TicketQueryFilter:
    """Filter options for listing tickets."""

    statuses: list[str | TicketStatus] | None = None
    criticalities: list[str | TicketCriticality] | None = None
    page_names: list[str] | None = None
    reported_bys: list[str] | None = None
    date_from: date | datetime | None = None
    date_to: date | datetime | None = None
    keyword: str | None = None


@dataclass(frozen=True)
class TicketView:
    """Read model exposed to the UI layer."""

    id: int
    ticket_code: str
    page_name: str
    reported_by: str
    criticality: str
    description: str
    ideal_closure_text: str
    status: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str


@dataclass(frozen=True)
class TicketEventView:
    """Read model for one ticket audit event."""

    id: int
    ticket_id: int
    event_type: str
    old_status: str | None
    new_status: str | None
    comment: str | None
    actor: str
    created_at: datetime


class TicketService:
    """Facade for ticketing operations and business policy checks."""

    def __init__(self, db_url: str | None = None) -> None:
        """Initialise the service with its own dedicated SQLAlchemy engine."""
        self._engine: Engine = build_tickets_engine(db_url=db_url)
        session_factory = build_tickets_session_factory(self._engine)
        self._repository = TicketRepository(session_factory=session_factory)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        """Create ticket tables if they do not already exist."""
        Base.metadata.create_all(bind=self._engine)

    def create_ticket(self, request: TicketCreateRequest) -> TicketView:
        """Create a ticket with default ``open`` status and audit event."""
        page_name = request.page_name.strip()
        reported_by = request.reported_by.strip()
        description = request.description.strip()
        ideal_closure_text = request.ideal_closure_text.strip()
        actor = (request.created_by or request.reported_by).strip()

        if not page_name:
            raise ValueError("Page name is required.")
        if not reported_by:
            raise ValueError("Reported by is required.")
        if not description:
            raise ValueError("Issue description is required.")
        if not ideal_closure_text:
            raise ValueError("Ideal closure text is required.")
        if not actor:
            raise ValueError("Creator identity is required.")

        criticality = self._coerce_criticality(request.criticality)
        ticket = self._repository.create_ticket(
            page_name=page_name,
            reported_by=reported_by,
            criticality=criticality,
            description=description,
            ideal_closure_text=ideal_closure_text,
            created_by=actor,
            initial_status=TicketStatus.OPEN,
        )
        return self._to_ticket_view(ticket)

    def list_tickets(self, query_filter: TicketQueryFilter | None = None) -> list[TicketView]:
        """List tickets with optional filters."""
        query_filter = query_filter or TicketQueryFilter()
        statuses = (
            [self._coerce_status(status) for status in query_filter.statuses]
            if query_filter.statuses
            else None
        )
        criticalities = (
            [
                self._coerce_criticality(criticality)
                for criticality in query_filter.criticalities
            ]
            if query_filter.criticalities
            else None
        )
        date_from = self._coerce_datetime_floor(query_filter.date_from)
        date_to = self._coerce_datetime_ceiling(query_filter.date_to)

        tickets = self._repository.list_tickets(
            statuses=statuses,
            criticalities=criticalities,
            page_names=query_filter.page_names,
            reported_bys=query_filter.reported_bys,
            date_from=date_from,
            date_to=date_to,
            keyword=query_filter.keyword,
        )
        return [self._to_ticket_view(ticket) for ticket in tickets]

    def update_status(self, request: TicketStatusUpdateRequest) -> TicketView:
        """Update ticket status if actor role is authorized."""
        actor_role = request.actor_role.strip().lower()
        if actor_role not in ALLOWED_STATUS_EDIT_ROLES:
            raise PermissionError(
                "Only admin and supervisor roles are allowed to update ticket status."
            )
        actor = request.actor.strip()
        if not actor:
            raise ValueError("Actor is required for status updates.")

        new_status = self._coerce_status(request.new_status)
        ticket = self._repository.update_status(
            ticket_id=request.ticket_id,
            new_status=new_status,
            actor=actor,
            comment=request.comment.strip() if request.comment else None,
        )
        return self._to_ticket_view(ticket)

    def list_events(self, ticket_id: int) -> list[TicketEventView]:
        """List audit events for a ticket."""
        events = self._repository.list_events(ticket_id=ticket_id)
        return [self._to_ticket_event_view(event) for event in events]

    @staticmethod
    def status_label(status: str | TicketStatus) -> str:
        """Render stored snake_case status as a UI-friendly label."""
        value = status.value if isinstance(status, TicketStatus) else status
        return value.replace("_", "-")

    @staticmethod
    def criticality_label(criticality: str | TicketCriticality) -> str:
        """Render criticality value as a title-cased display label."""
        value = (
            criticality.value if isinstance(criticality, TicketCriticality) else criticality
        )
        return value.strip().replace("_", " ").title()

    @staticmethod
    def _coerce_status(value: str | TicketStatus) -> TicketStatus:
        """Convert a status string/enum into :class:`TicketStatus`."""
        if isinstance(value, TicketStatus):
            return value
        try:
            return TicketStatus(value.strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported ticket status: {value}") from exc

    @staticmethod
    def _coerce_criticality(value: str | TicketCriticality) -> TicketCriticality:
        """Convert a criticality string/enum into :class:`TicketCriticality`."""
        if isinstance(value, TicketCriticality):
            return value
        try:
            return TicketCriticality(value.strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported ticket criticality: {value}") from exc

    @staticmethod
    def _coerce_datetime_floor(value: date | datetime | None) -> datetime | None:
        """Convert date-like filters to an inclusive start timestamp in UTC."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return datetime.combine(value, time.min, tzinfo=timezone.utc)

    @staticmethod
    def _coerce_datetime_ceiling(value: date | datetime | None) -> datetime | None:
        """Convert date-like filters to an inclusive end timestamp in UTC."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return datetime.combine(value, time.max, tzinfo=timezone.utc)

    @staticmethod
    def _to_ticket_view(ticket: Ticket) -> TicketView:
        """Map ORM ticket to immutable read model."""
        return TicketView(
            id=ticket.id,
            ticket_code=ticket.ticket_code or "",
            page_name=ticket.page_name,
            reported_by=ticket.reported_by,
            criticality=ticket.criticality.value,
            description=ticket.description,
            ideal_closure_text=ticket.ideal_closure_text,
            status=ticket.status.value,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
            created_by=ticket.created_by,
            updated_by=ticket.updated_by,
        )

    @staticmethod
    def _to_ticket_event_view(event: TicketEvent) -> TicketEventView:
        """Map ORM event to immutable read model."""
        return TicketEventView(
            id=event.id,
            ticket_id=event.ticket_id,
            event_type=event.event_type,
            old_status=event.old_status.value if event.old_status else None,
            new_status=event.new_status.value if event.new_status else None,
            comment=event.comment,
            actor=event.actor,
            created_at=event.created_at,
        )
