"""Business service layer for feedback tickets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
import mimetypes
from pathlib import Path
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.config_loader import load_config
from .engine import build_tickets_engine, build_tickets_session_factory
from .models import (
    Base,
    Ticket,
    TicketCriticality,
    TicketEvent,
    TicketImage,
    TicketStatus,
)
from .repository import TicketRepository

ALLOWED_STATUS_EDIT_ROLES = frozenset({"admin", "supervisor"})
ALLOWED_DELETE_ROLES = frozenset({"admin", "supervisor"})
ALLOWED_IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp"})
MAX_ATTACHMENTS_PER_TICKET = 5
MAX_ATTACHMENT_SIZE_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class TicketImageUpload:
    """One uploaded screenshot payload."""

    filename: str
    content: bytes


@dataclass(frozen=True)
class TicketCreateRequest:
    """Input payload for creating a new ticket."""

    page_name: str
    reported_by: str
    criticality: str | TicketCriticality
    description: str
    ideal_closure_text: str
    created_by: str | None = None
    reported_by_user_id: str | UUID | None = None
    created_by_user_id: str | UUID | None = None


@dataclass(frozen=True)
class TicketStatusUpdateRequest:
    """Input payload for updating a ticket status."""

    ticket_id: int
    new_status: str | TicketStatus
    actor: str
    actor_role: str
    actor_user_id: str | UUID | None = None
    comment: str | None = None


@dataclass(frozen=True)
class TicketDeleteRequest:
    """Input payload for deleting a ticket."""

    ticket_id: int
    actor: str
    actor_role: str


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
    reported_by_user_id: str | None
    criticality: str
    description: str
    ideal_closure_text: str
    status: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    updated_by_user_id: str | None


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
    actor_user_id: str | None
    created_at: datetime


@dataclass(frozen=True)
class TicketImageView:
    """Read model for one ticket screenshot entry."""

    id: int
    ticket_id: int
    original_filename: str
    uploaded_by: str
    uploaded_by_user_id: str | None
    mime_type: str
    size_bytes: int
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
        self._ensure_attachment_columns()

    def create_ticket(
        self,
        request: TicketCreateRequest,
        attachments: list[TicketImageUpload] | None = None,
    ) -> TicketView:
        """Create a ticket and optionally persist screenshot attachments."""
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

        validated_attachments = self._validate_attachments(attachments or [])
        criticality = self._coerce_criticality(request.criticality)
        ticket = self._repository.create_ticket(
            page_name=page_name,
            reported_by=reported_by,
            criticality=criticality,
            description=description,
            ideal_closure_text=ideal_closure_text,
            created_by=actor,
            reported_by_user_id=request.reported_by_user_id,
            created_by_user_id=request.created_by_user_id or request.reported_by_user_id,
            initial_status=TicketStatus.OPEN,
        )
        ticket_view = self._to_ticket_view(ticket)
        if validated_attachments:
            self._persist_attachments(
                ticket=ticket_view,
                uploaded_by=actor,
                uploaded_by_user_id=request.created_by_user_id or request.reported_by_user_id,
                attachments=validated_attachments,
            )
        return ticket_view

    def list_tickets(
        self, query_filter: TicketQueryFilter | None = None
    ) -> list[TicketView]:
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
            actor_user_id=request.actor_user_id,
            comment=request.comment.strip() if request.comment else None,
        )
        return self._to_ticket_view(ticket)

    def delete_ticket(self, request: TicketDeleteRequest) -> None:
        """Delete a ticket and linked database attachments."""
        actor_role = request.actor_role.strip().lower()
        if actor_role not in ALLOWED_DELETE_ROLES:
            raise PermissionError(
                "Only admin and supervisor roles are allowed to delete tickets."
            )
        actor = request.actor.strip()
        if not actor:
            raise ValueError("Actor is required for ticket deletion.")

        self._repository.delete_ticket(ticket_id=request.ticket_id)

    def list_events(self, ticket_id: int) -> list[TicketEventView]:
        """List audit events for a ticket."""
        events = self._repository.list_events(ticket_id=ticket_id)
        return [self._to_ticket_event_view(event) for event in events]

    def list_ticket_images(self, ticket_id: int) -> list[TicketImageView]:
        """List screenshot entries for one ticket."""
        images = self._repository.list_images(ticket_id=ticket_id)
        return [self._to_ticket_image_view(image) for image in images]

    def get_ticket_image_content(self, image_id: int) -> tuple[bytes, str, str] | None:
        """Fetch attachment bytes only when the detail panel renders them."""
        return self._repository.get_image_content(image_id=image_id)

    @staticmethod
    def status_label(status: str | TicketStatus) -> str:
        """Render stored snake_case status as a UI-friendly label."""
        value = status.value if isinstance(status, TicketStatus) else status
        return value.replace("_", "-")

    @staticmethod
    def criticality_label(criticality: str | TicketCriticality) -> str:
        """Render criticality value as a title-cased display label."""
        value = (
            criticality.value
            if isinstance(criticality, TicketCriticality)
            else criticality
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
    def _resolve_upload_extension(filename: str) -> str:
        """Return lower-case extension without leading dot."""
        return Path(filename).suffix.lower().lstrip(".")

    def _validate_attachments(
        self,
        attachments: list[TicketImageUpload],
    ) -> list[TicketImageUpload]:
        """Validate image upload constraints and normalize names/content."""
        max_files, max_size_bytes, allowed_extensions = self._attachment_limits()
        if len(attachments) > max_files:
            raise ValueError(
                f"Maximum {max_files} screenshots are allowed per ticket."
            )

        validated: list[TicketImageUpload] = []
        for attachment in attachments:
            filename = attachment.filename.strip()
            if not filename:
                raise ValueError("Attachment filename is required.")

            extension = self._resolve_upload_extension(filename)
            if extension not in allowed_extensions:
                allowed = ", ".join(sorted(allowed_extensions))
                raise ValueError(
                    f"Unsupported image format for '{filename}'. Allowed formats: {allowed}."
                )

            size_bytes = len(attachment.content)
            if size_bytes > max_size_bytes:
                max_size_mb = max_size_bytes // (1024 * 1024)
                raise ValueError(
                    f"File '{filename}' exceeds the {max_size_mb} MB size limit per screenshot."
                )
            if size_bytes <= 0:
                raise ValueError(f"File '{filename}' is empty.")

            validated.append(
                TicketImageUpload(filename=filename, content=attachment.content)
            )

        return validated

    def _persist_attachments(
        self,
        *,
        ticket: TicketView,
        uploaded_by: str,
        uploaded_by_user_id: str | UUID | None,
        attachments: list[TicketImageUpload],
    ) -> list[TicketImageView]:
        """Persist screenshot bytes and metadata in the ticket attachment table."""
        image_rows: list[dict[str, object]] = []

        for attachment in attachments:
            extension = self._resolve_upload_extension(attachment.filename)
            content_sha256 = hashlib.sha256(attachment.content).hexdigest()
            mime_type = (
                mimetypes.guess_type(attachment.filename)[0]
                or f"image/{'jpeg' if extension == 'jpg' else extension}"
            )
            image_rows.append(
                {
                    "image_path": f"db://feedback/ticket_attachments/{content_sha256}",
                    "original_filename": attachment.filename,
                    "uploaded_by": uploaded_by,
                    "uploaded_by_user_id": uploaded_by_user_id,
                    "mime_type": mime_type,
                    "file_extension": extension,
                    "size_bytes": len(attachment.content),
                    "content_sha256": content_sha256,
                    "image_bytes": attachment.content,
                    "metadata": {},
                }
            )

        images = self._repository.add_images(ticket_id=ticket.id, image_rows=image_rows)
        return [self._to_ticket_image_view(image) for image in images]

    @staticmethod
    def _attachment_limits() -> tuple[int, int, frozenset[str]]:
        """Return attachment limits from config with conservative defaults."""
        config = (
            load_config("setting_ds_dv.yml")
            .get("feedback", {})
            .get("attachments", {})
            or {}
        )
        max_files = int(config.get("max_files_per_ticket", MAX_ATTACHMENTS_PER_TICKET))
        max_mb = int(config.get("max_file_size_mb", 5))
        allowed = frozenset(
            str(ext).strip().lower().lstrip(".")
            for ext in config.get("allowed_extensions", ALLOWED_IMAGE_EXTENSIONS)
        )
        return max_files, max_mb * 1024 * 1024, allowed

    def _ensure_attachment_columns(self) -> None:
        """Add DB-backed attachment columns when an older table already exists."""
        if self._engine.dialect.name != "postgresql":
            return
        ddl = [
            """
            DO $$
            DECLARE
                seq_name text;
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'feedback'
                      AND table_name = 'tickets'
                      AND column_name = 'id'
                      AND is_identity = 'NO'
                      AND column_default IS NULL
                ) THEN
                    ALTER TABLE feedback.tickets ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY;
                END IF;
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'feedback'
                      AND table_name = 'ticket_events'
                      AND column_name = 'id'
                      AND is_identity = 'NO'
                      AND column_default IS NULL
                ) THEN
                    ALTER TABLE feedback.ticket_events ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY;
                END IF;
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'feedback'
                      AND table_name = 'ticket_attachments'
                      AND column_name = 'id'
                      AND is_identity = 'NO'
                      AND column_default IS NULL
                ) THEN
                    ALTER TABLE feedback.ticket_attachments ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY;
                END IF;

                seq_name := pg_get_serial_sequence('feedback.tickets', 'id');
                IF seq_name IS NOT NULL THEN
                    PERFORM setval(seq_name, GREATEST((SELECT COALESCE(MAX(id), 0) + 1 FROM feedback.tickets), 1), false);
                END IF;

                seq_name := pg_get_serial_sequence('feedback.ticket_events', 'id');
                IF seq_name IS NOT NULL THEN
                    PERFORM setval(seq_name, GREATEST((SELECT COALESCE(MAX(id), 0) + 1 FROM feedback.ticket_events), 1), false);
                END IF;

                seq_name := pg_get_serial_sequence('feedback.ticket_attachments', 'id');
                IF seq_name IS NOT NULL THEN
                    PERFORM setval(seq_name, GREATEST((SELECT COALESCE(MAX(id), 0) + 1 FROM feedback.ticket_attachments), 1), false);
                END IF;
            END $$;
            """,
            "ALTER TABLE feedback.ticket_attachments ADD COLUMN IF NOT EXISTS mime_type text",
            "ALTER TABLE feedback.ticket_attachments ADD COLUMN IF NOT EXISTS file_extension text",
            "ALTER TABLE feedback.ticket_attachments ADD COLUMN IF NOT EXISTS size_bytes integer",
            "ALTER TABLE feedback.ticket_attachments ADD COLUMN IF NOT EXISTS content_sha256 text",
            "ALTER TABLE feedback.ticket_attachments ADD COLUMN IF NOT EXISTS image_bytes bytea",
            'ALTER TABLE feedback.ticket_attachments ADD COLUMN IF NOT EXISTS "metadata" jsonb',
        ]
        with self._engine.begin() as conn:
            for statement in ddl:
                conn.execute(text(statement))

    @staticmethod
    def _to_ticket_view(ticket: Ticket) -> TicketView:
        """Map ORM ticket to immutable read model."""
        latest_actor = getattr(ticket, "_updated_by_name", None)
        if latest_actor is None:
            events = ticket.__dict__.get("events") or []
            latest_actor = next((event.actor for event in events if event.actor), None)
        return TicketView(
            id=ticket.id,
            ticket_code=ticket.ticket_code or "",
            page_name=ticket.page_name,
            reported_by=ticket.reported_by,
            reported_by_user_id=str(ticket.reported_by_user_id) if ticket.reported_by_user_id else None,
            criticality=str(ticket.criticality),
            description=ticket.description,
            ideal_closure_text=ticket.ideal_closure_text,
            status=str(ticket.status),
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
            created_by=ticket.reported_by,
            updated_by=latest_actor or ticket.reported_by,
            updated_by_user_id=str(ticket.updated_by_user_id) if ticket.updated_by_user_id else None,
        )

    @staticmethod
    def _to_ticket_event_view(event: TicketEvent) -> TicketEventView:
        """Map ORM event to immutable read model."""
        return TicketEventView(
            id=event.id,
            ticket_id=event.ticket_id,
            event_type=event.event_type,
            old_status=str(event.old_status) if event.old_status else None,
            new_status=str(event.new_status) if event.new_status else None,
            comment=event.comment,
            actor=event.actor or "",
            actor_user_id=str(event.actor_user_id) if event.actor_user_id else None,
            created_at=event.created_at,
        )

    @staticmethod
    def _to_ticket_image_view(image: TicketImage) -> TicketImageView:
        """Map ORM image metadata row to immutable read model."""
        return TicketImageView(
            id=image.id,
            ticket_id=image.ticket_id,
            original_filename=image.original_filename,
            uploaded_by=image.uploaded_by or "",
            uploaded_by_user_id=str(image.uploaded_by_user_id) if image.uploaded_by_user_id else None,
            mime_type=image.mime_type or "application/octet-stream",
            size_bytes=int(image.size_bytes or 0),
            created_at=image.created_at,
        )
