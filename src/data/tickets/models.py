"""SQLAlchemy models for feedback tickets and audit events."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base declarative class for ticketing tables."""


class TicketCriticality(str, Enum):
    """Supported ticket criticality levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    """Supported ticket status values."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"
    RESOLVED = "resolved"
    DEPENDENCY_CONFLICT = "dependency_conflict"


class Ticket(Base):
    """Ticket raised from the Feedback page."""

    __tablename__ = "tickets"
    __table_args__ = (
        Index("ix_tickets_status_updated_at", "status", "updated_at"),
        Index("ix_tickets_page_created_at", "page_name", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_code: Mapped[str | None] = mapped_column(
        String(16), unique=True, index=True, nullable=True
    )
    page_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    reported_by: Mapped[str] = mapped_column("reporter_name", Text, index=True, nullable=False)
    reported_by_user_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    criticality: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ideal_closure_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default=TicketStatus.OPEN.value,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        index=True,
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_by_user_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)

    events: Mapped[list["TicketEvent"]] = relationship(
        "TicketEvent",
        back_populates="ticket",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="desc(TicketEvent.created_at)",
    )
    images: Mapped[list["TicketImage"]] = relationship(
        "TicketImage",
        back_populates="ticket",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="desc(TicketImage.created_at)",
    )


class TicketEvent(Base):
    """Audit trail row for ticket lifecycle events."""

    __tablename__ = "ticket_events"
    __table_args__ = (
        Index("ix_ticket_events_ticket_created_at", "ticket_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    old_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor: Mapped[str | None] = mapped_column("actor_name", Text, nullable=True)
    actor_user_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="events")


class TicketImage(Base):
    """Screenshot attachment linked to one ticket."""

    __tablename__ = "ticket_attachments"
    __table_args__ = (
        CheckConstraint("size_bytes > 0", name="ck_ticket_attachment_size_positive"),
        Index("ix_ticket_attachments_ticket_id", "ticket_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False,
    )
    image_path: Mapped[str] = mapped_column("file_path", Text, nullable=False)
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    uploaded_by_user_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    uploaded_by: Mapped[str | None] = mapped_column("uploaded_by_name", Text, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_extension: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True, deferred=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="images")
