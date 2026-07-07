"""SQLAlchemy models for feedback tickets and audit events."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    DateTime,
    Enum as SqlEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
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


TICKET_CRITICALITY_ENUM = SqlEnum(
    TicketCriticality,
    name="ticket_criticality",
    native_enum=False,
    validate_strings=True,
)
TICKET_STATUS_ENUM = SqlEnum(
    TicketStatus,
    name="ticket_status",
    native_enum=False,
    validate_strings=True,
)


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
    reported_by: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    criticality: Mapped[TicketCriticality] = mapped_column(
        TICKET_CRITICALITY_ENUM,
        nullable=False,
        index=True,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ideal_closure_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[TicketStatus] = mapped_column(
        TICKET_STATUS_ENUM,
        nullable=False,
        default=TicketStatus.OPEN,
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
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False)

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
    old_status: Mapped[TicketStatus | None] = mapped_column(
        TICKET_STATUS_ENUM,
        nullable=True,
    )
    new_status: Mapped[TicketStatus | None] = mapped_column(
        TICKET_STATUS_ENUM,
        nullable=True,
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="events")


class TicketImage(Base):
    """Screenshot metadata linked to one ticket."""

    __tablename__ = "ticket_images"
    __table_args__ = (
        Index("ix_ticket_images_ticket_created_at", "ticket_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    uploaded_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="images")
