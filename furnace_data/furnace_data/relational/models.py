"""SQLAlchemy 2.0 ORM models for BF2 operational relational tables."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SqlEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base declarative class for relational ORM models."""


class UserRole(str, Enum):
    """Supported user roles for dashboard authorization."""

    ADMIN = "admin"
    SUPERVISOR = "supervisor"
    USER = "user"


USER_ROLE_ENUM = SqlEnum(
    UserRole,
    name="user_role",
    native_enum=False,
    values_callable=lambda enum: [e.value for e in enum],
)


class User(Base):
    """User credentials and role mapping."""

    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(128), primary_key=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[UserRole] = mapped_column(USER_ROLE_ENUM, nullable=False)


class HopperMaterialHistory(Base):
    """SCD Type-2 history for hopper to material assignments."""

    __tablename__ = "hopper_material_history"
    __table_args__ = (
        Index("ix_hopper_history_hopper_valid_from", "hopper", "valid_from"),
        Index("ix_hopper_history_hopper_valid_upto", "hopper", "valid_upto"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    hopper: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    material: Mapped[str] = mapped_column(String(256), nullable=False)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    valid_upto: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    modifier: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    ip_address: Mapped[str | None] = mapped_column(String(256), nullable=True)


class BurdenDistributionHistory(Base):
    """SCD Type-2 history for burden distribution fields."""

    __tablename__ = "burden_distribution_history"
    __table_args__ = (
        UniqueConstraint(
            "field_name",
            "valid_upto",
            name="uq_burden_active_record",
        ),
        Index("ix_burden_history_field_valid_from", "field_name", "valid_from"),
        Index("ix_burden_history_field_valid_upto", "field_name", "valid_upto"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    field_name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    field_value_float: Mapped[float | None] = mapped_column(Float, nullable=True)
    field_value_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    valid_upto: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    modifier: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    ip_address: Mapped[str | None] = mapped_column(String(256), nullable=True)


class Conversation(Base):
    """Persisted FurnaceMind chat conversation."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )

    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="medium")
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class ConversationMessage(Base):
    """One persisted FurnaceMind user, assistant, or tool message."""

    __tablename__ = "conversation_messages"
    __table_args__ = (
        Index("ix_conversation_messages_conversation_seq", "conversation_id", "sequence_num"),
    )

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_num: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class MemoryDocument(Base):
    """Uploaded knowledge document indexed for FurnaceMind retrieval."""

    __tablename__ = "memory_documents"
    __table_args__ = (
        Index("ix_memory_documents_user_active", "user_id", "is_active"),
    )

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    qdrant_collection: Mapped[str | None] = mapped_column(String(128), nullable=True)
    qdrant_point_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    token_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class MemorySummary(Base):
    """Compressed long-running conversation memory for FurnaceMind."""

    __tablename__ = "memory_summaries"
    __table_args__ = (
        Index("ix_memory_summaries_conversation", "conversation_id", "created_at"),
    )

    summary_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_message_id_start: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_message_id_end: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class LongTermMemory(Base):
    """Durable user memory extracted from FurnaceMind chat turns."""

    __tablename__ = "long_term_memories"
    __table_args__ = (
        Index("ix_long_term_memories_user", "user_id"),
    )

    memory_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    memory_text: Mapped[str] = mapped_column(Text, nullable=False)
    qdrant_collection: Mapped[str | None] = mapped_column(String(128), nullable=True)
    qdrant_point_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_conversation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_user_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_assistant_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class Skill(Base):
    """Built-in or uploaded FurnaceMind skill definition."""

    __tablename__ = "skills"
    __table_args__ = (
        Index("ix_skills_active", "is_active"),
    )

    skill_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="custom")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class FeedbackItem(Base):
    """Feedback attached to a FurnaceMind assistant message."""

    __tablename__ = "feedback_items"
    __table_args__ = (
        Index("ix_feedback_items_message", "message_id"),
        Index("ix_feedback_items_lesson_queue", "lesson_extracted", "created_at"),
        UniqueConstraint(
            "message_id",
            "user_id",
            "source",
            name="uq_feedback_items_message_user",
        ),
    )

    feedback_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(64), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    polarity: Mapped[str] = mapped_column(String(32), nullable=False)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_user_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    prev_assistant_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    extracted_lesson: Mapped[str | None] = mapped_column(Text, nullable=True)
    lesson_extracted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mem0_memory_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
