"""SQLAlchemy 2.0 ORM models for PostgreSQL-backed furnace app tables."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
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


class User(Base):
    """Application user stored in the identity schema."""

    __tablename__ = "users"
    __table_args__ = {"schema": "identity"}

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False, default=UserRole.USER.value)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class UserRoleAssignment(Base):
    """User-role mapping stored separately for future multi-role support."""

    __tablename__ = "user_roles"
    __table_args__ = {"schema": "identity"}

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("identity.users.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(Text, primary_key=True)
    assigned_by_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("identity.users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class MaterialCategory(Base):
    """Canonical material category."""

    __tablename__ = "material_categories"
    __table_args__ = {"schema": "plant_master"}

    category_code: Mapped[str] = mapped_column(Text, primary_key=True)
    category_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class Unit(Base):
    """Canonical unit of measure."""

    __tablename__ = "units"
    __table_args__ = {"schema": "plant_master"}

    unit_code: Mapped[str] = mapped_column(Text, primary_key=True)
    unit_name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class Material(Base):
    """Canonical material master row."""

    __tablename__ = "materials"
    __table_args__ = {"schema": "plant_master"}

    material_code: Mapped[str] = mapped_column(Text, primary_key=True)
    material_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    category_code: Mapped[str] = mapped_column(
        ForeignKey("plant_master.material_categories.category_code"), nullable=False
    )
    material_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    unit_code: Mapped[str] = mapped_column(
        ForeignKey("plant_master.units.unit_code"), nullable=False, default="MT"
    )


class Hopper(Base):
    """Canonical hopper master row."""

    __tablename__ = "hoppers"
    __table_args__ = {"schema": "plant_master"}

    hopper_code: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int | None] = mapped_column(Integer, nullable=True, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class HopperRawMaterialHistory(Base):
    """Wide hopper-to-material snapshot history."""

    __tablename__ = "hopper_raw_material_history"
    __table_args__ = (
        Index("ix_hopper_raw_material_history_date_time", "date_time"),
        {"schema": "ops_config"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    hopper_01: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_02: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_03: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_04: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_05: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_06: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_07: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_08: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_09: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_10: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_11: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_12: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_13: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_14: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_15: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_16: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_17: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_18: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_19: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_modified: Mapped[UUID | None] = mapped_column(
        ForeignKey("identity.users.id"), nullable=True
    )
    source_type: Mapped[str] = mapped_column(Text, nullable=False, default="webapp")
    import_batch_id: Mapped[UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class BurdenHistory(Base):
    """Wide burden distribution snapshot history."""

    __tablename__ = "burden_history"
    __table_args__ = (
        Index("ix_burden_history_date_time", "date_time"),
        {"schema": "ops_config"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    coke_p01_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p02_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p03_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p04_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p05_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p06_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p07_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p08_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p09_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p10_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p11_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p01_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p02_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p03_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p04_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p05_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p06_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p07_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p08_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p09_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p10_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p11_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_discharge_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_charge_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    noncoke_p01_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p02_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p03_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p04_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p05_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p06_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p07_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p08_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p09_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p10_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p11_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p01_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p02_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p03_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p04_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p05_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p06_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p07_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p08_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p09_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p10_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p11_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    non_coke_discharge_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    non_coke_charge_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    burden_changing_purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(Text, nullable=False, default="webapp")
    ip_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_modified: Mapped[UUID | None] = mapped_column(
        ForeignKey("identity.users.id"), nullable=True
    )
    import_batch_id: Mapped[UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


HOPPER_COLUMNS = tuple(f"hopper_{i:02d}" for i in range(1, 20))
BURDEN_VALUE_COLUMNS = tuple(
    col.name
    for col in BurdenHistory.__table__.columns
    if col.name
    not in {
        "id",
        "date_time",
        "source_type",
        "ip_address",
        "user_modified",
        "import_batch_id",
        "created_at",
    }
)

# Backward-compatible import names while callers migrate.
HopperMaterialHistory = HopperRawMaterialHistory
BurdenDistributionHistory = BurdenHistory


class Conversation(Base):
    """FurnaceMind chat conversation metadata."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
        {"schema": "furnace_mind"},
    )

    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("identity.users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
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
    """One persisted FurnaceMind chat message."""

    __tablename__ = "conversation_messages"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "sequence_num",
            name="uq_conversation_messages_sequence",
        ),
        Index(
            "ix_conversation_messages_conversation", "conversation_id", "sequence_num"
        ),
        {"schema": "furnace_mind"},
    )

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("identity.users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_num: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_calls: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class MemoryDocument(Base):
    """Uploaded FurnaceMind knowledge document metadata."""

    __tablename__ = "memory_documents"
    __table_args__ = (
        Index("ix_memory_documents_user_active", "user_id", "is_active"),
        {"schema": "furnace_mind"},
    )

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("identity.users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    qdrant_collection: Mapped[str | None] = mapped_column(String(128), nullable=True)
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

    @property
    def qdrant_point_ids(self) -> list:
        """Return Qdrant chunk point ids stored in metadata JSON."""
        metadata = self.metadata_json if isinstance(self.metadata_json, dict) else {}
        point_ids = metadata.get("qdrant_point_ids")
        return point_ids if isinstance(point_ids, list) else []


class MemorySummary(Base):
    """Compressed conversation summary for context management."""

    __tablename__ = "memory_summaries"
    __table_args__ = (
        Index("ix_memory_summaries_conversation", "conversation_id", "created_at"),
        {"schema": "furnace_mind"},
    )

    summary_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("identity.users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_message_id_start: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    source_message_id_end: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class MemoryFact(Base):
    """Durable user-specific memory fact stored before Qdrant indexing."""

    __tablename__ = "memory_facts"
    __table_args__ = {"schema": "furnace_mind"}

    fact_id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("identity.users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    fact_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_conversation_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    qdrant_collection: Mapped[str | None] = mapped_column(Text, nullable=True)
    qdrant_point_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, nullable=False)
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
        {"schema": "furnace_mind"},
    )

    skill_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    icon: Mapped[str | None] = mapped_column(String(64), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="uploaded"
    )
    qdrant_collection: Mapped[str | None] = mapped_column(String(128), nullable=True)
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
    """User feedback record and optional extracted lesson."""

    __tablename__ = "feedback_items"
    __table_args__ = (
        Index("ix_feedback_items_user_created", "user_id", "created_at"),
        Index("ix_feedback_items_pending_lesson", "lesson_extracted", "created_at"),
        {"schema": "furnace_mind"},
    )

    feedback_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("identity.users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    polarity: Mapped[str] = mapped_column(String(32), nullable=False)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_user_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    assistant_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    lesson_extracted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    extracted_lesson: Mapped[str | None] = mapped_column(Text, nullable=True)
    qdrant_collection: Mapped[str | None] = mapped_column(String(128), nullable=True)
    qdrant_point_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
