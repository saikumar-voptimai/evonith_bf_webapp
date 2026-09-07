"""SQLAlchemy 2.0 ORM models for PostgreSQL-backed furnace app tables."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")


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


class ScheduledJobStatus(str, Enum):
    """Lifecycle states for a persisted scheduled-job definition."""

    PENDING_PROVISIONING = "pending_provisioning"
    ACTIVE = "active"
    PAUSED = "paused"
    PROVISIONING_FAILED = "provisioning_failed"
    COMPLETED = "completed"
    EXECUTION_FAILED = "execution_failed"
    DELETED = "deleted"


class ScheduledJobRunStatus(str, Enum):
    """Lifecycle states for an individual scheduled-job execution."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ScheduledJobCommandAction(str, Enum):
    """Allow-listed control actions accepted from the web application."""

    PROVISION = "provision"
    PAUSE = "pause"
    RESUME = "resume"
    UPDATE = "update"
    ARCHIVE = "archive"


class ScheduledJobCommandStatus(str, Enum):
    """Durable processing states for Jetson lifecycle commands."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


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
    content_type: Mapped[str | None] = mapped_column(Text, nullable=True, deferred=True)
    file_size: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, deferred=True
    )
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, deferred=True)
    file_bytes: Mapped[bytes | None] = mapped_column(
        LargeBinary, nullable=True, deferred=True
    )
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
    symbol: Mapped[str | None] = mapped_column(String(16), nullable=True)
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


class ScheduledJob(Base):
    """Canonical definition and lifecycle state for one scheduled task."""

    __tablename__ = "scheduled_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending_provisioning', 'active', 'paused', "
            "'provisioning_failed', 'completed', 'execution_failed', 'deleted')",
            name="ck_automation_scheduled_jobs_status",
        ),
        CheckConstraint(
            "activation_generation >= 0 AND "
            "(status <> 'active' OR activation_generation > 0)",
            name="ck_automation_scheduled_jobs_activation_generation",
        ),
        CheckConstraint(
            "(status = 'active' AND is_active AND activated_at IS NOT NULL) OR "
            "(status <> 'active' AND NOT is_active AND activated_at IS NULL)",
            name="ck_automation_scheduled_jobs_activation_consistency",
        ),
        Index("ix_scheduled_jobs_creator_created", "owner_user_id", "created_at"),
        Index("ix_scheduled_jobs_status_created", "status", "created_at"),
        Index(
            "ix_scheduled_jobs_target_created_job",
            "target_device_id",
            "created_at",
            "job_id",
        ),
        {"schema": "automation"},
    )

    job_id: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    job_name: Mapped[str] = mapped_column(Text, nullable=False)
    job_type: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    definition_json: Mapped[dict] = mapped_column(
        "definition", JSON_DOCUMENT, nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ScheduledJobStatus.PENDING_PROVISIONING.value,
    )
    created_by_user_id: Mapped[UUID] = mapped_column(
        "owner_user_id",
        ForeignKey("identity.users.id"),
        nullable=False,
    )
    created_by_username: Mapped[str] = mapped_column(String(128), nullable=False)
    target_device_id: Mapped[str] = mapped_column(String(128), nullable=False)
    timer_unit_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provisioning_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    activation_generation: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class ScheduledJobRevision(Base):
    """Immutable snapshot of a scheduled-job definition revision."""

    __tablename__ = "scheduled_job_revisions"
    __table_args__ = (
        CheckConstraint(
            "revision_number >= 1",
            name="ck_scheduled_job_revisions_number_positive",
        ),
        CheckConstraint(
            "change_kind IN ('created', 'edited', 'rollback')",
            name="ck_scheduled_job_revisions_change_kind",
        ),
        UniqueConstraint(
            "job_id",
            "revision_number",
            name="uq_scheduled_job_revisions_job_number",
        ),
        Index("ix_scheduled_job_revisions_changer", "changed_by_user_id"),
        {"schema": "automation"},
    )

    revision_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("automation.scheduled_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
    )
    revision_number: Mapped[int] = mapped_column(BigInteger, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    definition_json: Mapped[dict] = mapped_column(
        "definition", JSON_DOCUMENT, nullable=False
    )
    changed_by_user_id: Mapped[UUID] = mapped_column(
        ForeignKey("identity.users.id"),
        nullable=False,
    )
    changed_by_username: Mapped[str] = mapped_column(String(128), nullable=False)
    change_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ScheduledJobCommand(Base):
    """Owner-requested lifecycle command consumed by the target Jetson."""

    __tablename__ = "scheduled_job_commands"
    __table_args__ = (
        CheckConstraint(
            "action IN ('provision', 'pause', 'resume', 'update', 'archive')",
            name="ck_scheduled_job_commands_action",
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'succeeded', 'failed')",
            name="ck_scheduled_job_commands_status",
        ),
        CheckConstraint(
            "requested_job_status IN ('pending_provisioning', 'active', 'paused', "
            "'provisioning_failed', 'completed', 'execution_failed')",
            name="ck_scheduled_job_commands_requested_job_status",
        ),
        CheckConstraint(
            "attempt_count >= 0 AND maximum_attempts BETWEEN 1 AND 10",
            name="ck_scheduled_job_commands_attempts",
        ),
        CheckConstraint(
            "(action = 'update' AND definition IS NOT NULL) OR "
            "(action <> 'update' AND definition IS NULL)",
            name="ck_scheduled_job_commands_update_definition",
        ),
        CheckConstraint(
            "(status = 'processing' AND lease_token IS NOT NULL AND "
            "lease_expires_at IS NOT NULL AND started_at IS NOT NULL) OR "
            "(status <> 'processing' AND lease_token IS NULL AND "
            "lease_expires_at IS NULL)",
            name="ck_scheduled_job_commands_lease_consistency",
        ),
        Index("ix_scheduled_job_commands_job_created", "job_id", "created_at"),
        Index("ix_scheduled_job_commands_requester", "requested_by_user_id"),
        Index(
            "ix_scheduled_job_commands_pending_target",
            "target_device_id",
            "available_at",
            "created_at",
            "command_id",
            postgresql_where=text("status = 'pending'"),
            sqlite_where=text("status = 'pending'"),
        ),
        Index(
            "ix_scheduled_job_commands_expired_lease",
            "target_device_id",
            "lease_expires_at",
            "created_at",
            "command_id",
            postgresql_where=text("status = 'processing'"),
            sqlite_where=text("status = 'processing'"),
        ),
        Index(
            "uq_scheduled_job_commands_one_open_per_job",
            "job_id",
            unique=True,
            postgresql_where=text("status IN ('pending', 'processing')"),
            sqlite_where=text("status IN ('pending', 'processing')"),
        ),
        {"schema": "automation"},
    )

    command_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("automation.scheduled_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
    )
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    requested_job_status: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ScheduledJobCommandStatus.PENDING.value,
    )
    target_device_id: Mapped[str] = mapped_column(String(128), nullable=False)
    requested_by_user_id: Mapped[UUID] = mapped_column(
        ForeignKey("identity.users.id"),
        nullable=False,
    )
    requested_by_username: Mapped[str] = mapped_column(String(128), nullable=False)
    expected_job_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    definition_json: Mapped[dict | None] = mapped_column(
        "definition",
        JSON(none_as_null=True).with_variant(
            JSONB(none_as_null=True),
            "postgresql",
        ),
        nullable=True,
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    maximum_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    lease_token: Mapped[UUID | None] = mapped_column(nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resulting_job_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ScheduledJobRun(Base):
    """One logical job occurrence, including its current internal attempt."""

    __tablename__ = "job_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', "
            "'timed_out', 'skipped', 'cancelled')",
            name="job_runs_status_check",
        ),
        CheckConstraint(
            "attempt_number >= 1",
            name="ck_job_runs_attempt_number_positive",
        ),
        CheckConstraint(
            "(status = 'running' AND completed_at IS NULL AND "
            "activation_generation > 0 AND lease_token IS NOT NULL AND "
            "lease_expires_at IS NOT NULL AND "
            "((retry_not_before_at IS NULL AND attempt_deadline_at IS NOT NULL) OR "
            "(retry_not_before_at IS NOT NULL AND attempt_deadline_at IS NULL))) OR "
            "(status <> 'running' AND lease_token IS NULL AND "
            "lease_expires_at IS NULL AND attempt_deadline_at IS NULL AND "
            "retry_not_before_at IS NULL)",
            name="ck_job_runs_execution_lease_consistency",
        ),
        UniqueConstraint(
            "job_id",
            "scheduled_for",
            name="uq_job_runs_job_scheduled_for",
        ),
        Index("ix_job_runs_job_created", "job_id", "created_at"),
        Index("ix_job_runs_status_scheduled", "status", "scheduled_for"),
        Index(
            "uq_job_runs_one_running_per_job",
            "job_id",
            unique=True,
            postgresql_where=text("status = 'running'"),
            sqlite_where=text("status = 'running'"),
        ),
        {"schema": "automation"},
    )

    run_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("automation.scheduled_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
    )
    scheduled_for: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ScheduledJobRunStatus.QUEUED.value
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    activation_generation: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
    )
    lease_token: Mapped[UUID | None] = mapped_column(nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempt_deadline_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retry_not_before_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    triggered_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ScheduledJobRunLog(Base):
    """Structured lifecycle message recorded for one scheduled-job run."""

    __tablename__ = "job_run_logs"
    __table_args__ = (
        CheckConstraint(
            "log_level IN ('debug', 'info', 'warning', 'error')",
            name="job_run_logs_log_level_check",
        ),
        Index("ix_job_run_logs_run_id", "run_id"),
        {"schema": "automation"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("automation.job_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    log_level: Mapped[str] = mapped_column(Text, nullable=False, default="info")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSON_DOCUMENT,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ScheduledJobOutput(Base):
    """Text, JSON, or artifact output produced by a scheduled-job run."""

    __tablename__ = "job_outputs"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_job_outputs_run_id"),
        {"schema": "automation"},
    )

    output_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("automation.job_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    output_type: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_json: Mapped[dict | None] = mapped_column(JSON_DOCUMENT, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSON_DOCUMENT,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
