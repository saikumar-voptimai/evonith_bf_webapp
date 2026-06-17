"""SQLAlchemy models for daemon job definitions and audit events."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base declarative class for daemon job tables."""


class DaemonJobKind(str, Enum):
    """Supported daemon job templates."""

    FURNACEMIND_SHIFT_REPORT = "furnacemind_shift_report"
    FURNACE_HEALTH_WATCH = "furnace_health_watch"
    HEATLOAD_WATCH = "heatload_watch"
    CHANNELING_WATCH = "channeling_watch"
    MATERIAL_BALANCE_REPORT = "material_balance_report"
    CUSTOM_LANGGRAPH_AGENT = "custom_langgraph_agent"


class DaemonJobScheduleType(str, Enum):
    """Supported scheduling modes."""

    SYSTEMD_TIMER = "systemd_timer"
    CRON_EXPRESSION = "cron_expression"
    MANUAL_ONLY = "manual_only"


class DaemonJobRestartPolicy(str, Enum):
    """Supported restart policies."""

    NO = "no"
    ON_FAILURE = "on-failure"
    ALWAYS = "always"


class DaemonJobConcurrencyPolicy(str, Enum):
    """Supported concurrency policies for future runners."""

    FORBID = "forbid"
    REPLACE = "replace"
    ALLOW = "allow"


class DaemonJobCriticality(str, Enum):
    """Supported daemon job criticality levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class DaemonJobAuditEventType(str, Enum):
    """Supported daemon job audit event types."""

    CREATED = "created"
    UPDATED = "updated"
    ENABLED = "enabled"
    DISABLED = "disabled"
    CLONED = "cloned"
    DELETED = "deleted"
    PREVIEWED = "previewed"


class DaemonJob(Base):
    """Persisted control-plane definition for one daemon job."""

    __tablename__ = "daemon_jobs"
    __table_args__ = (
        Index("ix_daemon_jobs_deleted_updated_at", "deleted", "updated_at"),
        Index("ix_daemon_jobs_kind_schedule", "job_kind", "schedule_type"),
        Index("ix_daemon_jobs_enabled_criticality", "enabled", "criticality"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    job_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    schedule_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    cron_expression: Mapped[str | None] = mapped_column(String(128), nullable=True)
    on_calendar: Mapped[str | None] = mapped_column(String(128), nullable=True)
    timezone: Mapped[str] = mapped_column(
        String(64), nullable=False, default="Asia/Kolkata"
    )
    systemd_unit_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
    )
    working_directory: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        default="/home/pi/evonith_bf_webapp",
    )
    python_executable: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        default="/home/pi/evonith_bf_webapp/.venv/bin/python",
    )
    module_path: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        default="src.jobs.runner",
    )
    job_args_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    env_file: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        default="/home/pi/evonith_bf_webapp/.env",
    )
    user_name: Mapped[str] = mapped_column(String(128), nullable=False, default="pi")
    group_name: Mapped[str] = mapped_column(String(128), nullable=False, default="pi")
    restart_policy: Mapped[str] = mapped_column(
        String(32), nullable=False, default="on-failure"
    )
    restart_sec: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    timeout_sec: Mapped[int] = mapped_column(Integer, nullable=False, default=900)
    max_runtime_sec: Mapped[int] = mapped_column(Integer, nullable=False, default=900)
    concurrency_policy: Mapped[str] = mapped_column(
        String(32), nullable=False, default="forbid"
    )
    criticality: Mapped[str] = mapped_column(
        String(32), nullable=False, default="normal"
    )
    tools_allowed_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    tools_blocked_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    memory_short_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    memory_long_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    reporting_rules_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )
    criticality_rules_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )
    persist_jobs_md_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        default="/home/pi/evonith_bf_webapp/persist_jobs.md",
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
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
    last_previewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class DaemonJobAuditEvent(Base):
    """Append-only audit event for daemon job changes."""

    __tablename__ = "daemon_job_audit_events"
    __table_args__ = (
        Index("ix_daemon_job_audit_job_created_at", "job_id", "created_at"),
        Index("ix_daemon_job_audit_event_type", "event_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )
    snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)
