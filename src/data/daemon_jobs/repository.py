"""Repository layer for daemon job CRUD and audit operations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .models import DaemonJob, DaemonJobAuditEvent, utc_now

EDITABLE_JOB_FIELDS = (
    "name",
    "description",
    "enabled",
    "job_kind",
    "schedule_type",
    "cron_expression",
    "on_calendar",
    "timezone",
    "systemd_unit_name",
    "working_directory",
    "python_executable",
    "module_path",
    "job_args_json",
    "env_file",
    "user_name",
    "group_name",
    "restart_policy",
    "restart_sec",
    "timeout_sec",
    "max_runtime_sec",
    "concurrency_policy",
    "criticality",
    "tools_allowed_json",
    "tools_blocked_json",
    "memory_short_json",
    "memory_long_json",
    "reporting_rules_json",
    "criticality_rules_json",
    "persist_jobs_md_path",
    "notes",
)

SNAPSHOT_FIELDS = (
    "id",
    *EDITABLE_JOB_FIELDS,
    "created_by",
    "updated_by",
    "created_at",
    "updated_at",
    "last_previewed_at",
    "deleted",
    "deleted_at",
)


class DaemonJobRepository:
    """Persistence operations for daemon jobs and audit events."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create repository with an injected SQLAlchemy session factory."""
        self._session_factory = session_factory

    def list_jobs(self, *, include_deleted: bool = False) -> list[DaemonJob]:
        """Return daemon jobs sorted by latest update."""
        with self._session_factory() as session:
            query = select(DaemonJob)
            if not include_deleted:
                query = query.where(DaemonJob.deleted.is_(False))
            query = query.order_by(DaemonJob.updated_at.desc(), DaemonJob.name.asc())
            jobs = list(session.execute(query).scalars().all())
            for job in jobs:
                session.expunge(job)
            return jobs

    def get_job(
        self, job_id: str, *, include_deleted: bool = False
    ) -> DaemonJob | None:
        """Fetch one daemon job by ID."""
        with self._session_factory() as session:
            job = session.get(DaemonJob, job_id)
            if job is None:
                return None
            if job.deleted and not include_deleted:
                return None
            session.expunge(job)
            return job

    def find_by_unit_name(
        self,
        unit_name: str,
        *,
        include_deleted: bool = True,
    ) -> DaemonJob | None:
        """Fetch a daemon job by unique systemd unit-name stem."""
        with self._session_factory() as session:
            query = select(DaemonJob).where(DaemonJob.systemd_unit_name == unit_name)
            if not include_deleted:
                query = query.where(DaemonJob.deleted.is_(False))
            job = session.execute(query).scalar_one_or_none()
            if job is None:
                return None
            session.expunge(job)
            return job

    def create_job(
        self,
        payload: dict[str, Any],
        *,
        actor: str | None = None,
        event_type: str = "created",
        message: str = "Daemon job created.",
    ) -> DaemonJob:
        """Insert one daemon job and append its first audit event."""
        with self._session_factory() as session:
            job = DaemonJob(
                id=str(uuid4()),
                created_by=actor,
                updated_by=actor,
                **{field: payload[field] for field in EDITABLE_JOB_FIELDS},
            )
            session.add(job)
            session.flush()
            self._add_event(
                session=session,
                job=job,
                event_type=event_type,
                message=message,
                actor=actor,
            )
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def update_job(
        self,
        job_id: str,
        payload: dict[str, Any],
        *,
        actor: str | None = None,
    ) -> DaemonJob:
        """Update one daemon job and append an update audit event."""
        with self._session_factory() as session:
            job = self._get_active_job_for_update(session=session, job_id=job_id)
            for field in EDITABLE_JOB_FIELDS:
                if field in payload:
                    setattr(job, field, payload[field])
            job.updated_by = actor
            job.updated_at = utc_now()
            self._add_event(
                session=session,
                job=job,
                event_type="updated",
                message="Daemon job updated.",
                actor=actor,
            )
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def set_enabled(
        self,
        job_id: str,
        enabled: bool,
        *,
        actor: str | None = None,
    ) -> DaemonJob:
        """Enable or disable one daemon job definition."""
        with self._session_factory() as session:
            job = self._get_active_job_for_update(session=session, job_id=job_id)
            job.enabled = enabled
            job.updated_by = actor
            job.updated_at = utc_now()
            event_type = "enabled" if enabled else "disabled"
            message = "Daemon job enabled." if enabled else "Daemon job disabled."
            self._add_event(
                session=session,
                job=job,
                event_type=event_type,
                message=message,
                actor=actor,
            )
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def delete_job(self, job_id: str, *, actor: str | None = None) -> DaemonJob:
        """Soft delete one daemon job definition and keep its audit history."""
        with self._session_factory() as session:
            job = self._get_active_job_for_update(session=session, job_id=job_id)
            now = utc_now()
            job.deleted = True
            job.deleted_at = now
            job.enabled = False
            job.updated_by = actor
            job.updated_at = now
            self._add_event(
                session=session,
                job=job,
                event_type="deleted",
                message="Daemon job soft-deleted.",
                actor=actor,
            )
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def mark_previewed(self, job_id: str, *, actor: str | None = None) -> DaemonJob:
        """Update preview timestamp and append a preview audit event."""
        with self._session_factory() as session:
            job = self._get_active_job_for_update(session=session, job_id=job_id)
            now = utc_now()
            job.last_previewed_at = now
            job.updated_by = actor
            job.updated_at = now
            self._add_event(
                session=session,
                job=job,
                event_type="previewed",
                message="Systemd preview generated.",
                actor=actor,
            )
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def add_audit_event(
        self,
        job_id: str,
        *,
        event_type: str,
        message: str,
        actor: str | None = None,
    ) -> None:
        """Append an audit event for an existing job."""
        with self._session_factory() as session:
            job = session.get(DaemonJob, job_id)
            if job is None:
                raise ValueError(f"Daemon job {job_id} not found.")
            self._add_event(
                session=session,
                job=job,
                event_type=event_type,
                message=message,
                actor=actor,
            )
            session.commit()

    def list_audit_events(self, job_id: str) -> list[DaemonJobAuditEvent]:
        """List daemon job audit events newest-first."""
        with self._session_factory() as session:
            query = (
                select(DaemonJobAuditEvent)
                .where(DaemonJobAuditEvent.job_id == job_id)
                .order_by(
                    DaemonJobAuditEvent.created_at.desc(), DaemonJobAuditEvent.id.desc()
                )
            )
            events = list(session.execute(query).scalars().all())
            for event in events:
                session.expunge(event)
            return events

    @staticmethod
    def _get_active_job_for_update(*, session: Session, job_id: str) -> DaemonJob:
        """Fetch an active job inside a transaction or raise a clear error."""
        job = session.get(DaemonJob, job_id)
        if job is None or job.deleted:
            raise ValueError(f"Daemon job {job_id} not found.")
        return job

    @staticmethod
    def _add_event(
        *,
        session: Session,
        job: DaemonJob,
        event_type: str,
        message: str,
        actor: str | None,
    ) -> None:
        """Insert one audit event row inside an active transaction."""
        session.add(
            DaemonJobAuditEvent(
                job_id=job.id,
                event_type=event_type,
                message=message,
                actor=actor,
                snapshot_json=_snapshot_job(job),
            )
        )


def _snapshot_job(job: DaemonJob) -> str:
    """Serialize the current job state for audit storage."""
    snapshot: dict[str, Any] = {}
    for field in SNAPSHOT_FIELDS:
        value = getattr(job, field)
        if isinstance(value, datetime):
            value = value.isoformat()
        snapshot[field] = value
    return json.dumps(snapshot, sort_keys=True)
