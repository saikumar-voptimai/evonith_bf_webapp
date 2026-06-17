"""Business service layer for daemon job definitions."""

from __future__ import annotations

from typing import Any

from sqlalchemy.engine import Engine

from .engine import build_daemon_jobs_engine, build_daemon_jobs_session_factory
from .models import Base, DaemonJob, DaemonJobAuditEvent, DaemonJobScheduleType
from .repository import DaemonJobRepository, EDITABLE_JOB_FIELDS
from .schemas import (
    DaemonJobAuditEventView,
    DaemonJobView,
    SystemdPreview,
    ValidationResult,
)
from .systemd_render import (
    render_install_commands,
    render_service_unit,
    render_timer_unit,
    render_uninstall_commands,
)
from .validators import suggest_systemd_unit_name, validate_daemon_job_payload


class DaemonJobService:
    """Facade for daemon job definition operations."""

    def __init__(self, db_url: str | None = None) -> None:
        """Initialise the service with its own dedicated SQLAlchemy engine."""
        self._engine: Engine = build_daemon_jobs_engine(db_url=db_url)
        session_factory = build_daemon_jobs_session_factory(self._engine)
        self._repository = DaemonJobRepository(session_factory=session_factory)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        """Create daemon job tables if they do not already exist."""
        Base.metadata.create_all(bind=self._engine)

    def list_jobs(self, include_deleted: bool = False) -> list[DaemonJobView]:
        """List daemon job definitions."""
        jobs = self._repository.list_jobs(include_deleted=include_deleted)
        return [self._to_job_view(job) for job in jobs]

    def get_job(self, job_id: str) -> DaemonJobView | None:
        """Fetch one active daemon job by ID."""
        job = self._repository.get_job(job_id)
        if job is None:
            return None
        return self._to_job_view(job)

    def create_job(
        self,
        payload: dict[str, Any],
        actor: str | None = None,
    ) -> DaemonJobView:
        """Validate and create one daemon job definition."""
        validation = self.validate_payload(payload)
        if not validation.is_valid:
            raise ValueError(_format_validation_errors(validation.errors))

        normalized = validation.normalized_payload
        self._ensure_unit_name_available(normalized["systemd_unit_name"])
        job = self._repository.create_job(normalized, actor=_clean_actor(actor))
        return self._to_job_view(job)

    def update_job(
        self,
        job_id: str,
        payload: dict[str, Any],
        actor: str | None = None,
    ) -> DaemonJobView:
        """Validate and update one daemon job definition."""
        existing = self._repository.get_job(job_id)
        if existing is None:
            raise ValueError(f"Daemon job {job_id} not found.")

        merged = self._job_to_payload(existing)
        merged.update(
            {key: value for key, value in payload.items() if key in EDITABLE_JOB_FIELDS}
        )
        validation = self.validate_payload(merged, existing_job_id=job_id)
        if not validation.is_valid:
            raise ValueError(_format_validation_errors(validation.errors))

        job = self._repository.update_job(
            job_id,
            validation.normalized_payload,
            actor=_clean_actor(actor),
        )
        return self._to_job_view(job)

    def set_enabled(
        self,
        job_id: str,
        enabled: bool,
        actor: str | None = None,
    ) -> DaemonJobView:
        """Enable or disable one daemon job definition."""
        job = self._repository.set_enabled(
            job_id,
            bool(enabled),
            actor=_clean_actor(actor),
        )
        return self._to_job_view(job)

    def clone_job(self, job_id: str, actor: str | None = None) -> DaemonJobView:
        """Clone one daemon job definition with a unique disabled copy."""
        existing = self._repository.get_job(job_id)
        if existing is None:
            raise ValueError(f"Daemon job {job_id} not found.")

        clone_payload = self._job_to_payload(existing)
        clone_payload["name"] = self._clone_name(existing.name)
        clone_payload["enabled"] = False
        clone_payload["systemd_unit_name"] = self._next_available_unit_name(
            existing.systemd_unit_name
        )
        validation = self.validate_payload(clone_payload)
        if not validation.is_valid:
            raise ValueError(_format_validation_errors(validation.errors))

        cleaned_actor = _clean_actor(actor)
        cloned = self._repository.create_job(
            validation.normalized_payload,
            actor=cleaned_actor,
            event_type="cloned",
            message=f"Daemon job cloned from {existing.id}.",
        )
        self._repository.add_audit_event(
            existing.id,
            event_type="cloned",
            message=f"Daemon job cloned to {cloned.id}.",
            actor=cleaned_actor,
        )
        return self._to_job_view(cloned)

    def delete_job(self, job_id: str, actor: str | None = None) -> DaemonJobView:
        """Soft delete one daemon job definition."""
        job = self._repository.delete_job(job_id, actor=_clean_actor(actor))
        return self._to_job_view(job)

    def preview_systemd(self, job_id: str, actor: str | None = None) -> SystemdPreview:
        """Render systemd previews and record a preview audit event."""
        job = self._repository.get_job(job_id)
        if job is None:
            raise ValueError(f"Daemon job {job_id} not found.")

        warnings = [
            "Step 1 only previews systemd units. It does not install anything on the Pi."
        ]
        if job.schedule_type == DaemonJobScheduleType.CRON_EXPRESSION.value:
            warnings.append(
                "cron_expression is stored for compatibility but Pi-side Step 2 should translate it to systemd OnCalendar or runner scheduling."
            )

        preview = SystemdPreview(
            job_id=job.id,
            service_unit=render_service_unit(job),
            timer_unit=render_timer_unit(job),
            install_commands=render_install_commands(job),
            uninstall_commands=render_uninstall_commands(job),
            warnings=warnings,
        )
        self._repository.mark_previewed(job_id, actor=_clean_actor(actor))
        return preview

    def validate_payload(
        self,
        payload: dict[str, Any],
        existing_job_id: str | None = None,
    ) -> ValidationResult:
        """Validate payload shape, policy, and unit-name uniqueness."""
        result = validate_daemon_job_payload(payload)
        if not result.is_valid:
            return result

        unit_name = str(result.normalized_payload["systemd_unit_name"])
        existing = self._repository.find_by_unit_name(unit_name, include_deleted=True)
        if existing is not None and existing.id != existing_job_id:
            result.errors.append(
                "systemd_unit_name is already used by another daemon job, including soft-deleted jobs."
            )
            result.is_valid = False
        return result

    def get_audit_events(self, job_id: str) -> list[DaemonJobAuditEventView]:
        """List audit events for one daemon job."""
        events = self._repository.list_audit_events(job_id)
        return [self._to_audit_event_view(event) for event in events]

    def _ensure_unit_name_available(self, unit_name: str) -> None:
        """Raise when a systemd unit-name stem is already present."""
        existing = self._repository.find_by_unit_name(unit_name, include_deleted=True)
        if existing is not None:
            raise ValueError(
                "systemd_unit_name is already used by another daemon job, including soft-deleted jobs."
            )

    def _next_available_unit_name(self, base_unit_name: str) -> str:
        """Return a unique clone unit-name stem."""
        root = base_unit_name[:92].rstrip("-_")
        for index in range(1, 1000):
            candidate = f"{root}-copy" if index == 1 else f"{root}-copy-{index}"
            if (
                self._repository.find_by_unit_name(candidate, include_deleted=True)
                is None
            ):
                return candidate
        raise ValueError("Could not find an available systemd_unit_name for clone.")

    @staticmethod
    def _clone_name(name: str) -> str:
        """Return a display name for a cloned job."""
        suffix = " (Copy)"
        if len(name) + len(suffix) <= 160:
            return f"{name}{suffix}"
        return f"{name[:160 - len(suffix)]}{suffix}"

    @staticmethod
    def _job_to_payload(job: DaemonJob | DaemonJobView) -> dict[str, Any]:
        """Map a job object back to editable payload fields."""
        return {field: getattr(job, field) for field in EDITABLE_JOB_FIELDS}

    @staticmethod
    def _to_job_view(job: DaemonJob) -> DaemonJobView:
        """Map ORM job to Pydantic read model."""
        return DaemonJobView.model_validate(job)

    @staticmethod
    def _to_audit_event_view(event: DaemonJobAuditEvent) -> DaemonJobAuditEventView:
        """Map ORM audit event to Pydantic read model."""
        return DaemonJobAuditEventView.model_validate(event)


def _clean_actor(actor: str | None) -> str | None:
    """Normalize optional actor identity."""
    if actor is None:
        return None
    cleaned = str(actor).strip()
    return cleaned or None


def _format_validation_errors(errors: list[str]) -> str:
    """Return validation errors as a compact exception message."""
    return " ".join(errors)


__all__ = ["DaemonJobService", "suggest_systemd_unit_name"]
