"""Durable control bridge between Streamlit Cloud and the BF2 Jetson.

The web process only validates and enqueues owner-scoped commands. A short-lived
Jetson process claims those commands with fenced leases, invokes the existing
systemd provisioning service, and records the outcome. This module contains no
systemd or subprocess calls.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from furnace_data.relational import (
    ScheduledJob,
    ScheduledJobCommand,
    ScheduledJobCommandRepository,
    ScheduledJobRevision,
    build_relational_engine,
    build_relational_session_factory,
)

from .service import (
    ScheduledJobPersistenceError,
    ScheduledJobValidationError,
    scheduled_job_definition_errors,
)


class ScheduledJobCommandConflictError(RuntimeError):
    """Raised when a request is stale or invalid for the current job state."""


@dataclass(frozen=True, slots=True)
class ScheduledJobCommandView:
    """Database-independent lifecycle command receipt and processing state."""

    command_id: str
    job_id: str
    action: str
    requested_job_status: str
    status: str
    target_device_id: str
    requested_by_user_id: str
    requested_by_username: str
    expected_job_updated_at: datetime
    definition: dict[str, object] | None
    attempt_count: int
    maximum_attempts: int
    resulting_job_status: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    lease_token: str | None


@dataclass(frozen=True, slots=True)
class ScheduledJobRevisionView:
    """Safe immutable definition-revision details for owner-facing history."""

    revision_id: str
    job_id: str
    revision_number: int
    schema_version: str
    definition: dict[str, object]
    changed_by_username: str
    change_kind: str
    created_at: datetime


def _utc_timestamp(value: datetime) -> datetime:
    """Normalize database timestamps, treating SQLite-naive values as UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class ScheduledJobCommandService:
    """Validate, enqueue, claim, and settle scheduled-job control commands."""

    def __init__(
        self,
        db_url: str | None = None,
        *,
        repository: ScheduledJobCommandRepository | None = None,
    ) -> None:
        """Create the service from a database URL or injected repository."""

        self._engine: Engine | None = None
        if repository is not None:
            self._repository = repository
            return
        self._engine = build_relational_engine(db_url=db_url)
        session_factory = build_relational_session_factory(self._engine)
        self._repository = ScheduledJobCommandRepository(session_factory)

    def dispose(self) -> None:
        """Dispose the SQLAlchemy engine owned by this service."""

        if self._engine is not None:
            self._engine.dispose()

    def request_action(
        self,
        *,
        job_id: str,
        action: str,
        owner_user_id: str,
        requested_by_username: str,
        expected_job_updated_at: datetime,
        definition: dict[str, object] | None = None,
        maximum_attempts: int = 3,
    ) -> ScheduledJobCommandView:
        """Validate and enqueue one lifecycle or definition-update request."""

        normalized_job_id = self._job_id(job_id)
        owner_id = self._owner_id(owner_user_id)
        username = str(requested_by_username).strip()
        if not username or len(username) > 128:
            raise ScheduledJobValidationError(
                ("requested_by_username: must be 1 to 128 characters",)
            )
        if not isinstance(expected_job_updated_at, datetime):
            raise ScheduledJobValidationError(
                ("expected_job_updated_at: timestamp is required",)
            )
        expected_at = _utc_timestamp(expected_job_updated_at)

        proposed_definition: dict[str, object] | None = None
        if definition is not None:
            proposed_definition = deepcopy(definition)
            errors = scheduled_job_definition_errors(proposed_definition)
            if errors:
                raise ScheduledJobValidationError(errors)

        try:
            command = self._repository.enqueue_for_owner(
                job_id=normalized_job_id,
                action=str(action).strip().lower(),
                owner_user_id=owner_id,
                requested_by_username=username,
                expected_job_updated_at=expected_at,
                definition=proposed_definition,
                maximum_attempts=maximum_attempts,
            )
        except ValueError as exc:
            raise ScheduledJobCommandConflictError(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job command could not be stored."
            ) from exc
        return self._to_command_view(command)

    def latest_for_owner(
        self,
        *,
        job_id: str,
        owner_user_id: str,
    ) -> ScheduledJobCommandView | None:
        """Return the newest command for one authenticated owner job."""

        try:
            command = self._repository.latest_for_owner(
                job_id=self._job_id(job_id),
                owner_user_id=self._owner_id(owner_user_id),
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job command status could not be loaded."
            ) from exc
        return self._to_command_view(command) if command is not None else None

    def list_revisions_for_owner(
        self,
        *,
        job_id: str,
        owner_user_id: str,
        limit: int = 25,
    ) -> tuple[ScheduledJobRevisionView, ...]:
        """Return bounded revision history for one authenticated owner job."""

        try:
            revisions = self._repository.list_revisions_for_owner(
                job_id=self._job_id(job_id),
                owner_user_id=self._owner_id(owner_user_id),
                limit=limit,
            )
        except (SQLAlchemyError, ValueError) as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job revision history could not be loaded."
            ) from exc
        return tuple(self._to_revision_view(revision) for revision in revisions)

    def claim_next(
        self,
        *,
        target_device_id: str,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> ScheduledJobCommandView | None:
        """Claim the next due target-device command for bounded processing."""

        try:
            command = self._repository.claim_next(
                target_device_id=target_device_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )
        except (SQLAlchemyError, ValueError) as exc:
            raise ScheduledJobPersistenceError(
                "The next scheduled-job command could not be claimed."
            ) from exc
        return self._to_command_view(command) if command is not None else None

    def apply_definition_revision(
        self,
        *,
        command: ScheduledJobCommandView,
        expected_job_updated_at: datetime,
        definition: dict[str, object],
        change_kind: str = "edited",
    ) -> ScheduledJob | None:
        """Apply a validated command definition with an immutable audit row."""

        errors = scheduled_job_definition_errors(definition)
        if errors:
            raise ScheduledJobValidationError(errors)
        try:
            return self._repository.apply_definition_revision(
                job_id=command.job_id,
                expected_updated_at=expected_job_updated_at,
                definition=deepcopy(definition),
                changed_by_user_id=UUID(command.requested_by_user_id),
                changed_by_username=command.requested_by_username,
                change_kind=change_kind,
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job definition revision could not be applied."
            ) from exc

    def mark_succeeded(
        self,
        command: ScheduledJobCommandView,
        *,
        resulting_job_status: str,
    ) -> ScheduledJobCommandView:
        """Mark a claimed command successful using its fenced lease token."""

        lease_token = self._required_lease_token(command)
        try:
            stored = self._repository.mark_succeeded(
                command_id=UUID(command.command_id),
                lease_token=lease_token,
                resulting_job_status=resulting_job_status,
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The command success result could not be stored."
            ) from exc
        if stored is None:
            raise ScheduledJobCommandConflictError(
                "The command lease was lost before success could be recorded."
            )
        return self._to_command_view(stored)

    def mark_failed_or_retry(
        self,
        command: ScheduledJobCommandView,
        *,
        error: str,
        retry_delay_seconds: int = 30,
        retry: bool = True,
    ) -> ScheduledJobCommandView:
        """Record a failed attempt, retrying only within the stored limit."""

        lease_token = self._required_lease_token(command)
        try:
            stored = self._repository.mark_failed_or_retry(
                command_id=UUID(command.command_id),
                lease_token=lease_token,
                error=error,
                retry_delay_seconds=retry_delay_seconds,
                retry=retry,
            )
        except (SQLAlchemyError, ValueError) as exc:
            raise ScheduledJobPersistenceError(
                "The command failure result could not be stored."
            ) from exc
        if stored is None:
            raise ScheduledJobCommandConflictError(
                "The command lease was lost before failure could be recorded."
            )
        return self._to_command_view(stored)

    @staticmethod
    def _required_lease_token(command: ScheduledJobCommandView) -> UUID:
        """Return a claimed command lease token or reject an unclaimed view."""

        if command.lease_token is None:
            raise ScheduledJobCommandConflictError(
                "A command must be claimed before it can be completed."
            )
        return UUID(command.lease_token)

    @staticmethod
    def _job_id(job_id: str) -> str:
        """Return a canonical scheduled-job UUID string."""

        try:
            return str(UUID(str(job_id).strip()))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScheduledJobValidationError(
                ("job_id: valid UUID is required",)
            ) from exc

    @staticmethod
    def _owner_id(owner_user_id: str) -> UUID:
        """Return a validated authenticated owner UUID."""

        try:
            return UUID(str(owner_user_id).strip())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScheduledJobValidationError(
                ("owner_user_id: authenticated user UUID is required",)
            ) from exc

    @staticmethod
    def _to_command_view(command: ScheduledJobCommand) -> ScheduledJobCommandView:
        """Convert a command ORM row into a detached immutable view."""

        return ScheduledJobCommandView(
            command_id=str(command.command_id),
            job_id=command.job_id,
            action=command.action,
            requested_job_status=command.requested_job_status,
            status=command.status,
            target_device_id=command.target_device_id,
            requested_by_user_id=str(command.requested_by_user_id),
            requested_by_username=command.requested_by_username,
            expected_job_updated_at=_utc_timestamp(command.expected_job_updated_at),
            definition=(
                deepcopy(command.definition_json)
                if command.definition_json is not None
                else None
            ),
            attempt_count=command.attempt_count,
            maximum_attempts=command.maximum_attempts,
            resulting_job_status=command.resulting_job_status,
            error_message=command.error_message,
            created_at=_utc_timestamp(command.created_at),
            started_at=(
                _utc_timestamp(command.started_at)
                if command.started_at is not None
                else None
            ),
            completed_at=(
                _utc_timestamp(command.completed_at)
                if command.completed_at is not None
                else None
            ),
            lease_token=str(command.lease_token) if command.lease_token else None,
        )

    @staticmethod
    def _to_revision_view(
        revision: ScheduledJobRevision,
    ) -> ScheduledJobRevisionView:
        """Convert a revision ORM row into a detached immutable view."""

        return ScheduledJobRevisionView(
            revision_id=str(revision.revision_id),
            job_id=revision.job_id,
            revision_number=revision.revision_number,
            schema_version=revision.schema_version,
            definition=deepcopy(revision.definition_json),
            changed_by_username=revision.changed_by_username,
            change_kind=revision.change_kind,
            created_at=_utc_timestamp(revision.created_at),
        )


__all__ = [
    "ScheduledJobCommandConflictError",
    "ScheduledJobCommandService",
    "ScheduledJobCommandView",
    "ScheduledJobRevisionView",
]
