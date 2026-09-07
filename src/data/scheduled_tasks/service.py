"""Application service for validated scheduled-task JSON definitions.

The service is the boundary used by Streamlit. It validates canonical JSON,
enforces owner identifiers, and exposes create/read/update/archive/delete
operations without importing any scheduling or execution runtime.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID

from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from furnace_data.relational import (
    ScheduledTaskDefinitionRecord,
    ScheduledTaskDefinitionRepository,
    build_relational_engine,
    build_relational_session_factory,
)
from utils.scheduled_tasks.scheduled_job_catalog import get_scheduled_job_catalog
from utils.scheduled_tasks.scheduled_task_definition import validate_task_definition


class ScheduledJobValidationError(ValueError):
    """Raised when a requested definition or identifier is invalid."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        """Store every validation issue for operator-friendly presentation."""

        self.errors = errors
        super().__init__(" ".join(errors))


class ScheduledJobPersistenceError(RuntimeError):
    """Raised when definition storage is unavailable."""


class ScheduledJobConflictError(RuntimeError):
    """Raised when a task changed after it was loaded by the operator."""


class _DefinitionRepository(Protocol):
    """Repository behaviour required by the definition service."""

    def create(
        self,
        *,
        definition: dict[str, object],
        owner_user_id: UUID,
        created_by_username: str,
    ) -> ScheduledTaskDefinitionRecord:
        """Store and return one definition record."""

        ...

    def get_for_owner(
        self, *, job_id: str, owner_user_id: UUID
    ) -> ScheduledTaskDefinitionRecord | None:
        """Return an owned record or ``None``."""

        ...

    def list_for_owner(
        self, *, owner_user_id: UUID, limit: int
    ) -> list[ScheduledTaskDefinitionRecord]:
        """Return a bounded owner task list."""

        ...

    def update_for_owner(
        self,
        *,
        job_id: str,
        owner_user_id: UUID,
        expected_updated_at: datetime,
        definition: dict[str, object],
        changed_by_username: str,
    ) -> ScheduledTaskDefinitionRecord | None:
        """Replace a definition when its stored version matches."""

        ...

    def archive_for_owner(
        self,
        *,
        job_id: str,
        owner_user_id: UUID,
        expected_updated_at: datetime,
    ) -> ScheduledTaskDefinitionRecord | None:
        """Archive a definition when its stored version matches."""

        ...

    def delete_for_owner(self, *, job_id: str, owner_user_id: UUID) -> bool:
        """Permanently delete an archived owner definition."""

        ...


@dataclass(frozen=True)
class ScheduledJobCreateRequest:
    """Authenticated request to store one scheduled-task definition."""

    definition: dict[str, object]
    created_by_user_id: str
    created_by_username: str


@dataclass(frozen=True)
class ScheduledJobView:
    """Database-independent representation of a stored definition."""

    job_id: str
    job_name: str
    schema_version: str
    definition: dict[str, object]
    status: str
    created_by_user_id: str
    created_by_username: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ScheduledJobListItem:
    """One saved definition plus empty future execution placeholders."""

    job: ScheduledJobView
    last_run_status: str | None = None
    last_run_at: datetime | None = None


def _database_timestamp(value: datetime) -> datetime:
    """Normalize a database timestamp to timezone-aware UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def scheduled_job_definition_errors(
    definition: dict[str, object],
) -> tuple[str, ...]:
    """Return schema and configured-catalog errors for a definition."""

    catalog = get_scheduled_job_catalog()
    errors = list(validate_task_definition(definition))
    job_type = definition.get("job_type")
    if isinstance(job_type, str) and catalog.job_type(job_type) is None:
        errors.append("job_type: task type is not configured")

    target = definition.get("target_device")
    if isinstance(target, dict):
        target_id = target.get("device_id")
        configured_target = (
            catalog.target_device(target_id) if isinstance(target_id, str) else None
        )
        if isinstance(target_id, str) and configured_target is None:
            errors.append("target_device.device_id: device is not configured")

    inputs = definition.get("inputs")
    if isinstance(inputs, dict):
        data_source = inputs.get("data_source")
        if isinstance(data_source, str) and catalog.data_source(data_source) is None:
            errors.append("inputs.data_source: data source is not configured")
        if job_type == "eta_co_report":
            signal = inputs.get("signal")
            if isinstance(signal, str) and catalog.eta_co_signal(signal) is None:
                errors.append("inputs.signal: ETA CO signal is not configured")
            aggregation = inputs.get("aggregation_interval")
            if (
                isinstance(aggregation, str)
                and catalog.aggregation_interval(aggregation) is None
            ):
                errors.append("inputs.aggregation_interval: interval is not configured")
    return tuple(dict.fromkeys(errors))


class ScheduledJobService:
    """Validate and manage owner-scoped scheduled-task definitions."""

    def __init__(
        self,
        db_url: str | None = None,
        *,
        repository: _DefinitionRepository | None = None,
    ) -> None:
        """Create the service with PostgreSQL or an injected test repository."""

        self._engine: Engine | None = None
        if repository is not None:
            self._repository = repository
            return
        self._engine = build_relational_engine(db_url=db_url)
        session_factory = build_relational_session_factory(self._engine)
        self._repository = ScheduledTaskDefinitionRepository(session_factory)

    def dispose(self) -> None:
        """Dispose the SQLAlchemy engine owned by this service."""

        if self._engine is not None:
            self._engine.dispose()

    def create_job(self, request: ScheduledJobCreateRequest) -> ScheduledJobView:
        """Validate and store a new definition in waiting state."""

        definition, owner_id, username = self._validated_request(request)
        try:
            record = self._repository.create(
                definition=definition,
                owner_user_id=owner_id,
                created_by_username=username,
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled task could not be stored in the database."
            ) from exc
        return self._to_view(record)

    def list_jobs_for_owner(
        self, owner_user_id: str, *, limit: int = 100
    ) -> tuple[ScheduledJobListItem, ...]:
        """Return a bounded newest-first list for one authenticated owner."""

        owner_id = self._owner_id(owner_user_id)
        if not 1 <= limit <= 100:
            raise ScheduledJobValidationError(("limit: must be between 1 and 100",))
        try:
            records = self._repository.list_for_owner(
                owner_user_id=owner_id,
                limit=limit,
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "Scheduled tasks could not be loaded from the database."
            ) from exc
        return tuple(ScheduledJobListItem(job=self._to_view(row)) for row in records)

    def get_job_for_owner(
        self, job_id: str, owner_user_id: str
    ) -> ScheduledJobView | None:
        """Return a task only when it belongs to the authenticated owner."""

        owner_id = self._owner_id(owner_user_id)
        normalized_job_id = self._job_id(job_id)
        try:
            record = self._repository.get_for_owner(
                job_id=normalized_job_id,
                owner_user_id=owner_id,
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled task could not be loaded from the database."
            ) from exc
        return self._to_view(record) if record is not None else None

    def update_job(
        self,
        *,
        job_id: str,
        owner_user_id: str,
        changed_by_username: str,
        expected_updated_at: datetime,
        definition: dict[str, object],
    ) -> ScheduledJobView:
        """Validate and replace one owner definition using optimistic locking."""

        request = ScheduledJobCreateRequest(
            definition=definition,
            created_by_user_id=owner_user_id,
            created_by_username=changed_by_username,
        )
        canonical, owner_id, username = self._validated_request(request)
        normalized_job_id = self._job_id(job_id)
        try:
            record = self._repository.update_for_owner(
                job_id=normalized_job_id,
                owner_user_id=owner_id,
                expected_updated_at=expected_updated_at,
                definition=canonical,
                changed_by_username=username,
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled task could not be updated."
            ) from exc
        if record is None:
            raise ScheduledJobConflictError(
                "The task changed or is no longer available; reload and try again."
            )
        return self._to_view(record)

    def archive_job(
        self,
        *,
        job_id: str,
        owner_user_id: str,
        expected_updated_at: datetime,
    ) -> ScheduledJobView:
        """Archive one owner definition without activating external behaviour."""

        try:
            record = self._repository.archive_for_owner(
                job_id=self._job_id(job_id),
                owner_user_id=self._owner_id(owner_user_id),
                expected_updated_at=expected_updated_at,
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled task could not be archived."
            ) from exc
        if record is None:
            raise ScheduledJobConflictError(
                "The task changed or is no longer available; reload and try again."
            )
        return self._to_view(record)

    def delete_job(self, *, job_id: str, owner_user_id: str) -> None:
        """Permanently delete one archived definition owned by the operator."""

        try:
            deleted = self._repository.delete_for_owner(
                job_id=self._job_id(job_id),
                owner_user_id=self._owner_id(owner_user_id),
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled task could not be deleted."
            ) from exc
        if not deleted:
            raise ScheduledJobConflictError(
                "Only an archived task owned by this operator can be deleted."
            )

    @staticmethod
    def _validated_request(
        request: ScheduledJobCreateRequest,
    ) -> tuple[dict[str, object], UUID, str]:
        """Validate a create/update request and return normalized values."""

        if not isinstance(request.definition, dict):
            raise ScheduledJobValidationError(
                ("definition: scheduled-task JSON object is required",)
            )
        definition = deepcopy(request.definition)
        errors = list(scheduled_job_definition_errors(definition))
        try:
            owner_id = UUID(str(request.created_by_user_id).strip())
        except (AttributeError, TypeError, ValueError):
            owner_id = None
            errors.append("created_by_user_id: authenticated user UUID is required")
        username = str(request.created_by_username).strip()
        if not username:
            errors.append("created_by_username: authenticated username is required")
        elif len(username) > 128:
            errors.append("created_by_username: must be 128 characters or fewer")
        if errors:
            raise ScheduledJobValidationError(tuple(dict.fromkeys(errors)))
        assert owner_id is not None
        return definition, owner_id, username

    @staticmethod
    def _owner_id(value: str) -> UUID:
        """Return a validated authenticated owner UUID."""

        try:
            return UUID(str(value).strip())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScheduledJobValidationError(
                ("owner_user_id: authenticated user UUID is required",)
            ) from exc

    @staticmethod
    def _job_id(value: str) -> str:
        """Return a normalized scheduled-task UUID string."""

        try:
            return str(UUID(str(value).strip()))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScheduledJobValidationError(
                ("job_id: valid UUID is required",)
            ) from exc

    @staticmethod
    def _to_view(record: ScheduledTaskDefinitionRecord) -> ScheduledJobView:
        """Convert an ORM record into the stable application result shape."""

        return ScheduledJobView(
            job_id=str(record.job_id),
            job_name=record.job_name,
            schema_version=record.schema_version,
            definition=deepcopy(record.definition_json),
            status=record.status,
            created_by_user_id=str(record.owner_user_id),
            created_by_username=record.created_by_username,
            created_at=_database_timestamp(record.created_at),
            updated_at=_database_timestamp(record.updated_at),
        )
