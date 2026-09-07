"""Validated persistence and owner-read boundary for Scheduled Tasks.

The service validates complete definitions, derives trusted searchable fields,
and stores canonical JSON in ``pending_provisioning`` state. It also returns
bounded, database-independent owner views for task management and run/output
history. Systemd mutation and FurnaceMind execution remain separate services.
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
    ScheduledJobRepository,
    build_relational_engine,
    build_relational_session_factory,
)
from utils.scheduled_tasks.scheduled_job_catalog import get_scheduled_job_catalog
from utils.scheduled_tasks.scheduled_task_definition import validate_task_definition


class ScheduledJobValidationError(ValueError):
    """Raised when a create request cannot be safely persisted."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        """Store all validation errors on the exception."""

        self.errors = errors
        super().__init__(" ".join(errors))


class ScheduledJobPersistenceError(RuntimeError):
    """Raised when the relational database cannot persist a valid job."""


def _utc_database_timestamp(value: datetime) -> datetime:
    """Normalize database timestamps, treating SQLite-naive values as UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ScheduledJobCreateRequest:
    """Authenticated request to persist one scheduled-job definition."""

    definition: dict[str, object]
    created_by_user_id: str
    created_by_username: str


@dataclass(frozen=True)
class ScheduledJobView:
    """Database-independent representation of a stored scheduled job."""

    job_id: str
    job_name: str
    schema_version: str
    definition: dict[str, object]
    status: str
    is_active: bool
    created_by_user_id: str
    created_by_username: str
    target_device_id: str
    timer_unit_name: str | None
    provisioning_error: str | None
    activated_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ScheduledJobListItem:
    """One owner job plus its latest execution status for list rendering."""

    job: ScheduledJobView
    last_run_status: str | None
    last_run_at: datetime | None


@dataclass(frozen=True)
class ScheduledJobRunHistoryItem:
    """Safe database-independent run and output details for one job occurrence."""

    run_id: str
    scheduled_for: datetime
    status: str
    attempt_number: int
    triggered_by: str | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    output_id: str | None
    output_type: str | None
    content: str | None
    content_json: dict[str, object] | None
    output_created_at: datetime | None


def scheduled_job_definition_errors(
    definition: dict[str, object],
) -> tuple[str, ...]:
    """Return schema and catalog errors for one proposed job definition."""

    catalog = get_scheduled_job_catalog()
    errors = list(validate_task_definition(definition))

    job_type = definition.get("job_type")
    if isinstance(job_type, str) and catalog.job_type(job_type) is None:
        errors.append("job_type: task type is not configured")

    target = definition.get("target_device")
    if isinstance(target, dict):
        target_device_id = target.get("device_id")
        target_device_type = target.get("device_type")
        configured_target = (
            catalog.target_device(target_device_id)
            if isinstance(target_device_id, str)
            else None
        )
        if isinstance(target_device_id, str) and configured_target is None:
            errors.append("target_device.device_id: device is not configured")
        elif (
            configured_target is not None
            and isinstance(target_device_type, str)
            and target_device_type != configured_target.device_type
        ):
            errors.append(
                "target_device.device_type: does not match the configured device"
            )

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
    """Validate and persist canonical scheduled-job definitions."""

    def __init__(
        self,
        db_url: str | None = None,
        *,
        repository: ScheduledJobRepository | None = None,
    ) -> None:
        """Create a service backed by PostgreSQL or an injected repository."""

        self._engine: Engine | None = None
        if repository is not None:
            self._repository = repository
            return

        self._engine = build_relational_engine(db_url=db_url)
        session_factory = build_relational_session_factory(self._engine)
        self._repository = ScheduledJobRepository(session_factory)

    def dispose(self) -> None:
        """Dispose the owned SQLAlchemy engine, when one was created."""

        if self._engine is not None:
            self._engine.dispose()

    def create_job(self, request: ScheduledJobCreateRequest) -> ScheduledJobView:
        """Validate and store a job awaiting systemd timer provisioning."""

        if not isinstance(request.definition, dict):
            raise ScheduledJobValidationError(
                ("definition: scheduled-job JSON object is required",)
            )
        definition = deepcopy(request.definition)
        errors = list(scheduled_job_definition_errors(definition))

        try:
            created_by_user_id = UUID(str(request.created_by_user_id).strip())
        except (AttributeError, TypeError, ValueError):
            created_by_user_id = None
            errors.append("created_by_user_id: authenticated user UUID is required")

        created_by_username = str(request.created_by_username).strip()
        if not created_by_username:
            errors.append("created_by_username: authenticated username is required")
        elif len(created_by_username) > 128:
            errors.append("created_by_username: must be 128 characters or fewer")

        job_name = definition.get("job_name")
        schema_version = definition.get("schema_version")
        target = definition.get("target_device")
        target_device_id = target.get("device_id") if isinstance(target, dict) else None

        if errors:
            raise ScheduledJobValidationError(tuple(dict.fromkeys(errors)))

        assert created_by_user_id is not None
        assert isinstance(job_name, str)
        assert isinstance(schema_version, str)
        assert isinstance(target_device_id, str)

        try:
            job = self._repository.create_job(
                job_name=job_name,
                schema_version=schema_version,
                definition=definition,
                created_by_user_id=created_by_user_id,
                created_by_username=created_by_username,
                target_device_id=target_device_id,
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled job could not be stored in the database."
            ) from exc
        return self._to_view(job)

    def get_job(self, job_id: str) -> ScheduledJobView | None:
        """Return a previously stored job by UUID."""

        try:
            normalized_job_id = UUID(str(job_id).strip())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScheduledJobValidationError(
                ("job_id: valid UUID is required",)
            ) from exc

        try:
            job = self._repository.get_job(str(normalized_job_id))
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled job could not be loaded from the database."
            ) from exc
        return self._to_view(job) if job is not None else None

    def list_jobs_for_owner(
        self,
        owner_user_id: str,
        *,
        limit: int = 50,
    ) -> tuple[ScheduledJobListItem, ...]:
        """Return a bounded newest-first list for one authenticated operator."""

        owner_id = self._owner_id(owner_user_id)
        if not 1 <= limit <= 100:
            raise ScheduledJobValidationError(("limit: must be between 1 and 100",))
        try:
            rows = self._repository.list_jobs_for_owner(
                owner_user_id=owner_id,
                limit=limit,
            )
        except (SQLAlchemyError, ValueError) as exc:
            raise ScheduledJobPersistenceError(
                "Scheduled jobs could not be loaded from the database."
            ) from exc
        return tuple(
            ScheduledJobListItem(
                job=self._to_view(job),
                last_run_status=run.status if run is not None else None,
                last_run_at=(
                    _utc_database_timestamp(
                        run.completed_at or run.started_at or run.scheduled_for
                    )
                    if run is not None
                    else None
                ),
            )
            for job, run in rows
        )

    def get_job_for_owner(
        self,
        job_id: str,
        owner_user_id: str,
    ) -> ScheduledJobView | None:
        """Return a job only when it belongs to the authenticated operator."""

        normalized_job_id = self._job_id(job_id)
        owner_id = self._owner_id(owner_user_id)
        try:
            job = self._repository.get_job_for_owner(
                job_id=normalized_job_id,
                owner_user_id=owner_id,
            )
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled job could not be loaded from the database."
            ) from exc
        return self._to_view(job) if job is not None else None

    def list_run_history(
        self,
        job_id: str,
        owner_user_id: str,
        *,
        limit: int = 25,
    ) -> tuple[ScheduledJobRunHistoryItem, ...]:
        """Return bounded execution and output history for one owner job."""

        normalized_job_id = self._job_id(job_id)
        owner_id = self._owner_id(owner_user_id)
        if not 1 <= limit <= 100:
            raise ScheduledJobValidationError(("limit: must be between 1 and 100",))
        try:
            rows = self._repository.list_run_history_for_owner(
                job_id=normalized_job_id,
                owner_user_id=owner_id,
                limit=limit,
            )
        except (SQLAlchemyError, ValueError) as exc:
            raise ScheduledJobPersistenceError(
                "Scheduled-job run history could not be loaded from the database."
            ) from exc
        return tuple(
            ScheduledJobRunHistoryItem(
                run_id=str(run.run_id),
                scheduled_for=_utc_database_timestamp(run.scheduled_for),
                status=run.status,
                attempt_number=run.attempt_number,
                triggered_by=run.triggered_by,
                started_at=(
                    _utc_database_timestamp(run.started_at)
                    if run.started_at is not None
                    else None
                ),
                completed_at=(
                    _utc_database_timestamp(run.completed_at)
                    if run.completed_at is not None
                    else None
                ),
                error_message=run.error_message,
                output_id=(str(output.output_id) if output is not None else None),
                output_type=output.output_type if output is not None else None,
                content=output.content if output is not None else None,
                content_json=(
                    deepcopy(output.content_json)
                    if output is not None and output.content_json is not None
                    else None
                ),
                output_created_at=(
                    _utc_database_timestamp(output.created_at)
                    if output is not None
                    else None
                ),
            )
            for run, output in rows
        )

    @staticmethod
    def _job_id(job_id: str) -> str:
        """Return a normalized scheduled-job UUID string."""

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
    def _to_view(job: ScheduledJob) -> ScheduledJobView:
        """Convert an ORM row into the stable service result shape."""

        return ScheduledJobView(
            job_id=str(job.job_id),
            job_name=job.job_name,
            schema_version=job.schema_version,
            definition=deepcopy(job.definition_json),
            status=job.status,
            is_active=job.is_active,
            created_by_user_id=str(job.created_by_user_id),
            created_by_username=job.created_by_username,
            target_device_id=job.target_device_id,
            timer_unit_name=job.timer_unit_name,
            provisioning_error=job.provisioning_error,
            activated_at=(
                _utc_database_timestamp(job.activated_at)
                if job.activated_at is not None
                else None
            ),
            created_at=_utc_database_timestamp(job.created_at),
            updated_at=_utc_database_timestamp(job.updated_at),
        )
