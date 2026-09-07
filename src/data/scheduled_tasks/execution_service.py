"""Database lifecycle boundary for one-shot scheduled-job executions.

The service keeps runner code independent of SQLAlchemy. It loads only active
job definitions, atomically claims each logical occurrence through the database
unique constraint, records internal retries, and persists either one output or
a terminal failure. It has no Streamlit or systemd dependency.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from furnace_data.relational import (
    ScheduledJobRepository,
    ScheduledJobRun,
    ScheduledJobRunRepository,
    ScheduledJobStatus,
    build_relational_engine,
    build_relational_session_factory,
)
from furnace_data.relational import (
    ScheduledJobRunLeaseBusyError as RelationalScheduledJobRunLeaseBusyError,
)

from .service import (
    ScheduledJobPersistenceError,
    ScheduledJobService,
    ScheduledJobValidationError,
    ScheduledJobView,
)

_AUTHORIZATION_TOKEN_PATTERN = re.compile(
    r"(?i)\b(?P<scheme>bearer|basic)\s+[A-Za-z0-9._~+/=-]+"
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?ix)"
    r"(?P<label_quote>[\"']?)"
    r"(?P<label>\b(?:[a-z][a-z0-9]*[\s_-])*"
    r"(?:api[\s_-]?key|password|passwd|token|secret|credentials?))"
    r"(?P=label_quote)"
    r"(?P<separator>\s*(?::|=|\bis\b)\s*)"
    r"(?P<value>\{[^}]*\}|\[[^]]*\]|\"(?:\\.|[^\"])*\"|"
    r"'(?:\\.|[^'])*'|[^\s,;]+)"
)
_URI_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(?P<scheme>[a-z][a-z0-9+.-]*://)[^\s/:@]+:[^\s/@]+@"
)
_KNOWN_CREDENTIAL_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|AKIA[0-9A-Z]{16}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"eyJ[A-Za-z0-9_-]{20,}(?:\.[A-Za-z0-9_-]+){1,2})"
)


class ScheduledJobNotFoundError(LookupError):
    """Raised when a runner receives an unknown scheduled-job identifier."""


class ScheduledJobNotActiveError(RuntimeError):
    """Raised when a runner is asked to execute a non-active job."""


class ScheduledJobRunStateError(RuntimeError):
    """Raised when a run cannot make the requested lifecycle transition."""


class ScheduledJobRunLeaseBusyError(RuntimeError):
    """Raised when another worker still owns a scheduled-job execution lease."""

    def __init__(self, lease_expires_at: datetime) -> None:
        """Store the database lease deadline for runner retry decisions."""

        self.lease_expires_at = _utc_instant(
            lease_expires_at,
            field_name="lease_expires_at",
        )
        super().__init__(
            "Scheduled-job execution lease remains owned until "
            f"{self.lease_expires_at.isoformat()}."
        )


class ScheduledJobOccurrenceAlreadyClaimedError(RuntimeError):
    """Raised when a requested claim is blocked by an existing job run."""

    def __init__(self, existing_run: ScheduledJobRunView) -> None:
        """Store the run that owns the occurrence or job execution slot."""

        self.existing_run = existing_run
        super().__init__(
            f"Scheduled-job claim was blocked by existing run {existing_run.run_id}."
        )


@dataclass(frozen=True, slots=True)
class ScheduledJobExecutionOutput:
    """Serializable output returned by a scheduled-task executor."""

    output_type: str
    content: str | None = None
    content_json: dict[str, object] | None = None
    artifact_path: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScheduledJobRunView:
    """Database-independent view of one scheduled-job occurrence."""

    run_id: str
    job_id: str
    scheduled_for: datetime
    status: str
    attempt_number: int
    activation_generation: int
    lease_token: str | None
    lease_expires_at: datetime | None
    attempt_deadline_at: datetime | None
    retry_not_before_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    triggered_by: str | None
    error_message: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ScheduledJobRunClaim:
    """Active job definition paired with its newly claimed run row."""

    job: ScheduledJobView
    run: ScheduledJobRunView


@dataclass(frozen=True, slots=True)
class ScheduledJobRunCompletion:
    """Completed run plus the persisted output identifier."""

    run: ScheduledJobRunView
    output_id: str


def _utc_instant(value: datetime, *, field_name: str) -> datetime:
    """Validate an aware timestamp and normalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ScheduledJobValidationError(
            (f"{field_name}: timezone-aware timestamp is required",)
        )
    return value.astimezone(timezone.utc)


def _stored_utc_instant(value: datetime, *, field_name: str) -> datetime:
    """Normalize an ORM timestamp, including SQLite's timezone-naive UTC form."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return _utc_instant(value, field_name=field_name)


def safe_scheduled_job_error_message(error: BaseException | str) -> str:
    """Return a redacted, bounded single-line error suitable for run history."""

    message = " ".join(str(error).split()).strip()
    message = _URI_CREDENTIAL_PATTERN.sub(r"\g<scheme>[REDACTED]@", message)
    message = _AUTHORIZATION_TOKEN_PATTERN.sub(
        lambda match: f"{match.group('scheme')} [REDACTED]",
        message,
    )
    message = _CREDENTIAL_ASSIGNMENT_PATTERN.sub(
        lambda match: (
            f"{match.group('label_quote')}{match.group('label')}"
            f"{match.group('label_quote')}{match.group('separator')}[REDACTED]"
        ),
        message,
    )
    message = _KNOWN_CREDENTIAL_PATTERN.sub("[REDACTED]", message)
    return (message or type(error).__name__)[:4000]


class ScheduledJobExecutionService:
    """Coordinate active-job checks and durable run lifecycle transitions."""

    def __init__(
        self,
        db_url: str | None = None,
        *,
        job_repository: ScheduledJobRepository | None = None,
        run_repository: ScheduledJobRunRepository | None = None,
    ) -> None:
        """Create a database-backed service or use injected test repositories."""

        self._engine: Engine | None = None
        if job_repository is not None or run_repository is not None:
            if job_repository is None or run_repository is None:
                raise ValueError("Both scheduled-job repositories must be provided.")
            self._job_service = ScheduledJobService(repository=job_repository)
            self._run_repository = run_repository
            return

        self._engine = build_relational_engine(db_url=db_url)
        session_factory = build_relational_session_factory(self._engine)
        self._job_service = ScheduledJobService(
            repository=ScheduledJobRepository(session_factory)
        )
        self._run_repository = ScheduledJobRunRepository(session_factory)

    def dispose(self) -> None:
        """Dispose the owned SQLAlchemy engine, when one was created."""

        if self._engine is not None:
            self._engine.dispose()

    def load_job(self, job_id: str) -> ScheduledJobView:
        """Return a stored job or raise a runner-specific lookup error."""

        job = self._job_service.get_job(job_id)
        if job is None:
            raise ScheduledJobNotFoundError(f"Scheduled job {job_id} was not found.")
        return job

    def load_active_job(self, job_id: str) -> ScheduledJobView:
        """Return an active job or raise a runner-specific lifecycle error."""

        job = self.load_job(job_id)
        if (
            job.status != ScheduledJobStatus.ACTIVE.value
            or not job.is_active
            or job.activated_at is None
        ):
            raise ScheduledJobNotActiveError(
                f"Scheduled job {job.job_id} is not active (status={job.status})."
            )
        return job

    def claim_run(
        self,
        *,
        job_id: str,
        scheduled_for: datetime,
        triggered_by: str = "systemd",
        timeout_seconds: int = 600,
        lease_seconds: int = 90,
    ) -> ScheduledJobRunClaim:
        """Claim an active occurrence with a fenced execution lease."""

        job = self.load_active_job(job_id)
        occurrence = _utc_instant(scheduled_for, field_name="scheduled_for")
        clean_trigger = triggered_by.strip()
        if not clean_trigger:
            raise ScheduledJobValidationError(
                ("triggered_by: non-empty value is required",)
            )

        try:
            run = self._run_repository.create_run(
                job_id=job.job_id,
                scheduled_for=occurrence,
                triggered_by=clean_trigger,
                timeout_seconds=timeout_seconds,
                lease_seconds=lease_seconds,
            )
        except ValueError as exc:
            raise ScheduledJobNotActiveError(str(exc)) from exc
        except IntegrityError as exc:
            try:
                existing = self._run_repository.get_run_for_occurrence(
                    job_id=job.job_id,
                    scheduled_for=occurrence,
                )
                if existing is None:
                    existing = self._run_repository.get_running_run_for_job(
                        job_id=job.job_id
                    )
            except SQLAlchemyError as lookup_exc:
                raise ScheduledJobPersistenceError(
                    "The existing scheduled-job occurrence could not be loaded."
                ) from lookup_exc
            if existing is None:
                raise ScheduledJobPersistenceError(
                    "The scheduled-job occurrence could not be claimed."
                ) from exc
            raise ScheduledJobOccurrenceAlreadyClaimedError(
                self._to_run_view(existing)
            ) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job occurrence could not be claimed."
            ) from exc
        return ScheduledJobRunClaim(job=job, run=self._to_run_view(run))

    def recover_or_get_running_run(
        self,
        *,
        job_id: str,
        stale_before: datetime,
        triggered_by: str,
        maximum_attempts: int,
        timeout_seconds: int | None = None,
        lease_seconds: int = 90,
        retry_interval_seconds: int | None = None,
    ) -> tuple[ScheduledJobRunView, bool] | None:
        """Return a live lease or atomically reclaim an expired occurrence."""

        normalized_job_id = self._job_id(job_id)
        cutoff = _utc_instant(stale_before, field_name="stale_before")
        clean_trigger = triggered_by.strip()
        if not clean_trigger:
            raise ScheduledJobValidationError(
                ("triggered_by: non-empty value is required",)
            )
        if maximum_attempts < 1:
            raise ScheduledJobValidationError(
                ("maximum_attempts: positive integer is required",)
            )
        try:
            resolution = self._run_repository.recover_or_get_running_run(
                job_id=normalized_job_id,
                stale_before=cutoff,
                triggered_by=clean_trigger,
                maximum_attempts=maximum_attempts,
                timeout_seconds=timeout_seconds,
                lease_seconds=lease_seconds,
                retry_interval_seconds=retry_interval_seconds,
            )
        except ValueError as exc:
            raise ScheduledJobNotActiveError(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The running scheduled-job occurrence could not be inspected."
            ) from exc
        if resolution is None:
            return None
        run, recovered = resolution
        return self._to_run_view(run), recovered

    def record_retry(
        self,
        *,
        run_id: str,
        attempt_number: int,
        previous_error: BaseException | str,
        expected_lease_token: str,
        expected_attempt_number: int,
        timeout_seconds: int = 600,
        lease_seconds: int = 90,
    ) -> ScheduledJobRunView:
        """Advance directly to a fenced retry after a completed backoff."""

        normalized_run_id = self._run_id(run_id)
        if attempt_number < 2:
            raise ScheduledJobValidationError(
                ("attempt_number: retry attempts begin at 2",)
            )
        try:
            run = self._run_repository.record_retry(
                run_id=normalized_run_id,
                attempt_number=attempt_number,
                previous_error=safe_scheduled_job_error_message(previous_error),
                expected_lease_token=self._lease_token(expected_lease_token),
                expected_attempt_number=expected_attempt_number,
                timeout_seconds=timeout_seconds,
                lease_seconds=lease_seconds,
            )
        except ValueError as exc:
            raise ScheduledJobRunStateError(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job retry could not be recorded."
            ) from exc
        if run is None:
            raise ScheduledJobNotFoundError(f"Scheduled run {run_id} was not found.")
        return self._to_run_view(run)

    def renew_lease(
        self,
        *,
        run_id: str,
        expected_lease_token: str,
        expected_attempt_number: int,
        lease_seconds: int = 90,
    ) -> ScheduledJobRunView:
        """Renew a fenced execution lease without reviving an old worker."""

        normalized_run_id = self._run_id(run_id)
        try:
            run = self._run_repository.renew_lease(
                run_id=normalized_run_id,
                expected_lease_token=self._lease_token(expected_lease_token),
                expected_attempt_number=expected_attempt_number,
                lease_seconds=lease_seconds,
            )
        except ValueError as exc:
            raise ScheduledJobRunStateError(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job execution lease could not be renewed."
            ) from exc
        if run is None:
            raise ScheduledJobNotFoundError(f"Scheduled run {run_id} was not found.")
        return self._to_run_view(run)

    def schedule_retry(
        self,
        *,
        run_id: str,
        expected_lease_token: str,
        expected_attempt_number: int,
        previous_error: BaseException | str,
        retry_interval_seconds: int,
        lease_seconds: int = 90,
    ) -> ScheduledJobRunView:
        """Persist a fenced retry and its not-before time before backoff."""

        normalized_run_id = self._run_id(run_id)
        try:
            run = self._run_repository.schedule_retry(
                run_id=normalized_run_id,
                expected_lease_token=self._lease_token(expected_lease_token),
                expected_attempt_number=expected_attempt_number,
                previous_error=safe_scheduled_job_error_message(previous_error),
                retry_interval_seconds=retry_interval_seconds,
                lease_seconds=lease_seconds,
            )
        except ValueError as exc:
            raise ScheduledJobRunStateError(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job retry could not be scheduled."
            ) from exc
        if run is None:
            raise ScheduledJobNotFoundError(f"Scheduled run {run_id} was not found.")
        return self._to_run_view(run)

    def begin_retry_attempt(
        self,
        *,
        run_id: str,
        expected_lease_token: str,
        expected_attempt_number: int,
        timeout_seconds: int,
        lease_seconds: int = 90,
    ) -> ScheduledJobRunView:
        """Begin a fenced retry after its durable backoff has elapsed."""

        normalized_run_id = self._run_id(run_id)
        try:
            run = self._run_repository.begin_retry_attempt(
                run_id=normalized_run_id,
                expected_lease_token=self._lease_token(expected_lease_token),
                expected_attempt_number=expected_attempt_number,
                timeout_seconds=timeout_seconds,
                lease_seconds=lease_seconds,
            )
        except RelationalScheduledJobRunLeaseBusyError as exc:
            raise ScheduledJobRunLeaseBusyError(exc.lease_expires_at) from exc
        except ValueError as exc:
            raise ScheduledJobRunStateError(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job retry could not begin."
            ) from exc
        if run is None:
            raise ScheduledJobNotFoundError(f"Scheduled run {run_id} was not found.")
        return self._to_run_view(run)

    def recover_stale_run(
        self,
        *,
        run_id: str,
        stale_before: datetime,
        triggered_by: str,
        maximum_attempts: int,
        timeout_seconds: int | None = None,
        lease_seconds: int = 90,
        retry_interval_seconds: int | None = None,
    ) -> ScheduledJobRunView:
        """Retry or time out a running occurrence whose claim became stale."""

        normalized_run_id = self._run_id(run_id)
        cutoff = _utc_instant(stale_before, field_name="stale_before")
        clean_trigger = triggered_by.strip()
        if not clean_trigger:
            raise ScheduledJobValidationError(
                ("triggered_by: non-empty value is required",)
            )
        if maximum_attempts < 1:
            raise ScheduledJobValidationError(
                ("maximum_attempts: positive integer is required",)
            )
        try:
            run = self._run_repository.recover_stale_run(
                run_id=normalized_run_id,
                stale_before=cutoff,
                triggered_by=clean_trigger,
                maximum_attempts=maximum_attempts,
                timeout_seconds=timeout_seconds,
                lease_seconds=lease_seconds,
                retry_interval_seconds=retry_interval_seconds,
            )
        except ValueError as exc:
            raise ScheduledJobNotActiveError(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The stale scheduled-job occurrence could not be recovered."
            ) from exc
        if run is None:
            raise ScheduledJobRunStateError(
                "The scheduled-job occurrence is no longer stale and running."
            )
        return self._to_run_view(run)

    def complete_run(
        self,
        *,
        run_id: str,
        expected_lease_token: str,
        expected_attempt_number: int,
        output: ScheduledJobExecutionOutput,
    ) -> ScheduledJobRunCompletion:
        """Persist one executor output and mark its running occurrence complete."""

        normalized_run_id = self._run_id(run_id)
        self.validate_output(output)
        try:
            completed = self._run_repository.complete_run(
                run_id=normalized_run_id,
                expected_lease_token=self._lease_token(expected_lease_token),
                expected_attempt_number=expected_attempt_number,
                output_type=output.output_type.strip(),
                content=output.content,
                content_json=output.content_json,
                artifact_path=output.artifact_path,
                output_metadata=dict(output.metadata),
            )
        except ValueError as exc:
            raise ScheduledJobRunStateError(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job completion could not be stored."
            ) from exc
        if completed is None:
            raise ScheduledJobNotFoundError(f"Scheduled run {run_id} was not found.")
        run, persisted_output = completed
        return ScheduledJobRunCompletion(
            run=self._to_run_view(run),
            output_id=str(persisted_output.output_id),
        )

    def cancel_run(
        self,
        *,
        run_id: str,
        expected_lease_token: str,
        expected_attempt_number: int,
        reason: BaseException | str,
    ) -> ScheduledJobRunView:
        """Cooperatively cancel one owned run and clear its execution lease."""

        normalized_run_id = self._run_id(run_id)
        try:
            run = self._run_repository.cancel_run(
                run_id=normalized_run_id,
                expected_lease_token=self._lease_token(expected_lease_token),
                expected_attempt_number=expected_attempt_number,
                reason=safe_scheduled_job_error_message(reason),
            )
        except ValueError as exc:
            raise ScheduledJobRunStateError(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job cancellation could not be stored."
            ) from exc
        if run is None:
            raise ScheduledJobNotFoundError(f"Scheduled run {run_id} was not found.")
        return self._to_run_view(run)

    def fail_run(
        self,
        *,
        run_id: str,
        expected_lease_token: str,
        expected_attempt_number: int,
        error: BaseException | str,
        timed_out: bool = False,
    ) -> ScheduledJobRunView:
        """Mark a running occurrence as failed or timed out."""

        normalized_run_id = self._run_id(run_id)
        try:
            run = self._run_repository.fail_run(
                run_id=normalized_run_id,
                expected_lease_token=self._lease_token(expected_lease_token),
                expected_attempt_number=expected_attempt_number,
                error_message=safe_scheduled_job_error_message(error),
                timed_out=timed_out,
            )
        except ValueError as exc:
            raise ScheduledJobRunStateError(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                "The scheduled-job failure could not be stored."
            ) from exc
        if run is None:
            raise ScheduledJobNotFoundError(f"Scheduled run {run_id} was not found.")
        return self._to_run_view(run)

    @staticmethod
    def _job_id(job_id: str) -> str:
        """Return a validated scheduled-job UUID as normalized text."""

        try:
            return str(UUID(str(job_id).strip()))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScheduledJobValidationError(
                ("job_id: valid UUID is required",)
            ) from exc

    @staticmethod
    def _run_id(run_id: str) -> UUID:
        """Return a validated run UUID."""

        try:
            return UUID(str(run_id).strip())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScheduledJobValidationError(
                ("run_id: valid UUID is required",)
            ) from exc

    @staticmethod
    def _lease_token(lease_token: str) -> UUID:
        """Return a validated opaque execution-lease token."""

        try:
            return UUID(str(lease_token).strip())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScheduledJobValidationError(
                ("expected_lease_token: valid UUID is required",)
            ) from exc

    @staticmethod
    def validate_output(output: ScheduledJobExecutionOutput) -> None:
        """Reject empty or non-JSON-serializable executor output."""

        if not isinstance(output.output_type, str) or not output.output_type.strip():
            raise ScheduledJobValidationError(
                ("output.output_type: non-empty value is required",)
            )
        if output.content is not None and not isinstance(output.content, str):
            raise ScheduledJobValidationError(
                ("output.content: string or null is required",)
            )
        if output.artifact_path is not None and not isinstance(
            output.artifact_path, str
        ):
            raise ScheduledJobValidationError(
                ("output.artifact_path: string or null is required",)
            )
        if output.content_json is not None and not isinstance(
            output.content_json, dict
        ):
            raise ScheduledJobValidationError(
                ("output.content_json: JSON object or null is required",)
            )
        if not isinstance(output.metadata, dict):
            raise ScheduledJobValidationError(
                ("output.metadata: JSON object is required",)
            )
        try:
            json.dumps(output.content_json, allow_nan=False)
            json.dumps(output.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ScheduledJobValidationError(
                ("output: JSON values must be finite and serializable",)
            ) from exc

    @staticmethod
    def _to_run_view(run: ScheduledJobRun) -> ScheduledJobRunView:
        """Convert a run ORM row into the stable service result shape."""

        return ScheduledJobRunView(
            run_id=str(run.run_id),
            job_id=run.job_id,
            scheduled_for=_stored_utc_instant(
                run.scheduled_for,
                field_name="scheduled_for",
            ),
            status=run.status,
            attempt_number=run.attempt_number,
            activation_generation=run.activation_generation,
            lease_token=(str(run.lease_token) if run.lease_token is not None else None),
            lease_expires_at=(
                _stored_utc_instant(
                    run.lease_expires_at,
                    field_name="lease_expires_at",
                )
                if run.lease_expires_at is not None
                else None
            ),
            attempt_deadline_at=(
                _stored_utc_instant(
                    run.attempt_deadline_at,
                    field_name="attempt_deadline_at",
                )
                if run.attempt_deadline_at is not None
                else None
            ),
            retry_not_before_at=(
                _stored_utc_instant(
                    run.retry_not_before_at,
                    field_name="retry_not_before_at",
                )
                if run.retry_not_before_at is not None
                else None
            ),
            started_at=(
                _stored_utc_instant(run.started_at, field_name="started_at")
                if run.started_at is not None
                else None
            ),
            completed_at=(
                _stored_utc_instant(run.completed_at, field_name="completed_at")
                if run.completed_at is not None
                else None
            ),
            triggered_by=run.triggered_by,
            error_message=run.error_message,
            created_at=_stored_utc_instant(run.created_at, field_name="created_at"),
        )
