"""Tests for the fenced Phase 5 one-shot scheduled-job runner and CLI."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import furnace_data.relational.repositories as relational_repositories
from data.scheduled_tasks import (
    ScheduledJobCreateRequest,
    ScheduledJobExecutionOutput,
    ScheduledJobExecutionService,
    ScheduledJobNotActiveError,
    ScheduledJobOccurrenceAlreadyClaimedError,
    ScheduledJobRunLeaseBusyError,
    ScheduledJobRunStateError,
    ScheduledJobService,
    ScheduledJobValidationError,
)
from furnace_data.relational import (
    ScheduledJob,
    ScheduledJobOutput,
    ScheduledJobRepository,
    ScheduledJobRevision,
    ScheduledJobRun,
    ScheduledJobRunLog,
    ScheduledJobRunRepository,
    ScheduledJobRunStatus,
    ScheduledJobStatus,
)
from utils.scheduled_tasks.job_runner import (
    ScheduledJobRunner,
    ScheduledJobRunnerResult,
    ScheduledTaskAttemptTimedOut,
    ScheduledTaskExecutionCancelled,
    ScheduledTaskExecutionFailed,
    main,
    scheduled_job_result_exit_code,
)
from utils.scheduled_tasks.schedule_occurrence import ScheduledOccurrenceNotDueError
from utils.scheduled_tasks.scheduled_task_definition import (
    ScheduledTaskInput,
    build_task_definition,
)

_USER_ID = "00000000-0000-0000-0000-000000000001"
_RUNNER_NOW = datetime(2026, 9, 5, 10, 7, tzinfo=timezone.utc)


class _ControlledClock:
    """Keep runner waits and repository database time on one deterministic clock."""

    def __init__(self, current: datetime) -> None:
        """Initialize the shared aware timestamp and recorded delays."""

        self.current = current
        self.delays: list[float] = []

    def now(self) -> datetime:
        """Return the current deterministic wall-clock instant."""

        return self.current

    def database_now(self, _session: Session) -> datetime:
        """Return the same instant for repository lease and retry decisions."""

        return self.current

    def sleep(self, seconds: float) -> None:
        """Advance both clocks without making the test process wait."""

        self.delays.append(seconds)
        self.current += timedelta(seconds=seconds)


def _definition(*, maximum_attempts: int = 3) -> dict[str, object]:
    """Build the representative hourly definition executed by runner tests."""

    task = ScheduledTaskInput(
        name="Hourly BF2 review",
        instructions="Prepare a concise BF2 operator report.",
        furnace="BF2",
        data_period="last_24_hours",
        output_format="operator_summary",
        schedule_kind="hourly",
        delivery_channel="in_app",
        job_type="furnace_summary",
        hourly_minute=5,
        maximum_attempts=maximum_attempts,
        retry_interval_seconds=1,
    )
    return build_task_definition(
        task,
        generated_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def runner_stack():
    """Create schema-translated automation tables for one runner test."""

    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        execution_options={
            "schema_translate_map": {"automation": None, "identity": None}
        },
    )
    ScheduledJob.__table__.create(engine)
    ScheduledJobRevision.__table__.create(engine)
    ScheduledJobRun.__table__.create(engine)
    ScheduledJobRunLog.__table__.create(engine)
    ScheduledJobOutput.__table__.create(engine)
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
        future=True,
    )
    job_repository = ScheduledJobRepository(factory)
    execution_service = ScheduledJobExecutionService(
        job_repository=job_repository,
        run_repository=ScheduledJobRunRepository(factory),
    )
    job_service = ScheduledJobService(repository=job_repository)
    try:
        yield job_service, execution_service, factory
    finally:
        engine.dispose()


@pytest.fixture
def controlled_clock(monkeypatch: pytest.MonkeyPatch) -> _ControlledClock:
    """Bind repository database time to the runner's deterministic test clock."""

    clock = _ControlledClock(_RUNNER_NOW)
    monkeypatch.setattr(
        relational_repositories,
        "_database_utc_now",
        clock.database_now,
    )
    return clock


def _create_job(
    stack,
    *,
    active: bool,
    maximum_attempts: int = 3,
):
    """Persist a test definition and optionally activate it for execution."""

    job_service, _, factory = stack
    job = job_service.create_job(
        ScheduledJobCreateRequest(
            definition=_definition(maximum_attempts=maximum_attempts),
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )
    if active:
        activated = ScheduledJobRepository(factory).activate_job(
            job_id=job.job_id,
            expected_status=job.status,
            expected_updated_at=job.updated_at,
            timer_unit_name=f"furnacemind-job-{job.job_id}.timer",
            activated_at=_RUNNER_NOW - timedelta(days=1),
        )
        assert activated is not None
    return job


class _SuccessfulExecutor:
    """Record calls and return a small structured report output."""

    def __init__(self) -> None:
        """Initialize an empty call list."""

        self.calls: list[tuple[str, datetime]] = []

    def execute(self, *, job, scheduled_for):
        """Return deterministic output for the claimed job occurrence."""

        self.calls.append((job.job_id, scheduled_for))
        return ScheduledJobExecutionOutput(
            output_type="operator_report",
            content="BF2 operation is stable.",
            content_json={"condition": "stable"},
        )


class _FailingExecutor:
    """Fail every execution attempt for retry lifecycle tests."""

    def __init__(self) -> None:
        """Initialize the attempt counter."""

        self.attempts = 0

    def execute(self, *, job, scheduled_for):
        """Raise a deterministic executor failure."""

        del job, scheduled_for
        self.attempts += 1
        raise RuntimeError("historian unavailable")


class _EventuallySuccessfulExecutor:
    """Fail once and then return a valid report output."""

    def __init__(self) -> None:
        """Initialize the attempt counter."""

        self.attempts = 0

    def execute(self, *, job, scheduled_for):
        """Return output on the second call."""

        del job, scheduled_for
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("temporary historian error")
        return ScheduledJobExecutionOutput(
            output_type="operator_report",
            content_json={"condition": "stable"},
        )


class _InvalidOutputExecutor:
    """Return an output that violates the executor contract."""

    def __init__(self) -> None:
        """Initialize the attempt counter."""

        self.attempts = 0

    def execute(self, *, job, scheduled_for):
        """Return an invalid empty output type."""

        del job, scheduled_for
        self.attempts += 1
        return ScheduledJobExecutionOutput(output_type="")


class _SensitiveFailingExecutor:
    """Raise an executor error containing representative credential formats."""

    def execute(self, *, job, scheduled_for):
        """Raise a message whose sensitive values must never be persisted or logged."""

        del job, scheduled_for
        raise RuntimeError(
            "historian password=hunter2 api_key='key value' token: abc123 "
            "Bearer eyJhbGciOiJIUzI1NiJ9.payload "
            "Basic dXNlcjpwYXNz credentials=ops:pw "
            "postgresql://operator:dbpass@db.local/furnace "
            "details={'password': 'nested-secret'}"
        )


def test_runner_completes_once_and_persists_logs_and_output(runner_stack) -> None:
    """A successful occurrence should produce one run, output, and audit trail."""

    job = _create_job(runner_stack, active=True)
    _, execution_service, factory = runner_stack
    executor = _SuccessfulExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=lambda _: None,
    )

    result = runner.run(job.job_id, now=_RUNNER_NOW)

    assert result.duplicate is False
    assert result.run.status == ScheduledJobRunStatus.COMPLETED.value
    assert result.output_id is not None
    assert len(executor.calls) == 1
    with factory() as session:
        run = session.execute(select(ScheduledJobRun)).scalar_one()
        logs = (
            session.execute(select(ScheduledJobRunLog).order_by(ScheduledJobRunLog.id))
            .scalars()
            .all()
        )
        output = session.execute(select(ScheduledJobOutput)).scalar_one()
    assert run.attempt_number == 1
    assert run.completed_at is not None
    assert [item.log_level for item in logs] == ["info", "info"]
    assert output.content_json == {"condition": "stable"}


def test_runner_does_not_commit_after_final_lease_check_fails(
    runner_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loss of fenced ownership after execution must prevent output persistence."""

    job = _create_job(runner_stack, active=True)
    _, execution_service, factory = runner_stack

    def _reject_renewal(**_kwargs):
        """Simulate another worker superseding this process's lease token."""

        raise ScheduledJobRunStateError("execution lease is no longer owned")

    monkeypatch.setattr(execution_service, "renew_lease", _reject_renewal)
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=_SuccessfulExecutor(),
    )

    with pytest.raises(ScheduledJobRunStateError, match="no longer owned"):
        runner.run(job.job_id, now=_RUNNER_NOW)

    with factory() as session:
        run = session.execute(select(ScheduledJobRun)).scalar_one()
        outputs = session.execute(select(ScheduledJobOutput)).scalars().all()
    assert run.status == ScheduledJobRunStatus.RUNNING.value
    assert outputs == []


def test_runner_cooperatively_cancels_an_owned_run(runner_stack) -> None:
    """A termination request must fence and persist cancellation before exit."""

    job = _create_job(runner_stack, active=True)
    _, execution_service, factory = runner_stack
    executor = _SuccessfulExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        cancellation_requested=lambda: True,
    )

    with pytest.raises(ScheduledTaskExecutionCancelled) as exc_info:
        runner.run(job.job_id, now=_RUNNER_NOW)

    assert exc_info.value.run.status == ScheduledJobRunStatus.CANCELLED.value
    assert executor.calls == []
    with factory() as session:
        run = session.execute(select(ScheduledJobRun)).scalar_one()
        outputs = session.execute(select(ScheduledJobOutput)).scalars().all()
    assert run.status == ScheduledJobRunStatus.CANCELLED.value
    assert run.lease_token is None
    assert outputs == []


def test_one_time_occurrence_equal_to_activation_boundary_executes(
    runner_stack,
) -> None:
    """A delayed initial one-time activation may claim its exact due instant."""

    job_service, execution_service, factory = runner_stack
    definition = build_task_definition(
        ScheduledTaskInput(
            name="One-time BF2 review",
            instructions="Prepare the requested one-time BF2 review.",
            furnace="BF2",
            data_period="last_24_hours",
            output_format="operator_summary",
            schedule_kind="once",
            delivery_channel="in_app",
            job_type="furnace_summary",
            run_date=date(2026, 9, 5),
            run_time=time(15, 37),
        ),
        generated_at=_RUNNER_NOW - timedelta(hours=1),
    )
    job = job_service.create_job(
        ScheduledJobCreateRequest(
            definition=definition,
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )
    activated = ScheduledJobRepository(factory).activate_job(
        job_id=job.job_id,
        expected_status=job.status,
        expected_updated_at=job.updated_at,
        timer_unit_name=f"furnacemind-job-{job.job_id}.timer",
        activated_at=_RUNNER_NOW,
    )
    assert activated is not None
    executor = _SuccessfulExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=lambda _: None,
    )

    result = runner.run(job.job_id, now=_RUNNER_NOW)

    assert result.run.scheduled_for.replace(tzinfo=timezone.utc) == _RUNNER_NOW
    assert executor.calls == [(job.job_id, _RUNNER_NOW)]
    terminal_job = job_service.get_job(job.job_id)
    assert terminal_job is not None
    assert terminal_job.status == ScheduledJobStatus.COMPLETED.value
    assert terminal_job.is_active is False
    assert terminal_job.activated_at is None
    assert terminal_job.timer_unit_name is None
    assert terminal_job.provisioning_error == "Timer cleanup is pending."


def test_one_time_terminal_failure_retires_the_job(runner_stack) -> None:
    """A final one-time executor failure must retire the timer-backed definition."""

    job_service, execution_service, factory = runner_stack
    definition = build_task_definition(
        ScheduledTaskInput(
            name="One-time failing BF2 review",
            instructions="Attempt the requested BF2 review once.",
            furnace="BF2",
            data_period="last_24_hours",
            output_format="operator_summary",
            schedule_kind="once",
            delivery_channel="in_app",
            job_type="furnace_summary",
            run_date=date(2026, 9, 5),
            run_time=time(15, 37),
            maximum_attempts=1,
        ),
        generated_at=_RUNNER_NOW - timedelta(hours=1),
    )
    job = job_service.create_job(
        ScheduledJobCreateRequest(
            definition=definition,
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )
    activated = ScheduledJobRepository(factory).activate_job(
        job_id=job.job_id,
        expected_status=job.status,
        expected_updated_at=job.updated_at,
        timer_unit_name=f"furnacemind-job-{job.job_id}.timer",
        activated_at=_RUNNER_NOW,
    )
    assert activated is not None
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=_FailingExecutor(),
    )

    with pytest.raises(ScheduledTaskExecutionFailed):
        runner.run(job.job_id, now=_RUNNER_NOW)

    terminal_job = job_service.get_job(job.job_id)
    assert terminal_job is not None
    assert terminal_job.status == ScheduledJobStatus.EXECUTION_FAILED.value
    assert terminal_job.is_active is False
    assert terminal_job.activated_at is None
    assert terminal_job.timer_unit_name is None
    assert terminal_job.provisioning_error == "Timer cleanup is pending."


def test_runner_treats_repeated_timer_delivery_as_idempotent(runner_stack) -> None:
    """The same job and logical occurrence must never execute twice."""

    job = _create_job(runner_stack, active=True)
    _, execution_service, factory = runner_stack
    executor = _SuccessfulExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=lambda _: None,
    )

    first = runner.run(job.job_id, now=_RUNNER_NOW)
    duplicate = runner.run(job.job_id, now=_RUNNER_NOW)

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.run.run_id == first.run.run_id
    assert len(executor.calls) == 1
    with factory() as session:
        assert len(session.execute(select(ScheduledJobRun)).scalars().all()) == 1


def test_different_occurrence_claim_reports_existing_running_overlap(
    runner_stack,
) -> None:
    """The one-running index race should resolve to the typed overlap signal."""

    job = _create_job(runner_stack, active=True)
    _, execution_service, _ = runner_stack
    existing = execution_service.claim_run(
        job_id=job.job_id,
        scheduled_for=datetime(2026, 9, 5, 8, 35, tzinfo=timezone.utc),
    )

    with pytest.raises(ScheduledJobOccurrenceAlreadyClaimedError) as exc_info:
        execution_service.claim_run(
            job_id=job.job_id,
            scheduled_for=datetime(2026, 9, 5, 9, 35, tzinfo=timezone.utc),
        )

    assert exc_info.value.existing_run.run_id == existing.run.run_id
    assert "claim was blocked by existing run" in str(exc_info.value)


def test_runner_records_retries_and_terminal_failure(
    runner_stack,
    controlled_clock: _ControlledClock,
) -> None:
    """Configured attempts should share one occurrence and end in failed state."""

    job = _create_job(runner_stack, active=True, maximum_attempts=3)
    _, execution_service, factory = runner_stack
    executor = _FailingExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=controlled_clock.sleep,
        wall_clock=controlled_clock.now,
    )

    with pytest.raises(ScheduledTaskExecutionFailed) as exc_info:
        runner.run(job.job_id, now=_RUNNER_NOW)

    assert exc_info.value.run.status == ScheduledJobRunStatus.FAILED.value
    assert executor.attempts == 3
    assert controlled_clock.delays == [1, 1]
    with factory() as session:
        run = session.execute(select(ScheduledJobRun)).scalar_one()
        logs = (
            session.execute(select(ScheduledJobRunLog).order_by(ScheduledJobRunLog.id))
            .scalars()
            .all()
        )
        outputs = session.execute(select(ScheduledJobOutput)).scalars().all()
    assert run.attempt_number == 3
    assert run.error_message == "historian unavailable"
    assert [item.log_level for item in logs] == [
        "info",
        "warning",
        "warning",
        "error",
    ]
    assert outputs == []


def test_executor_errors_are_redacted_in_database_and_process_logs(
    runner_stack,
    controlled_clock: _ControlledClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Credentials from executor exceptions must not reach history or journald."""

    job = _create_job(runner_stack, active=True, maximum_attempts=2)
    _, execution_service, factory = runner_stack
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=_SensitiveFailingExecutor(),
        sleep=controlled_clock.sleep,
        wall_clock=controlled_clock.now,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ScheduledTaskExecutionFailed):
            runner.run(job.job_id, now=_RUNNER_NOW)

    with factory() as session:
        run = session.execute(select(ScheduledJobRun)).scalar_one()
        logs = session.execute(select(ScheduledJobRunLog)).scalars().all()
    persisted_text = " ".join(
        [run.error_message or "", *(json.dumps(item.metadata_json) for item in logs)]
    )
    exposed_text = f"{persisted_text} {caplog.text}"
    for secret in (
        "hunter2",
        "key value",
        "abc123",
        "eyJhbGciOiJIUzI1NiJ9.payload",
        "dXNlcjpwYXNz",
        "ops:pw",
        "operator:dbpass",
        "nested-secret",
    ):
        assert secret not in exposed_text
    assert "[REDACTED]" in persisted_text
    assert "[REDACTED]" in caplog.text


def test_runner_can_complete_after_a_transient_failure(
    runner_stack,
    controlled_clock: _ControlledClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later successful attempt should clear the transient run error."""

    job = _create_job(runner_stack, active=True, maximum_attempts=3)
    _, execution_service, factory = runner_stack
    executor = _EventuallySuccessfulExecutor()
    transition_order: list[str] = []
    original_schedule_retry = execution_service.schedule_retry
    original_begin_retry = execution_service.begin_retry_attempt

    def _schedule_retry(**kwargs):
        """Record that retry intent became durable before the wait began."""

        transition_order.append("schedule-retry")
        return original_schedule_retry(**kwargs)

    def _begin_retry(**kwargs):
        """Record that the next attempt began only after its durable wait."""

        transition_order.append("begin-retry")
        return original_begin_retry(**kwargs)

    def _sleep(seconds: float) -> None:
        """Record and advance the clock shared by runner and repository."""

        transition_order.append("sleep")
        controlled_clock.sleep(seconds)

    monkeypatch.setattr(execution_service, "schedule_retry", _schedule_retry)
    monkeypatch.setattr(execution_service, "begin_retry_attempt", _begin_retry)
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=_sleep,
        wall_clock=controlled_clock.now,
    )

    result = runner.run(job.job_id, now=_RUNNER_NOW)

    assert result.run.status == ScheduledJobRunStatus.COMPLETED.value
    assert result.run.attempt_number == 2
    assert result.run.error_message is None
    assert executor.attempts == 2
    assert transition_order == ["schedule-retry", "sleep", "begin-retry"]
    with factory() as session:
        logs = (
            session.execute(select(ScheduledJobRunLog).order_by(ScheduledJobRunLog.id))
            .scalars()
            .all()
        )
    assert [item.log_level for item in logs] == ["info", "warning", "info"]


def test_invalid_executor_output_enters_retry_and_failure_lifecycle(
    runner_stack,
    controlled_clock: _ControlledClock,
) -> None:
    """Malformed executor output must not leave an occurrence running forever."""

    job = _create_job(runner_stack, active=True, maximum_attempts=2)
    _, execution_service, factory = runner_stack
    executor = _InvalidOutputExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=controlled_clock.sleep,
        wall_clock=controlled_clock.now,
    )

    with pytest.raises(ScheduledTaskExecutionFailed) as exc_info:
        runner.run(job.job_id, now=_RUNNER_NOW)

    assert executor.attempts == 2
    assert exc_info.value.run.status == ScheduledJobRunStatus.FAILED.value
    assert "output.output_type" in (exc_info.value.run.error_message or "")
    with factory() as session:
        assert session.execute(select(ScheduledJobOutput)).scalars().all() == []


@pytest.mark.parametrize(
    "output",
    [
        ScheduledJobExecutionOutput(
            output_type="operator_report",
            content_json={"reading": float("nan")},
        ),
        ScheduledJobExecutionOutput(
            output_type="operator_report",
            metadata={"limit": float("inf")},
        ),
    ],
)
def test_output_contract_rejects_non_finite_json_numbers(
    output: ScheduledJobExecutionOutput,
) -> None:
    """PostgreSQL-incompatible NaN and infinity values must fail validation."""

    with pytest.raises(ScheduledJobValidationError, match="finite"):
        ScheduledJobExecutionService.validate_output(output)


def test_attempt_timeout_retries_then_records_timed_out_state(
    runner_stack,
    controlled_clock: _ControlledClock,
) -> None:
    """A deadline error on the last configured attempt should persist timeout."""

    job = _create_job(runner_stack, active=True, maximum_attempts=2)
    _, execution_service, factory = runner_stack
    executor = _SuccessfulExecutor()

    def _time_out(*_args):
        """Simulate the Linux deadline interrupt deterministically."""

        raise ScheduledTaskAttemptTimedOut("execution deadline exceeded")

    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=controlled_clock.sleep,
        wall_clock=controlled_clock.now,
        execute_attempt=_time_out,
    )

    with pytest.raises(ScheduledTaskExecutionFailed) as exc_info:
        runner.run(job.job_id, now=_RUNNER_NOW)

    assert exc_info.value.run.status == ScheduledJobRunStatus.TIMED_OUT.value
    assert exc_info.value.run.attempt_number == 2
    assert executor.calls == []
    with factory() as session:
        logs = (
            session.execute(select(ScheduledJobRunLog).order_by(ScheduledJobRunLog.id))
            .scalars()
            .all()
        )
    assert [item.log_level for item in logs] == ["info", "warning", "error"]


def test_stale_running_occurrence_is_reclaimed_for_remaining_attempt(
    runner_stack,
    controlled_clock: _ControlledClock,
) -> None:
    """A crashed process claim should not permanently strand its occurrence."""

    job = _create_job(runner_stack, active=True, maximum_attempts=3)
    _, execution_service, factory = runner_stack
    claim = execution_service.claim_run(
        job_id=job.job_id,
        scheduled_for=datetime(2026, 9, 5, 9, 35, tzinfo=timezone.utc),
    )
    with factory() as session:
        session.execute(
            update(ScheduledJobRun)
            .where(ScheduledJobRun.run_id == UUID(claim.run.run_id))
            .values(lease_expires_at=_RUNNER_NOW - timedelta(seconds=1))
        )
        session.commit()
    executor = _SuccessfulExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=controlled_clock.sleep,
        wall_clock=controlled_clock.now,
    )

    result = runner.run(job.job_id, now=_RUNNER_NOW)

    assert result.run.run_id == claim.run.run_id
    assert result.run.status == ScheduledJobRunStatus.COMPLETED.value
    assert result.run.attempt_number == 2
    assert len(executor.calls) == 1
    with factory() as session:
        logs = (
            session.execute(select(ScheduledJobRunLog).order_by(ScheduledJobRunLog.id))
            .scalars()
            .all()
        )
    assert [item.log_level for item in logs] == ["info", "warning", "info"]


def test_older_stale_occurrence_is_recovered_before_newer_due_occurrence(
    runner_stack,
    controlled_clock: _ControlledClock,
) -> None:
    """A new timer event should finish an older stale claim instead of replacing it."""

    job = _create_job(runner_stack, active=True, maximum_attempts=3)
    _, execution_service, factory = runner_stack
    older_occurrence = datetime(2026, 9, 5, 8, 35, tzinfo=timezone.utc)
    claim = execution_service.claim_run(
        job_id=job.job_id,
        scheduled_for=older_occurrence,
    )
    with factory() as session:
        session.execute(
            update(ScheduledJobRun)
            .where(ScheduledJobRun.run_id == UUID(claim.run.run_id))
            .values(lease_expires_at=_RUNNER_NOW - timedelta(seconds=1))
        )
        session.commit()
    executor = _SuccessfulExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=controlled_clock.sleep,
        wall_clock=controlled_clock.now,
    )

    result = runner.run(job.job_id, now=_RUNNER_NOW)

    assert result.run.run_id == claim.run.run_id
    assert result.run.scheduled_for.replace(tzinfo=timezone.utc) == older_occurrence
    assert result.run.attempt_number == 2
    assert executor.calls == [(job.job_id, older_occurrence)]
    with factory() as session:
        runs = session.execute(select(ScheduledJobRun)).scalars().all()
    assert len(runs) == 1
    assert runs[0].scheduled_for == older_occurrence.replace(tzinfo=None)


def test_pre_resume_stale_occurrence_is_cancelled_before_current_run(
    runner_stack,
    controlled_clock: _ControlledClock,
) -> None:
    """Recovery must never execute work from before the latest activation."""

    job = _create_job(runner_stack, active=True, maximum_attempts=3)
    _, execution_service, factory = runner_stack
    old_occurrence = datetime(2026, 9, 5, 8, 35, tzinfo=timezone.utc)
    claim = execution_service.claim_run(
        job_id=job.job_id,
        scheduled_for=old_occurrence,
    )
    with factory() as session:
        session.execute(
            update(ScheduledJobRun)
            .where(ScheduledJobRun.run_id == UUID(claim.run.run_id))
            .values(lease_expires_at=_RUNNER_NOW - timedelta(seconds=1))
        )
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(
                activated_at=datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc),
                activation_generation=ScheduledJob.activation_generation + 1,
            )
        )
        session.commit()
    executor = _SuccessfulExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=controlled_clock.sleep,
        wall_clock=controlled_clock.now,
    )

    result = runner.run(job.job_id, now=_RUNNER_NOW)

    current_occurrence = datetime(2026, 9, 5, 9, 35, tzinfo=timezone.utc)
    assert result.run.status == ScheduledJobRunStatus.COMPLETED.value
    assert executor.calls == [(job.job_id, current_occurrence)]
    with factory() as session:
        runs = (
            session.execute(
                select(ScheduledJobRun).order_by(ScheduledJobRun.scheduled_for)
            )
            .scalars()
            .all()
        )
    assert [run.status for run in runs] == [
        ScheduledJobRunStatus.CANCELLED.value,
        ScheduledJobRunStatus.COMPLETED.value,
    ]
    assert runs[0].error_message == (
        "Occurrence predates the job's current activation."
    )


def test_older_fresh_occurrence_reports_retryable_lease_contention(
    runner_stack,
    controlled_clock: _ControlledClock,
) -> None:
    """A live older lease must be retried, never reported as successful duplication."""

    job = _create_job(runner_stack, active=True)
    _, execution_service, factory = runner_stack
    older_occurrence = datetime(2026, 9, 5, 8, 35, tzinfo=timezone.utc)
    claim = execution_service.claim_run(
        job_id=job.job_id,
        scheduled_for=older_occurrence,
    )
    executor = _SuccessfulExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=controlled_clock.sleep,
        wall_clock=controlled_clock.now,
    )

    with pytest.raises(ScheduledJobRunLeaseBusyError) as exc_info:
        runner.run(job.job_id, now=_RUNNER_NOW)

    assert exc_info.value.lease_expires_at == claim.run.lease_expires_at
    assert executor.calls == []
    with factory() as session:
        assert len(session.execute(select(ScheduledJobRun)).scalars().all()) == 1


def test_stale_final_attempt_is_closed_as_timed_out(
    runner_stack,
    controlled_clock: _ControlledClock,
) -> None:
    """A stale occurrence with no attempts left should close without re-execution."""

    job = _create_job(runner_stack, active=True, maximum_attempts=1)
    _, execution_service, factory = runner_stack
    claim = execution_service.claim_run(
        job_id=job.job_id,
        scheduled_for=datetime(2026, 9, 5, 9, 35, tzinfo=timezone.utc),
    )
    with factory() as session:
        session.execute(
            update(ScheduledJobRun)
            .where(ScheduledJobRun.run_id == UUID(claim.run.run_id))
            .values(lease_expires_at=_RUNNER_NOW - timedelta(seconds=1))
        )
        session.commit()
    executor = _SuccessfulExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=controlled_clock.sleep,
        wall_clock=controlled_clock.now,
    )

    with pytest.raises(ScheduledTaskExecutionFailed) as exc_info:
        runner.run(job.job_id, now=_RUNNER_NOW)

    assert exc_info.value.run.status == ScheduledJobRunStatus.TIMED_OUT.value
    assert exc_info.value.run.attempt_number == 1
    assert executor.calls == []


@pytest.mark.parametrize(
    ("timed_out", "expected_status"),
    [
        (False, ScheduledJobRunStatus.FAILED.value),
        (True, ScheduledJobRunStatus.TIMED_OUT.value),
    ],
)
def test_terminal_duplicate_is_reported_as_non_success(
    runner_stack,
    timed_out: bool,
    expected_status: str,
) -> None:
    """A duplicate failed occurrence must remain a non-success for service callers."""

    job = _create_job(runner_stack, active=True, maximum_attempts=1)
    _, execution_service, _ = runner_stack
    claim = execution_service.claim_run(
        job_id=job.job_id,
        scheduled_for=datetime(2026, 9, 5, 9, 35, tzinfo=timezone.utc),
    )
    assert claim.run.lease_token is not None
    failed = execution_service.fail_run(
        run_id=claim.run.run_id,
        expected_lease_token=claim.run.lease_token,
        expected_attempt_number=claim.run.attempt_number,
        error="execution failed",
        timed_out=timed_out,
    )
    result = ScheduledJobRunnerResult(run=failed, output_id=None, duplicate=True)
    assert scheduled_job_result_exit_code(result) == 2
    executor = _SuccessfulExecutor()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=executor,
        sleep=lambda _: None,
    )

    with pytest.raises(ScheduledTaskExecutionFailed) as exc_info:
        runner.run(job.job_id, now=_RUNNER_NOW)

    assert exc_info.value.run.status == expected_status
    assert executor.calls == []


def test_runner_rejects_a_job_until_timer_provisioning_activates_it(
    runner_stack,
) -> None:
    """Pending definitions must not execute before Phase 4 activation."""

    job = _create_job(runner_stack, active=False)
    _, execution_service, factory = runner_stack
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=_SuccessfulExecutor(),
        sleep=lambda _: None,
    )

    with pytest.raises(ScheduledJobNotActiveError):
        runner.run(job.job_id, now=_RUNNER_NOW)

    with factory() as session:
        assert session.execute(select(ScheduledJobRun)).scalars().all() == []


def test_runner_rejects_occurrences_from_before_activation(runner_stack) -> None:
    """First activation must not backfill an earlier calendar occurrence."""

    job = _create_job(runner_stack, active=True)
    _, execution_service, factory = runner_stack
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(activated_at=_RUNNER_NOW - timedelta(minutes=1))
        )
        session.commit()
    runner = ScheduledJobRunner(
        execution_service=execution_service,
        executor=_SuccessfulExecutor(),
        sleep=lambda _: None,
    )

    with pytest.raises(
        ScheduledOccurrenceNotDueError,
        match="since this job was activated",
    ):
        runner.run(job.job_id, now=_RUNNER_NOW)

    with factory() as session:
        assert session.execute(select(ScheduledJobRun)).scalars().all() == []


def test_claim_rechecks_activation_boundary_inside_locked_transaction(
    runner_stack,
) -> None:
    """A claim must reject a stale pre-resume occurrence at the database boundary."""

    job = _create_job(runner_stack, active=True)
    _, execution_service, factory = runner_stack
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(activated_at=_RUNNER_NOW)
        )
        session.commit()

    with pytest.raises(ScheduledJobNotActiveError, match="not active for"):
        execution_service.claim_run(
            job_id=job.job_id,
            scheduled_for=_RUNNER_NOW - timedelta(minutes=1),
        )

    with factory() as session:
        assert session.execute(select(ScheduledJobRun)).scalars().all() == []


def test_validation_cli_is_read_only_and_does_not_consume_an_occurrence(
    runner_stack,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validation should emit a receipt without creating production run state."""

    job = _create_job(runner_stack, active=False)
    _, execution_service, factory = runner_stack

    exit_code = main(
        ["--job-id", job.job_id, "--validate-only"],
        execution_service=execution_service,
        now=_RUNNER_NOW,
    )

    assert exit_code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["job_id"] == job.job_id
    assert document["status"] == "validated"
    assert document["job_status"] == ScheduledJobStatus.PENDING_PROVISIONING.value
    assert document["production_policy_valid"] is True
    assert document["database_write"] is False
    assert document["validated_at"] == _RUNNER_NOW.isoformat()
    with factory() as session:
        assert session.execute(select(ScheduledJobRun)).scalars().all() == []
        assert session.execute(select(ScheduledJobOutput)).scalars().all() == []


def test_validation_cli_accepts_a_future_one_time_job(
    runner_stack,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Read-only validation should not require a scheduled occurrence to be due."""

    job_service, execution_service, factory = runner_stack
    definition = build_task_definition(
        ScheduledTaskInput(
            name="Future BF2 review",
            instructions="Prepare the requested BF2 review.",
            furnace="BF2",
            data_period="last_24_hours",
            output_format="operator_summary",
            schedule_kind="once",
            delivery_channel="in_app",
            job_type="furnace_summary",
            run_date=date(2026, 9, 6),
            run_time=time(12, 0),
        ),
        generated_at=_RUNNER_NOW,
    )
    job = job_service.create_job(
        ScheduledJobCreateRequest(
            definition=definition,
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )

    exit_code = main(
        ["--job-id", job.job_id, "--validate-only"],
        execution_service=execution_service,
        now=_RUNNER_NOW,
    )

    assert exit_code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "validated"
    assert document["validated_at"] == _RUNNER_NOW.isoformat()
    with factory() as session:
        assert session.execute(select(ScheduledJobRun)).scalars().all() == []


def test_validation_cli_handles_malformed_database_definition_cleanly(
    runner_stack,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Corrupt stored JSON should produce a concise failure without a traceback."""

    job = _create_job(runner_stack, active=False)
    _, execution_service, factory = runner_stack
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(definition_json=["password=hunter2"])
        )
        session.commit()

    with caplog.at_level(logging.ERROR):
        exit_code = main(
            ["--job-id", job.job_id, "--validate-only"],
            execution_service=execution_service,
            now=_RUNNER_NOW,
        )

    assert exit_code == 2
    assert capsys.readouterr().out == ""
    assert "could not be processed (ValueError)" in caplog.text
    assert "hunter2" not in caplog.text
    assert "Traceback" not in caplog.text


def test_production_cli_executes_and_prints_only_a_safe_receipt(
    runner_stack,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default command should persist work without printing report content."""

    job = _create_job(runner_stack, active=True)
    _, execution_service, factory = runner_stack
    executor = _SuccessfulExecutor()

    exit_code = main(
        ["--job-id", job.job_id],
        execution_service=execution_service,
        executor=executor,
        now=_RUNNER_NOW,
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    document = json.loads(stdout)
    assert set(document) == {
        "attempt_number",
        "duplicate",
        "job_id",
        "output_id",
        "run_id",
        "scheduled_for",
        "status",
    }
    assert document["job_id"] == job.job_id
    assert document["status"] == ScheduledJobRunStatus.COMPLETED.value
    assert document["attempt_number"] == 1
    assert document["duplicate"] is False
    assert document["output_id"] is not None
    assert "BF2 operation is stable" not in stdout
    assert '"condition"' not in stdout
    with factory() as session:
        run = session.execute(select(ScheduledJobRun)).scalar_one()
        output = session.execute(select(ScheduledJobOutput)).scalar_one()
    assert run.status == ScheduledJobRunStatus.COMPLETED.value
    assert str(output.output_id) == document["output_id"]


def test_production_cli_reports_a_live_lease_as_retryable(
    runner_stack,
    controlled_clock: _ControlledClock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second systemd delivery must return retryable contention, not success."""

    job = _create_job(runner_stack, active=True)
    _, execution_service, factory = runner_stack
    claimed = execution_service.claim_run(
        job_id=job.job_id,
        scheduled_for=datetime(2026, 9, 5, 8, 35, tzinfo=timezone.utc),
    )

    exit_code = main(
        ["--job-id", job.job_id],
        execution_service=execution_service,
        executor=_SuccessfulExecutor(),
        now=_RUNNER_NOW,
    )

    assert exit_code == 1
    assert capsys.readouterr().out == ""
    with factory() as session:
        runs = session.execute(select(ScheduledJobRun)).scalars().all()
        outputs = session.execute(select(ScheduledJobOutput)).scalars().all()
    assert [str(run.run_id) for run in runs] == [claimed.run.run_id]
    assert runs[0].status == ScheduledJobRunStatus.RUNNING.value
    assert outputs == []


def test_stable_runner_wrapper_help_works_without_pythonpath(
    tmp_path: Path,
) -> None:
    """The Phase 4 ExecStart wrapper should resolve local packages by itself."""

    wrapper = (
        Path(__file__).resolve().parents[1] / "scripts" / "furnacemind_job_runner.py"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(wrapper), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "furnacemind-job-runner" in completed.stdout
