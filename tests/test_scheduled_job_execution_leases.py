"""Tests for Phase 5 fenced leases and one-time terminal persistence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from furnace_data.relational import (
    ScheduledJob,
    ScheduledJobOutput,
    ScheduledJobRepository,
    ScheduledJobRevision,
    ScheduledJobRun,
    ScheduledJobRunLeaseBusyError,
    ScheduledJobRunLog,
    ScheduledJobRunRepository,
    ScheduledJobRunStatus,
    ScheduledJobStatus,
)
from furnace_data.relational.repositories import _database_utc_now

_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest.mark.parametrize(
    ("dialect_name", "expected_function"),
    (("postgresql", "clock_timestamp"), ("sqlite", "CURRENT_TIMESTAMP")),
)
def test_database_clock_uses_wall_time_for_postgresql_leases(
    dialect_name: str,
    expected_function: str,
) -> None:
    """PostgreSQL lease decisions must not reuse transaction-start time."""

    timestamp = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)
    result = Mock()
    result.scalar_one.return_value = timestamp
    session = Mock()
    session.get_bind.return_value = SimpleNamespace(
        dialect=SimpleNamespace(name=dialect_name)
    )
    session.execute.return_value = result

    assert _database_utc_now(session) == timestamp
    statement = session.execute.call_args.args[0]
    assert expected_function in str(statement)


@pytest.fixture
def execution_repositories():
    """Create the scheduled execution tables in an isolated SQLite database."""

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
    try:
        yield (
            ScheduledJobRepository(factory),
            ScheduledJobRunRepository(factory),
            factory,
        )
    finally:
        engine.dispose()


def _create_active_job(execution_repositories, *, one_time: bool) -> ScheduledJob:
    """Create and activate one repository-level scheduled job."""

    job_repository, _, _ = execution_repositories
    trigger = (
        {"type": "once", "run_at": "2026-09-05T10:00:00+00:00"}
        if one_time
        else {"type": "cron", "expression": "5 * * * *"}
    )
    pending = job_repository.create_job(
        job_name="Lease test",
        schema_version="scheduled-job-definition/v1",
        definition={
            "job_type": "custom_report",
            "instructions": "Review BF2 conditions.",
            "schedule": {
                "timezone": "UTC",
                "trigger": trigger,
            },
            "retry": {
                "maximum_attempts": 3,
                "retry_interval_seconds": 60,
                "timeout_seconds": 600,
            },
        },
        created_by_user_id=_USER_ID,
        created_by_username="operator.test",
        target_device_id="bf2-jetson-01",
    )
    active = job_repository.activate_job(
        job_id=pending.job_id,
        expected_status=pending.status,
        expected_updated_at=pending.updated_at,
        timer_unit_name=f"furnacemind-job-{pending.job_id}.timer",
        activated_at=datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc),
    )
    assert active is not None
    return active


def _claim(execution_repositories, job: ScheduledJob) -> ScheduledJobRun:
    """Claim one deterministic occurrence and return its fenced run."""

    _, run_repository, _ = execution_repositories
    return run_repository.create_run(
        job_id=job.job_id,
        scheduled_for=datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc),
        triggered_by="test",
        timeout_seconds=600,
        lease_seconds=90,
    )


def test_claim_copies_activation_generation_and_creates_a_fenced_lease(
    execution_repositories,
) -> None:
    """A claim should persist ownership and hard-deadline fields atomically."""

    job = _create_active_job(execution_repositories, one_time=False)
    run = _claim(execution_repositories, job)

    assert job.activation_generation == 1
    assert run.activation_generation == job.activation_generation
    assert run.lease_token is not None
    assert run.lease_expires_at is not None
    assert run.attempt_deadline_at is not None
    assert run.lease_expires_at <= run.attempt_deadline_at
    assert run.retry_not_before_at is None


def test_recovery_rotates_lease_and_fences_the_previous_worker(
    execution_repositories,
) -> None:
    """A reclaimed attempt must reject completion from its former owner."""

    job = _create_active_job(execution_repositories, one_time=False)
    run = _claim(execution_repositories, job)
    assert run.lease_token is not None
    old_token = run.lease_token
    _, run_repository, factory = execution_repositories
    with factory() as session:
        session.execute(
            update(ScheduledJobRun)
            .where(ScheduledJobRun.run_id == run.run_id)
            .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
        )
        session.commit()

    resolution = run_repository.recover_or_get_running_run(
        job_id=job.job_id,
        triggered_by="recovery",
        maximum_attempts=3,
        timeout_seconds=600,
        lease_seconds=90,
        retry_interval_seconds=60,
    )

    assert resolution is not None
    recovered, acquired = resolution
    assert acquired is True
    assert recovered.attempt_number == 2
    assert recovered.lease_token not in {None, old_token}
    assert recovered.retry_not_before_at is not None

    with pytest.raises(ValueError, match="no longer owned"):
        run_repository.complete_run(
            run_id=run.run_id,
            expected_lease_token=old_token,
            expected_attempt_number=1,
            output_type="text",
            content="stale result",
            content_json=None,
            artifact_path=None,
            output_metadata={},
        )
    with factory() as session:
        assert session.execute(select(ScheduledJobOutput)).scalars().all() == []


def test_one_time_success_retires_job_in_the_output_transaction(
    execution_repositories,
) -> None:
    """One-time success should atomically persist output and completed job state."""

    job = _create_active_job(execution_repositories, one_time=True)
    run = _claim(execution_repositories, job)
    assert run.lease_token is not None
    job_repository, run_repository, factory = execution_repositories

    completed = run_repository.complete_run(
        run_id=run.run_id,
        expected_lease_token=run.lease_token,
        expected_attempt_number=run.attempt_number,
        output_type="operator_summary",
        content="BF2 is stable.",
        content_json={"stable": True},
        artifact_path=None,
        output_metadata={},
    )

    assert completed is not None
    completed_run, output = completed
    assert completed_run.status == ScheduledJobRunStatus.COMPLETED.value
    assert output.run_id == run.run_id
    terminal_job = job_repository.get_job(job.job_id)
    assert terminal_job is not None
    assert terminal_job.status == ScheduledJobStatus.COMPLETED.value
    assert terminal_job.is_active is False
    assert terminal_job.activated_at is None
    assert terminal_job.timer_unit_name is None
    assert terminal_job.provisioning_error == "Timer cleanup is pending."
    with factory() as session:
        assert len(session.execute(select(ScheduledJobOutput)).scalars().all()) == 1


def test_one_time_failure_and_cancellation_both_retire_job(
    execution_repositories,
) -> None:
    """Any consumed one-time occurrence must leave the job in a terminal state."""

    job_repository, run_repository, _ = execution_repositories
    failed_job = _create_active_job(execution_repositories, one_time=True)
    failed_run = _claim(execution_repositories, failed_job)
    assert failed_run.lease_token is not None
    failed = run_repository.fail_run(
        run_id=failed_run.run_id,
        expected_lease_token=failed_run.lease_token,
        expected_attempt_number=failed_run.attempt_number,
        error_message="agent failed",
    )
    assert failed is not None
    terminal_job = job_repository.get_job(failed_job.job_id)
    assert terminal_job is not None
    assert terminal_job.status == ScheduledJobStatus.EXECUTION_FAILED.value

    cancelled_job = _create_active_job(execution_repositories, one_time=True)
    cancelled_run = _claim(execution_repositories, cancelled_job)
    assert cancelled_run.lease_token is not None
    cancelled = run_repository.cancel_run(
        run_id=cancelled_run.run_id,
        expected_lease_token=cancelled_run.lease_token,
        expected_attempt_number=cancelled_run.attempt_number,
        reason="service received SIGTERM",
    )
    assert cancelled is not None
    assert cancelled.status == ScheduledJobRunStatus.CANCELLED.value
    cancelled_terminal = job_repository.get_job(cancelled_job.job_id)
    assert cancelled_terminal is not None
    assert cancelled_terminal.status == ScheduledJobStatus.EXECUTION_FAILED.value
    assert cancelled_terminal.is_active is False
    assert cancelled_terminal.activated_at is None
    assert cancelled_terminal.timer_unit_name is None
    assert cancelled_terminal.provisioning_error == "Timer cleanup is pending."


def test_resume_rejects_live_run_and_atomically_cancels_expired_run(
    execution_repositories,
) -> None:
    """Resume should never overlap an earlier leased execution."""

    job_repository, _, factory = execution_repositories
    job = _create_active_job(execution_repositories, one_time=False)
    run = _claim(execution_repositories, job)
    paused = job_repository.pause_job(
        job_id=job.job_id,
        expected_status=job.status,
        expected_updated_at=job.updated_at,
    )
    assert paused is not None
    timer_name = f"furnacemind-job-{job.job_id}.timer"

    with pytest.raises(ScheduledJobRunLeaseBusyError):
        job_repository.settle_and_activate_job(
            job_id=paused.job_id,
            expected_status=paused.status,
            expected_updated_at=paused.updated_at,
            timer_unit_name=timer_name,
        )

    with factory() as session:
        session.execute(
            update(ScheduledJobRun)
            .where(ScheduledJobRun.run_id == run.run_id)
            .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
        )
        session.commit()

    settlement = job_repository.settle_and_activate_job(
        job_id=paused.job_id,
        expected_status=paused.status,
        expected_updated_at=paused.updated_at,
        timer_unit_name=timer_name,
    )

    assert settlement is not None
    assert settlement.job.status == ScheduledJobStatus.ACTIVE.value
    assert settlement.job.activation_generation == job.activation_generation + 1
    assert settlement.cancelled_run is not None
    assert settlement.cancelled_run.status == ScheduledJobRunStatus.CANCELLED.value
    assert settlement.cancelled_run.lease_token is None


def test_recurring_completion_preserves_job_lifecycle_state(
    execution_repositories,
) -> None:
    """Completing a recurring occurrence should leave its schedule active."""

    job_repository, run_repository, _ = execution_repositories
    job = _create_active_job(execution_repositories, one_time=False)
    run = _claim(execution_repositories, job)
    assert run.lease_token is not None

    run_repository.complete_run(
        run_id=run.run_id,
        expected_lease_token=run.lease_token,
        expected_attempt_number=run.attempt_number,
        output_type="operator_summary",
        content="done",
        content_json=None,
        artifact_path=None,
        output_metadata={},
    )

    active = job_repository.get_job(job.job_id)
    assert active is not None
    assert active.status == ScheduledJobStatus.ACTIVE.value
    assert active.is_active is True


def test_wrong_lease_cannot_renew_or_cancel_an_execution(
    execution_repositories,
) -> None:
    """Every mutable run transition should require the current opaque token."""

    job = _create_active_job(execution_repositories, one_time=False)
    run = _claim(execution_repositories, job)
    _, run_repository, _ = execution_repositories

    with pytest.raises(ValueError, match="no longer owned"):
        run_repository.renew_lease(
            run_id=run.run_id,
            expected_lease_token=uuid4(),
            expected_attempt_number=run.attempt_number,
        )
    with pytest.raises(ValueError, match="no longer owned"):
        run_repository.cancel_run(
            run_id=run.run_id,
            expected_lease_token=uuid4(),
            expected_attempt_number=run.attempt_number,
            reason="not the owner",
        )


def test_expired_lease_cannot_complete_with_the_previous_token(
    execution_repositories,
) -> None:
    """A token must not authorize writes after its database-clock lease expires."""

    job = _create_active_job(execution_repositories, one_time=False)
    run = _claim(execution_repositories, job)
    assert run.lease_token is not None
    _, run_repository, factory = execution_repositories
    with factory() as session:
        session.execute(
            update(ScheduledJobRun)
            .where(ScheduledJobRun.run_id == run.run_id)
            .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
        )
        session.commit()

    with pytest.raises(ValueError, match="lease has expired"):
        run_repository.complete_run(
            run_id=run.run_id,
            expected_lease_token=run.lease_token,
            expected_attempt_number=run.attempt_number,
            output_type="operator_summary",
            content="must not persist",
            content_json=None,
            artifact_path=None,
            output_metadata={},
        )


def test_retry_is_durable_and_cannot_begin_before_its_database_deadline(
    execution_repositories,
) -> None:
    """A retry backoff should survive process exit and enforce its not-before time."""

    job = _create_active_job(execution_repositories, one_time=False)
    run = _claim(execution_repositories, job)
    assert run.lease_token is not None
    _, run_repository, factory = execution_repositories

    waiting = run_repository.schedule_retry(
        run_id=run.run_id,
        expected_lease_token=run.lease_token,
        expected_attempt_number=run.attempt_number,
        previous_error="temporary executor failure",
        retry_interval_seconds=60,
        lease_seconds=90,
    )

    assert waiting is not None
    assert waiting.attempt_number == 2
    assert waiting.lease_token == run.lease_token
    assert waiting.retry_not_before_at is not None
    assert waiting.attempt_deadline_at is None
    with pytest.raises(ScheduledJobRunLeaseBusyError):
        run_repository.begin_retry_attempt(
            run_id=waiting.run_id,
            expected_lease_token=waiting.lease_token,
            expected_attempt_number=waiting.attempt_number,
            timeout_seconds=600,
            lease_seconds=90,
        )

    with factory() as session:
        session.execute(
            update(ScheduledJobRun)
            .where(ScheduledJobRun.run_id == waiting.run_id)
            .values(
                retry_not_before_at=datetime.now(timezone.utc) - timedelta(seconds=1)
            )
        )
        session.commit()

    executing = run_repository.begin_retry_attempt(
        run_id=waiting.run_id,
        expected_lease_token=waiting.lease_token,
        expected_attempt_number=waiting.attempt_number,
        timeout_seconds=600,
        lease_seconds=90,
    )

    assert executing is not None
    assert executing.attempt_number == 2
    assert executing.retry_not_before_at is None
    assert executing.attempt_deadline_at is not None
    assert executing.lease_expires_at is not None
    assert executing.lease_expires_at <= executing.attempt_deadline_at


def test_lease_renewal_is_capped_by_and_rejects_an_elapsed_attempt_deadline(
    execution_repositories,
) -> None:
    """Heartbeat renewal must not extend or revive the hard attempt timeout."""

    _, run_repository, factory = execution_repositories
    job = _create_active_job(execution_repositories, one_time=False)
    run = run_repository.create_run(
        job_id=job.job_id,
        scheduled_for=datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc),
        triggered_by="test",
        timeout_seconds=10,
        lease_seconds=2,
    )
    assert run.lease_token is not None

    renewed = run_repository.renew_lease(
        run_id=run.run_id,
        expected_lease_token=run.lease_token,
        expected_attempt_number=run.attempt_number,
        lease_seconds=90,
    )

    assert renewed is not None
    assert renewed.lease_expires_at is not None
    assert renewed.attempt_deadline_at is not None
    assert renewed.lease_expires_at <= renewed.attempt_deadline_at

    elapsed = datetime.now(timezone.utc) - timedelta(seconds=1)
    with factory() as session:
        session.execute(
            update(ScheduledJobRun)
            .where(ScheduledJobRun.run_id == run.run_id)
            .values(attempt_deadline_at=elapsed, lease_expires_at=elapsed)
        )
        session.commit()

    with pytest.raises(ValueError, match="deadline has elapsed"):
        run_repository.renew_lease(
            run_id=run.run_id,
            expected_lease_token=run.lease_token,
            expected_attempt_number=run.attempt_number,
            lease_seconds=90,
        )


def test_output_conflict_rolls_back_run_and_one_time_job_terminalization(
    execution_repositories,
) -> None:
    """Output uniqueness failure must not partially commit terminal lifecycle state."""

    job_repository, run_repository, factory = execution_repositories
    job = _create_active_job(execution_repositories, one_time=True)
    run = _claim(execution_repositories, job)
    assert run.lease_token is not None
    with factory() as session:
        session.add(
            ScheduledJobOutput(
                run_id=run.run_id,
                output_type="preexisting",
                content="conflicting output",
            )
        )
        session.commit()

    with pytest.raises(IntegrityError) as exception_info:
        run_repository.complete_run(
            run_id=run.run_id,
            expected_lease_token=run.lease_token,
            expected_attempt_number=run.attempt_number,
            output_type="operator_summary",
            content="should roll back",
            content_json=None,
            artifact_path=None,
            output_metadata={},
        )
    assert "UNIQUE constraint failed" in str(exception_info.value)

    persisted_job = job_repository.get_job(job.job_id)
    assert persisted_job is not None
    assert persisted_job.status == ScheduledJobStatus.ACTIVE.value
    with factory() as session:
        persisted_run = session.get(ScheduledJobRun, run.run_id)
        assert persisted_run is not None
        assert persisted_run.status == ScheduledJobRunStatus.RUNNING.value
        assert persisted_run.lease_token == run.lease_token
        outputs = session.execute(select(ScheduledJobOutput)).scalars().all()
        assert len(outputs) == 1
