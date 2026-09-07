"""Tests for optimistic scheduled-job provisioning lifecycle transitions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import furnace_data.relational.repositories as relational_repositories
from furnace_data.relational import (
    ScheduledJob,
    ScheduledJobRepository,
    ScheduledJobRevision,
    ScheduledJobRun,
    ScheduledJobRunLog,
    ScheduledJobStatus,
)

_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def lifecycle_repository():
    """Create a scheduled-job repository backed by isolated in-memory SQLite."""

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
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
        future=True,
    )
    try:
        yield ScheduledJobRepository(factory)
    finally:
        engine.dispose()


def _create_pending_job(repository: ScheduledJobRepository) -> ScheduledJob:
    """Persist and return a minimal pending job through the public repository."""

    return repository.create_job(
        job_name="Hourly BF2 check",
        schema_version="1.0",
        definition={
            "job_type": "eta_co_report",
            "instructions": "Review BF2 conditions.",
            "schedule": {
                "timezone": "Asia/Kolkata",
                "trigger": {"type": "cron", "expression": "5 * * * *"},
            },
        },
        created_by_user_id=_USER_ID,
        created_by_username="operator.test",
        target_device_id="bf2-jetson-01",
    )


def _assert_inactive_consistency(job: ScheduledJob) -> None:
    """Assert the database activation invariant for a non-active job."""

    assert job.status != ScheduledJobStatus.ACTIVE.value
    assert job.is_active is False
    assert job.activated_at is None


def test_reconciliation_scan_uses_a_stable_target_keyset(
    lifecycle_repository: ScheduledJobRepository,
) -> None:
    """Keyset pages must not skip tied timestamps or include later/other jobs."""

    jobs = [_create_pending_job(lifecycle_repository) for _ in range(5)]
    base = datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc)
    session_factory = lifecycle_repository._session_factory  # noqa: SLF001
    with session_factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id.in_([jobs[0].job_id, jobs[1].job_id]))
            .values(created_at=base)
        )
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == jobs[1].job_id)
            .values(
                status=ScheduledJobStatus.DELETED.value,
                provisioning_error="Timer cleanup is pending.",
            )
        )
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == jobs[2].job_id)
            .values(created_at=base, target_device_id="another-device")
        )
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == jobs[3].job_id)
            .values(created_at=base + timedelta(minutes=1))
        )
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == jobs[4].job_id)
            .values(created_at=base, status=ScheduledJobStatus.DELETED.value)
        )
        session.commit()

    rows: list[tuple[str, datetime]] = []
    after_created_at = None
    after_job_id = None
    while True:
        page = lifecycle_repository.list_reconciliation_job_ids(
            target_device_id="bf2-jetson-01",
            created_through=base + timedelta(seconds=1),
            after_created_at=after_created_at,
            after_job_id=after_job_id,
            limit=1,
        )
        if not page:
            break
        rows.extend(page)
        after_job_id, after_created_at = page[-1]

    assert [job_id for job_id, _ in rows] == sorted([jobs[0].job_id, jobs[1].job_id])
    assert all(created_at == base for _, created_at in rows)


def test_reconciliation_cutoff_uses_an_aware_database_timestamp(
    lifecycle_repository: ScheduledJobRepository,
) -> None:
    """A scan cutoff must come from the database and be normalized to UTC."""

    cutoff = lifecycle_repository.reconciliation_cutoff()

    assert cutoff.tzinfo is not None
    assert cutoff.utcoffset() == timedelta(0)


def test_activate_pause_and_resume_preserve_activation_consistency(
    lifecycle_repository: ScheduledJobRepository,
) -> None:
    """Valid lifecycle changes should keep status and activation fields aligned."""

    pending = _create_pending_job(lifecycle_repository)

    active = lifecycle_repository.activate_job(
        job_id=pending.job_id,
        expected_status=pending.status,
        expected_updated_at=pending.updated_at,
        timer_unit_name=f"furnacemind-job-{pending.job_id}.timer",
    )

    assert active is not None
    assert active.status == ScheduledJobStatus.ACTIVE.value
    assert active.is_active is True
    assert active.activated_at is not None
    assert active.activation_generation == 1
    assert active.provisioning_error is None
    assert active.timer_unit_name == f"furnacemind-job-{pending.job_id}.timer"

    paused = lifecycle_repository.pause_job(
        job_id=active.job_id,
        expected_status=active.status,
        expected_updated_at=active.updated_at,
    )

    assert paused is not None
    _assert_inactive_consistency(paused)
    assert paused.status == ScheduledJobStatus.PAUSED.value
    assert paused.timer_unit_name == active.timer_unit_name
    assert paused.activation_generation == active.activation_generation

    resumed = lifecycle_repository.activate_job(
        job_id=paused.job_id,
        expected_status=paused.status,
        expected_updated_at=paused.updated_at,
        timer_unit_name=paused.timer_unit_name or "missing.timer",
    )

    assert resumed is not None
    assert resumed.status == ScheduledJobStatus.ACTIVE.value
    assert resumed.is_active is True
    assert resumed.activated_at is not None
    assert resumed.activation_generation == active.activation_generation + 1


def test_active_timer_receipt_can_only_be_repaired_to_the_canonical_unit(
    lifecycle_repository: ScheduledJobRepository,
) -> None:
    """Receipt repair must remain active and reject arbitrary unit names."""

    pending = _create_pending_job(lifecycle_repository)
    canonical_timer = f"furnacemind-job-{pending.job_id}.timer"
    active = lifecycle_repository.activate_job(
        job_id=pending.job_id,
        expected_status=pending.status,
        expected_updated_at=pending.updated_at,
        timer_unit_name=canonical_timer,
    )
    assert active is not None

    with pytest.raises(ValueError, match="derived"):
        lifecycle_repository.repair_timer_unit_name(
            job_id=active.job_id,
            expected_status=active.status,
            expected_updated_at=active.updated_at,
            timer_unit_name="operator-controlled.timer",
        )

    assert (
        lifecycle_repository.repair_timer_unit_name(
            job_id=active.job_id,
            expected_status=active.status,
            expected_updated_at=active.updated_at - timedelta(seconds=1),
            timer_unit_name=canonical_timer,
        )
        is None
    )

    repaired = lifecycle_repository.repair_timer_unit_name(
        job_id=active.job_id,
        expected_status=active.status,
        expected_updated_at=active.updated_at,
        timer_unit_name=canonical_timer,
    )

    assert repaired is not None
    assert repaired.status == ScheduledJobStatus.ACTIVE.value
    assert repaired.timer_unit_name == canonical_timer
    assert repaired.updated_at > active.updated_at

    paused = lifecycle_repository.pause_job(
        job_id=repaired.job_id,
        expected_status=repaired.status,
        expected_updated_at=repaired.updated_at,
    )
    assert paused is not None
    assert (
        lifecycle_repository.repair_timer_unit_name(
            job_id=paused.job_id,
            expected_status=paused.status,
            expected_updated_at=paused.updated_at,
            timer_unit_name=canonical_timer,
        )
        is None
    )


def test_stale_timestamp_and_invalid_source_states_are_conflicts(
    lifecycle_repository: ScheduledJobRepository,
) -> None:
    """CAS updates should return None without changing a stale or invalid row."""

    pending = _create_pending_job(lifecycle_repository)

    stale = lifecycle_repository.activate_job(
        job_id=pending.job_id,
        expected_status=pending.status,
        expected_updated_at=pending.updated_at - timedelta(seconds=1),
        timer_unit_name=f"furnacemind-job-{pending.job_id}.timer",
    )
    invalid = lifecycle_repository.pause_job(
        job_id=pending.job_id,
        expected_status=pending.status,
        expected_updated_at=pending.updated_at,
    )

    assert stale is None
    assert invalid is None
    unchanged = lifecycle_repository.get_job(pending.job_id)
    assert unchanged is not None
    assert unchanged.status == ScheduledJobStatus.PENDING_PROVISIONING.value
    _assert_inactive_consistency(unchanged)


def test_provisioning_failure_is_bounded_and_can_refresh_retry_error(
    lifecycle_repository: ScheduledJobRepository,
) -> None:
    """Provisioning failures should be safe to store and refresh after retries."""

    pending = _create_pending_job(lifecycle_repository)
    failed = lifecycle_repository.mark_provisioning_failed(
        job_id=pending.job_id,
        expected_status=pending.status,
        expected_updated_at=pending.updated_at,
        error=f"  {'x' * 5000}  ",
    )

    assert failed is not None
    assert failed.status == ScheduledJobStatus.PROVISIONING_FAILED.value
    _assert_inactive_consistency(failed)
    assert failed.provisioning_error == "x" * 4096

    retried_failure = lifecycle_repository.mark_provisioning_failed(
        job_id=failed.job_id,
        expected_status=failed.status,
        expected_updated_at=failed.updated_at,
        error="second attempt failed",
    )

    assert retried_failure is not None
    assert retried_failure.status == ScheduledJobStatus.PROVISIONING_FAILED.value
    assert retried_failure.provisioning_error == "second attempt failed"
    _assert_inactive_consistency(retried_failure)


def test_active_job_can_be_compensated_to_provisioning_failed(
    lifecycle_repository: ScheduledJobRepository,
) -> None:
    """A timer-start failure should safely compensate an active database row."""

    pending = _create_pending_job(lifecycle_repository)
    active = lifecycle_repository.activate_job(
        job_id=pending.job_id,
        expected_status=pending.status,
        expected_updated_at=pending.updated_at,
        timer_unit_name=f"furnacemind-job-{pending.job_id}.timer",
    )
    assert active is not None

    failed = lifecycle_repository.mark_provisioning_failed(
        job_id=active.job_id,
        expected_status=active.status,
        expected_updated_at=active.updated_at,
        error="systemctl start failed",
    )

    assert failed is not None
    assert failed.status == ScheduledJobStatus.PROVISIONING_FAILED.value
    assert failed.timer_unit_name == active.timer_unit_name
    assert failed.provisioning_error == "systemctl start failed"
    _assert_inactive_consistency(failed)


def test_external_error_updates_preserve_lifecycle_and_use_cas(
    lifecycle_repository: ScheduledJobRepository,
) -> None:
    """External error metadata should not alter state and should reject stale CAS."""

    pending = _create_pending_job(lifecycle_repository)
    with_error = lifecycle_repository.record_external_error(
        job_id=pending.job_id,
        expected_status=pending.status,
        expected_updated_at=pending.updated_at,
        error="  ",
    )

    assert with_error is not None
    assert with_error.status == pending.status
    assert with_error.provisioning_error == (
        "External scheduler operation failed without an error message."
    )
    assert (
        lifecycle_repository.clear_external_error(
            job_id=pending.job_id,
            expected_status=pending.status,
            expected_updated_at=pending.updated_at,
        )
        is None
    )

    cleared = lifecycle_repository.clear_external_error(
        job_id=with_error.job_id,
        expected_status=with_error.status,
        expected_updated_at=with_error.updated_at,
    )

    assert cleared is not None
    assert cleared.status == pending.status
    assert cleared.provisioning_error is None
    _assert_inactive_consistency(cleared)


def test_cas_revision_advances_when_database_clock_repeats(
    lifecycle_repository: ScheduledJobRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every transition must invalidate stale writers even at one clock instant."""

    pending = _create_pending_job(lifecycle_repository)

    def _frozen_now() -> datetime:
        """Return the row's existing revision to exercise monotonic fallback."""

        return pending.updated_at

    monkeypatch.setattr(relational_repositories, "utc_now", _frozen_now)
    updated = lifecycle_repository.record_external_error(
        job_id=pending.job_id,
        expected_status=pending.status,
        expected_updated_at=pending.updated_at,
        error="first writer",
    )

    assert updated is not None
    assert updated.updated_at > pending.updated_at
    assert (
        lifecycle_repository.clear_external_error(
            job_id=pending.job_id,
            expected_status=pending.status,
            expected_updated_at=pending.updated_at,
        )
        is None
    )


@pytest.mark.parametrize(
    "source_status",
    [
        ScheduledJobStatus.PENDING_PROVISIONING.value,
        ScheduledJobStatus.ACTIVE.value,
        ScheduledJobStatus.PAUSED.value,
        ScheduledJobStatus.PROVISIONING_FAILED.value,
    ],
)
def test_mark_deleted_accepts_each_nonterminal_state(
    lifecycle_repository: ScheduledJobRepository,
    source_status: str,
) -> None:
    """Archiving should accept every nonterminal scheduled-job state."""

    job = _create_pending_job(lifecycle_repository)
    if source_status == ScheduledJobStatus.ACTIVE.value:
        transitioned = lifecycle_repository.activate_job(
            job_id=job.job_id,
            expected_status=job.status,
            expected_updated_at=job.updated_at,
            timer_unit_name=f"furnacemind-job-{job.job_id}.timer",
        )
        assert transitioned is not None
        job = transitioned
    elif source_status == ScheduledJobStatus.PAUSED.value:
        active = lifecycle_repository.activate_job(
            job_id=job.job_id,
            expected_status=job.status,
            expected_updated_at=job.updated_at,
            timer_unit_name=f"furnacemind-job-{job.job_id}.timer",
        )
        assert active is not None
        transitioned = lifecycle_repository.pause_job(
            job_id=active.job_id,
            expected_status=active.status,
            expected_updated_at=active.updated_at,
        )
        assert transitioned is not None
        job = transitioned
    elif source_status == ScheduledJobStatus.PROVISIONING_FAILED.value:
        transitioned = lifecycle_repository.mark_provisioning_failed(
            job_id=job.job_id,
            expected_status=job.status,
            expected_updated_at=job.updated_at,
            error="failed",
        )
        assert transitioned is not None
        job = transitioned

    deleted = lifecycle_repository.mark_deleted(
        job_id=job.job_id,
        expected_status=job.status,
        expected_updated_at=job.updated_at,
    )

    assert deleted is not None
    assert deleted.status == ScheduledJobStatus.DELETED.value
    assert deleted.timer_unit_name is None
    assert deleted.provisioning_error == "Timer cleanup is pending."
    _assert_inactive_consistency(deleted)


def test_deleted_state_is_terminal_but_keeps_cleanup_error_metadata(
    lifecycle_repository: ScheduledJobRepository,
) -> None:
    """Deleted jobs must stay deleted while cleanup reconciliation is recorded."""

    pending = _create_pending_job(lifecycle_repository)
    deleted = lifecycle_repository.mark_deleted(
        job_id=pending.job_id,
        expected_status=pending.status,
        expected_updated_at=pending.updated_at,
    )
    assert deleted is not None

    assert (
        lifecycle_repository.activate_job(
            job_id=deleted.job_id,
            expected_status=deleted.status,
            expected_updated_at=deleted.updated_at,
            timer_unit_name=f"furnacemind-job-{deleted.job_id}.timer",
        )
        is None
    )
    assert (
        lifecycle_repository.mark_provisioning_failed(
            job_id=deleted.job_id,
            expected_status=deleted.status,
            expected_updated_at=deleted.updated_at,
            error="must not reopen",
        )
        is None
    )
    assert (
        lifecycle_repository.pause_job(
            job_id=deleted.job_id,
            expected_status=deleted.status,
            expected_updated_at=deleted.updated_at,
        )
        is None
    )
    assert (
        lifecycle_repository.mark_deleted(
            job_id=deleted.job_id,
            expected_status=deleted.status,
            expected_updated_at=deleted.updated_at,
        )
        is None
    )

    cleanup_error = lifecycle_repository.record_external_error(
        job_id=deleted.job_id,
        expected_status=deleted.status,
        expected_updated_at=deleted.updated_at,
        error="unit file cleanup failed",
    )

    assert cleanup_error is not None
    assert cleanup_error.status == ScheduledJobStatus.DELETED.value
    assert cleanup_error.provisioning_error == "unit file cleanup failed"
    _assert_inactive_consistency(cleanup_error)

    reconciled = lifecycle_repository.clear_external_error(
        job_id=cleanup_error.job_id,
        expected_status=cleanup_error.status,
        expected_updated_at=cleanup_error.updated_at,
    )
    assert reconciled is not None
    assert reconciled.status == ScheduledJobStatus.DELETED.value
    assert reconciled.provisioning_error is None
    _assert_inactive_consistency(reconciled)
