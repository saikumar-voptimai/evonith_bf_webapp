"""Tests for the PostgreSQL-backed Streamlit-to-Jetson control bridge."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from data.scheduled_tasks import (
    ScheduledJobCommandConflictError,
    ScheduledJobCommandProcessor,
    ScheduledJobCommandService,
    ScheduledJobCreateRequest,
    ScheduledJobService,
)
from furnace_data.relational import (
    ScheduledJob,
    ScheduledJobCommand,
    ScheduledJobCommandRepository,
    ScheduledJobRevision,
    ScheduledJobRepository,
)
from utils.scheduled_tasks.scheduled_task_definition import (
    ScheduledTaskInput,
    build_task_definition,
)

_USER_ID = "00000000-0000-0000-0000-000000000001"


def _definition(name: str = "Hourly BF2 ETA CO") -> dict[str, object]:
    """Build a complete production-shaped definition for queue tests."""

    return build_task_definition(
        ScheduledTaskInput(
            name=name,
            instructions="Review BF2 ETA CO and prepare an operator report.",
            furnace="BF2",
            data_period="",
            output_format="operator_summary",
            schedule_kind="hourly",
            delivery_channel="in_app",
            job_type="eta_co_report",
            hourly_minute=5,
        ),
        generated_at=datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def command_stack():
    """Create schema-translated job, revision, and command tables."""

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
    ScheduledJobCommand.__table__.create(engine)
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
        future=True,
    )
    job_service = ScheduledJobService(repository=ScheduledJobRepository(factory))
    command_service = ScheduledJobCommandService(
        repository=ScheduledJobCommandRepository(factory)
    )
    try:
        yield job_service, command_service, factory
    finally:
        engine.dispose()


def _create_job(job_service: ScheduledJobService):
    """Persist one pending job owned by the test operator."""

    return job_service.create_job(
        ScheduledJobCreateRequest(
            definition=_definition(),
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )


def test_create_records_revision_and_open_commands_are_idempotent(
    command_stack,
) -> None:
    """Creation should be audited and repeated open requests should collapse."""

    job_service, command_service, factory = command_stack
    job = _create_job(job_service)

    first = command_service.request_action(
        job_id=job.job_id,
        action="provision",
        owner_user_id=_USER_ID,
        requested_by_username="operator.test",
        expected_job_updated_at=job.updated_at,
    )
    duplicate = command_service.request_action(
        job_id=job.job_id,
        action="archive",
        owner_user_id=_USER_ID,
        requested_by_username="operator.test",
        expected_job_updated_at=job.updated_at,
    )

    assert duplicate.command_id == first.command_id
    assert duplicate.action == "provision"
    with factory() as session:
        revision = session.execute(select(ScheduledJobRevision)).scalar_one()
        assert revision.revision_number == 1
        assert revision.change_kind == "created"


def test_command_claim_is_target_scoped_and_fenced(command_stack) -> None:
    """Only the addressed Jetson should claim and settle a leased command."""

    job_service, command_service, _ = command_stack
    job = _create_job(job_service)
    queued = command_service.request_action(
        job_id=job.job_id,
        action="provision",
        owner_user_id=_USER_ID,
        requested_by_username="operator.test",
        expected_job_updated_at=job.updated_at,
    )

    assert (
        command_service.claim_next(
            target_device_id="another-device",
            worker_id="worker-a",
        )
        is None
    )
    claimed = command_service.claim_next(
        target_device_id="bf2-jetson-01",
        worker_id="worker-a",
    )

    assert claimed is not None
    assert claimed.command_id == queued.command_id
    assert claimed.status == "processing"
    assert claimed.lease_token is not None
    settled = command_service.mark_succeeded(
        claimed,
        resulting_job_status="active",
    )
    assert settled.status == "succeeded"
    assert settled.resulting_job_status == "active"


def test_invalid_state_and_wrong_owner_are_rejected_before_enqueue(
    command_stack,
) -> None:
    """Queue requests must not bypass job ownership or lifecycle rules."""

    job_service, command_service, _ = command_stack
    job = _create_job(job_service)

    with pytest.raises(ScheduledJobCommandConflictError, match="not found"):
        command_service.request_action(
            job_id=job.job_id,
            action="provision",
            owner_user_id="00000000-0000-0000-0000-000000000002",
            requested_by_username="operator.other",
            expected_job_updated_at=job.updated_at,
        )
    with pytest.raises(ScheduledJobCommandConflictError, match="Cannot pause"):
        command_service.request_action(
            job_id=job.job_id,
            action="pause",
            owner_user_id=_USER_ID,
            requested_by_username="operator.test",
            expected_job_updated_at=job.updated_at,
        )


def test_processor_applies_pending_edit_and_appends_revision(command_stack) -> None:
    """A Jetson pass should apply an edit and retain both definition versions."""

    job_service, command_service, _ = command_stack
    job = _create_job(job_service)
    edited = deepcopy(job.definition)
    edited["job_name"] = "Edited hourly BF2 ETA CO"
    edited["instructions"] = "Review the signal and prepare the revised report."
    command_service.request_action(
        job_id=job.job_id,
        action="update",
        owner_user_id=_USER_ID,
        requested_by_username="operator.test",
        expected_job_updated_at=job.updated_at,
        definition=edited,
    )
    unused_provisioner = SimpleNamespace()
    processor = ScheduledJobCommandProcessor(
        command_service=command_service,
        job_service=job_service,
        provisioning_service=unused_provisioner,
        target_device_id="bf2-jetson-01",
        worker_id="jetson-test",
    )

    result = processor.process_batch()

    assert result.claimed == 1
    assert result.succeeded == 1
    stored = job_service.get_job(job.job_id)
    assert stored is not None
    assert stored.job_name == "Edited hourly BF2 ETA CO"
    revisions = command_service.list_revisions_for_owner(
        job_id=job.job_id,
        owner_user_id=_USER_ID,
    )
    assert [item.revision_number for item in revisions] == [2, 1]
    assert revisions[0].change_kind == "edited"


def test_terminal_stale_command_does_not_retry(command_stack) -> None:
    """A stale command should fail once instead of retrying obsolete intent."""

    job_service, command_service, factory = command_stack
    job = _create_job(job_service)
    command_service.request_action(
        job_id=job.job_id,
        action="provision",
        owner_user_id=_USER_ID,
        requested_by_username="operator.test",
        expected_job_updated_at=job.updated_at,
    )
    with factory() as session:
        row = session.get(ScheduledJob, job.job_id)
        assert row is not None
        row.job_name = "Changed elsewhere"
        row.updated_at = datetime(2026, 9, 7, 6, 30, tzinfo=timezone.utc)
        session.commit()

    processor = ScheduledJobCommandProcessor(
        command_service=command_service,
        job_service=job_service,
        provisioning_service=SimpleNamespace(),
        target_device_id="bf2-jetson-01",
        worker_id="jetson-test",
    )
    result = processor.process_batch()

    assert result.failed == 1
    latest = command_service.latest_for_owner(
        job_id=job.job_id,
        owner_user_id=_USER_ID,
    )
    assert latest is not None
    assert latest.status == "failed"
    assert latest.attempt_count == 1


def test_active_edit_recovers_after_interruption_immediately_after_pause(
    command_stack,
) -> None:
    """A later Jetson pass should finish an active edit left paused by a crash."""

    job_service, command_service, factory = command_stack
    pending = _create_job(job_service)
    activated_at = datetime(2026, 9, 7, 6, 10, tzinfo=timezone.utc)
    with factory() as session:
        row = session.get(ScheduledJob, pending.job_id)
        assert row is not None
        row.status = "active"
        row.is_active = True
        row.activated_at = activated_at
        row.activation_generation = 1
        row.timer_unit_name = f"furnacemind-job-{pending.job_id}.timer"
        row.updated_at = activated_at
        session.commit()
    active = job_service.get_job(pending.job_id)
    assert active is not None
    edited = deepcopy(active.definition)
    edited["job_name"] = "Recovered active edit"
    command_service.request_action(
        job_id=active.job_id,
        action="update",
        owner_user_id=_USER_ID,
        requested_by_username="operator.test",
        expected_job_updated_at=active.updated_at,
        definition=edited,
    )

    paused_at = datetime(2026, 9, 7, 6, 11, tzinfo=timezone.utc)
    with factory() as session:
        row = session.get(ScheduledJob, active.job_id)
        assert row is not None
        row.status = "paused"
        row.is_active = False
        row.activated_at = None
        row.updated_at = paused_at
        session.commit()

    class _ResumeProvisioner:
        """Activate the edited row without exercising already-tested systemd code."""

        def resume(self, job_id: str):
            """Move the paused test row back to active and return a receipt."""

            with factory() as session:
                row = session.get(ScheduledJob, job_id)
                assert row is not None
                row.status = "active"
                row.is_active = True
                row.activated_at = datetime(2026, 9, 7, 6, 12, tzinfo=timezone.utc)
                row.activation_generation += 1
                row.updated_at = datetime(2026, 9, 7, 6, 12, tzinfo=timezone.utc)
                session.commit()
            return SimpleNamespace(job=job_service.get_job(job_id))

    processor = ScheduledJobCommandProcessor(
        command_service=command_service,
        job_service=job_service,
        provisioning_service=_ResumeProvisioner(),
        target_device_id="bf2-jetson-01",
        worker_id="jetson-recovery-test",
    )

    result = processor.process_batch()

    assert result.succeeded == 1
    stored = job_service.get_job(active.job_id)
    assert stored is not None
    assert stored.status == "active"
    assert stored.job_name == "Recovered active edit"
