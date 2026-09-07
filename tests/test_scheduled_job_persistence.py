"""Tests for Phase 2 scheduled-job database persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from data.scheduled_tasks import (
    ScheduledJobCreateRequest,
    ScheduledJobService,
    ScheduledJobValidationError,
)
from furnace_data.relational import (
    ScheduledJob,
    ScheduledJobOutput,
    ScheduledJobRepository,
    ScheduledJobRevision,
    ScheduledJobRun,
    ScheduledJobRunStatus,
    ScheduledJobStatus,
)
from utils.scheduled_tasks.scheduled_task_definition import (
    ScheduledTaskInput,
    build_task_definition,
)

_USER_ID = "00000000-0000-0000-0000-000000000001"


def _valid_definition() -> dict[str, object]:
    """Build a representative definition through the production contract."""

    task = ScheduledTaskInput(
        name="Hourly BF2 ETA CO",
        instructions="Review BF2 ETA CO and prepare a concise operator report.",
        furnace="BF2",
        data_period="",
        output_format="operator_summary",
        schedule_kind="hourly",
        delivery_channel="in_app",
        job_type="eta_co_report",
        hourly_minute=5,
    )
    return build_task_definition(
        task,
        generated_at=datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def persistence_stack():
    """Create schema-translated scheduled tables in an in-memory database."""

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
    ScheduledJobOutput.__table__.create(engine)
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
        future=True,
    )
    service = ScheduledJobService(
        repository=ScheduledJobRepository(factory),
    )
    try:
        yield service, factory
    finally:
        engine.dispose()


def test_create_job_persists_canonical_definition_and_pending_status(
    persistence_stack,
) -> None:
    """A valid create request should persist trusted fields and canonical JSON."""

    service, factory = persistence_stack
    definition = _valid_definition()

    created = service.create_job(
        ScheduledJobCreateRequest(
            definition=definition,
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )

    assert UUID(created.job_id)
    assert created.status == ScheduledJobStatus.PENDING_PROVISIONING.value
    assert created.is_active is False
    assert created.definition == definition
    assert created.target_device_id == "bf2-jetson-01"
    assert created.timer_unit_name is None
    assert created.provisioning_error is None
    assert created.activated_at is None

    with factory() as session:
        row = session.execute(select(ScheduledJob)).scalar_one()
        assert str(row.job_id) == created.job_id
        assert row.created_by_user_id == UUID(_USER_ID)
        assert row.created_by_username == "operator.test"
        assert row.definition_json == definition
        assert row.activated_at is None


def test_get_job_returns_the_stored_definition(persistence_stack) -> None:
    """Stored scheduled jobs should be retrievable by their public UUID."""

    service, _ = persistence_stack
    created = service.create_job(
        ScheduledJobCreateRequest(
            definition=_valid_definition(),
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )

    loaded = service.get_job(created.job_id)

    assert loaded is not None
    assert loaded == created


def test_owner_job_list_is_bounded_and_includes_only_the_latest_run(
    persistence_stack,
) -> None:
    """Management listing should remain owner-scoped and avoid per-job queries."""

    service, factory = persistence_stack
    first = service.create_job(
        ScheduledJobCreateRequest(
            definition=_valid_definition(),
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )
    second_definition = _valid_definition()
    second_definition["job_name"] = "Second owner job"
    second = service.create_job(
        ScheduledJobCreateRequest(
            definition=second_definition,
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )
    other_definition = _valid_definition()
    other_definition["job_name"] = "Different operator job"
    service.create_job(
        ScheduledJobCreateRequest(
            definition=other_definition,
            created_by_user_id="00000000-0000-0000-0000-000000000002",
            created_by_username="operator.other",
        )
    )
    with factory() as session:
        session.add_all(
            [
                ScheduledJobRun(
                    job_id=first.job_id,
                    scheduled_for=datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc),
                    status=ScheduledJobRunStatus.FAILED.value,
                    completed_at=datetime(2026, 9, 5, 8, 1, tzinfo=timezone.utc),
                    created_at=datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc),
                ),
                ScheduledJobRun(
                    job_id=first.job_id,
                    scheduled_for=datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc),
                    status=ScheduledJobRunStatus.COMPLETED.value,
                    completed_at=datetime(2026, 9, 5, 9, 1, tzinfo=timezone.utc),
                    created_at=datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc),
                ),
            ]
        )
        session.commit()

    items = service.list_jobs_for_owner(_USER_ID)

    assert {item.job.job_id for item in items} == {first.job_id, second.job_id}
    first_item = next(item for item in items if item.job.job_id == first.job_id)
    assert first_item.last_run_status == ScheduledJobRunStatus.COMPLETED.value
    assert first_item.last_run_at == datetime(2026, 9, 5, 9, 1, tzinfo=timezone.utc)


def test_owner_job_details_and_run_output_history_are_access_controlled(
    persistence_stack,
) -> None:
    """A management read must never expose another operator's job or output."""

    service, factory = persistence_stack
    created = service.create_job(
        ScheduledJobCreateRequest(
            definition=_valid_definition(),
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )
    run_id = UUID("00000000-0000-0000-0000-000000000301")
    with factory() as session:
        session.add(
            ScheduledJobRun(
                run_id=run_id,
                job_id=created.job_id,
                scheduled_for=datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc),
                status=ScheduledJobRunStatus.COMPLETED.value,
                attempt_number=2,
                completed_at=datetime(2026, 9, 5, 10, 2, tzinfo=timezone.utc),
                triggered_by="systemd",
            )
        )
        session.add(
            ScheduledJobOutput(
                run_id=run_id,
                output_type="operator_summary",
                content="BF2 remained stable.",
                content_json={"row_count": 12},
            )
        )
        session.commit()

    detail = service.get_job_for_owner(created.job_id, _USER_ID)
    history = service.list_run_history(created.job_id, _USER_ID)

    assert detail == created
    assert len(history) == 1
    assert history[0].run_id == str(run_id)
    assert history[0].attempt_number == 2
    assert history[0].content == "BF2 remained stable."
    assert history[0].content_json == {"row_count": 12}
    assert (
        service.get_job_for_owner(
            created.job_id,
            "00000000-0000-0000-0000-000000000002",
        )
        is None
    )
    assert (
        service.list_run_history(
            created.job_id,
            "00000000-0000-0000-0000-000000000002",
        )
        == ()
    )


def test_create_job_rejects_invalid_definition_before_persistence(
    persistence_stack,
) -> None:
    """Backend validation should reject modified or incomplete JSON."""

    service, factory = persistence_stack
    invalid_definition = _valid_definition()
    invalid_definition["schema_version"] = "unsupported-version"

    with pytest.raises(ScheduledJobValidationError) as exc_info:
        service.create_job(
            ScheduledJobCreateRequest(
                definition=invalid_definition,
                created_by_user_id=_USER_ID,
                created_by_username="operator.test",
            )
        )

    assert any("schema_version" in error for error in exc_info.value.errors)
    with factory() as session:
        assert session.execute(select(ScheduledJob)).scalars().all() == []


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("job_type", "unknown_report", "job_type"),
        ("target_device.device_id", "unknown-device", "target_device.device_id"),
        ("inputs.data_source", "unknown_source", "inputs.data_source"),
    ],
)
def test_create_job_rejects_values_missing_from_runtime_catalog(
    persistence_stack,
    field: str,
    value: str,
    expected_error: str,
) -> None:
    """Backend validation should enforce the deployed runtime catalog."""

    service, _ = persistence_stack
    definition = _valid_definition()
    owner, child = field.split(".", 1) if "." in field else (None, field)
    if owner is None:
        definition[child] = value
    else:
        nested = definition[owner]
        assert isinstance(nested, dict)
        nested[child] = value

    with pytest.raises(ScheduledJobValidationError) as exc_info:
        service.create_job(
            ScheduledJobCreateRequest(
                definition=definition,
                created_by_user_id=_USER_ID,
                created_by_username="operator.test",
            )
        )

    assert any(expected_error in error for error in exc_info.value.errors)


@pytest.mark.parametrize(
    ("user_id", "username", "expected_error"),
    [
        ("not-a-uuid", "operator.test", "created_by_user_id"),
        (_USER_ID, "  ", "created_by_username"),
    ],
)
def test_create_job_requires_authenticated_operator_identity(
    persistence_stack,
    user_id: str,
    username: str,
    expected_error: str,
) -> None:
    """Persistence should reject missing or malformed ownership information."""

    service, _ = persistence_stack

    with pytest.raises(ScheduledJobValidationError) as exc_info:
        service.create_job(
            ScheduledJobCreateRequest(
                definition=_valid_definition(),
                created_by_user_id=user_id,
                created_by_username=username,
            )
        )

    assert any(expected_error in error for error in exc_info.value.errors)
