"""Persistence tests for owner-scoped scheduled-task JSON definitions."""

from __future__ import annotations

from datetime import datetime, time, timezone
from uuid import UUID

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from furnace_data.relational.models import User
from data.scheduled_tasks import (
    ScheduledJobConflictError,
    ScheduledJobCreateRequest,
    ScheduledJobService,
    ScheduledJobValidationError,
)
from furnace_data.relational import (
    ScheduledTaskDefinitionRecord,
    ScheduledTaskDefinitionRepository,
)
from utils.scheduled_tasks.scheduled_task_definition import (
    ScheduledTaskInput,
    build_task_definition,
)

_OWNER_ID = UUID("00000000-0000-0000-0000-000000000001")
_OTHER_ID = UUID("00000000-0000-0000-0000-000000000002")


@pytest.fixture
def service() -> ScheduledJobService:
    """Return a service backed by isolated SQLite tables with translated schemas."""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        execution_options={
            "schema_translate_map": {"identity": None, "automation": None}
        },
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        """Enable SQLite foreign-key behaviour for this test connection."""

        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    User.__table__.create(engine)
    ScheduledTaskDefinitionRecord.__table__.create(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        session.add_all(
            [
                User(
                    id=_OWNER_ID,
                    username="operator.one",
                    password_hash="unused",
                    role="user",
                ),
                User(
                    id=_OTHER_ID,
                    username="operator.two",
                    password_hash="unused",
                    role="user",
                ),
            ]
        )
        session.commit()

    return ScheduledJobService(repository=ScheduledTaskDefinitionRepository(factory))


def _definition(name: str = "Daily BF2 summary") -> dict[str, object]:
    """Build one valid canonical definition for storage tests."""

    return build_task_definition(
        ScheduledTaskInput(
            name=name,
            instructions="Review the previous BF2 day and prepare an operator summary.",
            furnace="BF2",
            data_period="previous_day",
            output_format="operator_summary",
            schedule_kind="daily",
            delivery_channel="in_app",
            job_type="daily_report",
            analysis_level="low",
            data_source="online_process_data",
            target_device_id="bf2-jetson-01",
            target_device_type="jetson",
            run_time=time(7, 0),
        ),
        generated_at=datetime(2026, 9, 7, 3, 0, tzinfo=timezone.utc),
    )


def _create(service: ScheduledJobService) -> object:
    """Persist one definition for the primary test owner."""

    return service.create_job(
        ScheduledJobCreateRequest(
            definition=_definition(),
            created_by_user_id=str(_OWNER_ID),
            created_by_username="operator.one",
        )
    )


def test_create_list_and_get_are_owner_scoped(service: ScheduledJobService) -> None:
    """Persist canonical JSON and prevent a different owner from reading it."""

    created = _create(service)

    assert created.status == "pending_provisioning"
    assert created.definition == _definition()
    assert service.get_job_for_owner(created.job_id, str(_OWNER_ID)) == created
    assert service.get_job_for_owner(created.job_id, str(_OTHER_ID)) is None
    assert [
        item.job.job_id for item in service.list_jobs_for_owner(str(_OWNER_ID))
    ] == [created.job_id]
    assert service.list_jobs_for_owner(str(_OTHER_ID)) == ()


def test_update_uses_owner_and_optimistic_version(service: ScheduledJobService) -> None:
    """Replace valid JSON while rejecting stale or cross-owner mutations."""

    created = _create(service)
    revised = _definition("Revised daily BF2 summary")
    updated = service.update_job(
        job_id=created.job_id,
        owner_user_id=str(_OWNER_ID),
        changed_by_username="operator.one",
        expected_updated_at=created.updated_at,
        definition=revised,
    )

    assert updated.job_name == "Revised daily BF2 summary"
    assert updated.definition == revised
    with pytest.raises(ScheduledJobConflictError):
        service.update_job(
            job_id=created.job_id,
            owner_user_id=str(_OWNER_ID),
            changed_by_username="operator.one",
            expected_updated_at=created.updated_at,
            definition=_definition("Stale edit"),
        )
    with pytest.raises(ScheduledJobConflictError):
        service.update_job(
            job_id=created.job_id,
            owner_user_id=str(_OTHER_ID),
            changed_by_username="operator.two",
            expected_updated_at=updated.updated_at,
            definition=_definition("Cross-owner edit"),
        )


def test_archive_then_permanently_delete(service: ScheduledJobService) -> None:
    """Archive stored UI state before allowing irreversible deletion."""

    created = _create(service)
    archived = service.archive_job(
        job_id=created.job_id,
        owner_user_id=str(_OWNER_ID),
        expected_updated_at=created.updated_at,
    )

    assert archived.status == "deleted"
    with pytest.raises(ScheduledJobConflictError):
        service.delete_job(job_id=created.job_id, owner_user_id=str(_OTHER_ID))
    service.delete_job(job_id=created.job_id, owner_user_id=str(_OWNER_ID))
    assert service.get_job_for_owner(created.job_id, str(_OWNER_ID)) is None


def test_invalid_definition_and_identity_never_reach_storage(
    service: ScheduledJobService,
) -> None:
    """Reject malformed JSON and unauthenticated owner identifiers."""

    invalid = _definition()
    invalid.pop("instructions")

    with pytest.raises(ScheduledJobValidationError):
        service.create_job(
            ScheduledJobCreateRequest(
                definition=invalid,
                created_by_user_id=str(_OWNER_ID),
                created_by_username="operator.one",
            )
        )
    with pytest.raises(ScheduledJobValidationError):
        service.create_job(
            ScheduledJobCreateRequest(
                definition=_definition(),
                created_by_user_id="not-a-uuid",
                created_by_username="operator.one",
            )
        )
