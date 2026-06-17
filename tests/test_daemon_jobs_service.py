"""Tests for daemon job definition service behavior."""

from __future__ import annotations

import pytest

from src.data.daemon_jobs.service import DaemonJobService
from src.data.daemon_jobs.validators import DEFAULT_JOB_PAYLOAD


@pytest.fixture
def daemon_job_service() -> DaemonJobService:
    """Create a daemon job service bound to an in-memory SQLite database."""
    return DaemonJobService(db_url="sqlite:///:memory:")


def _valid_payload(**overrides: object) -> dict[str, object]:
    """Return a valid daemon job payload for tests."""
    payload = dict(DEFAULT_JOB_PAYLOAD)
    payload.update(
        {
            "name": "FurnaceMind Shift Report",
            "systemd_unit_name": "evonith-furnacemind-shift-report",
            "job_kind": "furnacemind_shift_report",
            "schedule_type": "systemd_timer",
            "on_calendar": "hourly",
        }
    )
    payload.update(overrides)
    return payload


def test_create_job_persists_definition_and_audit_event(
    daemon_job_service: DaemonJobService,
) -> None:
    """Creating a daemon job should persist the definition and audit row."""
    created = daemon_job_service.create_job(_valid_payload(), actor="admin_1")

    assert created.id
    assert created.enabled is False
    assert created.systemd_unit_name == "evonith-furnacemind-shift-report"

    listed = daemon_job_service.list_jobs()
    assert len(listed) == 1
    assert listed[0].id == created.id

    events = daemon_job_service.get_audit_events(created.id)
    assert len(events) == 1
    assert events[0].event_type == "created"
    assert events[0].actor == "admin_1"


def test_validation_rejects_unsafe_unit_path_json_and_critical_rules(
    daemon_job_service: DaemonJobService,
) -> None:
    """Validation should block unsafe names, paths, JSON, and critical jobs."""
    result = daemon_job_service.validate_payload(
        _valid_payload(
            systemd_unit_name="Bad Name.service",
            working_directory="/tmp/app;rm",
            job_args_json="{bad json",
            criticality="critical",
            reporting_rules_json="{}",
        )
    )

    assert not result.is_valid
    assert any("systemd_unit_name" in error for error in result.errors)
    assert any("working_directory" in error for error in result.errors)
    assert any("job_args_json" in error for error in result.errors)
    assert any("Critical jobs" in error for error in result.errors)


def test_preview_clone_enable_and_soft_delete_flow(
    daemon_job_service: DaemonJobService,
) -> None:
    """Lifecycle helpers should render previews and preserve soft-deleted jobs."""
    created = daemon_job_service.create_job(_valid_payload(), actor="admin_1")
    preview = daemon_job_service.preview_systemd(created.id, actor="admin_1")

    assert "ExecStart=" in preview.service_unit
    assert "OnCalendar=hourly" in preview.timer_unit
    assert "Step 1 only previews systemd units" in preview.warnings[0]

    enabled = daemon_job_service.set_enabled(created.id, True, actor="admin_1")
    assert enabled.enabled is True

    cloned = daemon_job_service.clone_job(created.id, actor="admin_1")
    assert cloned.enabled is False
    assert cloned.systemd_unit_name == "evonith-furnacemind-shift-report-copy"

    deleted = daemon_job_service.delete_job(cloned.id, actor="admin_1")
    assert deleted.deleted is True

    assert [job.id for job in daemon_job_service.list_jobs()] == [created.id]
    assert len(daemon_job_service.list_jobs(include_deleted=True)) == 2

    event_types = [
        event.event_type for event in daemon_job_service.get_audit_events(created.id)
    ]
    assert {"created", "previewed", "enabled", "cloned"}.issubset(event_types)
