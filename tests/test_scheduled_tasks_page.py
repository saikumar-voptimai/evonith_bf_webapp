"""Streamlit tests for scheduled-task JSON creation and management."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
_TEST_JOB_ID = "00000000-0000-0000-0000-000000000101"
_TEST_USER_ID = "00000000-0000-0000-0000-000000000001"


class _FakeScheduledJobService:
    """Capture UI persistence calls without requiring PostgreSQL."""

    requests: list[object] = []
    updates: list[dict[str, object]] = []
    archives: list[dict[str, object]] = []
    deletes: list[dict[str, object]] = []
    items: tuple[SimpleNamespace, ...] = ()

    def create_job(self, request: object) -> SimpleNamespace:
        """Return a waiting-state receipt for one captured create request."""

        self.__class__.requests.append(request)
        return SimpleNamespace(
            job_id=_TEST_JOB_ID,
            status="pending_provisioning",
            definition=request.definition,
        )

    def list_jobs_for_owner(
        self, owner_user_id: str, *, limit: int = 100
    ) -> tuple[SimpleNamespace, ...]:
        """Return the configured owner-scoped task list."""

        assert owner_user_id == _TEST_USER_ID
        assert 1 <= limit <= 100
        return self.__class__.items

    def get_job_for_owner(
        self, job_id: str, owner_user_id: str
    ) -> SimpleNamespace | None:
        """Return a configured task only for the authenticated test owner."""

        assert owner_user_id == _TEST_USER_ID
        return next(
            (item.job for item in self.__class__.items if item.job.job_id == job_id),
            None,
        )

    def update_job(self, **kwargs: object) -> SimpleNamespace:
        """Capture a direct JSON update and return its receipt."""

        self.__class__.updates.append(kwargs)
        return SimpleNamespace(
            job_id=str(kwargs["job_id"]),
            status="pending_provisioning",
            definition=kwargs["definition"],
        )

    def archive_job(self, **kwargs: object) -> SimpleNamespace:
        """Capture an archive request that changes stored UI state only."""

        self.__class__.archives.append(kwargs)
        return SimpleNamespace(job_id=kwargs["job_id"], status="deleted")

    def delete_job(self, **kwargs: object) -> None:
        """Capture permanent deletion of an archived definition."""

        self.__class__.deletes.append(kwargs)


@pytest.fixture(autouse=True)
def _stub_scheduled_job_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject authenticated identity and fake definition persistence."""

    import ui.scheduled_tasks as scheduled_tasks_ui

    _FakeScheduledJobService.requests = []
    _FakeScheduledJobService.updates = []
    _FakeScheduledJobService.archives = []
    _FakeScheduledJobService.deletes = []
    _FakeScheduledJobService.items = ()
    monkeypatch.setattr(scheduled_tasks_ui, "current_user_id", lambda: _TEST_USER_ID)
    monkeypatch.setattr(
        scheduled_tasks_ui,
        "_get_scheduled_job_service",
        lambda: _FakeScheduledJobService(),
    )


def _stored_job_item(status: str = "pending_provisioning") -> SimpleNamespace:
    """Return one stored definition list item for management-view tests."""

    created_at = datetime(2026, 9, 5, 4, 0, tzinfo=timezone.utc)
    definition = {
        "schema_version": "scheduled-job-definition/v1",
        "job_name": "Hourly BF2 ETA CO",
        "instructions": "Review ETA CO and prepare a concise operator report.",
        "job_type": "eta_co_report",
        "analysis_level": "low",
        "schedule": {
            "frequency": "hourly",
            "timezone": "Asia/Kolkata",
            "overlap_policy": "skip",
            "misfire_policy": "fire_once_latest",
            "trigger": {"type": "cron", "expression": "5 * * * *"},
        },
        "target_device": {
            "device_id": "bf2-jetson-01",
            "device_type": "jetson",
        },
        "inputs": {
            "furnace": "BF2",
            "data_source": "online_process_data",
            "output_format": "operator_summary",
            "signal": "body_etaco",
            "report_duration_minutes": 60,
            "aggregation_interval": "1min",
            "warning_threshold": 42.0,
            "critical_threshold": 40.0,
            "include_graph": False,
            "include_ai_summary": True,
        },
        "delivery": {
            "channel": "in_app",
            "notify_on_failure": True,
            "destination_ref": "task_owner",
        },
        "retry": {
            "maximum_attempts": 3,
            "retry_interval_seconds": 60,
            "timeout_seconds": 600,
        },
        "policy_profile": "bf_operator_read_only_v1",
    }
    job = SimpleNamespace(
        job_id=_TEST_JOB_ID,
        job_name=definition["job_name"],
        schema_version=definition["schema_version"],
        definition=definition,
        status=status,
        created_by_user_id=_TEST_USER_ID,
        created_by_username="operator.test",
        created_at=created_at,
        updated_at=created_at,
    )
    return SimpleNamespace(job=job, last_run_status=None, last_run_at=None)


def _app() -> AppTest:
    """Return an authenticated Scheduled Tasks app test instance."""

    sys.modules.pop("utils.session", None)
    app = AppTest.from_file("src/custom_pages/10_Scheduled_Tasks.py")
    app.session_state["auth_user"] = "operator.test"
    app.session_state["role"] = "user"
    app.run(timeout=30)
    assert not app.exception
    return app


def test_operator_can_create_and_download_validated_json() -> None:
    """Generate, persist, preview, and offer one canonical task JSON file."""

    app = _app()
    app.text_input(key="scheduled_task_name").set_value("Daily BF2 health")
    app.text_area(key="scheduled_task_instructions").set_value(
        "Review BF2 ETA CO and prepare the operator health summary."
    )
    app.button(key="scheduled_task_create").click().run(timeout=30)

    assert not app.exception
    assert not app.error
    generated = app.session_state["scheduled_task_generated_definition"]
    payload = json.loads(generated["json"])
    assert generated["job_id"] == _TEST_JOB_ID
    assert generated["filename"] == "daily-bf2-health.scheduled-job.json"
    assert payload["schema_version"] == "scheduled-job-definition/v1"
    assert payload["job_name"] == "Daily BF2 health"
    assert payload["target_device"]["device_id"] == "bf2-jetson-01"
    assert len(_FakeScheduledJobService.requests) == 1
    assert any(
        item.label == "Download task JSON" for item in app.get("download_button")
    )


def test_task_type_fields_and_clear_confirmation_remain_operator_friendly() -> None:
    """Show type-specific controls and clear only after explicit confirmation."""

    app = _app()
    assert any(item.key == "scheduled_task_eta_duration" for item in app.number_input)
    assert not any(item.key == "scheduled_task_target_device" for item in app.selectbox)

    app.selectbox(key="scheduled_task_job_type").select("Daily Furnace Summary").run(
        timeout=30
    )
    assert not any(
        item.key == "scheduled_task_eta_duration" for item in app.number_input
    )
    assert not any(item.key == "scheduled_task_data_period" for item in app.selectbox)

    app.text_input(key="scheduled_task_name").set_value("Daily report")
    app.text_area(key="scheduled_task_instructions").set_value(
        "Prepare the previous day's BF2 report."
    )
    app.button(key="scheduled_task_clear").click().run(timeout=30)
    assert app.text_input(key="scheduled_task_name").value == "Daily report"
    app.button(key="scheduled_task_clear_confirm").click().run(timeout=30)
    assert app.text_input(key="scheduled_task_name").value == ""


def test_management_keeps_future_controls_disabled_and_history_empty() -> None:
    """Retain lifecycle UI without claiming that execution is available."""

    _FakeScheduledJobService.items = (_stored_job_item(),)
    app = _app()
    app.session_state["scheduled_task_page_view"] = "Your tasks"
    app.run(timeout=30)

    activate = app.button(key=f"scheduled_task_action_{_TEST_JOB_ID}")
    assert activate.label == "Activate"
    assert activate.disabled
    assert not app.button(key=f"scheduled_task_archive_{_TEST_JOB_ID}").disabled
    assert app.button(key=f"scheduled_task_delete_{_TEST_JOB_ID}").disabled
    assert any("Timer controls are read-only" in item.value for item in app.caption)
    assert any("has not run yet" in item.value for item in app.info)
    assert any("scheduled-job-definition/v1" in item.value for item in app.code)


def test_clone_and_direct_edit_manage_stored_json() -> None:
    """Clone independently and save edits without a runtime command queue."""

    _FakeScheduledJobService.items = (_stored_job_item(),)
    app = _app()
    app.session_state["scheduled_task_page_view"] = "Your tasks"
    app.run(timeout=30)

    app.button(key=f"scheduled_task_clone_{_TEST_JOB_ID}").click().run(timeout=30)
    assert app.session_state["scheduled_task_page_view"] == "Create task"
    assert app.text_input(key="scheduled_task_name").value == (
        "Copy of Hourly BF2 ETA CO"
    )

    app.session_state["scheduled_task_page_view"] = "Your tasks"
    app.run(timeout=30)
    app.button(key=f"scheduled_task_edit_{_TEST_JOB_ID}").click().run(timeout=30)
    app.text_input(key="scheduled_task_name").set_value("Revised BF2 ETA CO")
    app.button(key="scheduled_task_create").click().run(timeout=30)

    assert not app.exception
    update = _FakeScheduledJobService.updates[-1]
    assert update["job_id"] == _TEST_JOB_ID
    assert update["definition"]["job_name"] == "Revised BF2 ETA CO"
    assert app.session_state["scheduled_task_generated_definition"]["kind"] == "update"


def test_archive_and_delete_require_separate_confirmations() -> None:
    """Archive definitions first and permanently delete only archived tasks."""

    _FakeScheduledJobService.items = (_stored_job_item(),)
    app = _app()
    app.session_state["scheduled_task_page_view"] = "Your tasks"
    app.run(timeout=30)
    app.button(key=f"scheduled_task_archive_{_TEST_JOB_ID}").click().run(timeout=30)
    assert not _FakeScheduledJobService.archives
    app.button(key=f"scheduled_task_archive_confirm_{_TEST_JOB_ID}").click().run(
        timeout=30
    )
    assert _FakeScheduledJobService.archives[-1]["job_id"] == _TEST_JOB_ID

    _FakeScheduledJobService.items = (_stored_job_item(status="deleted"),)
    app.run(timeout=30)
    delete = app.button(key=f"scheduled_task_delete_{_TEST_JOB_ID}")
    assert not delete.disabled
    delete.click().run(timeout=30)
    assert not _FakeScheduledJobService.deletes
    app.button(key=f"scheduled_task_delete_confirm_{_TEST_JOB_ID}").click().run(
        timeout=30
    )
    assert _FakeScheduledJobService.deletes[-1]["job_id"] == _TEST_JOB_ID


def test_page_copy_and_theme_match_the_definition_only_scope() -> None:
    """Guard terminology, adaptive palette, and absence of runtime imports."""

    source = (ROOT / "src" / "ui" / "scheduled_tasks.py").read_text(encoding="utf-8")
    css = (ROOT / "src" / "assets" / "css" / "scheduled_tasks_style.css").read_text(
        encoding="utf-8"
    )

    assert "Scheduled Tasks" in source
    assert '"Schedule",' in source
    assert "Generate JSON" not in source
    assert "Cron expression" not in source
    assert "Timer controls are read-only in this milestone." in source
    assert "ScheduledJobCommandService" not in source
    assert "list_run_history" not in source
    assert "light-dark(" in css
    assert '[data-testid="stApp"]:has(.st-key-scheduled_task_shell)' in css
