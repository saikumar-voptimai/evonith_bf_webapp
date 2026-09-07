"""Streamlit tests for Scheduled Tasks creation and owner management."""

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
    """Capture page persistence requests without requiring PostgreSQL."""

    requests = []
    items = ()
    history = ()

    def create_job(self, request):
        """Return a pending job receipt for a validated definition."""

        self.__class__.requests.append(request)
        return SimpleNamespace(
            job_id=_TEST_JOB_ID,
            status="pending_provisioning",
            definition=request.definition,
        )

    def list_jobs_for_owner(self, owner_user_id, *, limit=50):
        """Return the configured owner-scoped task list."""

        assert owner_user_id == _TEST_USER_ID
        assert 1 <= limit <= 100
        return self.__class__.items

    def get_job_for_owner(self, job_id, owner_user_id):
        """Return a configured task only for the authenticated test owner."""

        assert owner_user_id == _TEST_USER_ID
        return next(
            (item.job for item in self.__class__.items if item.job.job_id == job_id),
            None,
        )

    def list_run_history(self, job_id, owner_user_id, *, limit=25):
        """Return the configured bounded run history."""

        assert job_id == _TEST_JOB_ID
        assert owner_user_id == _TEST_USER_ID
        assert 1 <= limit <= 100
        return self.__class__.history


class _FakeScheduledJobCommandService:
    """Capture lifecycle requests without requiring a target Jetson."""

    calls = []
    commands = []

    def request_action(self, **kwargs):
        """Return a queued command receipt for an owner-scoped request."""

        self.__class__.calls.append(kwargs)
        command = SimpleNamespace(
            command_id="00000000-0000-0000-0000-000000000401",
            job_id=kwargs["job_id"],
            action=kwargs["action"],
            status="pending",
            definition=kwargs.get("definition"),
            error_message=None,
        )
        self.__class__.commands.append(command)
        return command

    def latest_for_owner(self, *, job_id, owner_user_id):
        """Return the latest captured command for the selected owner job."""

        assert owner_user_id == _TEST_USER_ID
        return next(
            (
                command
                for command in reversed(self.__class__.commands)
                if command.job_id == job_id
            ),
            None,
        )

    def list_revisions_for_owner(self, *, job_id, owner_user_id, limit=25):
        """Return one initial revision for management-page rendering."""

        assert job_id == _TEST_JOB_ID
        assert owner_user_id == _TEST_USER_ID
        assert limit == 25
        return (
            SimpleNamespace(
                revision_number=1,
                change_kind="created",
                created_at=datetime(2026, 9, 5, 4, 0, tzinfo=timezone.utc),
                changed_by_username="operator.test",
            ),
        )


@pytest.fixture(autouse=True)
def _stub_scheduled_job_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject authenticated database identity and persistence for page tests."""

    import ui.scheduled_tasks as scheduled_tasks_ui

    _FakeScheduledJobService.requests = []
    _FakeScheduledJobService.items = ()
    _FakeScheduledJobService.history = ()
    _FakeScheduledJobCommandService.calls = []
    _FakeScheduledJobCommandService.commands = []
    monkeypatch.setattr(scheduled_tasks_ui, "current_user_id", lambda: _TEST_USER_ID)
    monkeypatch.setattr(
        scheduled_tasks_ui,
        "_get_scheduled_job_service",
        lambda: _FakeScheduledJobService(),
    )
    monkeypatch.setattr(
        scheduled_tasks_ui,
        "_get_scheduled_job_command_service",
        lambda: _FakeScheduledJobCommandService(),
    )


def _stored_job_item(status: str = "active") -> SimpleNamespace:
    """Return one stored job list item for management-view tests."""

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
        "policy": {"profile": "bf_operator_read_only_v1"},
    }
    job = SimpleNamespace(
        job_id=_TEST_JOB_ID,
        job_name=definition["job_name"],
        schema_version=definition["schema_version"],
        definition=definition,
        status=status,
        is_active=status == "active",
        created_by_user_id=_TEST_USER_ID,
        created_by_username="operator.test",
        target_device_id="bf2-jetson-01",
        timer_unit_name=(f"furnacemind-job-{_TEST_JOB_ID}.timer"),
        provisioning_error=None,
        activated_at=created_at,
        created_at=created_at,
        updated_at=created_at,
    )
    return SimpleNamespace(
        job=job,
        last_run_status="succeeded",
        last_run_at=created_at,
    )


def _stored_run_history() -> tuple[SimpleNamespace, ...]:
    """Return one successful stored run and output for management-view tests."""

    occurred_at = datetime(2026, 9, 5, 4, 5, tzinfo=timezone.utc)
    return (
        SimpleNamespace(
            run_id="00000000-0000-0000-0000-000000000201",
            scheduled_for=occurred_at,
            status="succeeded",
            attempt_number=1,
            triggered_by="systemd_timer",
            started_at=occurred_at,
            completed_at=occurred_at,
            error_message=None,
            output_id="00000000-0000-0000-0000-000000000301",
            output_type="operator_summary",
            content="ETA CO remained within the configured limits.",
            content_json={"result": "normal"},
            output_created_at=occurred_at,
        ),
    )


def _app() -> AppTest:
    """Return an authenticated Scheduled Tasks app test instance."""

    # Session-unit tests temporarily import this module against a Streamlit
    # stub, so reload it against the real AppTest runtime when suites are mixed.
    sys.modules.pop("utils.session", None)
    app = AppTest.from_file("src/custom_pages/10_Scheduled_Tasks.py")
    app.session_state["auth_user"] = "operator.test"
    app.session_state["role"] = "user"
    app.run(timeout=30)
    assert not app.exception
    return app


def test_operator_can_create_an_eta_co_job_definition_in_session() -> None:
    """Verify ETA CO creation and per-user session-state isolation."""

    app = _app()

    assert app.selectbox(key="scheduled_task_job_type").value == "ETA CO Report"
    assert app.number_input(key="scheduled_task_eta_duration").value == 60
    assert app.selectbox(key="scheduled_task_eta_signal").value == "Body ETA CO"
    assert not any(item.key == "scheduled_task_target_device" for item in app.selectbox)

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
    assert generated["status"] == "pending_provisioning"
    assert len(_FakeScheduledJobService.requests) == 1
    assert _FakeScheduledJobService.requests[0].created_by_username == "operator.test"
    assert generated["filename"] == "daily-bf2-health.scheduled-job.json"
    assert payload["schema_version"] == "scheduled-job-definition/v1"
    assert payload["job_name"] == "Daily BF2 health"
    assert payload["job_type"] == "eta_co_report"
    assert payload["schedule"]["trigger"]["type"] == "cron"
    assert payload["inputs"]["signal"] == "body_etaco"
    assert payload["target_device"] == {
        "device_id": "bf2-jetson-01",
        "device_type": "jetson",
    }
    assert payload["retry"]["maximum_attempts"] == 3
    assert payload["analysis_level"] == "low"
    assert "metadata" not in payload
    assert "summary" not in payload["schedule"]
    assert "cron_format" not in payload["schedule"]["trigger"]

    app.session_state["auth_user"] = "operator.two"
    app.run(timeout=30)

    assert "scheduled_task_generated_definition" not in app.session_state
    assert app.text_input(key="scheduled_task_name").value == ""


def test_hourly_email_controls_generate_complete_delivery_json() -> None:
    """Verify hourly email controls produce complete delivery JSON."""

    app = _app()
    app.selectbox(key="scheduled_task_repeat").select("Every hour")
    app.selectbox(key="scheduled_task_delivery_channel").select("Email")
    app.run(timeout=30)

    app.text_input(key="scheduled_task_name").set_value("Hourly ETA CO")
    app.text_area(key="scheduled_task_instructions").set_value(
        "Check ETA CO and prepare a brief hourly report."
    )
    app.number_input(key="scheduled_task_hourly_minute").set_value(5)
    app.text_area(key="scheduled_task_email_recipients").set_value(
        "lead@example.com; team@example.com"
    )
    app.text_input(key="scheduled_task_email_subject").set_value("Hourly BF2 ETA CO")
    app.multiselect(key="scheduled_task_email_attachments").set_value(
        ["Report JSON", "Trend graph (PNG)"]
    )
    app.button(key="scheduled_task_create").click().run(timeout=30)

    assert not app.exception
    assert not app.error
    payload = json.loads(
        app.session_state["scheduled_task_generated_definition"]["json"]
    )
    assert payload["schedule"]["frequency"] == "hourly"
    assert payload["schedule"]["trigger"]["expression"] == "5 * * * *"
    assert payload["delivery"] == {
        "channel": "email",
        "notify_on_failure": True,
        "recipients": ["lead@example.com", "team@example.com"],
        "subject": "Hourly BF2 ETA CO",
        "attachments": ["json", "png"],
    }


def test_switching_task_type_changes_inputs_and_clear_resets_all_state() -> None:
    """Verify task-type switching and confirmed clearing reset form state."""

    app = _app()
    assert any(item.key == "scheduled_task_eta_duration" for item in app.number_input)

    app.selectbox(key="scheduled_task_job_type").select("Daily Furnace Summary").run(
        timeout=30
    )

    assert not any(
        item.key == "scheduled_task_eta_duration" for item in app.number_input
    )
    assert not any(item.key == "scheduled_task_data_period" for item in app.selectbox)

    app.text_input(key="scheduled_task_name").set_value("Daily report")
    app.text_area(key="scheduled_task_instructions").set_value(
        "Prepare a daily BF2 report."
    )
    app.button(key="scheduled_task_create").click().run(timeout=30)
    assert "scheduled_task_generated_definition" in app.session_state
    payload = json.loads(
        app.session_state["scheduled_task_generated_definition"]["json"]
    )
    assert payload["job_type"] == "daily_report"
    assert payload["inputs"]["data_period"] == "previous_day"

    app.button(key="scheduled_task_clear").click().run(timeout=30)

    assert not app.exception
    assert app.text_input(key="scheduled_task_name").value == "Daily report"
    assert "scheduled_task_generated_definition" in app.session_state

    app.button(key="scheduled_task_clear_confirm").click().run(timeout=30)

    assert not app.exception
    assert app.text_input(key="scheduled_task_name").value == ""
    assert "scheduled_task_generated_definition" not in app.session_state


def test_task_type_controls_fixed_and_selectable_data_periods() -> None:
    """Verify fixed and flexible task types show the right period controls."""

    app = _app()

    app.selectbox(key="scheduled_task_job_type").select("Shift Handover Summary").run(
        timeout=30
    )
    assert not any(item.key == "scheduled_task_data_period" for item in app.selectbox)

    app.text_input(key="scheduled_task_name").set_value("Shift handover")
    app.text_area(key="scheduled_task_instructions").set_value(
        "Prepare the previous shift handover summary."
    )
    app.button(key="scheduled_task_create").click().run(timeout=30)
    payload = json.loads(
        app.session_state["scheduled_task_generated_definition"]["json"]
    )
    assert payload["job_type"] == "shift_report"
    assert payload["inputs"]["data_period"] == "previous_shift"

    app.selectbox(key="scheduled_task_job_type").select(
        "Furnace Performance Summary"
    ).run(timeout=30)
    assert app.selectbox(key="scheduled_task_data_period").value

    app.selectbox(key="scheduled_task_job_type").select("Custom Report").run(timeout=30)
    assert app.selectbox(key="scheduled_task_data_period").value


def test_shift_handover_example_uses_the_renamed_task_type() -> None:
    """Verify the shift example uses its renamed type and shift cadence."""

    app = _app()
    app.selectbox(key="scheduled_task_template").select("Shift handover summary")
    app.button(key="scheduled_task_use_template").click().run(timeout=30)

    assert (
        app.selectbox(key="scheduled_task_job_type").value == "Shift Handover Summary"
    )
    assert app.selectbox(key="scheduled_task_repeat").value == "After every shift"


def test_custom_schedule_is_absent_and_stale_state_resets_to_daily() -> None:
    """Verify custom scheduling is absent and stale state resets to daily."""

    sys.modules.pop("utils.session", None)
    app = AppTest.from_file("src/custom_pages/10_Scheduled_Tasks.py")
    app.session_state["auth_user"] = "operator.test"
    app.session_state["role"] = "user"
    app.session_state["scheduled_task_state_owner"] = "operator.test"
    app.session_state["scheduled_task_repeat"] = "Custom schedule"
    app.run(timeout=30)

    assert not app.exception
    repeat = app.selectbox(key="scheduled_task_repeat")
    assert "Custom schedule" not in repeat.options
    assert repeat.value == "Every day"


def test_management_view_shows_owner_job_history_and_queued_controls() -> None:
    """Verify stored jobs expose history and database-backed controls."""

    _FakeScheduledJobService.items = (_stored_job_item(),)
    _FakeScheduledJobService.history = _stored_run_history()
    app = _app()
    app.session_state["scheduled_task_page_view"] = "Your tasks"
    app.run(timeout=30)

    assert not app.exception
    assert app.selectbox(key="scheduled_task_status_filter").value == "All tasks"
    assert app.selectbox(key="scheduled_task_selected_job").value.startswith(
        "Hourly BF2 ETA CO"
    )
    pause = app.button(key=f"scheduled_task_action_{_TEST_JOB_ID}")
    archive = app.button(key=f"scheduled_task_archive_{_TEST_JOB_ID}")
    assert pause.label == "Pause"
    assert not pause.disabled
    assert not archive.disabled
    assert any("ETA CO remained" in item.value for item in app.markdown)
    assert any("scheduled-job-definition/v1" in item.value for item in app.code)


def test_management_lifecycle_actions_are_confirmed_and_allow_listed() -> None:
    """Verify pause and confirmed archive actions call the narrow control client."""

    _FakeScheduledJobService.items = (_stored_job_item(),)
    app = _app()
    app.session_state["scheduled_task_page_view"] = "Your tasks"
    app.run(timeout=30)

    app.button(key=f"scheduled_task_action_{_TEST_JOB_ID}").click().run(timeout=30)
    assert not app.exception
    assert [call["action"] for call in _FakeScheduledJobCommandService.calls] == [
        "pause"
    ]
    assert any("Pause request queued" in item.value for item in app.success)

    _FakeScheduledJobCommandService.commands = []
    app.run(timeout=30)
    app.button(key=f"scheduled_task_archive_{_TEST_JOB_ID}").click().run(timeout=30)
    assert not app.exception
    assert _FakeScheduledJobCommandService.calls[-1]["action"] != "archive"
    app.button(key=f"scheduled_task_archive_confirm_{_TEST_JOB_ID}").click().run(
        timeout=30
    )
    assert not app.exception
    assert _FakeScheduledJobCommandService.calls[-1]["action"] == "archive"


def test_clone_restores_definition_as_a_new_creation_form() -> None:
    """Verify cloning preserves stored settings without modifying the original."""

    _FakeScheduledJobService.items = (_stored_job_item(),)
    app = _app()
    app.session_state["scheduled_task_page_view"] = "Your tasks"
    app.run(timeout=30)

    app.button(key=f"scheduled_task_clone_{_TEST_JOB_ID}").click().run(timeout=30)

    assert not app.exception
    assert app.session_state["scheduled_task_page_view"] == "Create task"
    assert app.text_input(key="scheduled_task_name").value == (
        "Copy of Hourly BF2 ETA CO"
    )
    assert app.selectbox(key="scheduled_task_repeat").value == "Every hour"
    assert app.number_input(key="scheduled_task_hourly_minute").value == 5
    assert app.selectbox(key="scheduled_task_analysis_level").value == "Brief analysis"
    assert "scheduled_task_generated_definition" not in app.session_state


def test_edit_loads_existing_definition_and_queues_full_revision() -> None:
    """Verify editing keeps identity and queues a validated replacement."""

    _FakeScheduledJobService.items = (_stored_job_item(status="paused"),)
    app = _app()
    app.session_state["scheduled_task_page_view"] = "Your tasks"
    app.run(timeout=30)

    app.button(key=f"scheduled_task_edit_{_TEST_JOB_ID}").click().run(timeout=30)

    assert not app.exception
    assert app.session_state["scheduled_task_page_view"] == "Create task"
    assert app.text_input(key="scheduled_task_name").value == "Hourly BF2 ETA CO"
    assert app.button(key="scheduled_task_create").label == "Save changes"
    app.text_input(key="scheduled_task_name").set_value("Revised BF2 ETA CO")
    app.button(key="scheduled_task_create").click().run(timeout=30)

    assert not app.exception
    request = _FakeScheduledJobCommandService.calls[-1]
    assert request["action"] == "update"
    assert request["job_id"] == _TEST_JOB_ID
    assert request["definition"]["job_name"] == "Revised BF2 ETA CO"
    assert app.session_state["scheduled_task_generated_definition"]["kind"] == (
        "update"
    )


def test_page_has_distinct_lightweight_palette_without_prototype_copy() -> None:
    """Verify page copy, layout, typography, and palette follow the design."""

    source = (ROOT / "src" / "ui" / "scheduled_tasks.py").read_text(encoding="utf-8")
    css = (ROOT / "src" / "assets" / "css" / "scheduled_tasks_style.css").read_text(
        encoding="utf-8"
    )

    assert "Scheduled Tasks" in source
    assert "Create a scheduled task" in source
    assert '"Schedule",' in source
    assert "Scheduled Reports" not in source
    assert "Generate JSON" not in source
    assert "Custom schedule" not in source
    assert "Cron expression" not in source
    assert "scheduled_task_custom_" not in source
    assert "linear-gradient(" not in css
    assert "#3b5563" in css
    assert "#26343c" in css
    assert "#f7f8f7" in css
    assert "#161f24" in css
    assert "light-dark(" in css
    assert '[data-testid="stApp"]:has(.st-key-scheduled_task_shell)' in css
    assert "st.context.theme" not in source
    assert "scheduled-task-theme-" not in source
    assert "scheduled-task-theme-" not in css
    assert ".stApp:has(.st-key-scheduled_task_shell)" not in css
    assert '[data-testid="stTooltipIcon"]' in css
    assert '--schedule-font: "Aptos", "Segoe UI", sans-serif' in css
    assert "--schedule-text-control: 14px" in css
    assert ".st-key-scheduled_task_shell textarea::placeholder" in css
    assert 'key="scheduled_task_creator"' in source
    assert 'key="scheduled_task_quick_start"' not in source
    assert 'key="scheduled_task_review"' not in source
    assert 'key="scheduled_task_target"' not in source
    assert 'key="scheduled_task_target_device"' not in source
    assert "form_col, review_col" not in source

    for bmo_color in ("#0f2940", "#19507a", "#1f7a5a"):
        assert bmo_color not in css

    for prototype_phrase in (
        "JSON-only milestone",
        "READ-ONLY REQUEST",
        "Session only",
        "Not connected",
        "notifications are not active yet",
    ):
        assert prototype_phrase not in source
