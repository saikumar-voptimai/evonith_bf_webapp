"""Focused tests for scheduled-job documents and their relational facade."""

from __future__ import annotations

import copy
import json
from datetime import datetime

import pytest
from jsonschema import ValidationError

import data.db as db_module
from data.db import ScheduledJobService
from utils.scheduled_jobs import (
    LOCAL_TIMEZONE,
    LOCAL_TIMEZONE_NAME,
    archive_job,
    build_delivery_configuration,
    build_job_document,
    build_retry_configuration,
    build_schedule,
    clone_job_document,
    generate_job_id,
    normalize_email_recipients,
    pause_job,
    request_run_now,
    resume_job,
    revise_job,
    validate_cron_expression,
    validate_job_document,
    validate_required_fields,
)

NOW = LOCAL_TIMEZONE.localize(datetime(2026, 9, 9, 10, 15))


def _job(status: str = "draft") -> dict:
    return build_job_document(
        job_name="ETA CO Shift Report",
        instructions="Generate the ETA CO report and trend graph.",
        model_level="medium",
        status=status,
        schedule=build_schedule("Every Hour", execution_minute=30),
        delivery=build_delivery_configuration(),
        retry=build_retry_configuration(),
        created_by="qa",
        job_id="Job-1001",
        now=NOW,
    )


def test_job_id_format_and_initial_revision_and_statuses() -> None:
    assert generate_job_id(1001) == "Job-1001"
    assert generate_job_id(1002) == "Job-1002"
    with pytest.raises(ValueError, match="1001 or greater"):
        generate_job_id(1000)
    assert _job("draft")["revision"] == 1
    assert _job("draft")["status"] == "draft"
    active = _job("active")
    assert active["status"] == "active"
    assert active["model_level"] == "medium"
    assert "job_type" not in active
    assert "job_inputs" not in active
    assert "target_device" not in active
    assert list(active)[-1] == "retry"


def test_required_fields_are_field_specific() -> None:
    errors = validate_required_fields(
        job_name="",
        instructions="",
        frequency="",
    )
    assert errors == [
        "Job Name is required.",
        "Job Instructions are required.",
        "Frequency is required.",
    ]


@pytest.mark.parametrize("model_level", ["low", "medium", "high"])
def test_supported_model_levels(model_level: str) -> None:
    job = _job()
    job["model_level"] = model_level
    validate_job_document(job)

    job["model_level"] = "extreme"
    with pytest.raises(ValidationError):
        validate_job_document(job)


def test_schedules_use_kolkata_and_calculate_standard_next_runs() -> None:
    hourly = build_schedule("Every Hour", execution_minute=20)
    daily = build_schedule("Every Day", execution_time="11:30")
    weekly = build_schedule("Every Week", weekday="Wednesday", execution_time="11:30")
    shifts = build_schedule("Every Shift")

    assert LOCAL_TIMEZONE_NAME == "Asia/Kolkata"
    assert all(
        schedule["timezone"] == "Asia/Kolkata"
        for schedule in (hourly, daily, weekly, shifts)
    )
    assert _job()["next_run"].endswith("+05:30")
    assert shifts["shifts"]

    with pytest.raises(ValueError, match="0 to 59"):
        build_schedule("Every Hour", execution_minute=60)
    with pytest.raises(ValueError, match="Weekday is required"):
        build_schedule("Every Week", weekday="Funday", execution_time="10:00")


@pytest.mark.parametrize(
    "expression",
    ["0 6 * * 1", "*/15 * * * *", "0,30 6-18/2 * * 1-5"],
)
def test_valid_cron_expressions(expression: str) -> None:
    assert validate_cron_expression(expression)
    schedule = build_schedule("Custom Schedule", cron_expression=expression)
    assert schedule["cron"] == expression


@pytest.mark.parametrize(
    "expression",
    ["* * *", "60 * * * *", "* 24 * * *", "* * 0 * *", "*/0 * * * *"],
)
def test_invalid_cron_expressions(expression: str) -> None:
    assert not validate_cron_expression(expression)
    with pytest.raises(ValueError, match="Invalid cron expression"):
        build_schedule("Custom Schedule", cron_expression=expression)


def test_delivery_configuration_normalizes_all_supported_channels() -> None:
    assert normalize_email_recipients(
        "FIRST@example.com; second@example.com\nfirst@example.com"
    ) == ["first@example.com", "second@example.com"]

    configured = build_delivery_configuration(
        {
            "email": {
                "to": "first@example.com,second@example.com",
                "cc": "manager@example.com",
                "bcc": "",
                "subject": "Report",
                "attachment_formats": ["png", "csv", "json"],
            },
            "slack": {"id": "C012345"},
            "telegram": {"user_id": "12345", "channel_id": ""},
            "whatsapp": {
                "group_or_user_name": "BF Operations",
            },
        }
    )
    assert configured["email"]["to"] == [
        "first@example.com",
        "second@example.com",
    ]
    assert configured["email"]["cc"] == ["manager@example.com"]
    assert configured["email"]["attachment_formats"] == ["png", "csv", "json"]
    assert configured["slack"]["id"] == "C012345"
    assert configured["telegram"]["user_id"] == "12345"
    assert configured["whatsapp"] == {
        "group_or_user_name": "BF Operations"
    }
    assert build_delivery_configuration(
        {
            "whatsapp": {
                "group_name": "Legacy Group",
                "user_name": "",
                "phone_number": "",
            }
        }
    )["whatsapp"] == {"group_or_user_name": "Legacy Group"}
    assert build_delivery_configuration() == {}
    job = _job()
    job["delivery"] = configured
    validate_job_document(job)

    with pytest.raises(ValueError, match="email To recipient"):
        build_delivery_configuration(
            {"email": {"to": "", "subject": "Report"}}
        )
    with pytest.raises(ValueError, match="Slack ID"):
        build_delivery_configuration({"slack": {"id": ""}})
    with pytest.raises(ValueError, match="Telegram User ID or Channel ID"):
        build_delivery_configuration({"telegram": {}})
    with pytest.raises(ValueError, match="WhatsApp Group/User Name"):
        build_delivery_configuration({"whatsapp": {}})
    with pytest.raises(ValueError, match="Invalid email address: invalid"):
        normalize_email_recipients("valid@example.com; invalid")


def test_legacy_email_configuration_is_converted_for_delivery_agents() -> None:
    legacy = _job()
    legacy.pop("model_level")
    legacy.pop("delivery")
    legacy.pop("delivery_history")
    legacy["email"] = {
        "enabled": True,
        "recipients": ["operator@example.com"],
        "subject": "Legacy report",
        "attachment_formats": ["csv"],
    }
    legacy["email_history"] = [{"status": "sent"}]

    migrated = ScheduledJobService._job_from_database(legacy)

    assert migrated["delivery"]["email"] == {
        "to": ["operator@example.com"],
        "cc": [],
        "bcc": [],
        "subject": "Legacy report",
        "attachment_formats": ["csv"],
    }
    assert migrated["delivery_history"] == [{"status": "sent"}]
    assert migrated["model_level"] == "medium"
    assert "email" not in migrated
    assert "email_history" not in migrated
    validate_job_document(migrated)


def test_retry_defaults_and_custom_values() -> None:
    assert build_retry_configuration() == {
        "maximum_attempts": 3,
        "retry_interval_seconds": 60,
        "timeout_seconds": 300,
        "notify_on_failure": True,
    }
    assert (
        build_retry_configuration(
            maximum_attempts=5,
            retry_interval_seconds=120,
            timeout_seconds=600,
            notify_on_failure=False,
        )["maximum_attempts"]
        == 5
    )
    with pytest.raises(ValueError, match="positive integer"):
        build_retry_configuration(timeout_seconds=0)


def test_json_schema_and_sensitive_key_rejection() -> None:
    job = _job()
    validate_job_document(job)

    missing = copy.deepcopy(job)
    del missing["schedule"]
    with pytest.raises(ValidationError):
        validate_job_document(missing)

    sensitive = copy.deepcopy(job)
    sensitive["transport"] = {"apiKey": "do-not-store"}
    with pytest.raises(ValueError, match="Sensitive field is not allowed"):
        validate_job_document(sensitive)

    for removed_field in ("job_type", "job_inputs", "target_device"):
        legacy = copy.deepcopy(job)
        legacy[removed_field] = {}
        with pytest.raises(ValidationError):
            validate_job_document(legacy)

    embedded_report = copy.deepcopy(job)
    embedded_report["report_history"] = [
        {"graph": {"base64": "data:image/png;base64,AAAA"}}
    ]
    with pytest.raises(ValueError, match="metadata or file paths only"):
        validate_job_document(embedded_report)


def test_revision_history_preserves_revision_one_and_creates_revision_two() -> None:
    original = _job("active")
    original_before = copy.deepcopy(original)
    revised = revise_job(
        original,
        {
            "job_name": "Revised ETA CO Report",
            "model_level": "high",
        },
        saved_by="editor",
        now=NOW,
    )

    assert original == original_before
    assert revised["revision"] == 2
    assert revised["revision_history"][0]["revision"] == 1
    snapshot = revised["revision_history"][0]["configuration"]
    assert snapshot["job_name"] == original["job_name"]
    assert snapshot["model_level"] == "medium"
    assert revised["model_level"] == "high"
    assert "revision_history" not in snapshot


def test_clone_gets_new_identity_revision_one_and_empty_histories() -> None:
    source = request_run_now(_job("active"), actor="operator", now=NOW)
    source["run_history"].append({"status": "completed"})
    cloned = clone_job_document(
        source,
        created_by="operator",
        job_id="Job-1002",
        now=NOW,
    )

    assert cloned["job_id"] != source["job_id"]
    assert cloned["revision"] == 1
    assert cloned["status"] == "draft"
    for history in (
        "revision_history",
        "execution_requests",
        "run_history",
        "delivery_history",
        "report_history",
        "error_history",
        "action_history",
    ):
        assert cloned[history] == []


def test_pause_resume_and_run_now_do_not_change_revision_or_schedule() -> None:
    job = _job("active")
    schedule = copy.deepcopy(job["schedule"])
    paused = pause_job(job, actor="operator", now=NOW)
    resumed = resume_job(paused, actor="operator", now=NOW)
    requested = request_run_now(resumed, actor="operator", now=NOW)

    assert paused["status"] == "paused"
    assert paused["next_run"] is None
    assert resumed["status"] == "active"
    assert requested["revision"] == job["revision"]
    assert requested["schedule"] == schedule
    assert requested["execution_requests"][-1]["manual"] is True
    assert requested["execution_requests"][-1]["status"] == "pending"


def test_archive_preserves_all_histories() -> None:
    job = _job("active")
    for history in (
        "revision_history",
        "run_history",
        "delivery_history",
        "report_history",
        "error_history",
    ):
        job[history].append({"marker": history})
    before = {
        name: copy.deepcopy(job[name]) for name in job if name.endswith("history")
    }
    archived = archive_job(job, actor="operator", now=NOW)

    assert archived["status"] == "archived"
    assert archived["next_run"] is None
    for history, value in before.items():
        if history != "action_history":
            assert archived[history] == value
    assert archived["action_history"][-1]["action"] == "archive"
    with pytest.raises(ValueError, match="Archived jobs cannot be resumed"):
        resume_job(archived, actor="operator", now=NOW)


class _FakeResult:
    def __init__(self, rows=None, scalar_value=False, rowcount=1):
        self._rows = rows or []
        self._scalar = scalar_value
        self.rowcount = rowcount

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar


class _FakeEngine:
    def __init__(self, job: dict):
        self.job = job
        self.calls = []

    def begin(self):
        return self

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, params=None):
        sql = str(statement)
        self.calls.append((sql, params or {}))
        if "SELECT EXISTS" in sql:
            return _FakeResult(scalar_value=True)
        if "nextval('automation.scheduled_job_id_seq')" in sql:
            return _FakeResult(scalar_value=1001)
        if "SELECT job_description" in sql:
            return _FakeResult(
                rows=[{"job_description": json.dumps(self.job)}]
            )
        return _FakeResult()

    def dispose(self):
        pass


def test_scheduled_job_service_uses_bound_ordered_json_sql_without_postgres(
    monkeypatch,
) -> None:
    job = _job()
    retry_first_job = {
        "retry": job["retry"],
        **{key: value for key, value in job.items() if key != "retry"},
    }
    assert list(
        json.loads(ScheduledJobService._serialized_job(retry_first_job))
    )[-1] == "retry"
    legacy_job = {
        **job,
        "job_type": "ETA CO Report",
        "job_inputs": {"signal": "body_etaco"},
        "target_device": {"device_id": "bf2-pi-01"},
        "email": {
            "enabled": False,
            "recipients": [],
            "subject": "",
            "attachment_formats": [],
        },
        "email_history": [],
    }
    legacy_job.pop("delivery")
    legacy_job.pop("delivery_history")
    engine = _FakeEngine(legacy_job)
    monkeypatch.setattr(
        db_module, "build_relational_engine", lambda db_url=None: engine
    )
    monkeypatch.setattr(
        db_module,
        "build_relational_session_factory",
        lambda relational_engine: object(),
    )
    service = ScheduledJobService(db_url="postgresql://example")

    assert service.reserve_job_id() == "Job-1001"
    service.create_job(job)
    assert service.list_jobs() == [job]
    assert service.get_job(job["job_id"]) == job
    assert service.job_name_exists("name' OR TRUE --") is True
    service.update_job(job["job_id"], job)
    service.delete_job(job["job_id"])

    all_sql = "\n".join(sql for sql, _ in engine.calls)
    assert "automation.scheduled_jobs" in all_sql
    assert "offline_feed.scheduled_jobs" not in all_sql
    assert "ORDER BY date_time DESC" in all_sql
    assert "CREATE SEQUENCE IF NOT EXISTS automation.scheduled_job_id_seq" in all_sql
    assert "DELETE FROM automation.scheduled_jobs" in all_sql
    assert "scheduled-jobs" not in all_sql
    assert "name' OR TRUE --" not in all_sql
    assert any(
        params.get("job_name") == "name' OR TRUE --" for _, params in engine.calls
    )
    assert any(
        "CAST(:job_description AS JSON)" in sql
        and isinstance(params.get("job_description"), str)
        and list(json.loads(params["job_description"]))[-1] == "retry"
        for sql, params in engine.calls
    )
    assert any(
        "DELETE FROM automation.scheduled_jobs" in sql
        and params.get("job_id") == job["job_id"]
        for sql, params in engine.calls
    )
