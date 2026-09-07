"""Tests for the JSON-only scheduled-job definition builder."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timezone

import pytest

from utils.scheduled_tasks.scheduled_task_definition import (
    POLICY_PROFILE,
    SCHEMA_VERSION,
    ScheduledTaskInput,
    ScheduledTaskValidationError,
    build_task_definition,
    describe_schedule,
    parse_email_recipients,
    task_definition_filename,
    task_definition_json,
    upcoming_run_times,
    validate_task_definition,
    validate_task_input,
)

NOW_UTC = datetime(2026, 9, 3, 3, 0, tzinfo=timezone.utc)  # 08:30 IST


def _credential_fixture(*parts: str) -> str:
    """Build synthetic credential text without storing scanner-shaped literals."""

    return "".join(parts)


def _task(**overrides: object) -> ScheduledTaskInput:
    """Return a baseline scheduled-task input with supplied overrides."""

    values: dict[str, object] = {
        "name": "Daily BF2 furnace health report",
        "instructions": "Review BF2 data and prepare a concise furnace health report.",
        "furnace": "BF2",
        "data_period": "previous_day",
        "output_format": "operator_summary",
        "schedule_kind": "daily",
        "delivery_channel": "in_app",
        "job_type": "daily_report",
        "analysis_level": "low",
        "data_source": "online_process_data",
        "target_device_id": "bf2-jetson-01",
        "target_device_type": "jetson",
        "run_time": time(7, 0),
    }
    values.update(overrides)
    return ScheduledTaskInput(**values)  # type: ignore[arg-type]


def _eta_task(**overrides: object) -> ScheduledTaskInput:
    """Return an ETA CO task input with supplied overrides."""

    values: dict[str, object] = {
        "name": "Hourly ETA CO report",
        "instructions": "Review the BF2 body ETA CO trend and explain limit breaches.",
        "job_type": "eta_co_report",
        "output_format": "detailed_report",
        "eta_co_signal": "body_etaco",
        "report_duration_minutes": 60,
        "aggregation_interval": "5min",
        "warning_threshold": 42.0,
        "critical_threshold": 40.0,
        "include_graph": True,
        "include_ai_summary": True,
        "analysis_level": "medium",
    }
    values.update(overrides)
    return _task(**values)


def test_builds_generic_definition_with_nested_schedule_and_policy() -> None:
    """Verify generic definitions contain the expected nested contract."""

    definition = build_task_definition(_task(), generated_at=NOW_UTC)

    assert definition["schema_version"] == SCHEMA_VERSION
    assert definition["job_type"] == "daily_report"
    assert definition["analysis_level"] == "low"
    assert definition["schedule"] == {
        "frequency": "daily",
        "timezone": "Asia/Kolkata",
        "overlap_policy": "skip",
        "misfire_policy": "fire_once_latest",
        "trigger": {
            "type": "cron",
            "expression": "0 7 * * *",
        },
    }
    assert definition["target_device"] == {
        "device_id": "bf2-jetson-01",
        "device_type": "jetson",
    }
    assert definition["inputs"] == {
        "furnace": "BF2",
        "data_source": "online_process_data",
        "output_format": "operator_summary",
        "data_period": "previous_day",
        "data_period_anchor": "scheduled_for",
    }
    assert definition["retry"] == {
        "maximum_attempts": 3,
        "retry_interval_seconds": 60,
        "timeout_seconds": 600,
    }
    assert definition["policy_profile"] == POLICY_PROFILE
    assert "metadata" not in definition
    assert not ({"job_id", "status", "revision", "enabled"} & definition.keys())
    assert validate_task_definition(definition) == ()


def test_non_object_definition_returns_validation_errors_without_crashing() -> None:
    """Corrupt stored JSON should fail validation before dict-only semantics."""

    errors = validate_task_definition(["malformed"])

    assert errors
    assert errors[0].startswith("$:")


def test_builds_eta_co_definition_with_job_specific_inputs() -> None:
    """Verify ETA CO definitions contain their job-specific inputs."""

    definition = build_task_definition(_eta_task(), generated_at=NOW_UTC)

    assert definition["inputs"] == {
        "furnace": "BF2",
        "data_source": "online_process_data",
        "output_format": "detailed_report",
        "signal": "body_etaco",
        "report_duration_minutes": 60,
        "aggregation_interval": "5min",
        "warning_threshold": 42.0,
        "critical_threshold": 40.0,
        "include_graph": True,
        "include_ai_summary": True,
    }
    assert "data_period" not in definition["inputs"]
    assert definition["analysis_level"] == "medium"
    assert validate_task_definition(definition) == ()


def test_eta_ai_summary_requires_an_analysis_level_in_the_json_contract() -> None:
    """Verify ETA AI summaries require an active analysis level."""

    definition = deepcopy(build_task_definition(_eta_task(), generated_at=NOW_UTC))
    definition["analysis_level"] = "none"

    errors = validate_task_definition(definition)

    assert any(error.startswith("analysis_level:") for error in errors)


def test_definition_validator_enforces_eta_cross_field_rules() -> None:
    """Verify ETA threshold and aggregation cross-field validation."""

    definition = deepcopy(build_task_definition(_eta_task(), generated_at=NOW_UTC))
    definition["inputs"]["critical_threshold"] = 42.0  # type: ignore[index]
    definition["inputs"]["report_duration_minutes"] = 30  # type: ignore[index]
    definition["inputs"]["aggregation_interval"] = "1h"  # type: ignore[index]

    errors = validate_task_definition(definition)

    assert "inputs.critical_threshold: must be lower than warning_threshold" in errors
    assert (
        "inputs.aggregation_interval: cannot exceed report_duration_minutes" in errors
    )


def test_selected_days_are_compiled_inside_the_schedule_trigger() -> None:
    """Verify selected weekdays compile into sorted cron days and a summary."""

    task = _task(
        schedule_kind="selected_days",
        days_of_week=("Sunday", "Wednesday", "Monday"),
    )
    definition = build_task_definition(task, generated_at=NOW_UTC)

    schedule = definition["schedule"]
    assert schedule["frequency"] == "selected_days"  # type: ignore[index]
    assert schedule["trigger"]["expression"] == "0 7 * * 0,1,3"  # type: ignore[index]
    assert describe_schedule(task) == "Every Monday, Wednesday, Sunday at 07:00 IST"


@pytest.mark.parametrize(
    ("overrides", "expected_expression"),
    [
        ({"schedule_kind": "weekdays"}, "0 7 * * 1-5"),
        (
            {"schedule_kind": "weekly", "days_of_week": ("Friday",)},
            "0 7 * * 5",
        ),
        ({"schedule_kind": "monthly", "day_of_month": 28}, "0 7 28 * *"),
    ],
)
def test_calendar_schedule_variants_build_expected_cron(
    overrides: dict[str, object], expected_expression: str
) -> None:
    """Verify calendar schedule variants produce the expected cron."""

    definition = build_task_definition(_task(**overrides), generated_at=NOW_UTC)

    assert definition["schedule"]["trigger"]["expression"] == (  # type: ignore[index]
        expected_expression
    )


def test_shift_end_definition_keeps_its_derivation_nested_in_trigger() -> None:
    """Verify shift-end derivation stays nested in the schedule trigger."""

    definition = build_task_definition(
        _task(
            job_type="shift_report",
            schedule_kind="shift_end",
            data_period="previous_shift",
            run_time=None,
            shift_labels=("A", "B", "C"),
            shift_delay_minutes=10,
        ),
        generated_at=NOW_UTC,
    )

    trigger = definition["schedule"]["trigger"]  # type: ignore[index]
    assert trigger == {
        "type": "cron",
        "expression": "10 6,14,22 * * *",
        "derived_from": {
            "type": "plant_shift_end",
            "shifts": [
                {"label": "A", "end_time": "14:00"},
                {"label": "B", "end_time": "22:00"},
                {"label": "C", "end_time": "06:00"},
            ],
            "delay_minutes": 10,
        },
    }

    without_derivation = deepcopy(definition)
    del without_derivation["schedule"]["trigger"]["derived_from"]  # type: ignore[index]
    assert any(
        error.startswith("schedule.trigger:")
        for error in validate_task_definition(without_derivation)
    )


def test_hourly_schedule_builds_cron_and_upcoming_preview() -> None:
    """Verify hourly schedules produce a cron trigger and run preview."""

    task = _task(schedule_kind="hourly", run_time=None, hourly_minute=5)

    definition = build_task_definition(task, generated_at=NOW_UTC)
    runs = upcoming_run_times(task, count=3, now=NOW_UTC)

    assert describe_schedule(task) == "Every hour at minute 05 IST"
    assert definition["schedule"]["trigger"] == {  # type: ignore[index]
        "type": "cron",
        "expression": "5 * * * *",
    }
    assert [run.isoformat(timespec="minutes") for run in runs] == [
        "2026-09-03T09:05+05:30",
        "2026-09-03T10:05+05:30",
        "2026-09-03T11:05+05:30",
    ]


def test_upcoming_preview_honors_the_requested_count() -> None:
    """Verify schedule previews have no hidden search-window truncation."""

    runs = upcoming_run_times(_task(), count=3701, now=NOW_UTC)

    assert len(runs) == 3701


def test_interval_trigger_and_preview_exclude_an_occurrence_equal_to_now() -> None:
    """Verify interval previews exclude an occurrence equal to now."""

    task = _task(
        schedule_kind="interval_hours",
        run_date=date(2026, 9, 3),
        run_time=time(8, 30),
        interval_hours=8,
    )

    definition = build_task_definition(task, generated_at=NOW_UTC)
    runs = upcoming_run_times(task, count=2, now=NOW_UTC)

    assert definition["schedule"]["trigger"] == {  # type: ignore[index]
        "type": "interval",
        "interval_seconds": 28_800,
        "anchor_at": "2026-09-03T08:30+05:30",
    }
    assert [run.isoformat(timespec="minutes") for run in runs] == [
        "2026-09-03T16:30+05:30",
        "2026-09-04T00:30+05:30",
    ]


def test_one_time_trigger_preserves_the_plant_timezone_offset() -> None:
    """Verify one-time triggers retain the plant timezone offset."""

    definition = build_task_definition(
        _task(
            schedule_kind="once",
            run_date=date(2026, 9, 4),
            run_time=time(9, 30),
        ),
        generated_at=NOW_UTC,
    )

    assert definition["schedule"]["trigger"] == {  # type: ignore[index]
        "type": "once",
        "run_at": "2026-09-04T09:30+05:30",
    }


def test_custom_schedule_is_not_supported_in_milestone_one() -> None:
    """Verify input and build validation reject custom schedules."""

    task = _task(schedule_kind="custom")

    assert validate_task_input(task, now=NOW_UTC) == (
        "Choose a supported repeat option.",
    )
    with pytest.raises(ScheduledTaskValidationError):
        build_task_definition(task, generated_at=NOW_UTC)


def test_schema_rejects_custom_schedule_frequency() -> None:
    """Verify the schema rejects the unsupported custom frequency."""

    definition = deepcopy(build_task_definition(_task(), generated_at=NOW_UTC))
    definition["schedule"]["frequency"] = "custom"  # type: ignore[index]

    errors = validate_task_definition(definition)

    assert any(error.startswith("schedule.frequency:") for error in errors)


def test_timezone_must_be_selected_from_the_configured_iana_zones() -> None:
    """Verify tasks allow only configured IANA timezones."""

    errors = validate_task_input(
        _task(timezone_name="Europe/London"),
        now=NOW_UTC,
    )

    assert errors == ("Choose a configured timezone.",)

    definition = deepcopy(build_task_definition(_task(), generated_at=NOW_UTC))
    definition["schedule"]["timezone"] = "Europe/London"  # type: ignore[index]
    assert "schedule.timezone: timezone is not configured" in (
        validate_task_definition(definition)
    )


def test_email_delivery_contains_recipients_subject_and_attachments() -> None:
    """Verify email definitions contain normalized delivery details."""

    definition = build_task_definition(
        _eta_task(
            delivery_channel="email",
            email_recipients=("OPERATOR@EXAMPLE.COM", "supervisor@example.com"),
            email_subject="  BF2 ETA CO report  ",
            email_attachments=("png", "csv", "json"),
            notify_on_failure=False,
        ),
        generated_at=NOW_UTC,
    )

    assert definition["delivery"] == {
        "channel": "email",
        "notify_on_failure": False,
        "recipients": ["operator@example.com", "supervisor@example.com"],
        "subject": "BF2 ETA CO report",
        "attachments": ["png", "csv", "json"],
    }


def test_definition_validator_rejects_case_insensitive_email_duplicates() -> None:
    """Verify definition validation rejects equivalent email addresses."""

    definition = deepcopy(
        build_task_definition(
            _task(
                delivery_channel="email",
                email_recipients=("lead@example.com",),
                email_subject="BF2 report",
            ),
            generated_at=NOW_UTC,
        )
    )
    definition["delivery"]["recipients"] = [  # type: ignore[index]
        "lead@example.com",
        "LEAD@example.com",
    ]

    assert (
        "delivery.recipients: duplicate email recipients are not allowed"
        in validate_task_definition(definition)
    )


@pytest.mark.parametrize(
    ("channel", "destination", "expected_destination"),
    [
        ("whatsapp", "+91 98765 43210", "+919876543210"),
        ("telegram", "@bf2_operator", "@bf2_operator"),
        ("telegram", "-123456789", "-123456789"),
    ],
)
def test_message_delivery_destinations_are_validated_and_normalized(
    channel: str, destination: str, expected_destination: str
) -> None:
    """Verify message destinations are validated and normalized."""

    definition = build_task_definition(
        _task(delivery_channel=channel, delivery_destination=destination),
        generated_at=NOW_UTC,
    )

    assert definition["delivery"]["destination"] == expected_destination  # type: ignore[index]


def test_rejects_malformed_message_destinations() -> None:
    """Verify malformed destinations produce channel-specific errors."""

    whatsapp_errors = validate_task_input(
        _task(delivery_channel="whatsapp", delivery_destination="abcdefgh12345678"),
        now=NOW_UTC,
    )
    telegram_errors = validate_task_input(
        _task(delivery_channel="telegram", delivery_destination="not valid!"),
        now=NOW_UTC,
    )

    assert any("valid WhatsApp" in error for error in whatsapp_errors)
    assert any("valid Telegram" in error for error in telegram_errors)


def test_email_recipient_list_parses_separators_and_rejects_duplicates() -> None:
    """Verify recipient parsing handles separators and detects duplicates."""

    recipients = parse_email_recipients(
        "lead@example.com; TEAM@example.com\nteam@example.com"
    )

    assert recipients == (
        "lead@example.com",
        "TEAM@example.com",
        "team@example.com",
    )
    errors = validate_task_input(
        _task(
            delivery_channel="email",
            email_recipients=recipients,
            email_subject="BF2 report",
        ),
        now=NOW_UTC,
    )
    assert "Remove duplicate email recipients." in errors


def test_email_recipient_length_gets_a_field_level_validation_error() -> None:
    """Verify overlong email recipients produce a field-level error."""

    oversized_recipient = f"{'a' * 245}@example.com"

    errors = validate_task_input(
        _task(
            delivery_channel="email",
            email_recipients=(oversized_recipient,),
            email_subject="BF2 report",
        ),
        now=NOW_UTC,
    )

    assert "Keep each email recipient to 254 characters or fewer." in errors


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        (
            {"critical_threshold": 42.0, "warning_threshold": 42.0},
            "ETA CO critical threshold must be lower than the warning threshold.",
        ),
        (
            {"critical_threshold": -1.0},
            "Keep ETA CO thresholds between 0 and 100 percent.",
        ),
        (
            {"report_duration_minutes": 30, "aggregation_interval": "1h"},
            "The aggregation interval cannot exceed the data window.",
        ),
        (
            {"aggregation_interval": "2min"},
            "Choose a configured ETA CO aggregation interval.",
        ),
    ],
)
def test_rejects_invalid_eta_threshold_and_aggregation_settings(
    overrides: dict[str, object], expected_error: str
) -> None:
    """Verify invalid ETA thresholds and aggregation settings are rejected."""

    errors = validate_task_input(_eta_task(**overrides), now=NOW_UTC)

    assert expected_error in errors


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        (
            {"target_device_id": "unknown-device"},
            "Choose a configured target device.",
        ),
        (
            {"target_device_type": "vm"},
            "The target device type does not match its configured device ID.",
        ),
    ],
)
def test_rejects_unknown_or_mismatched_target_devices(
    overrides: dict[str, object], expected_error: str
) -> None:
    """Verify unknown devices and mismatched device types are rejected."""

    assert expected_error in validate_task_input(_task(**overrides), now=NOW_UTC)


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"maximum_attempts": 0}, "Choose 1 to 10 maximum attempts."),
        (
            {"retry_interval_seconds": 3601},
            "Choose a retry interval from 1 to 3,600 seconds.",
        ),
        (
            {"timeout_seconds": 59},
            "Choose a timeout from 60 to 86,400 seconds.",
        ),
    ],
)
def test_rejects_invalid_retry_settings(
    overrides: dict[str, object], expected_error: str
) -> None:
    """Verify retry settings remain within supported bounds."""

    assert expected_error in validate_task_input(_task(**overrides), now=NOW_UTC)


def test_rejects_invalid_email_and_incompatible_graph_attachment() -> None:
    """Verify invalid email settings and incompatible PNG attachments fail."""

    task = _eta_task(
        include_graph=False,
        delivery_channel="email",
        email_recipients=("not-an-email",),
        email_subject="ETA CO",
        email_attachments=("png",),
    )

    errors = validate_task_input(task, now=NOW_UTC)

    assert "Enter valid email recipients separated by commas." in errors
    assert "Turn on the ETA CO trend graph before attaching a PNG graph." in errors
    with pytest.raises(ScheduledTaskValidationError):
        build_task_definition(task, generated_at=NOW_UTC)


@pytest.mark.parametrize(
    "secret_text",
    [
        _credential_fixture("api_", "key=", "synthetic-value"),
        _credential_fixture("database cred", "entials: ", "synthetic-value"),
        _credential_fixture("Authorization: Bear", "er ", "synthetic-value-123"),
        _credential_fixture("AWS key AK", "IA", "ABCDEFGHIJKLMNOP"),
        _credential_fixture("-----BEGIN PRIV", "ATE KEY-----"),
    ],
)
def test_rejects_credentials_before_serialization(secret_text: str) -> None:
    """Verify credential-like instructions are rejected before serialization."""

    task = _task(instructions=f"Prepare the report using {secret_text}")

    errors = validate_task_input(task, now=NOW_UTC)

    assert any("must not contain secrets" in error for error in errors)
    with pytest.raises(ScheduledTaskValidationError):
        build_task_definition(task, generated_at=NOW_UTC)


def test_json_schema_reports_path_aware_contract_errors() -> None:
    """Verify schema errors identify invalid and unexpected field paths."""

    definition = deepcopy(build_task_definition(_task(), generated_at=NOW_UTC))
    definition["target_device"]["device_type"] = "laptop"  # type: ignore[index]
    definition["unexpected_runtime_state"] = "active"

    errors = validate_task_definition(definition)

    assert any(error.startswith("target_device.device_type:") for error in errors)
    assert any("unexpected_runtime_state" in error for error in errors)


def test_schema_accepts_future_configured_common_job_type_identifiers() -> None:
    """Verify the schema accepts future valid common task-type identifiers."""

    definition = deepcopy(build_task_definition(_task(), generated_at=NOW_UTC))
    definition["job_type"] = "future_common_report"

    assert validate_task_definition(definition) == ()


def test_definition_validator_rejects_embedded_credentials_without_echoing() -> None:
    """Verify embedded credentials are rejected without exposing values."""

    definition = deepcopy(build_task_definition(_task(), generated_at=NOW_UTC))
    credential_value = _credential_fixture("to", "ken=", "do-not-echo-this")
    definition["target_device"]["device_id"] = (  # type: ignore[index]
        credential_value
    )

    errors = validate_task_definition(definition)

    assert errors == ("$: scheduled tasks must not contain credentials",)
    assert "do-not-echo-this" not in " ".join(errors)


def test_definition_validator_rejects_invalid_and_mismatched_cron_semantics() -> None:
    """Verify invalid and frequency-mismatched cron values are rejected."""

    invalid_clock = deepcopy(build_task_definition(_task(), generated_at=NOW_UTC))
    invalid_clock["schedule"]["trigger"]["expression"] = (  # type: ignore[index]
        "99 99 * * *"
    )
    wrong_weekday = deepcopy(
        build_task_definition(
            _task(schedule_kind="weekdays"),
            generated_at=NOW_UTC,
        )
    )
    wrong_weekday["schedule"]["trigger"]["expression"] = (  # type: ignore[index]
        "0 7 * * *"
    )

    expected_error = (
        "schedule.trigger.expression: cron expression does not match the "
        "selected frequency"
    )
    assert expected_error in validate_task_definition(invalid_clock)
    assert expected_error in validate_task_definition(wrong_weekday)


def test_hidden_delivery_values_do_not_block_or_enter_in_app_json() -> None:
    """Verify stale delivery fields are omitted from in-app definitions."""

    definition = build_task_definition(
        _task(
            delivery_channel="in_app",
            email_subject=_credential_fixture("api_", "key=", "hidden-stale-value"),
            delivery_destination=_credential_fixture(
                "to", "ken=", "another-hidden-value"
            ),
        ),
        generated_at=NOW_UTC,
    )

    serialized = task_definition_json(definition)
    assert "hidden-stale-value" not in serialized
    assert "another-hidden-value" not in serialized


def test_custom_lookback_is_preserved_for_non_eta_jobs() -> None:
    """Verify non-ETA tasks preserve custom lookback settings."""

    definition = build_task_definition(
        _task(
            job_type="furnace_summary",
            data_period="custom_lookback",
            custom_lookback_value=12,
            custom_lookback_unit="hours",
        ),
        generated_at=NOW_UTC,
    )

    assert definition["inputs"]["lookback"] == {  # type: ignore[index]
        "value": 12,
        "unit": "hours",
    }


@pytest.mark.parametrize(
    ("job_type", "wrong_period", "expected_message"),
    [
        (
            "shift_report",
            "previous_day",
            "Shift Handover Summary always uses previous completed shift.",
        ),
        (
            "daily_report",
            "previous_shift",
            "Daily Furnace Summary always uses previous calendar day.",
        ),
    ],
)
def test_fixed_scope_task_types_reject_mismatched_data_periods(
    job_type: str,
    wrong_period: str,
    expected_message: str,
) -> None:
    """Verify fixed-scope task types reject mismatched data periods."""

    errors = validate_task_input(
        _task(job_type=job_type, data_period=wrong_period),
        now=NOW_UTC,
    )

    assert expected_message in errors


def test_definition_validation_rejects_a_mismatched_fixed_data_period() -> None:
    """Verify definitions enforce the fixed daily-report data period."""

    definition = deepcopy(build_task_definition(_task(), generated_at=NOW_UTC))
    definition["inputs"]["data_period"] = "current_shift"  # type: ignore[index]

    assert (
        "inputs.data_period: must be previous_day for daily_report"
        in validate_task_definition(definition)
    )


def test_generated_timestamp_must_be_timezone_aware() -> None:
    """Verify the definition validation clock includes timezone information."""

    with pytest.raises(ValueError, match="must include a timezone"):
        build_task_definition(_task(), generated_at=datetime(2026, 9, 3, 3, 0))


def test_json_and_filename_are_download_safe() -> None:
    """Verify serialized JSON and filenames are safe for download."""

    definition = build_task_definition(_task(), generated_at=NOW_UTC)

    assert task_definition_json(definition).endswith("}\n")
    assert task_definition_filename("  BF2 / Daily: Health?  ") == (
        "bf2-daily-health.scheduled-job.json"
    )
    assert task_definition_filename("////") == "scheduled-job.scheduled-job.json"
