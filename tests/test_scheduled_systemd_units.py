"""Tests for pure, safe scheduled-job systemd unit rendering."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timezone
from pathlib import PurePosixPath
from uuid import UUID

import pytest

from utils.scheduled_tasks.scheduled_task_definition import (
    ScheduledTaskInput,
    build_task_definition,
)
from utils.scheduled_tasks.systemd_units import (
    CONTROL_SYNC_SERVICE_UNIT_NAME,
    CONTROL_SYNC_TIMER_UNIT_NAME,
    MANAGED_UNIT_MARKER,
    SERVICE_TEMPLATE_UNIT_NAME,
    SystemdProvisioningSettings,
    build_control_sync_artifacts,
    build_provisioning_plan,
    compile_on_calendar,
    render_service_template,
    render_timer_unit,
    service_instance_name,
    timer_unit_name,
)

_JOB_ID = "12345678-1234-5678-9234-567812345678"
_NOW_UTC = datetime(2026, 9, 3, 3, 0, tzinfo=timezone.utc)


def _definition(**overrides: object) -> dict[str, object]:
    """Build a representative valid scheduled-job definition for rendering tests."""

    values: dict[str, object] = {
        "name": "BF2 operating review",
        "instructions": "Review BF2 performance and prepare an operator summary.",
        "furnace": "BF2",
        "data_period": "last_24_hours",
        "output_format": "operator_summary",
        "schedule_kind": "daily",
        "delivery_channel": "in_app",
        "job_type": "furnace_summary",
        "run_time": time(7, 0),
    }
    values.update(overrides)
    return build_task_definition(
        ScheduledTaskInput(**values),  # type: ignore[arg-type]
        generated_at=_NOW_UTC,
    )


def test_unit_names_are_derived_only_from_a_canonical_uuid() -> None:
    """Canonical job ids should produce stable timer and service-instance names."""

    assert timer_unit_name(_JOB_ID) == f"furnacemind-job-{_JOB_ID}.timer"
    assert service_instance_name(_JOB_ID) == f"furnacemind-job@{_JOB_ID}.service"
    assert timer_unit_name(UUID(_JOB_ID)) == f"furnacemind-job-{_JOB_ID}.timer"


@pytest.mark.parametrize(
    "job_id",
    [
        "12345678123456789234567812345678",
        "12345678-1234-5678-9234-567812345678.timer",
        "12345678-1234-5678-9234-567812345678/../../unsafe",
        "12345678-1234-5678-9234-567812345678\n[Service]",
        "not-a-uuid",
    ],
)
def test_unit_names_reject_noncanonical_or_unsafe_job_ids(job_id: str) -> None:
    """Untrusted identifiers must never become systemd unit basenames."""

    with pytest.raises(ValueError, match="UUID"):
        timer_unit_name(job_id)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"schedule_kind": "hourly", "run_time": None, "hourly_minute": 5},
            ("*-*-* *:05:00 Asia/Kolkata",),
        ),
        ({}, ("*-*-* 07:00:00 Asia/Kolkata",)),
        (
            {"schedule_kind": "weekdays"},
            ("Mon..Fri *-*-* 07:00:00 Asia/Kolkata",),
        ),
        (
            {
                "schedule_kind": "selected_days",
                "days_of_week": ("Sunday", "Wednesday", "Monday"),
            },
            ("Sun,Mon,Wed *-*-* 07:00:00 Asia/Kolkata",),
        ),
        (
            {"schedule_kind": "weekly", "days_of_week": ("Friday",)},
            ("Fri *-*-* 07:00:00 Asia/Kolkata",),
        ),
        (
            {"schedule_kind": "monthly", "day_of_month": 28},
            ("*-*-28 07:00:00 Asia/Kolkata",),
        ),
    ],
)
def test_generated_cron_variants_compile_to_systemd_calendar_values(
    overrides: dict[str, object],
    expected: tuple[str, ...],
) -> None:
    """Every UI-generated cron variant should preserve its wall-clock meaning."""

    assert compile_on_calendar(_definition(**overrides)) == expected


def test_shift_end_compiles_to_one_calendar_line_per_selected_run_time() -> None:
    """Shift triggers should remain explicit when multiple daily times are selected."""

    definition = _definition(
        job_type="shift_report",
        data_period="previous_shift",
        schedule_kind="shift_end",
        run_time=None,
        shift_labels=("A", "B", "C"),
        shift_delay_minutes=10,
    )

    assert compile_on_calendar(definition) == (
        "*-*-* 06:10:00 Asia/Kolkata",
        "*-*-* 14:10:00 Asia/Kolkata",
        "*-*-* 22:10:00 Asia/Kolkata",
    )


def test_one_time_schedule_preserves_the_declared_calendar_timezone() -> None:
    """One-time timers should represent their instant in the configured timezone."""

    definition = _definition(
        schedule_kind="once",
        run_date=date(2026, 9, 4),
        run_time=time(9, 30),
    )

    assert compile_on_calendar(definition) == ("2026-09-04 09:30:00 Asia/Kolkata",)


@pytest.mark.parametrize(
    ("interval_hours", "expected"),
    [
        (8, "*-*-* 03,11,19:00:00 UTC"),
        (12, "*-*-* 03,15:00:00 UTC"),
        (48, "*-*-* 03:00:00 UTC"),
        (168, "*-*-* 03:00:00 UTC"),
    ],
)
def test_interval_schedule_uses_an_anchored_utc_probe_superset(
    interval_hours: int,
    expected: str,
) -> None:
    """Interval probes should include all elapsed-time occurrences without drift."""

    definition = _definition(
        schedule_kind="interval_hours",
        run_date=date(2026, 9, 3),
        run_time=time(8, 30),
        interval_hours=interval_hours,
    )

    assert compile_on_calendar(definition) == (expected,)


def test_schedule_compiler_rejects_a_raw_cron_frequency_mismatch() -> None:
    """A schema-shaped cron value must still agree with its selected frequency."""

    definition = deepcopy(_definition())
    definition["schedule"]["trigger"]["expression"] = "0 7 * * 1-5"  # type: ignore[index]

    with pytest.raises(ValueError, match="does not match the selected frequency"):
        compile_on_calendar(definition)


def test_schedule_compiler_rejects_unvalidated_definitions_and_secrets() -> None:
    """Invalid or credential-bearing definitions must not reach unit rendering."""

    definition = deepcopy(_definition())
    definition["instructions"] = "password=hunter2"

    with pytest.raises(ValueError, match="must not contain credentials") as exc_info:
        compile_on_calendar(definition)

    assert "hunter2" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("field_name", "unsafe_value"),
    [
        ("unit_directory", "etc/systemd/system"),
        ("working_directory", "/opt/furnacemind/../unsafe"),
        ("runner_script", "/opt/furnacemind/job runner.py"),
        ("control_sync_script", "/opt/furnacemind/control sync.py"),
        ("operation_lock_directory", "/run/lock/../unsafe"),
        ("environment_file", "/etc/furnacemind/env\n[Service]"),
        ("python_executable", "/opt/%i/python"),
        ("systemctl_path", r"C:\Windows\systemctl.exe"),
    ],
)
def test_settings_reject_unsafe_or_non_posix_paths(
    field_name: str,
    unsafe_value: str,
) -> None:
    """Deployment settings should reject path traversal and directive injection."""

    with pytest.raises(ValueError, match="safe absolute POSIX path"):
        SystemdProvisioningSettings(**{field_name: unsafe_value})  # type: ignore[arg-type]


def test_settings_reject_unsafe_service_account_names() -> None:
    """Systemd account directives should accept only conservative Linux names."""

    with pytest.raises(ValueError, match="account name"):
        SystemdProvisioningSettings(service_user="root\nExecStart=/bin/sh")


@pytest.mark.parametrize("field_name", ["service_user", "service_group"])
def test_settings_reject_literal_root_service_identity(field_name: str) -> None:
    """Natural-language agent jobs must never be configured to execute as root."""

    with pytest.raises(ValueError, match="unprivileged"):
        SystemdProvisioningSettings(**{field_name: "root"})  # type: ignore[arg-type]


@pytest.mark.parametrize("interval", [4, 3601])
def test_settings_reject_unsafe_control_sync_intervals(interval: int) -> None:
    """The deployment timer interval must stay within a bounded range."""

    with pytest.raises(ValueError, match="control_sync_interval_seconds"):
        SystemdProvisioningSettings(control_sync_interval_seconds=interval)


def test_shared_service_template_has_a_stable_runner_contract_and_hardening() -> None:
    """The shared oneshot service should use safe settings and the UUID instance."""

    settings = SystemdProvisioningSettings(
        working_directory="/srv/furnacemind",
        python_executable="/srv/furnacemind/.venv/bin/python",
        runner_script="/srv/furnacemind/scripts/furnacemind_job_runner.py",
        environment_file="/etc/furnacemind/runtime.env",
        service_user="fm-agent",
        service_group="fm-agent",
    )

    content = render_service_template(settings)

    assert content.startswith(f"{MANAGED_UNIT_MARKER}\n")
    assert "Type=oneshot" in content
    assert "User=fm-agent" in content
    assert "Group=fm-agent" in content
    assert "WorkingDirectory=/srv/furnacemind" in content
    assert "EnvironmentFile=/etc/furnacemind/runtime.env" in content
    assert (
        "ExecStart=/srv/furnacemind/.venv/bin/python "
        "/srv/furnacemind/scripts/furnacemind_job_runner.py --job-id %i"
    ) in content
    assert "NoNewPrivileges=true" in content
    assert "PrivateTmp=true" in content
    assert "ProtectSystem=full" in content
    assert "ProtectHome=true" in content
    assert "CapabilityBoundingSet=" in content
    assert "KillMode=control-group" in content
    assert "Restart=on-failure" in content
    assert "RestartSec=30s" in content
    assert "RestartPreventExitStatus=2" in content
    assert "StartLimitBurst=5" in content
    assert "UMask=0077" in content
    assert "--validate-only" not in content


def test_control_sync_units_run_a_bounded_privileged_bridge() -> None:
    """The bridge should run briefly as root and never execute operator text."""

    settings = SystemdProvisioningSettings(
        working_directory="/srv/furnacemind",
        python_executable="/srv/furnacemind/.venv/bin/python",
        control_sync_script="/srv/furnacemind/scripts/furnacemind_control_sync.py",
        environment_file="/etc/furnacemind/runtime.env",
        unit_directory="/etc/systemd/system",
        operation_lock_directory="/run/lock/furnacemind-scheduled-tasks",
        control_sync_interval_seconds=45,
    )

    service, timer = build_control_sync_artifacts(settings)

    assert service.unit_name == CONTROL_SYNC_SERVICE_UNIT_NAME
    assert "Type=oneshot" in service.content
    assert "User=root" in service.content
    assert "furnacemind_control_sync.py --max-commands 25" in service.content
    assert "ProtectSystem=strict" in service.content
    assert (
        "ReadWritePaths=/etc/systemd/system /run/lock/furnacemind-scheduled-tasks"
        in service.content
    )
    assert timer.unit_name == CONTROL_SYNC_TIMER_UNIT_NAME
    assert "OnUnitInactiveSec=45s" in timer.content
    assert f"Unit={CONTROL_SYNC_SERVICE_UNIT_NAME}" in timer.content


def test_timer_unit_references_only_the_uuid_service_instance() -> None:
    """Per-job timers must not expose operator names, instructions, or secrets."""

    definition = _definition(
        name="Confidential operator title",
        instructions="Prepare the private internal operating summary.",
    )

    content = render_timer_unit(_JOB_ID, definition)

    assert "OnCalendar=*-*-* 07:00:00 Asia/Kolkata" in content
    assert f"Unit={service_instance_name(_JOB_ID)}" in content
    assert "Persistent=true" in content
    assert "AccuracySec=1s" in content
    assert "RemainAfterElapse=true" in content
    assert "Confidential operator title" not in content
    assert "private internal operating summary" not in content


def test_provisioning_plan_contains_the_shared_service_and_per_job_timer() -> None:
    """A pure plan should contain exactly two managed artifacts at trusted paths."""

    settings = SystemdProvisioningSettings(unit_directory="/run/fm-systemd")

    plan = build_provisioning_plan(_JOB_ID, _definition(), settings)

    assert plan.job_id == _JOB_ID
    assert plan.service_instance == service_instance_name(_JOB_ID)
    assert plan.timer_unit == timer_unit_name(_JOB_ID)
    assert plan.on_calendar == ("*-*-* 07:00:00 Asia/Kolkata",)
    assert [artifact.unit_name for artifact in plan.artifacts] == [
        SERVICE_TEMPLATE_UNIT_NAME,
        timer_unit_name(_JOB_ID),
    ]
    assert [artifact.destination_path for artifact in plan.artifacts] == [
        PurePosixPath("/run/fm-systemd/furnacemind-job@.service"),
        PurePosixPath(f"/run/fm-systemd/{timer_unit_name(_JOB_ID)}"),
    ]
    assert all(
        artifact.content.startswith(f"{MANAGED_UNIT_MARKER}\n")
        for artifact in plan.artifacts
    )
