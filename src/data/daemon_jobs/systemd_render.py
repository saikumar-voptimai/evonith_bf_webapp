"""Render systemd preview text for daemon job definitions."""

from __future__ import annotations

from typing import Any

from .models import DaemonJobScheduleType


def render_service_unit(job: Any) -> str:
    """Render a systemd service unit preview for one daemon job."""
    unit_name = _job_value(job, "systemd_unit_name")
    schedule_type = _job_value(job, "schedule_type")
    restart_policy = _job_value(job, "restart_policy")

    lines = [
        "[Unit]",
        f"Description=Evonith Agent Job - {_systemd_text(_job_value(job, 'name'))}",
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=oneshot",
        f"WorkingDirectory={_job_value(job, 'working_directory')}",
        f"EnvironmentFile={_job_value(job, 'env_file')}",
        (
            f"ExecStart={_job_value(job, 'python_executable')} "
            f"-m {_job_value(job, 'module_path')} --job-id {_job_value(job, 'id')}"
        ),
        f"User={_job_value(job, 'user_name')}",
        f"Group={_job_value(job, 'group_name')}",
        f"TimeoutStartSec={_job_value(job, 'timeout_sec')}",
        f"RuntimeMaxSec={_job_value(job, 'max_runtime_sec')}",
    ]

    if (
        schedule_type == DaemonJobScheduleType.MANUAL_ONLY.value
        and restart_policy == "on-failure"
    ):
        lines.extend(
            [
                "Restart=on-failure",
                f"RestartSec={_job_value(job, 'restart_sec')}",
            ]
        )

    lines.extend(
        [
            "",
            "[Install]",
            "WantedBy=multi-user.target",
        ]
    )
    return "\n".join(lines)


def render_timer_unit(job: Any) -> str:
    """Render a systemd timer unit preview when the job uses systemd_timer."""
    if _job_value(job, "schedule_type") != DaemonJobScheduleType.SYSTEMD_TIMER.value:
        return ""

    return "\n".join(
        [
            "[Unit]",
            f"Description=Timer for Evonith Agent Job - {_systemd_text(_job_value(job, 'name'))}",
            "",
            "[Timer]",
            f"OnCalendar={_job_value(job, 'on_calendar')}",
            "Persistent=true",
            f"Unit={_job_value(job, 'systemd_unit_name')}.service",
            "",
            "[Install]",
            "WantedBy=timers.target",
        ]
    )


def render_install_commands(job: Any) -> str:
    """Render Pi-side install commands as preview text only."""
    unit_name = _job_value(job, "systemd_unit_name")
    schedule_type = _job_value(job, "schedule_type")
    enabled = bool(_job_value(job, "enabled"))

    commands = [
        "# Step 1 preview only. Do not run these from the dashboard.",
        f"sudo install -m 0644 {unit_name}.service /etc/systemd/system/{unit_name}.service",
    ]

    if schedule_type == DaemonJobScheduleType.SYSTEMD_TIMER.value:
        commands.extend(
            [
                f"sudo install -m 0644 {unit_name}.timer /etc/systemd/system/{unit_name}.timer",
                "sudo systemctl daemon-reload",
            ]
        )
        if enabled:
            commands.append(f"sudo systemctl enable --now {unit_name}.timer")
        else:
            commands.append(f"sudo systemctl disable --now {unit_name}.timer")
    elif schedule_type == DaemonJobScheduleType.CRON_EXPRESSION.value:
        commands.extend(
            [
                "sudo systemctl daemon-reload",
                "# cron_expression is stored for compatibility.",
                "# Step 2 should translate it to systemd OnCalendar or runner scheduling.",
            ]
        )
    else:
        commands.extend(
            [
                "sudo systemctl daemon-reload",
                f"# Manual-only job. Step 2 may run: sudo systemctl start {unit_name}.service",
            ]
        )

    return "\n".join(commands)


def render_uninstall_commands(job: Any) -> str:
    """Render Pi-side uninstall commands as preview text only."""
    unit_name = _job_value(job, "systemd_unit_name")
    schedule_type = _job_value(job, "schedule_type")

    commands = ["# Step 1 preview only. Do not run these from the dashboard."]
    if schedule_type == DaemonJobScheduleType.SYSTEMD_TIMER.value:
        commands.append(f"sudo systemctl disable --now {unit_name}.timer")
        commands.append(f"sudo rm -f /etc/systemd/system/{unit_name}.timer")
    commands.extend(
        [
            f"sudo systemctl stop {unit_name}.service",
            f"sudo rm -f /etc/systemd/system/{unit_name}.service",
            "sudo systemctl daemon-reload",
            "sudo systemctl reset-failed",
        ]
    )
    return "\n".join(commands)


def _job_value(job: Any, field: str) -> Any:
    """Read a field from an ORM object, Pydantic model, or dict."""
    if isinstance(job, dict):
        return job[field]
    return getattr(job, field)


def _systemd_text(value: Any) -> str:
    """Sanitize descriptive text for unit metadata lines."""
    return str(value).replace("\n", " ").replace("\r", " ").strip()
