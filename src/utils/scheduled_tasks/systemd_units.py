"""Render safe systemd units for database-backed scheduled jobs.

This module is the pure boundary between a validated scheduled-job definition and
the Linux unit files used to trigger it.  It performs no file, database, or
``systemctl`` operations.  Unit names are derived only from canonical UUIDs, unit
contents contain no operator-entered text, and all configurable paths are checked
as trusted absolute POSIX paths before they enter a systemd directive.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from uuid import UUID
from zoneinfo import ZoneInfo

from utils.scheduled_tasks.scheduled_task_definition import validate_task_definition

MANAGED_UNIT_MARKER = "# Managed by FurnaceMind scheduled tasks. Do not edit."
SERVICE_TEMPLATE_UNIT_NAME = "furnacemind-job@.service"
CONTROL_SYNC_SERVICE_UNIT_NAME = "furnacemind-control-sync.service"
CONTROL_SYNC_TIMER_UNIT_NAME = "furnacemind-control-sync.timer"

_SAFE_ACCOUNT_PATTERN = re.compile(r"^[a-z_][a-z0-9_-]{0,30}\$?$")
_SAFE_UNIT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]+$")
_CRON_WEEKDAY_NAMES = {
    0: "Sun",
    1: "Mon",
    2: "Tue",
    3: "Wed",
    4: "Thu",
    5: "Fri",
    6: "Sat",
}


def _validated_absolute_path(value: object, *, field_name: str) -> PurePosixPath:
    """Return a safe absolute POSIX path for a trusted deployment setting."""

    if isinstance(value, PurePosixPath):
        raw_value = str(value)
    elif isinstance(value, str):
        raw_value = value
    else:
        raise TypeError(f"{field_name} must be a POSIX path string.")

    if (
        not raw_value
        or "\\" in raw_value
        or any(character.isspace() for character in raw_value)
        or any(character in raw_value for character in ("\x00", '"', "'", "%", "$"))
    ):
        raise ValueError(f"{field_name} must be a safe absolute POSIX path.")

    path = PurePosixPath(raw_value)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise ValueError(f"{field_name} must be a safe absolute POSIX path.")
    return path


def _validated_account_name(value: object, *, field_name: str) -> str:
    """Return a conservative, explicitly non-root Linux account name."""

    if not isinstance(value, str) or not _SAFE_ACCOUNT_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a safe Linux account name.")
    if value == "root":
        raise ValueError(f"{field_name} must name a dedicated unprivileged account.")
    return value


def _canonical_job_id(job_id: object) -> str:
    """Return a canonical lowercase UUID or reject an ambiguous unit identifier."""

    if isinstance(job_id, UUID):
        return str(job_id)
    if not isinstance(job_id, str):
        raise TypeError("job_id must be a canonical UUID string.")
    try:
        parsed = UUID(job_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("job_id must be a canonical UUID string.") from exc
    canonical = str(parsed)
    if job_id != canonical:
        raise ValueError("job_id must use canonical lowercase UUID formatting.")
    return canonical


@dataclass(frozen=True, slots=True)
class SystemdProvisioningSettings:
    """Trusted Linux paths and service identity used by unit rendering."""

    unit_directory: PurePosixPath | str = PurePosixPath("/etc/systemd/system")
    working_directory: PurePosixPath | str = PurePosixPath("/opt/furnacemind")
    python_executable: PurePosixPath | str = PurePosixPath(
        "/opt/furnacemind/.venv/bin/python"
    )
    runner_script: PurePosixPath | str = PurePosixPath(
        "/opt/furnacemind/scripts/furnacemind_job_runner.py"
    )
    control_sync_script: PurePosixPath | str = PurePosixPath(
        "/opt/furnacemind/scripts/furnacemind_control_sync.py"
    )
    operation_lock_directory: PurePosixPath | str = PurePosixPath(
        "/run/lock/furnacemind-scheduled-tasks"
    )
    environment_file: PurePosixPath | str = PurePosixPath(
        "/etc/furnacemind/furnacemind.env"
    )
    systemctl_path: PurePosixPath | str = PurePosixPath("/usr/bin/systemctl")
    systemd_analyze_path: PurePosixPath | str = PurePosixPath(
        "/usr/bin/systemd-analyze"
    )
    service_user: str = "furnacemind"
    service_group: str = "furnacemind"
    control_sync_interval_seconds: int = 30

    def __post_init__(self) -> None:
        """Normalize and validate every setting before it enters a unit or command."""

        for field_name in (
            "unit_directory",
            "working_directory",
            "python_executable",
            "runner_script",
            "control_sync_script",
            "operation_lock_directory",
            "environment_file",
            "systemctl_path",
            "systemd_analyze_path",
        ):
            object.__setattr__(
                self,
                field_name,
                _validated_absolute_path(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        object.__setattr__(
            self,
            "service_user",
            _validated_account_name(self.service_user, field_name="service_user"),
        )
        object.__setattr__(
            self,
            "service_group",
            _validated_account_name(self.service_group, field_name="service_group"),
        )
        if not 5 <= self.control_sync_interval_seconds <= 3600:
            raise ValueError(
                "control_sync_interval_seconds must be between 5 and 3600."
            )


@dataclass(frozen=True, slots=True)
class SystemdUnitArtifact:
    """One rendered, managed systemd unit and its absolute destination path."""

    unit_name: str
    destination_path: PurePosixPath
    content: str

    def __post_init__(self) -> None:
        """Enforce safe artifact names, paths, and the managed-file marker."""

        if not _SAFE_UNIT_NAME_PATTERN.fullmatch(self.unit_name):
            raise ValueError("unit_name must be a safe systemd unit basename.")
        if self.destination_path.name != self.unit_name:
            raise ValueError("destination_path must end with unit_name.")
        if (
            not self.destination_path.is_absolute()
            or ".." in self.destination_path.parts
        ):
            raise ValueError("destination_path must be an absolute POSIX path.")
        if not self.content.startswith(f"{MANAGED_UNIT_MARKER}\n"):
            raise ValueError(
                "Managed unit content must start with its ownership marker."
            )
        if "\x00" in self.content or not self.content.endswith("\n"):
            raise ValueError(
                "Managed unit content must be safe newline-terminated text."
            )


@dataclass(frozen=True, slots=True)
class ProvisioningPlan:
    """Pure rendering result consumed by the privileged provisioning layer."""

    job_id: str
    service_instance: str
    timer_unit: str
    on_calendar: tuple[str, ...]
    artifacts: tuple[SystemdUnitArtifact, ...]

    def __post_init__(self) -> None:
        """Check that plan identifiers and artifacts agree with the canonical job id."""

        canonical = _canonical_job_id(self.job_id)
        if self.service_instance != service_instance_name(canonical):
            raise ValueError("service_instance does not match job_id.")
        if self.timer_unit != timer_unit_name(canonical):
            raise ValueError("timer_unit does not match job_id.")
        if not self.on_calendar:
            raise ValueError(
                "A provisioning plan requires at least one calendar event."
            )
        expected_units = {SERVICE_TEMPLATE_UNIT_NAME, self.timer_unit}
        actual_units = {artifact.unit_name for artifact in self.artifacts}
        if actual_units != expected_units or len(self.artifacts) != 2:
            raise ValueError("A provisioning plan requires one service and one timer.")


def timer_unit_name(job_id: str | UUID) -> str:
    """Return the canonical per-job timer unit name for ``job_id``."""

    return f"furnacemind-job-{_canonical_job_id(job_id)}.timer"


def service_instance_name(job_id: str | UUID) -> str:
    """Return the shared service-template instance name for ``job_id``."""

    return f"furnacemind-job@{_canonical_job_id(job_id)}.service"


def _validated_schedule(
    definition: dict[str, object],
) -> tuple[str, str, dict[str, object]]:
    """Validate a complete definition and return its trusted schedule fields."""

    if not isinstance(definition, dict):
        raise TypeError("Scheduled-job definition must be a JSON object.")
    errors = validate_task_definition(definition)
    if errors:
        raise ValueError("Scheduled-job definition is invalid: " + " ".join(errors))

    schedule = definition["schedule"]
    assert isinstance(schedule, dict)
    frequency = schedule["frequency"]
    timezone_name = schedule["timezone"]
    trigger = schedule["trigger"]
    assert isinstance(frequency, str)
    assert isinstance(timezone_name, str)
    assert isinstance(trigger, dict)
    return frequency, timezone_name, trigger


def _aware_datetime(value: object, *, field_name: str) -> datetime:
    """Parse an ISO timestamp and require an explicit UTC offset."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a UTC offset.")
    if parsed.microsecond:
        raise ValueError(f"{field_name} must use whole-second precision.")
    return parsed


def _clock(hour: int | str, minute: int, second: int = 0) -> str:
    """Format a systemd calendar clock with a numeric or wildcard hour."""

    hour_text = hour if isinstance(hour, str) else f"{hour:02d}"
    return f"{hour_text}:{minute:02d}:{second:02d}"


def _weekday_names(cron_value: str) -> str:
    """Convert a validated numeric cron weekday list into systemd names."""

    return ",".join(_CRON_WEEKDAY_NAMES[int(item)] for item in cron_value.split(","))


def _compile_cron_calendar(
    *,
    frequency: str,
    expression: str,
    timezone_name: str,
) -> tuple[str, ...]:
    """Compile one validated, UI-generated cron expression for systemd."""

    minute_text, hour_text, day_text, _month_text, weekday_text = expression.split()
    minute = int(minute_text)
    timezone_suffix = f" {timezone_name}"

    if frequency == "hourly":
        return (f"*-*-* {_clock('*', minute)}{timezone_suffix}",)

    if frequency == "shift_end":
        return tuple(
            f"*-*-* {_clock(int(hour), minute)}{timezone_suffix}"
            for hour in hour_text.split(",")
        )

    hour = int(hour_text)
    clock = _clock(hour, minute)
    if frequency == "daily":
        calendar_prefix = "*-*-*"
    elif frequency == "weekdays":
        calendar_prefix = "Mon..Fri *-*-*"
    elif frequency in {"selected_days", "weekly"}:
        calendar_prefix = f"{_weekday_names(weekday_text)} *-*-*"
    elif frequency == "monthly":
        calendar_prefix = f"*-*-{int(day_text):02d}"
    else:
        raise ValueError(f"Unsupported cron schedule frequency: {frequency}")
    return (f"{calendar_prefix} {clock}{timezone_suffix}",)


def _compile_interval_calendar(trigger: dict[str, object]) -> tuple[str, ...]:
    """Compile an elapsed-time interval into a safe superset of UTC probes.

    A calendar timer cannot represent every anchored N-hour sequence directly
    when N does not divide a day.  The greatest-common-divisor hour set contains
    every real occurrence; the database runner then ignores probe invocations
    that resolve to an already-claimed logical occurrence.
    """

    interval_seconds = trigger["interval_seconds"]
    assert isinstance(interval_seconds, int)
    interval_hours = interval_seconds // 3600
    anchor = _aware_datetime(
        trigger["anchor_at"],
        field_name="schedule.trigger.anchor_at",
    ).astimezone(timezone.utc)
    hour_stride = math.gcd(interval_hours, 24)
    probe_hours = sorted(
        {
            (anchor.hour + (index * hour_stride)) % 24
            for index in range(24 // hour_stride)
        }
    )
    hour_field = ",".join(f"{hour:02d}" for hour in probe_hours)
    return (f"*-*-* {_clock(hour_field, anchor.minute, anchor.second)} UTC",)


def compile_on_calendar(definition: dict[str, object]) -> tuple[str, ...]:
    """Compile a validated scheduled-job definition into ``OnCalendar`` values."""

    frequency, timezone_name, trigger = _validated_schedule(definition)
    trigger_type = trigger["type"]

    if trigger_type == "once":
        run_at = _aware_datetime(
            trigger["run_at"],
            field_name="schedule.trigger.run_at",
        ).astimezone(ZoneInfo(timezone_name))
        return (
            f"{run_at:%Y-%m-%d} {_clock(run_at.hour, run_at.minute, run_at.second)} "
            f"{timezone_name}",
        )

    if trigger_type == "interval":
        return _compile_interval_calendar(trigger)

    if trigger_type == "cron":
        expression = trigger["expression"]
        assert isinstance(expression, str)
        return _compile_cron_calendar(
            frequency=frequency,
            expression=expression,
            timezone_name=timezone_name,
        )

    raise ValueError(f"Unsupported schedule trigger type: {trigger_type}")


def render_service_template(settings: SystemdProvisioningSettings) -> str:
    """Render the one shared, hardened one-shot FurnaceMind service template."""

    if not isinstance(settings, SystemdProvisioningSettings):
        raise TypeError("settings must be SystemdProvisioningSettings.")
    return "\n".join(
        (
            MANAGED_UNIT_MARKER,
            "[Unit]",
            "Description=Run FurnaceMind scheduled job %i",
            "Wants=network-online.target",
            "After=network-online.target",
            "StartLimitIntervalSec=15min",
            "StartLimitBurst=5",
            "",
            "[Service]",
            "Type=oneshot",
            f"User={settings.service_user}",
            f"Group={settings.service_group}",
            f"WorkingDirectory={settings.working_directory}",
            f"EnvironmentFile={settings.environment_file}",
            f"ExecStart={settings.python_executable} {settings.runner_script} --job-id %i",
            "TimeoutStartSec=infinity",
            "Restart=on-failure",
            "RestartSec=30s",
            "RestartPreventExitStatus=2",
            "KillMode=control-group",
            "UMask=0077",
            "CapabilityBoundingSet=",
            "AmbientCapabilities=",
            "NoNewPrivileges=true",
            "PrivateTmp=true",
            "ProtectSystem=full",
            "ProtectHome=true",
            "ProtectControlGroups=true",
            "ProtectKernelModules=true",
            "ProtectKernelTunables=true",
            "RestrictSUIDSGID=true",
            "LockPersonality=true",
            "StandardOutput=journal",
            "StandardError=journal",
            "SyslogIdentifier=furnacemind-job-%i",
            "",
        )
    )


def build_service_template_artifact(
    settings: SystemdProvisioningSettings,
) -> SystemdUnitArtifact:
    """Build the deployment-owned shared service-template artifact."""

    if not isinstance(settings, SystemdProvisioningSettings):
        raise TypeError("settings must be SystemdProvisioningSettings.")
    return SystemdUnitArtifact(
        unit_name=SERVICE_TEMPLATE_UNIT_NAME,
        destination_path=settings.unit_directory / SERVICE_TEMPLATE_UNIT_NAME,
        content=render_service_template(settings),
    )


def render_control_sync_service(settings: SystemdProvisioningSettings) -> str:
    """Render the privileged one-shot bridge that applies queued commands."""

    if not isinstance(settings, SystemdProvisioningSettings):
        raise TypeError("settings must be SystemdProvisioningSettings.")
    return "\n".join(
        (
            MANAGED_UNIT_MARKER,
            "[Unit]",
            "Description=Apply queued FurnaceMind scheduled-task controls",
            "Wants=network-online.target",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=oneshot",
            "User=root",
            "Group=root",
            f"WorkingDirectory={settings.working_directory}",
            f"EnvironmentFile={settings.environment_file}",
            (
                f"ExecStart={settings.python_executable} "
                f"{settings.control_sync_script} --max-commands 25"
            ),
            "TimeoutStartSec=10min",
            "UMask=0077",
            "NoNewPrivileges=true",
            "PrivateTmp=true",
            "ProtectSystem=strict",
            "ProtectHome=true",
            (
                f"ReadWritePaths={settings.unit_directory} "
                f"{settings.operation_lock_directory}"
            ),
            "ProtectControlGroups=true",
            "ProtectKernelModules=true",
            "ProtectKernelTunables=true",
            "RestrictSUIDSGID=true",
            "LockPersonality=true",
            "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
            "StandardOutput=journal",
            "StandardError=journal",
            "SyslogIdentifier=furnacemind-control-sync",
            "",
        )
    )


def render_control_sync_timer(settings: SystemdProvisioningSettings) -> str:
    """Render the periodic trigger for the short-lived command bridge."""

    if not isinstance(settings, SystemdProvisioningSettings):
        raise TypeError("settings must be SystemdProvisioningSettings.")
    interval = settings.control_sync_interval_seconds
    return "\n".join(
        (
            MANAGED_UNIT_MARKER,
            "[Unit]",
            "Description=Check for queued FurnaceMind task controls",
            "",
            "[Timer]",
            "OnBootSec=30s",
            f"OnUnitInactiveSec={interval}s",
            "AccuracySec=1s",
            "RandomizedDelaySec=3s",
            f"Unit={CONTROL_SYNC_SERVICE_UNIT_NAME}",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        )
    )


def build_control_sync_artifacts(
    settings: SystemdProvisioningSettings,
) -> tuple[SystemdUnitArtifact, SystemdUnitArtifact]:
    """Build the deployment-owned command bridge service and timer artifacts."""

    if not isinstance(settings, SystemdProvisioningSettings):
        raise TypeError("settings must be SystemdProvisioningSettings.")
    return (
        SystemdUnitArtifact(
            unit_name=CONTROL_SYNC_SERVICE_UNIT_NAME,
            destination_path=(settings.unit_directory / CONTROL_SYNC_SERVICE_UNIT_NAME),
            content=render_control_sync_service(settings),
        ),
        SystemdUnitArtifact(
            unit_name=CONTROL_SYNC_TIMER_UNIT_NAME,
            destination_path=settings.unit_directory / CONTROL_SYNC_TIMER_UNIT_NAME,
            content=render_control_sync_timer(settings),
        ),
    )


def _render_timer_from_calendar(job_id: str, on_calendar: tuple[str, ...]) -> str:
    """Render a per-job timer from already compiled calendar expressions."""

    calendar_lines = tuple(f"OnCalendar={value}" for value in on_calendar)
    return "\n".join(
        (
            MANAGED_UNIT_MARKER,
            "[Unit]",
            f"Description=Schedule FurnaceMind job {job_id}",
            "",
            "[Timer]",
            *calendar_lines,
            "Persistent=true",
            "AccuracySec=1s",
            "RandomizedDelaySec=0",
            "RemainAfterElapse=true",
            f"Unit={service_instance_name(job_id)}",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        )
    )


def render_timer_unit(job_id: str | UUID, definition: dict[str, object]) -> str:
    """Render one managed timer for a canonical job and validated definition."""

    canonical = _canonical_job_id(job_id)
    return _render_timer_from_calendar(canonical, compile_on_calendar(definition))


def build_provisioning_plan(
    job_id: str | UUID,
    definition: dict[str, object],
    settings: SystemdProvisioningSettings,
) -> ProvisioningPlan:
    """Build the complete pure unit-installation plan for one scheduled job."""

    if not isinstance(settings, SystemdProvisioningSettings):
        raise TypeError("settings must be SystemdProvisioningSettings.")
    canonical = _canonical_job_id(job_id)
    on_calendar = compile_on_calendar(definition)
    timer_unit = timer_unit_name(canonical)
    service_instance = service_instance_name(canonical)
    service_artifact = build_service_template_artifact(settings)
    timer_artifact = SystemdUnitArtifact(
        unit_name=timer_unit,
        destination_path=settings.unit_directory / timer_unit,
        content=_render_timer_from_calendar(canonical, on_calendar),
    )
    return ProvisioningPlan(
        job_id=canonical,
        service_instance=service_instance,
        timer_unit=timer_unit,
        on_calendar=on_calendar,
        artifacts=(service_artifact, timer_artifact),
    )


__all__ = [
    "CONTROL_SYNC_SERVICE_UNIT_NAME",
    "CONTROL_SYNC_TIMER_UNIT_NAME",
    "MANAGED_UNIT_MARKER",
    "SERVICE_TEMPLATE_UNIT_NAME",
    "ProvisioningPlan",
    "SystemdProvisioningSettings",
    "SystemdUnitArtifact",
    "build_provisioning_plan",
    "build_control_sync_artifacts",
    "build_service_template_artifact",
    "compile_on_calendar",
    "render_service_template",
    "render_control_sync_service",
    "render_control_sync_timer",
    "render_timer_unit",
    "service_instance_name",
    "timer_unit_name",
]
