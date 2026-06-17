"""Validation helpers for daemon job definitions."""

from __future__ import annotations

import json
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import (
    DaemonJobConcurrencyPolicy,
    DaemonJobCriticality,
    DaemonJobKind,
    DaemonJobRestartPolicy,
    DaemonJobScheduleType,
)
from .schemas import DaemonJobPayload, ValidationResult

ALLOWED_JOB_KINDS = tuple(kind.value for kind in DaemonJobKind)
ALLOWED_SCHEDULE_TYPES = tuple(schedule.value for schedule in DaemonJobScheduleType)
ALLOWED_RESTART_POLICIES = tuple(policy.value for policy in DaemonJobRestartPolicy)
ALLOWED_CONCURRENCY_POLICIES = tuple(
    policy.value for policy in DaemonJobConcurrencyPolicy
)
ALLOWED_CRITICALITIES = tuple(level.value for level in DaemonJobCriticality)

JSON_FIELDS = (
    "job_args_json",
    "tools_allowed_json",
    "tools_blocked_json",
    "memory_short_json",
    "memory_long_json",
    "reporting_rules_json",
    "criticality_rules_json",
)
PATH_FIELDS = (
    "working_directory",
    "python_executable",
    "env_file",
    "persist_jobs_md_path",
)
PAYLOAD_FIELDS = tuple(DaemonJobPayload.model_fields.keys())

DEFAULT_JOB_PAYLOAD: dict[str, Any] = {
    "name": "",
    "description": None,
    "enabled": False,
    "job_kind": DaemonJobKind.CUSTOM_LANGGRAPH_AGENT.value,
    "schedule_type": DaemonJobScheduleType.SYSTEMD_TIMER.value,
    "cron_expression": None,
    "on_calendar": "hourly",
    "timezone": "Asia/Kolkata",
    "systemd_unit_name": "",
    "working_directory": "/home/pi/evonith_bf_webapp",
    "python_executable": "/home/pi/evonith_bf_webapp/.venv/bin/python",
    "module_path": "src.jobs.runner",
    "job_args_json": "{}",
    "env_file": "/home/pi/evonith_bf_webapp/.env",
    "user_name": "pi",
    "group_name": "pi",
    "restart_policy": DaemonJobRestartPolicy.ON_FAILURE.value,
    "restart_sec": 10,
    "timeout_sec": 900,
    "max_runtime_sec": 900,
    "concurrency_policy": DaemonJobConcurrencyPolicy.FORBID.value,
    "criticality": DaemonJobCriticality.NORMAL.value,
    "tools_allowed_json": "[]",
    "tools_blocked_json": "[]",
    "memory_short_json": "[]",
    "memory_long_json": "[]",
    "reporting_rules_json": "{}",
    "criticality_rules_json": "{}",
    "persist_jobs_md_path": "/home/pi/evonith_bf_webapp/persist_jobs.md",
    "notes": None,
}

UNIT_NAME_RE = re.compile(r"^[a-z0-9_-]+$")
MODULE_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
ACCOUNT_NAME_RE = re.compile(r"^[a-z_][a-z0-9_-]*[$]?$")
SYSTEMD_CALENDAR_RE = re.compile(r"^[A-Za-z0-9*,:./ _+\-~]+$")
PATH_FORBIDDEN_RE = re.compile(r"""[\s;&|`$<>'"\\\n\r\t]""")
TEXT_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def suggest_systemd_unit_name(name: str) -> str:
    """Return a safe unit-name stem derived from a display name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    slug = slug.strip("-_")
    if not slug:
        slug = "agent-job"
    if not slug.startswith("evonith-"):
        slug = f"evonith-{slug}"
    return slug[:96].strip("-_") or "evonith-agent-job"


def validate_daemon_job_payload(payload: dict[str, Any]) -> ValidationResult:
    """Validate and normalize one daemon job payload."""
    errors: list[str] = []
    warnings: list[str] = []
    normalized = dict(DEFAULT_JOB_PAYLOAD)

    unknown_fields = sorted(set(payload) - set(PAYLOAD_FIELDS))
    if unknown_fields:
        errors.append(f"Unsupported field(s): {', '.join(unknown_fields)}.")

    for field in PAYLOAD_FIELDS:
        if field in payload:
            normalized[field] = payload[field]

    _normalize_strings(normalized)
    _normalize_booleans_and_ints(normalized, errors)

    if not normalized["name"]:
        errors.append("Name is required.")
    elif TEXT_CONTROL_RE.search(normalized["name"]):
        errors.append("Name cannot contain control characters.")
    elif len(normalized["name"]) > 160:
        errors.append("Name must be 160 characters or fewer.")

    _validate_unit_name(str(normalized.get("systemd_unit_name") or ""), errors)
    _validate_enum("job_kind", normalized["job_kind"], ALLOWED_JOB_KINDS, errors)
    _validate_enum(
        "schedule_type",
        normalized["schedule_type"],
        ALLOWED_SCHEDULE_TYPES,
        errors,
    )
    _validate_enum(
        "restart_policy",
        normalized["restart_policy"],
        ALLOWED_RESTART_POLICIES,
        errors,
    )
    _validate_enum(
        "concurrency_policy",
        normalized["concurrency_policy"],
        ALLOWED_CONCURRENCY_POLICIES,
        errors,
    )
    _validate_enum(
        "criticality", normalized["criticality"], ALLOWED_CRITICALITIES, errors
    )

    _validate_schedule(normalized, errors, warnings)
    _validate_timing(normalized, errors)
    _validate_paths(normalized, errors)
    _validate_module_path(str(normalized.get("module_path") or ""), errors)
    _validate_account_name("user_name", str(normalized.get("user_name") or ""), errors)
    _validate_account_name(
        "group_name", str(normalized.get("group_name") or ""), errors
    )
    _validate_timezone(str(normalized.get("timezone") or ""), errors)

    parsed_json = _validate_json_fields(normalized, errors)
    if normalized["criticality"] == DaemonJobCriticality.CRITICAL.value:
        reporting_rules = parsed_json.get("reporting_rules_json")
        if not reporting_rules:
            errors.append("Critical jobs require non-empty reporting rules JSON.")

    if not errors:
        try:
            normalized = DaemonJobPayload.model_validate(normalized).model_dump()
        except Exception as exc:  # pragma: no cover - defensive pydantic guard
            errors.append(f"Payload schema validation failed: {exc}")

    return ValidationResult(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        normalized_payload=normalized,
    )


def _normalize_strings(payload: dict[str, Any]) -> None:
    """Trim string inputs and normalize nullable blank fields."""
    nullable_text_fields = {"description", "cron_expression", "on_calendar", "notes"}
    for field in PAYLOAD_FIELDS:
        value = payload.get(field)
        if isinstance(value, str):
            value = value.strip()
            if field in nullable_text_fields and value == "":
                value = None
            payload[field] = value


def _normalize_booleans_and_ints(payload: dict[str, Any], errors: list[str]) -> None:
    """Coerce simple bool/int fields from Streamlit inputs."""
    enabled = payload.get("enabled")
    if isinstance(enabled, str):
        if enabled.strip().lower() in {"true", "1", "yes", "y"}:
            payload["enabled"] = True
        elif enabled.strip().lower() in {"false", "0", "no", "n"}:
            payload["enabled"] = False
        else:
            errors.append("Enabled must be true or false.")
    elif not isinstance(enabled, bool):
        payload["enabled"] = bool(enabled)

    for field in ("restart_sec", "timeout_sec", "max_runtime_sec"):
        try:
            payload[field] = int(payload[field])
        except (TypeError, ValueError):
            errors.append(f"{field} must be an integer.")


def _validate_unit_name(unit_name: str, errors: list[str]) -> None:
    """Validate the systemd unit-name stem."""
    if not unit_name:
        errors.append("systemd_unit_name is required.")
        return
    if unit_name.endswith((".service", ".timer")):
        errors.append("systemd_unit_name must not include .service or .timer.")
    if "/" in unit_name or "\\" in unit_name:
        errors.append("systemd_unit_name must not contain path separators.")
    if not UNIT_NAME_RE.fullmatch(unit_name):
        errors.append(
            "systemd_unit_name may contain only lowercase letters, digits, dash, and underscore."
        )


def _validate_enum(
    field: str,
    value: Any,
    allowed_values: tuple[str, ...],
    errors: list[str],
) -> None:
    """Validate a string enum field."""
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        errors.append(f"{field} must be one of: {allowed}.")


def _validate_schedule(
    payload: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate scheduling fields for the selected scheduling mode."""
    schedule_type = payload.get("schedule_type")
    cron_expression = payload.get("cron_expression")
    on_calendar = payload.get("on_calendar")

    if schedule_type == DaemonJobScheduleType.MANUAL_ONLY.value:
        if cron_expression or on_calendar:
            errors.append(
                "manual_only jobs must not set cron_expression or on_calendar."
            )
        return

    if schedule_type == DaemonJobScheduleType.CRON_EXPRESSION.value:
        if not cron_expression:
            errors.append("cron_expression schedule requires cron_expression.")
        elif not _is_valid_cron_expression(str(cron_expression)):
            errors.append("cron_expression must be a valid five-field cron expression.")
        if on_calendar:
            errors.append("cron_expression jobs must not set on_calendar.")
        warnings.append(
            "cron_expression is stored for compatibility but Step 2 should translate it to systemd OnCalendar or runner scheduling."
        )
        return

    if schedule_type == DaemonJobScheduleType.SYSTEMD_TIMER.value:
        if cron_expression:
            errors.append("systemd_timer jobs must not set cron_expression.")
        if not on_calendar:
            errors.append("systemd_timer schedule requires on_calendar.")
        elif not _is_valid_on_calendar(str(on_calendar)):
            errors.append("on_calendar must be a valid-ish systemd OnCalendar value.")


def _is_valid_cron_expression(expression: str) -> bool:
    """Validate a simple five-field cron expression without extra dependencies."""
    parts = expression.split()
    if len(parts) != 5:
        return False
    ranges = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
    return all(
        _is_valid_cron_field(part, low, high)
        for part, (low, high) in zip(parts, ranges)
    )


def _is_valid_cron_field(field: str, low: int, high: int) -> bool:
    """Validate one numeric cron field supporting ranges, lists, and steps."""
    if not field:
        return False

    for item in field.split(","):
        if not item:
            return False
        base, sep, step = item.partition("/")
        if sep:
            if not step.isdigit() or int(step) < 1:
                return False
        if base == "*":
            continue
        if "-" in base:
            start, end = base.split("-", 1)
            if not (start.isdigit() and end.isdigit()):
                return False
            if not (low <= int(start) <= int(end) <= high):
                return False
            continue
        if not base.isdigit():
            return False
        if not (low <= int(base) <= high):
            return False
    return True


def _is_valid_on_calendar(value: str) -> bool:
    """Apply a conservative allowlist for systemd OnCalendar preview text."""
    value = value.strip()
    if not value or len(value) > 128:
        return False
    if PATH_FORBIDDEN_RE.search(value.replace(" ", "")):
        return False
    if not SYSTEMD_CALENDAR_RE.fullmatch(value):
        return False
    return any(ch.isalnum() or ch == "*" for ch in value)


def _validate_timing(payload: dict[str, Any], errors: list[str]) -> None:
    """Validate restart and timeout bounds."""
    restart_sec = payload.get("restart_sec")
    timeout_sec = payload.get("timeout_sec")
    max_runtime_sec = payload.get("max_runtime_sec")

    if isinstance(restart_sec, int) and not (1 <= restart_sec <= 300):
        errors.append("restart_sec must be between 1 and 300.")
    if isinstance(timeout_sec, int) and not (60 <= timeout_sec <= 21600):
        errors.append("timeout_sec must be between 60 and 21600.")
    if isinstance(max_runtime_sec, int) and not (60 <= max_runtime_sec <= 21600):
        errors.append("max_runtime_sec must be between 60 and 21600.")


def _validate_paths(payload: dict[str, Any], errors: list[str]) -> None:
    """Validate Pi-side absolute path fields."""
    for field in PATH_FIELDS:
        value = str(payload.get(field) or "")
        if not value:
            errors.append(f"{field} is required.")
            continue
        if not value.startswith("/"):
            errors.append(f"{field} must be an absolute POSIX path.")
        if PATH_FORBIDDEN_RE.search(value):
            errors.append(f"{field} contains unsafe path characters.")


def _validate_module_path(module_path: str, errors: list[str]) -> None:
    """Validate the Python module path used in ExecStart."""
    if not module_path:
        errors.append("module_path is required.")
    elif not MODULE_PATH_RE.fullmatch(module_path):
        errors.append("module_path must be a safe dotted Python module path.")


def _validate_account_name(field: str, value: str, errors: list[str]) -> None:
    """Validate system account names for preview rendering."""
    if not value:
        errors.append(f"{field} is required.")
    elif not ACCOUNT_NAME_RE.fullmatch(value):
        errors.append(f"{field} must be a safe system account name.")


def _validate_timezone(timezone: str, errors: list[str]) -> None:
    """Validate IANA timezone values where zoneinfo data is available."""
    if not timezone:
        errors.append("timezone is required.")
        return
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        errors.append("timezone must be a valid IANA timezone name.")


def _validate_json_fields(
    payload: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    """Validate JSON text fields and return parsed values by field name."""
    parsed: dict[str, Any] = {}
    for field in JSON_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must contain valid JSON text.")
            continue
        try:
            parsed[field] = json.loads(value)
        except json.JSONDecodeError as exc:
            errors.append(f"{field} is invalid JSON: {exc.msg}.")
    return parsed
