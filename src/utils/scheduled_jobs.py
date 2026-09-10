"""Pure helpers for scheduled-job configuration documents.

The Streamlit page and a future worker share the JSON contract in this module.
This module deliberately does not execute jobs.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from datetime import datetime, time, timedelta
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator, FormatChecker

from utils.shift_windows import (
    LOCAL_TIMEZONE,
    LOCAL_TIMEZONE_NAME,
    SHIFT_LABELS,
    SHIFT_WINDOWS,
)

SCHEMA_VERSION = "1.0"
FREQUENCIES = (
    "Every Hour",
    "Every Day",
    "Every Shift",
    "Every Week",
    "Custom Schedule",
)
JOB_STATUSES = ("draft", "active", "paused", "archived")
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

_EMAIL_RE = re.compile(
    r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}$",
    re.IGNORECASE,
)
_JOB_ID_RE = re.compile(r"^JOB-[A-F0-9]{12}$")
_RUN_ID_RE = re.compile(r"^RUN-[A-F0-9]{12}$")
_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "auth_token",
    "client_secret",
    "credential",
    "credentials",
    "connection_string",
    "database_credential",
    "database_credentials",
    "database_password",
    "database_url",
    "db_url",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "smtp_credential",
    "smtp_credentials",
    "smtp_host",
    "smtp_password",
    "smtp_port",
    "smtp_server",
    "smtp_username",
    "token",
}
_SENSITIVE_KEYS_COLLAPSED = {
    sensitive.replace("_", "") for sensitive in _SENSITIVE_KEYS
}
_SENSITIVE_SUFFIXES = (
    "_password",
    "_secret",
    "_token",
    "_credential",
    "_credentials",
    "_api_key",
    "_private_key",
)
_CONFIGURATION_FIELDS = (
    "job_name",
    "instructions",
    "timezone",
    "schedule",
    "email",
    "retry",
)


JOB_DOCUMENT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Scheduled job document",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "job_id",
        "revision",
        "job_name",
        "instructions",
        "status",
        "created_at",
        "updated_at",
        "created_by",
        "timezone",
        "schedule",
        "email",
        "retry",
        "last_run",
        "next_run",
        "revision_history",
        "execution_requests",
        "run_history",
        "email_history",
        "report_history",
        "error_history",
        "action_history",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "job_id": {"type": "string", "pattern": _JOB_ID_RE.pattern},
        "revision": {"type": "integer", "minimum": 1},
        "job_name": {"type": "string", "minLength": 1},
        "instructions": {"type": "string", "minLength": 1},
        "status": {"enum": list(JOB_STATUSES)},
        "created_at": {"type": "string", "format": "date-time"},
        "updated_at": {"type": "string", "format": "date-time"},
        "created_by": {"type": "string", "minLength": 1},
        "timezone": {"const": LOCAL_TIMEZONE_NAME},
        "schedule": {
            "type": "object",
            "required": ["frequency", "timezone"],
            "properties": {
                "frequency": {"enum": list(FREQUENCIES)},
                "timezone": {"const": LOCAL_TIMEZONE_NAME},
                "execution_minute": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 59,
                },
                "execution_time": {
                    "type": "string",
                    "pattern": r"^([01]\d|2[0-3]):[0-5]\d$",
                },
                "weekday": {"enum": list(WEEKDAYS)},
                "cron": {"type": "string", "minLength": 1},
                "shifts": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["label", "start_hour", "end_hour"],
                        "properties": {
                            "label": {"type": "string"},
                            "start_hour": {"type": "integer"},
                            "end_hour": {"type": "integer"},
                        },
                        "additionalProperties": True,
                    },
                },
            },
            "additionalProperties": False,
        },
        "email": {
            "type": "object",
            "required": ["enabled", "recipients", "subject", "attachment_formats"],
            "properties": {
                "enabled": {"type": "boolean"},
                "recipients": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "format": "email"},
                },
                "subject": {"type": "string"},
                "attachment_formats": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"enum": ["png", "csv", "json"]},
                },
            },
            "additionalProperties": False,
        },
        "retry": {
            "type": "object",
            "required": [
                "maximum_attempts",
                "retry_interval_seconds",
                "timeout_seconds",
                "notify_on_failure",
            ],
            "properties": {
                "maximum_attempts": {"type": "integer", "minimum": 1},
                "retry_interval_seconds": {"type": "integer", "minimum": 1},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "notify_on_failure": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "last_run": {
            "anyOf": [
                {"type": "null"},
                {"type": "string", "format": "date-time"},
            ]
        },
        "next_run": {
            "anyOf": [
                {"type": "null"},
                {"type": "string", "format": "date-time"},
            ]
        },
        "revision_history": {"type": "array", "items": {"type": "object"}},
        "execution_requests": {"type": "array", "items": {"type": "object"}},
        "run_history": {"type": "array", "items": {"type": "object"}},
        "email_history": {"type": "array", "items": {"type": "object"}},
        "report_history": {"type": "array", "items": {"type": "object"}},
        "error_history": {"type": "array", "items": {"type": "object"}},
        "action_history": {"type": "array", "items": {"type": "object"}},
    },
}

_JOB_VALIDATOR = Draft202012Validator(
    JOB_DOCUMENT_SCHEMA,
    format_checker=FormatChecker(),
)


def _now(now: datetime | None = None) -> datetime:
    """Return a timezone-aware timestamp in the configured local timezone."""
    if now is None:
        return datetime.now(LOCAL_TIMEZONE)
    if now.tzinfo is None:
        return LOCAL_TIMEZONE.localize(now)
    return now.astimezone(LOCAL_TIMEZONE)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def generate_job_id() -> str:
    """Generate a stable, display-friendly unique job identifier."""
    return f"JOB-{uuid4().hex[:12].upper()}"


def generate_run_request_id() -> str:
    """Generate a unique manual-run request identifier."""
    return f"RUN-{uuid4().hex[:12].upper()}"


def normalize_email_recipients(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Split, validate, de-duplicate, and normalize email recipients."""
    if isinstance(value, str):
        candidates = re.split(r"[,;\n]+", value)
    else:
        candidates = list(value)

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        address = str(candidate).strip().lower()
        if not address:
            continue
        local_part = address.partition("@")[0]
        if (
            not _EMAIL_RE.fullmatch(address)
            or local_part.startswith(".")
            or local_part.endswith(".")
            or ".." in local_part
        ):
            raise ValueError(f"Invalid email address: {address}")
        if address not in seen:
            normalized.append(address)
            seen.add(address)
    return normalized


def build_email_configuration(
    *,
    enabled: bool = False,
    recipients: str | list[str] | tuple[str, ...] = "",
    subject: str = "",
    attachment_formats: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build email delivery settings without accepting transport credentials."""
    normalized_recipients = normalize_email_recipients(recipients) if enabled else []
    if enabled and not normalized_recipients:
        raise ValueError("Enter at least one email recipient.")

    formats = list(dict.fromkeys(str(item).lower() for item in attachment_formats))
    invalid_formats = sorted(set(formats) - {"png", "csv", "json"})
    if invalid_formats:
        raise ValueError(f"Invalid attachment format: {invalid_formats[0]}")

    return {
        "enabled": bool(enabled),
        "recipients": normalized_recipients if enabled else [],
        "subject": str(subject).strip() if enabled else "",
        "attachment_formats": formats if enabled else [],
    }


def build_retry_configuration(
    *,
    maximum_attempts: int = 3,
    retry_interval_seconds: int = 60,
    timeout_seconds: int = 300,
    notify_on_failure: bool = True,
) -> dict[str, Any]:
    """Build and validate retry/timeout settings."""
    values = {
        "maximum_attempts": maximum_attempts,
        "retry_interval_seconds": retry_interval_seconds,
        "timeout_seconds": timeout_seconds,
    }
    for field, raw_value in values.items():
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, int)
            or raw_value < 1
        ):
            label = field.replace("_", " ").capitalize()
            raise ValueError(f"{label} must be a positive integer.")
    return {**values, "notify_on_failure": bool(notify_on_failure)}


def _cron_value(value: str, minimum: int, maximum: int) -> bool:
    if not value.isdigit():
        return False
    return minimum <= int(value) <= maximum


def _cron_part_is_valid(part: str, minimum: int, maximum: int) -> bool:
    for item in part.split(","):
        if not item or item.count("/") > 1:
            return False
        base, separator, step = item.partition("/")
        if separator and (not step.isdigit() or int(step) < 1):
            return False
        if base == "*":
            continue
        if base.count("-") == 1:
            start, end = base.split("-", 1)
            if not (
                _cron_value(start, minimum, maximum)
                and _cron_value(end, minimum, maximum)
                and int(start) <= int(end)
            ):
                return False
            continue
        if not _cron_value(base, minimum, maximum):
            return False
    return True


def cron_validation_error(expression: str) -> str | None:
    """Return a concise error for an unsupported/malformed five-field cron."""
    fields = str(expression or "").strip().split()
    if len(fields) != 5:
        return "expected exactly 5 fields."

    ranges = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
    labels = ("minute", "hour", "day of month", "month", "day of week")
    for field, bounds, label in zip(fields, ranges, labels, strict=True):
        if not _cron_part_is_valid(field, *bounds):
            return f"invalid {label} field '{field}'."
    return None


def validate_cron_expression(expression: str) -> bool:
    """Return whether *expression* is a supported standard five-field cron."""
    return cron_validation_error(expression) is None


def _time_text(value: time | str | None) -> str:
    if isinstance(value, time):
        return value.strftime("%H:%M")
    text_value = str(value or "").strip()
    try:
        return datetime.strptime(text_value, "%H:%M").strftime("%H:%M")
    except ValueError as exc:
        raise ValueError("Execution time is required.") from exc


def build_schedule(
    frequency: str,
    *,
    execution_minute: int | None = None,
    execution_time: time | str | None = None,
    weekday: str | None = None,
    cron_expression: str | None = None,
) -> dict[str, Any]:
    """Build and validate a supported schedule configuration."""
    if not str(frequency or "").strip():
        raise ValueError("Frequency is required.")
    if frequency not in FREQUENCIES:
        raise ValueError(f"Unsupported frequency: {frequency}")

    schedule: dict[str, Any] = {
        "frequency": frequency,
        "timezone": LOCAL_TIMEZONE_NAME,
    }
    if frequency == "Every Hour":
        if (
            isinstance(execution_minute, bool)
            or not isinstance(execution_minute, int)
            or not 0 <= execution_minute <= 59
        ):
            raise ValueError("Execution minute must be an integer from 0 to 59.")
        schedule["execution_minute"] = execution_minute
    elif frequency == "Every Day":
        schedule["execution_time"] = _time_text(execution_time)
    elif frequency == "Every Shift":
        if not SHIFT_WINDOWS or not SHIFT_LABELS:
            raise ValueError("No production shifts are configured.")
        schedule["shifts"] = [copy.deepcopy(dict(window)) for window in SHIFT_WINDOWS]
    elif frequency == "Every Week":
        if weekday not in WEEKDAYS:
            raise ValueError("Weekday is required.")
        schedule["weekday"] = weekday
        schedule["execution_time"] = _time_text(execution_time)
    else:
        cron = str(cron_expression or "").strip()
        error = cron_validation_error(cron)
        if error:
            raise ValueError(f"Invalid cron expression: {error}")
        schedule["cron"] = cron
    return schedule


def validate_schedule(schedule: Mapping[str, Any]) -> None:
    """Validate an already-built schedule with the same rules as creation."""
    frequency = str(schedule.get("frequency") or "")
    rebuilt = build_schedule(
        frequency,
        execution_minute=schedule.get("execution_minute"),
        execution_time=schedule.get("execution_time"),
        weekday=schedule.get("weekday"),
        cron_expression=schedule.get("cron"),
    )
    if schedule.get("timezone") != LOCAL_TIMEZONE_NAME:
        raise ValueError(f"Schedule timezone must be {LOCAL_TIMEZONE_NAME}.")
    if frequency == "Every Shift" and schedule.get("shifts") != rebuilt["shifts"]:
        raise ValueError(
            "Every Shift must use the configured production shift windows."
        )


def calculate_next_run(
    schedule: Mapping[str, Any], now: datetime | None = None
) -> str | None:
    """Calculate the next local run for standard frequencies; cron stays worker-owned."""
    validate_schedule(schedule)
    current = _now(now)
    frequency = schedule["frequency"]

    if frequency == "Custom Schedule":
        return None
    if frequency == "Every Hour":
        candidate = current.replace(
            minute=int(schedule["execution_minute"]), second=0, microsecond=0
        )
        if candidate <= current:
            candidate += timedelta(hours=1)
        return _iso(candidate)

    if frequency in {"Every Day", "Every Week"}:
        execution_time = datetime.strptime(
            str(schedule["execution_time"]), "%H:%M"
        ).time()
        candidate_date = current.date()
        if frequency == "Every Week":
            days_ahead = WEEKDAYS.index(str(schedule["weekday"])) - current.weekday()
            candidate_date += timedelta(days=days_ahead % 7)
        candidate = LOCAL_TIMEZONE.localize(
            datetime.combine(candidate_date, execution_time)
        )
        if candidate <= current:
            candidate += timedelta(days=7 if frequency == "Every Week" else 1)
        return _iso(candidate)

    candidates: list[datetime] = []
    for day_offset in (0, 1):
        candidate_date = current.date() + timedelta(days=day_offset)
        for window in SHIFT_WINDOWS:
            candidate = LOCAL_TIMEZONE.localize(
                datetime.combine(
                    candidate_date,
                    time(hour=int(window["start_hour"])),
                )
            )
            if candidate > current:
                candidates.append(candidate)
    return _iso(min(candidates))


def validate_required_fields(
    *,
    job_name: str,
    instructions: str,
    frequency: str,
) -> list[str]:
    """Return clear required-field validation messages for the page."""
    errors: list[str] = []
    if not str(job_name or "").strip():
        errors.append("Job Name is required.")
    if not str(instructions or "").strip():
        errors.append("Job Instructions are required.")
    if not str(frequency or "").strip():
        errors.append("Frequency is required.")
    return errors


def _normalized_key(value: Any) -> str:
    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value).strip())
    return re.sub(r"[^a-z0-9]+", "_", snake_case.lower()).strip("_")


def find_sensitive_keys(value: Any, path: str = "$") -> list[str]:
    """Recursively find credential-like keys anywhere in a JSON-compatible value."""
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            collapsed = normalized.replace("_", "")
            child_path = f"{path}.{key}"
            if (
                normalized in _SENSITIVE_KEYS
                or collapsed in _SENSITIVE_KEYS_COLLAPSED
                or normalized.endswith(_SENSITIVE_SUFFIXES)
            ):
                findings.append(child_path)
            findings.extend(find_sensitive_keys(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(find_sensitive_keys(child, f"{path}[{index}]"))
    return findings


def assert_no_sensitive_keys(value: Any) -> None:
    """Reject secrets and credential fields before preview/download/persistence."""
    findings = find_sensitive_keys(value)
    if findings:
        raise ValueError(f"Sensitive field is not allowed: {findings[0]}")


def _assert_report_value_is_metadata(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if (
                "base64" in normalized
                or normalized == "binary"
                or normalized.endswith("_bytes")
                or normalized in {"image_data", "file_content"}
            ):
                raise ValueError(
                    "Report history may contain metadata or file paths only: "
                    f"{child_path}"
                )
            _assert_report_value_is_metadata(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_report_value_is_metadata(child, f"{path}[{index}]")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"Report history may not contain binary file content: {path}")
    elif isinstance(value, str) and value.strip().lower().startswith("data:image/"):
        raise ValueError(
            "Report history may contain metadata or file paths only: " f"{path}"
        )


def _assert_report_metadata_only(job: Mapping[str, Any]) -> None:
    _assert_report_value_is_metadata(job.get("report_history", []), "$.report_history")


def validate_job_document(job: Mapping[str, Any]) -> None:
    """Validate security, schedule semantics, and the complete JSON Schema."""
    assert_no_sensitive_keys(job)
    _assert_report_metadata_only(job)
    _JOB_VALIDATOR.validate(job)
    validate_schedule(job["schedule"])
    if job["email"]["enabled"]:
        if not job["email"]["recipients"]:
            raise ValueError("Enter at least one email recipient.")
        normalize_email_recipients(job["email"]["recipients"])


def build_job_document(
    *,
    job_name: str,
    instructions: str,
    status: str,
    schedule: Mapping[str, Any],
    email: Mapping[str, Any],
    retry: Mapping[str, Any],
    created_by: str,
    job_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build Revision 1 of a complete scheduled-job document."""
    errors = validate_required_fields(
        job_name=job_name,
        instructions=instructions,
        frequency=str(schedule.get("frequency") or ""),
    )
    if errors:
        raise ValueError(errors[0])
    if status not in {"draft", "active"}:
        raise ValueError("New job status must be draft or active.")
    actor = str(created_by or "").strip()
    if not actor:
        raise ValueError("Created By is required.")

    timestamp = _now(now)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id or generate_job_id(),
        "revision": 1,
        "job_name": str(job_name).strip(),
        "instructions": str(instructions).strip(),
        "status": status,
        "created_at": _iso(timestamp),
        "updated_at": _iso(timestamp),
        "created_by": actor,
        "timezone": LOCAL_TIMEZONE_NAME,
        "schedule": copy.deepcopy(dict(schedule)),
        "email": copy.deepcopy(dict(email)),
        "retry": copy.deepcopy(dict(retry)),
        "last_run": None,
        "next_run": calculate_next_run(schedule, timestamp),
        "revision_history": [],
        "execution_requests": [],
        "run_history": [],
        "email_history": [],
        "report_history": [],
        "error_history": [],
        "action_history": [],
    }
    validate_job_document(document)
    return document


def configuration_snapshot(job: Mapping[str, Any]) -> dict[str, Any]:
    """Return only editable configuration fields, with no histories."""
    return {field: copy.deepcopy(job[field]) for field in _CONFIGURATION_FIELDS}


def revise_job(
    job: Mapping[str, Any],
    configuration_updates: Mapping[str, Any],
    *,
    saved_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create the next revision while preserving an immutable prior snapshot."""
    unsupported = sorted(set(configuration_updates) - set(_CONFIGURATION_FIELDS))
    if unsupported:
        raise ValueError(f"Not an editable configuration field: {unsupported[0]}")

    revised = copy.deepcopy(dict(job))
    timestamp = _now(now)
    prior_revision = int(revised["revision"])
    revised["revision_history"].append(
        {
            "revision": prior_revision,
            "saved_at": _iso(timestamp),
            "saved_by": str(saved_by).strip(),
            "configuration": configuration_snapshot(job),
        }
    )
    for field, value in configuration_updates.items():
        revised[field] = copy.deepcopy(value)
    revised["revision"] = prior_revision + 1
    revised["updated_at"] = _iso(timestamp)
    revised["next_run"] = (
        None
        if revised["status"] in {"paused", "archived"}
        else calculate_next_run(revised["schedule"], timestamp)
    )
    validate_job_document(revised)
    return revised


def _append_action(
    job: Mapping[str, Any], action: str, actor: str, now: datetime | None = None
) -> tuple[dict[str, Any], datetime]:
    updated = copy.deepcopy(dict(job))
    timestamp = _now(now)
    updated["updated_at"] = _iso(timestamp)
    updated["action_history"].append(
        {
            "action": action,
            "performed_at": _iso(timestamp),
            "performed_by": str(actor).strip(),
        }
    )
    return updated, timestamp


def pause_job(
    job: Mapping[str, Any], *, actor: str, now: datetime | None = None
) -> dict[str, Any]:
    """Pause future execution without changing the configuration revision."""
    if job.get("status") == "archived":
        raise ValueError("Archived jobs cannot be paused.")
    updated, _ = _append_action(job, "pause", actor, now)
    updated["status"] = "paused"
    updated["next_run"] = None
    validate_job_document(updated)
    return updated


def resume_job(
    job: Mapping[str, Any], *, actor: str, now: datetime | None = None
) -> dict[str, Any]:
    """Resume/activate scheduling without incrementing the revision."""
    if job.get("status") == "archived":
        raise ValueError("Archived jobs cannot be resumed.")
    action = "activate" if job.get("status") == "draft" else "resume"
    updated, timestamp = _append_action(job, action, actor, now)
    updated["status"] = "active"
    updated["next_run"] = calculate_next_run(updated["schedule"], timestamp)
    validate_job_document(updated)
    return updated


def request_run_now(
    job: Mapping[str, Any], *, actor: str, now: datetime | None = None
) -> dict[str, Any]:
    """Queue a manual execution request; never execute the report here."""
    if job.get("status") == "archived":
        raise ValueError("Archived jobs cannot be run.")
    updated, timestamp = _append_action(job, "run_now_requested", actor, now)
    request_id = generate_run_request_id()
    if not _RUN_ID_RE.fullmatch(request_id):  # Defensive if generation changes.
        raise ValueError("Invalid run request ID.")
    updated["execution_requests"].append(
        {
            "request_id": request_id,
            "requested_at": _iso(timestamp),
            "requested_by": str(actor).strip(),
            "manual": True,
            "status": "pending",
        }
    )
    validate_job_document(updated)
    return updated


def archive_job(
    job: Mapping[str, Any], *, actor: str, now: datetime | None = None
) -> dict[str, Any]:
    """Archive a job in place while preserving all histories."""
    updated, _ = _append_action(job, "archive", actor, now)
    updated["status"] = "archived"
    updated["next_run"] = None
    validate_job_document(updated)
    return updated


def clone_job_document(
    job: Mapping[str, Any], *, created_by: str, now: datetime | None = None
) -> dict[str, Any]:
    """Create an independent Revision 1 draft from another job's configuration."""
    timestamp = _now(now)
    cloned = build_job_document(
        job_name=f"{job['job_name']} (Copy)",
        instructions=job["instructions"],
        status="draft",
        schedule=copy.deepcopy(job["schedule"]),
        email=copy.deepcopy(job["email"]),
        retry=copy.deepcopy(job["retry"]),
        created_by=created_by,
        now=timestamp,
    )
    return cloned
