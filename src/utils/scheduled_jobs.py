"""Build and validate scheduled-job documents from shared YAML metadata.

The module owns configuration documents and state transitions only; a worker is
responsible for executing jobs and delivering their output.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from datetime import datetime, time, timedelta
from typing import Any
from uuid import uuid4

import pytz
from jsonschema import Draft202012Validator, FormatChecker

from config.config_loader import load_config
from utils.shift_windows import SHIFT_LABELS, SHIFT_WINDOWS

SCHEDULED_JOBS_CONFIG: dict[str, Any] = load_config("scheduled_jobs.yml") or {}
SCHEMA_VERSION = str(SCHEDULED_JOBS_CONFIG["schema_version"])
LOCAL_TIMEZONE_NAME = str(SCHEDULED_JOBS_CONFIG["timezone"])
LOCAL_TIMEZONE = pytz.timezone(LOCAL_TIMEZONE_NAME)

FREQUENCY_SETTINGS: dict[str, dict[str, Any]] = SCHEDULED_JOBS_CONFIG["frequencies"]
FREQUENCIES = tuple(FREQUENCY_SETTINGS)
DEFAULT_FREQUENCY = str(SCHEDULED_JOBS_CONFIG["default_frequency"])
WEEKDAYS = tuple(SCHEDULED_JOBS_CONFIG["weekdays"])
JOB_STATUSES = tuple(SCHEDULED_JOBS_CONFIG["job_statuses"])
MODEL_LEVEL_HINTS: dict[str, str] = SCHEDULED_JOBS_CONFIG["model_levels"]
MODEL_LEVELS = tuple(MODEL_LEVEL_HINTS)
DEFAULT_MODEL_LEVEL = str(SCHEDULED_JOBS_CONFIG["default_model_level"])
DELIVERY_CHANNEL_SETTINGS: dict[str, dict[str, Any]] = SCHEDULED_JOBS_CONFIG[
    "delivery_channels"
]
DELIVERY_CHANNELS = tuple(DELIVERY_CHANNEL_SETTINGS)
DELIVERY_CHANNEL_LABELS = {
    name: str(settings["label"]) for name, settings in DELIVERY_CHANNEL_SETTINGS.items()
}
RETRY_FIELDS: dict[str, dict[str, Any]] = SCHEDULED_JOBS_CONFIG["retry_fields"]
RETRY_DEFAULTS = {name: settings["default"] for name, settings in RETRY_FIELDS.items()}

_CONFIGURATION_FIELDS = tuple(SCHEDULED_JOBS_CONFIG["editable_fields"])
_HISTORY_FIELDS = tuple(SCHEDULED_JOBS_CONFIG["history_fields"])
_JOB_ID_RE = re.compile(SCHEDULED_JOBS_CONFIG["id_patterns"]["job"])
_RUN_ID_RE = re.compile(SCHEDULED_JOBS_CONFIG["id_patterns"]["run"])
_EMAIL_RE = re.compile(
    r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}$",
    re.IGNORECASE,
)
_SENSITIVE_KEYS = set(SCHEDULED_JOBS_CONFIG["sensitive_keys"])
_SENSITIVE_KEYS_COLLAPSED = {key.replace("_", "") for key in _SENSITIVE_KEYS}
_SENSITIVE_SUFFIXES = tuple(SCHEDULED_JOBS_CONFIG["sensitive_suffixes"])
_CRON_FIELDS = tuple(SCHEDULED_JOBS_CONFIG["cron_fields"])


def _frequency_for(kind: str) -> str:
    """Return the configured frequency label for an internal schedule kind."""
    return next(
        name for name, values in FREQUENCY_SETTINGS.items() if values["kind"] == kind
    )


_HOURLY, _DAILY, _SHIFT, _WEEKLY, _CRON = (
    _frequency_for(kind) for kind in ("hourly", "daily", "shift", "weekly", "cron")
)

JOB_DOCUMENT_SCHEMA: dict[str, Any] = copy.deepcopy(
    SCHEDULED_JOBS_CONFIG["document_schema"]
)
_PROPERTIES = JOB_DOCUMENT_SCHEMA["properties"]
_PROPERTIES["schema_version"]["const"] = SCHEMA_VERSION
_PROPERTIES["job_id"]["pattern"] = _JOB_ID_RE.pattern
_PROPERTIES["model_level"]["enum"] = list(MODEL_LEVELS)
_PROPERTIES["status"]["enum"] = list(JOB_STATUSES)
_PROPERTIES["timezone"]["const"] = LOCAL_TIMEZONE_NAME
_SCHEDULE_SCHEMA = _PROPERTIES["schedule"]["properties"]
_SCHEDULE_SCHEMA["frequency"]["enum"] = list(FREQUENCIES)
_SCHEDULE_SCHEMA["timezone"]["const"] = LOCAL_TIMEZONE_NAME
_SCHEDULE_SCHEMA["weekday"]["enum"] = list(WEEKDAYS)
_PROPERTIES["delivery"]["properties"] = {
    name: copy.deepcopy(settings["schema"])
    for name, settings in DELIVERY_CHANNEL_SETTINGS.items()
}
Draft202012Validator.check_schema(JOB_DOCUMENT_SCHEMA)
_JOB_VALIDATOR = Draft202012Validator(
    JOB_DOCUMENT_SCHEMA, format_checker=FormatChecker()
)


def _now(value: datetime | None = None) -> datetime:
    """Return *value* as a timezone-aware configured-local timestamp."""
    if value is None:
        return datetime.now(LOCAL_TIMEZONE)
    return (
        LOCAL_TIMEZONE.localize(value)
        if value.tzinfo is None
        else value.astimezone(LOCAL_TIMEZONE)
    )


def _iso(value: datetime) -> str:
    """Serialize a timestamp without unnecessary microseconds."""
    return value.isoformat(timespec="seconds")


def generate_job_id(job_number: int) -> str:
    """Format a database-reserved sequential job number."""
    if isinstance(job_number, bool) or not isinstance(job_number, int):
        raise ValueError("Job number must be an integer.")
    if job_number < 1001:
        raise ValueError("Job number must be 1001 or greater.")
    return f"Job-{job_number}"


def generate_run_request_id() -> str:
    """Generate a unique manual-run request identifier."""
    return f"RUN-{uuid4().hex[:12].upper()}"


def normalize_email_recipients(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Split, validate, lowercase, and de-duplicate email recipients."""
    candidates = re.split(r"[,;\n]+", value) if isinstance(value, str) else value
    addresses = [str(candidate).strip().lower() for candidate in candidates]
    for address in filter(None, addresses):
        local_part = address.partition("@")[0]
        if (
            not _EMAIL_RE.fullmatch(address)
            or local_part.startswith(".")
            or local_part.endswith(".")
            or ".." in local_part
        ):
            raise ValueError(f"Invalid email address: {address}")
    return list(dict.fromkeys(filter(None, addresses)))


def _channel(delivery: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """Return one channel mapping and reject non-object settings."""
    value = delivery.get(name)
    if value is not None and not isinstance(value, Mapping):
        raise ValueError(f"{name.title()} delivery settings must be an object.")
    return value


def _required_text(settings: Mapping[str, Any], field: str, label: str) -> str:
    """Normalize a required text field."""
    value = str(settings.get(field) or "").strip()
    if not value:
        raise ValueError(f"{label} is required.")
    return value


def build_delivery_configuration(
    delivery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize worker-safe delivery settings for every channel."""
    if delivery is not None and not isinstance(delivery, Mapping):
        raise ValueError("Delivery settings must be an object.")
    raw = delivery or {}
    unknown = sorted(set(raw) - set(DELIVERY_CHANNELS))
    if unknown:
        raise ValueError(f"Unsupported delivery channel: {unknown[0]}")

    configured: dict[str, Any] = {}
    if email := _channel(raw, "email"):
        recipients = normalize_email_recipients(email.get("to", ""))
        if not recipients:
            raise ValueError("Enter at least one email To recipient.")
        formats = list(
            dict.fromkeys(
                str(item).lower() for item in email.get("attachment_formats", ())
            )
        )
        allowed_formats = set(
            DELIVERY_CHANNEL_SETTINGS["email"]["fields"]["attachment_formats"][
                "options"
            ]
        )
        if invalid := sorted(set(formats) - allowed_formats):
            raise ValueError(f"Invalid attachment format: {invalid[0]}")
        configured["email"] = {
            "to": recipients,
            "cc": normalize_email_recipients(email.get("cc", "")),
            "bcc": normalize_email_recipients(email.get("bcc", "")),
            "subject": _required_text(email, "subject", "Email Subject"),
            "attachment_formats": formats,
        }
    elif "email" in raw:
        raise ValueError("Enter at least one email To recipient.")

    if slack := _channel(raw, "slack"):
        configured["slack"] = {"id": _required_text(slack, "id", "Slack ID")}
    elif "slack" in raw:
        raise ValueError("Slack ID is required.")

    if telegram := _channel(raw, "telegram"):
        user_id = str(telegram.get("user_id") or "").strip()
        channel_id = str(telegram.get("channel_id") or "").strip()
        if not (user_id or channel_id):
            raise ValueError("Enter a Telegram User ID or Channel ID.")
        configured["telegram"] = {"user_id": user_id, "channel_id": channel_id}
    elif "telegram" in raw:
        raise ValueError("Enter a Telegram User ID or Channel ID.")

    if whatsapp := _channel(raw, "whatsapp"):
        recipient = next(
            (
                str(whatsapp.get(key) or "").strip()
                for key in (
                    "group_or_user_name",
                    "group_name",
                    "user_name",
                    "phone_number",
                )
                if whatsapp.get(key)
            ),
            "",
        )
        if not recipient:
            raise ValueError("WhatsApp Group/User Name is required.")
        configured["whatsapp"] = {"group_or_user_name": recipient}
    elif "whatsapp" in raw:
        raise ValueError("WhatsApp Group/User Name is required.")
    return configured


def build_retry_configuration(
    *,
    maximum_attempts: int = int(RETRY_DEFAULTS["maximum_attempts"]),
    retry_interval_seconds: int = int(RETRY_DEFAULTS["retry_interval_seconds"]),
    timeout_seconds: int = int(RETRY_DEFAULTS["timeout_seconds"]),
    notify_on_failure: bool = bool(RETRY_DEFAULTS["notify_on_failure"]),
) -> dict[str, Any]:
    """Build validated retry and timeout settings using YAML-backed defaults."""
    values = {
        "maximum_attempts": maximum_attempts,
        "retry_interval_seconds": retry_interval_seconds,
        "timeout_seconds": timeout_seconds,
    }
    for field, value in values.items():
        minimum = int(RETRY_FIELDS[field].get("minimum", 1))
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"{field.replace('_', ' ').capitalize()} must be a positive integer."
            )
    return {**values, "notify_on_failure": bool(notify_on_failure)}


def _cron_part_is_valid(part: str, minimum: int, maximum: int) -> bool:
    """Validate comma, range, wildcard, and step syntax for one cron field."""

    def valid_number(value: str) -> bool:
        return value.isdigit() and minimum <= int(value) <= maximum

    for item in part.split(","):
        base, separator, step = item.partition("/")
        if (
            not item
            or item.count("/") > 1
            or (separator and (not step.isdigit() or int(step) < 1))
        ):
            return False
        if base == "*":
            continue
        if base.count("-") == 1:
            start, end = base.split("-", 1)
            if not (
                valid_number(start) and valid_number(end) and int(start) <= int(end)
            ):
                return False
        elif not valid_number(base):
            return False
    return True


def cron_validation_error(expression: str) -> str | None:
    """Return a concise error for an unsupported or malformed five-field cron."""
    fields = str(expression or "").strip().split()
    if len(fields) != len(_CRON_FIELDS):
        return f"expected exactly {len(_CRON_FIELDS)} fields."
    for value, settings in zip(fields, _CRON_FIELDS, strict=True):
        if not _cron_part_is_valid(value, settings["minimum"], settings["maximum"]):
            return f"invalid {settings['label']} field '{value}'."
    return None


def validate_cron_expression(expression: str) -> bool:
    """Return whether *expression* is a supported standard five-field cron."""
    return cron_validation_error(expression) is None


def _time_text(value: time | str | None) -> str:
    """Normalize a time object or HH:MM string."""
    if isinstance(value, time):
        return value.strftime("%H:%M")
    try:
        return datetime.strptime(str(value or "").strip(), "%H:%M").strftime("%H:%M")
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
    """Build a normalized schedule for any YAML-configured frequency."""
    if not str(frequency or "").strip():
        raise ValueError("Frequency is required.")
    if frequency not in FREQUENCY_SETTINGS:
        raise ValueError(f"Unsupported frequency: {frequency}")

    schedule: dict[str, Any] = {"frequency": frequency, "timezone": LOCAL_TIMEZONE_NAME}
    if frequency == _HOURLY:
        if (
            isinstance(execution_minute, bool)
            or not isinstance(execution_minute, int)
            or not 0 <= execution_minute <= 59
        ):
            raise ValueError("Execution minute must be an integer from 0 to 59.")
        schedule["execution_minute"] = execution_minute
    elif frequency == _DAILY:
        schedule["execution_time"] = _time_text(execution_time)
    elif frequency == _SHIFT:
        if not SHIFT_WINDOWS or not SHIFT_LABELS:
            raise ValueError("No production shifts are configured.")
        schedule["shifts"] = copy.deepcopy(list(SHIFT_WINDOWS))
    elif frequency == _WEEKLY:
        if weekday not in WEEKDAYS:
            raise ValueError("Weekday is required.")
        schedule.update(weekday=weekday, execution_time=_time_text(execution_time))
    else:
        cron = str(cron_expression or "").strip()
        if error := cron_validation_error(cron):
            raise ValueError(f"Invalid cron expression: {error}")
        schedule["cron"] = cron
    return schedule


def validate_schedule(schedule: Mapping[str, Any]) -> None:
    """Validate stored schedule semantics with the creation rules."""
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
    if frequency == _SHIFT and schedule.get("shifts") != rebuilt["shifts"]:
        raise ValueError(
            "Every Shift must use the configured production shift windows."
        )


def calculate_next_run(
    schedule: Mapping[str, Any], now: datetime | None = None
) -> str | None:
    """Calculate the next local run; custom cron evaluation remains worker-owned."""
    validate_schedule(schedule)
    current, frequency = _now(now), schedule["frequency"]
    if frequency == _CRON:
        return None
    if frequency == _HOURLY:
        candidate = current.replace(
            minute=int(schedule["execution_minute"]), second=0, microsecond=0
        )
        return _iso(candidate + timedelta(hours=candidate <= current))
    if frequency in {_DAILY, _WEEKLY}:
        run_time = datetime.strptime(str(schedule["execution_time"]), "%H:%M").time()
        candidate_date = current.date()
        if frequency == _WEEKLY:
            offset = WEEKDAYS.index(str(schedule["weekday"])) - current.weekday()
            candidate_date += timedelta(days=offset % 7)
        candidate = LOCAL_TIMEZONE.localize(datetime.combine(candidate_date, run_time))
        if candidate <= current:
            candidate += timedelta(days=7 if frequency == _WEEKLY else 1)
        return _iso(candidate)

    candidates = [
        LOCAL_TIMEZONE.localize(
            datetime.combine(
                current.date() + timedelta(days=offset), time(int(window["start_hour"]))
            )
        )
        for offset in (0, 1)
        for window in SHIFT_WINDOWS
    ]
    return _iso(min(candidate for candidate in candidates if candidate > current))


def validate_required_fields(
    *, job_name: str, instructions: str, frequency: str
) -> list[str]:
    """Return field-specific messages for missing required editor values."""
    fields = {
        "Job Name is required.": job_name,
        "Job Instructions are required.": instructions,
        "Frequency is required.": frequency,
    }
    return [
        message for message, value in fields.items() if not str(value or "").strip()
    ]


def _normalized_key(value: Any) -> str:
    """Convert a mapping key to normalized snake case for security checks."""
    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value).strip())
    return re.sub(r"[^a-z0-9]+", "_", snake_case.lower()).strip("_")


def find_sensitive_keys(value: Any, path: str = "$") -> list[str]:
    """Recursively locate credential-like keys in a JSON-compatible value."""
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized, child_path = _normalized_key(key), f"{path}.{key}"
            if (
                normalized in _SENSITIVE_KEYS
                or normalized.replace("_", "") in _SENSITIVE_KEYS_COLLAPSED
                or normalized.endswith(_SENSITIVE_SUFFIXES)
            ):
                findings.append(child_path)
            findings.extend(find_sensitive_keys(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(find_sensitive_keys(child, f"{path}[{index}]"))
    return findings


def assert_no_sensitive_keys(value: Any) -> None:
    """Reject secrets and credential fields before preview or persistence."""
    if findings := find_sensitive_keys(value):
        raise ValueError(f"Sensitive field is not allowed: {findings[0]}")


def _assert_report_value_is_metadata(value: Any, path: str) -> None:
    """Reject embedded binary report content while allowing paths and metadata."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized, child_path = _normalized_key(key), f"{path}.{key}"
            if (
                "base64" in normalized
                or normalized == "binary"
                or normalized.endswith("_bytes")
                or normalized in {"image_data", "file_content"}
            ):
                raise ValueError(
                    f"Report history may contain metadata or file paths only: {child_path}"
                )
            _assert_report_value_is_metadata(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_report_value_is_metadata(child, f"{path}[{index}]")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"Report history may not contain binary file content: {path}")
    elif isinstance(value, str) and value.strip().lower().startswith("data:image/"):
        raise ValueError(
            f"Report history may contain metadata or file paths only: {path}"
        )


def validate_job_document(job: Mapping[str, Any]) -> None:
    """Validate security, schedule semantics, delivery, and the JSON Schema."""
    assert_no_sensitive_keys(job)
    _assert_report_value_is_metadata(job.get("report_history", []), "$.report_history")
    _JOB_VALIDATOR.validate(job)
    validate_schedule(job["schedule"])
    build_delivery_configuration(job["delivery"])


def build_job_document(
    *,
    job_name: str,
    instructions: str,
    model_level: str,
    status: str,
    schedule: Mapping[str, Any],
    delivery: Mapping[str, Any],
    retry: Mapping[str, Any],
    created_by: str,
    job_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build and validate revision one of a complete scheduled-job document."""
    errors = validate_required_fields(
        job_name=job_name,
        instructions=instructions,
        frequency=str(schedule.get("frequency") or ""),
    )
    if errors:
        raise ValueError(errors[0])
    if status not in {"draft", "active"}:
        raise ValueError("New job status must be draft or active.")
    actor, level = str(created_by or "").strip(), str(model_level or "").strip().lower()
    if not actor:
        raise ValueError("Created By is required.")
    if level not in MODEL_LEVELS:
        raise ValueError(f"Model Level must be {', '.join(MODEL_LEVELS)}.")

    timestamp = _now(now)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job_id": str(job_id),
        "revision": 1,
        "job_name": str(job_name).strip(),
        "instructions": str(instructions).strip(),
        "model_level": level,
        "status": status,
        "created_at": _iso(timestamp),
        "updated_at": _iso(timestamp),
        "created_by": actor,
        "timezone": LOCAL_TIMEZONE_NAME,
        "schedule": copy.deepcopy(dict(schedule)),
        "delivery": build_delivery_configuration(delivery),
        "last_run": None,
        "next_run": calculate_next_run(schedule, timestamp),
        **{field: [] for field in _HISTORY_FIELDS},
        "retry": copy.deepcopy(dict(retry)),
    }
    validate_job_document(document)
    return document


def configuration_snapshot(job: Mapping[str, Any]) -> dict[str, Any]:
    """Copy editable configuration fields without operational histories."""
    return {field: copy.deepcopy(job[field]) for field in _CONFIGURATION_FIELDS}


def revise_job(
    job: Mapping[str, Any],
    configuration_updates: Mapping[str, Any],
    *,
    saved_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create the next revision while preserving an immutable prior snapshot."""
    if unsupported := sorted(set(configuration_updates) - set(_CONFIGURATION_FIELDS)):
        raise ValueError(f"Not an editable configuration field: {unsupported[0]}")
    revised, timestamp = copy.deepcopy(dict(job)), _now(now)
    revised["revision_history"].append(
        {
            "revision": revised["revision"],
            "saved_at": _iso(timestamp),
            "saved_by": str(saved_by).strip(),
            "configuration": configuration_snapshot(job),
        }
    )
    revised.update(copy.deepcopy(dict(configuration_updates)))
    revised["revision"] += 1
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
    """Copy a job and append a timestamped audit action."""
    updated, timestamp = copy.deepcopy(dict(job)), _now(now)
    updated["updated_at"] = _iso(timestamp)
    updated["action_history"].append(
        {
            "action": action,
            "performed_at": _iso(timestamp),
            "performed_by": str(actor).strip(),
        }
    )
    return updated, timestamp


def _change_status(
    job: Mapping[str, Any], status: str, action: str, actor: str, now: datetime | None
) -> dict[str, Any]:
    """Apply an audited status transition without changing the revision."""
    updated, timestamp = _append_action(job, action, actor, now)
    updated["status"] = status
    updated["next_run"] = (
        calculate_next_run(updated["schedule"], timestamp)
        if status == "active"
        else None
    )
    validate_job_document(updated)
    return updated


def pause_job(
    job: Mapping[str, Any], *, actor: str, now: datetime | None = None
) -> dict[str, Any]:
    """Pause future execution without changing the configuration revision."""
    if job.get("status") == "archived":
        raise ValueError("Archived jobs cannot be paused.")
    return _change_status(job, "paused", "pause", actor, now)


def resume_job(
    job: Mapping[str, Any], *, actor: str, now: datetime | None = None
) -> dict[str, Any]:
    """Activate or resume scheduling without changing the revision."""
    if job.get("status") == "archived":
        raise ValueError("Archived jobs cannot be resumed.")
    action = "activate" if job.get("status") == "draft" else "resume"
    return _change_status(job, "active", action, actor, now)


def request_run_now(
    job: Mapping[str, Any], *, actor: str, now: datetime | None = None
) -> dict[str, Any]:
    """Queue a manual execution request without running the report in-process."""
    if job.get("status") == "archived":
        raise ValueError("Archived jobs cannot be run.")
    updated, timestamp = _append_action(job, "run_now_requested", actor, now)
    request_id = generate_run_request_id()
    if not _RUN_ID_RE.fullmatch(request_id):
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
    """Archive a job while preserving its configuration and histories."""
    return _change_status(job, "archived", "archive", actor, now)


def clone_job_document(
    job: Mapping[str, Any],
    *,
    created_by: str,
    job_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create an independent revision-one draft from another job."""
    return build_job_document(
        job_name=f"{job['job_name']} (Copy)",
        instructions=job["instructions"],
        model_level=job["model_level"],
        status="draft",
        schedule=job["schedule"],
        delivery=job["delivery"],
        retry=job["retry"],
        created_by=created_by,
        job_id=job_id,
        now=_now(now),
    )
