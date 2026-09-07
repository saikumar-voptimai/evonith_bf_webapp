"""Canonical contract builder and validator for scheduled-task JSON.

This module validates operator inputs, generates a portable definition payload, and
performs lightweight schedule calculations. It intentionally has no Streamlit or
execution dependencies so the resulting logic can be reused by backend workers or
control-plane APIs.

Workflow:
1. UI values are normalized into :class:`ScheduledTaskInput`.
2. ``validate_task_input`` produces operator-facing validation errors.
3. ``build_task_definition`` creates a complete payload and validates it again
   against ``scheduled_job_definition.schema.json``.
4. Serialization helpers produce a stable JSON file and filename for download.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jsonschema import Draft202012Validator, FormatChecker

from utils.scheduled_tasks.scheduled_job_catalog import get_scheduled_job_catalog
from utils.shift_windows import (
    LOCAL_TIMEZONE_NAME,
    SHIFT_WINDOWS,
)
from utils.shift_windows import (
    SHIFT_LABELS as CONFIGURED_SHIFT_LABELS,
)

SCHEMA_VERSION = "scheduled-job-definition/v1"
POLICY_PROFILE = "bf_operator_read_only_v1"
DEFAULT_TIMEZONE = LOCAL_TIMEZONE_NAME

DATA_PERIOD_LABELS: dict[str, str] = {
    "previous_shift": "Previous completed shift",
    "current_shift": "Current shift",
    "previous_day": "Previous calendar day",
    "last_24_hours": "Last 24 hours",
    "custom_lookback": "Custom lookback",
}

_FIXED_DATA_PERIOD_BY_JOB_TYPE: dict[str, str] = {
    "shift_report": "previous_shift",
    "daily_report": "previous_day",
}

OUTPUT_FORMAT_LABELS: dict[str, str] = {
    "operator_summary": "Concise operator summary",
    "detailed_report": "Detailed furnace report",
    "exception_alert": "Exceptions and alerts only",
}

ANALYSIS_LEVEL_LABELS: dict[str, str] = {
    "none": "No AI analysis",
    "low": "Brief analysis",
    "medium": "Standard analysis",
    "high": "Detailed analysis",
}

SCHEDULE_KIND_LABELS: dict[str, str] = {
    "once": "Once",
    "shift_end": "After every shift",
    "hourly": "Every hour",
    "daily": "Every day",
    "weekdays": "Monday to Friday",
    "selected_days": "Selected days",
    "weekly": "Every week",
    "monthly": "Every month",
    "interval_hours": "Every N hours",
}

DELIVERY_CHANNEL_LABELS: dict[str, str] = {
    "in_app": "In-app",
    "email": "Email",
    "whatsapp": "WhatsApp",
    "telegram": "Telegram",
}

EMAIL_ATTACHMENT_LABELS: dict[str, str] = {
    "json": "Report JSON",
    "png": "Trend graph (PNG)",
    "csv": "Source data (CSV)",
}

WEEKDAY_NAMES: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

_WEEKDAY_TO_CRON = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 0,
}
_WEEKDAY_INDEX = {name: index for index, name in enumerate(WEEKDAY_NAMES)}
_SHIFT_END_HOURS = {
    str(window["label"]): int(window["end_hour"]) for window in SHIFT_WINDOWS
}
_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_WHATSAPP_PATTERN = re.compile(r"^\+[1-9][0-9 ()-]{7,20}$")
_TELEGRAM_USERNAME_PATTERN = re.compile(r"^@?[A-Za-z][A-Za-z0-9_]{4,31}$")
_TELEGRAM_CHAT_ID_PATTERN = re.compile(r"^-?[0-9]{5,20}$")
_SAFE_FILENAME_PATTERN = re.compile(r"[^a-z0-9]+")
_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(?:api[\s_-]?key|access[\s_-]?key|account[\s_-]?key|"
        r"client[\s_-]?secret|connection[\s_-]?string|"
        r"smtp[\s_-]?(?:password|credentials?|secret)|"
        r"(?:db|database)[\s_-]?(?:password|credentials?|secret)|password|passwd|"
        r"secret|token)\s*(?::|=|\bis\b)\s*\S+"
    ),
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{12,}|AKIA[0-9A-Z]{16}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{20,}\.)"
    ),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@"),
)


@dataclass(frozen=True, slots=True)
class ScheduledTaskInput:
    """Normalized, immutable input contract collected from the Scheduled Tasks UI."""

    name: str
    instructions: str
    furnace: str
    data_period: str
    output_format: str
    schedule_kind: str
    delivery_channel: str
    job_type: str = "custom_report"
    analysis_level: str = "low"
    data_source: str = "online_process_data"
    target_device_id: str = "bf2-jetson-01"
    target_device_type: str = "jetson"
    timezone_name: str = DEFAULT_TIMEZONE
    run_date: date | None = None
    run_time: time | None = None
    hourly_minute: int = 0
    days_of_week: tuple[str, ...] = ()
    day_of_month: int | None = None
    interval_hours: int | None = None
    shift_labels: tuple[str, ...] = ()
    shift_delay_minutes: int = 0
    custom_lookback_value: int | None = None
    custom_lookback_unit: str | None = None
    eta_co_signal: str = "body_etaco"
    report_duration_minutes: int = 60
    aggregation_interval: str = "1min"
    warning_threshold: float = 42.0
    critical_threshold: float = 40.0
    include_graph: bool = True
    include_ai_summary: bool = True
    delivery_destination: str = ""
    email_recipients: tuple[str, ...] = ()
    email_subject: str = ""
    email_attachments: tuple[str, ...] = ()
    notify_on_failure: bool = True
    maximum_attempts: int = 3
    retry_interval_seconds: int = 60
    timeout_seconds: int = 600


class ScheduledTaskValidationError(ValueError):
    """Raised when operator inputs cannot produce a safe job definition."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        """Store individual validation errors and create a combined message."""

        self.errors = errors
        super().__init__(" ".join(errors))


def fixed_data_period_for_job_type(job_type: str) -> str | None:
    """Return the enforced data period for task types with a fixed scope."""

    return _FIXED_DATA_PERIOD_BY_JOB_TYPE.get(job_type)


def _timezone(timezone_name: str) -> ZoneInfo:
    """Resolve an IANA timezone name or raise a clear configuration error."""

    try:
        return ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc


def _now_local(now: datetime | None, timezone_name: str) -> datetime:
    """Return the reference time expressed in the selected plant timezone."""

    tz = _timezone(timezone_name)
    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def _local_datetime(
    selected_date: date,
    selected_time: time,
    timezone_name: str,
) -> datetime:
    """Combine local date and time values in the selected timezone."""

    clean_time = selected_time.replace(second=0, microsecond=0, tzinfo=None)
    return datetime.combine(selected_date, clean_time).replace(
        tzinfo=_timezone(timezone_name)
    )


def parse_email_recipients(raw_value: str) -> tuple[str, ...]:
    """Split an operator-entered email list while preserving invalid entries."""

    parts = re.split(r"[,;\n]", raw_value)
    return tuple(part.strip() for part in parts if part.strip())


def _normalized_email_recipients(recipients: tuple[str, ...]) -> tuple[str, ...]:
    """Trim, lowercase, and deduplicate email recipients in input order."""

    result: list[str] = []
    seen: set[str] = set()
    for recipient in recipients:
        normalized = recipient.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def _contains_secret(value: str) -> bool:
    """Return whether text resembles a credential or embedded secret."""

    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def _string_values(value: object) -> tuple[str, ...]:
    """Recursively collect strings from a JSON-compatible value."""

    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(
            text
            for key, item in value.items()
            for text in (*_string_values(key), *_string_values(item))
        )
    if isinstance(value, (list, tuple)):
        return tuple(text for item in value for text in _string_values(item))
    return ()


def _schedule_validation_errors(
    task: ScheduledTaskInput,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Return operator-facing errors for frequency-specific schedule fields."""

    errors: list[str] = []
    if task.schedule_kind not in SCHEDULE_KIND_LABELS:
        return ["Choose a supported repeat option."]

    catalog = get_scheduled_job_catalog()
    if catalog.timezone(task.timezone_name) is None:
        return ["Choose a configured timezone."]

    try:
        current = _now_local(now, task.timezone_name)
    except ValueError:
        return ["The selected timezone is not recognized."]

    timed_kinds = {
        "once",
        "daily",
        "weekdays",
        "selected_days",
        "weekly",
        "monthly",
        "interval_hours",
    }
    if task.schedule_kind in timed_kinds and task.run_time is None:
        errors.append("Choose the time when the task should run.")

    if task.schedule_kind == "hourly" and not 0 <= task.hourly_minute <= 59:
        errors.append("Choose a minute past each hour from 0 to 59.")

    if task.schedule_kind == "once":
        if task.run_date is None:
            errors.append("Choose the date for the one-time task.")
        elif task.run_time is not None:
            run_at = _local_datetime(task.run_date, task.run_time, task.timezone_name)
            if run_at <= current:
                errors.append("The one-time run must be in the future.")

    if task.schedule_kind == "selected_days":
        if not task.days_of_week:
            errors.append("Choose at least one day of the week.")
        elif any(day not in WEEKDAY_NAMES for day in task.days_of_week):
            errors.append("One or more selected weekdays are not supported.")

    if task.schedule_kind == "weekly":
        if len(task.days_of_week) != 1:
            errors.append("Choose one weekday for the weekly task.")
        elif task.days_of_week[0] not in WEEKDAY_NAMES:
            errors.append("Choose a supported weekday for the weekly task.")

    if task.schedule_kind == "monthly":
        if task.day_of_month is None or not 1 <= task.day_of_month <= 28:
            errors.append("Choose a day of the month from 1 to 28.")

    if task.schedule_kind == "interval_hours":
        if task.interval_hours is None or not 1 <= task.interval_hours <= 168:
            errors.append("Enter an interval from 1 to 168 hours.")
        if task.run_date is None:
            errors.append("Choose when the repeating interval should start.")

    if task.schedule_kind == "shift_end":
        if not task.shift_labels:
            errors.append("Choose at least one shift.")
        elif any(label not in CONFIGURED_SHIFT_LABELS for label in task.shift_labels):
            errors.append("One or more selected shifts are not configured for BF2.")
        elif len(set(task.shift_labels)) != len(task.shift_labels):
            errors.append("Choose each shift only once.")
        if not 0 <= task.shift_delay_minutes <= 180:
            errors.append("The after-shift delay must be from 0 to 180 minutes.")

    return errors


def validate_task_input(
    task: ScheduledTaskInput,
    *,
    now: datetime | None = None,
) -> tuple[str, ...]:
    """Validate untrusted form values and return operator-facing errors.

    ``now`` is the optional reference time used to reject past one-time runs.
    Naive values are interpreted in the task timezone.
    """

    errors: list[str] = []
    name = task.name.strip()
    instructions = task.instructions.strip()
    catalog = get_scheduled_job_catalog()

    if not name:
        errors.append("Enter a task name.")
    elif len(name) > 80:
        errors.append("Keep the task name to 80 characters or fewer.")

    if not instructions:
        errors.append("Describe what FurnaceMind should do.")
    elif len(instructions) > 4000:
        errors.append("Keep the instructions to 4,000 characters or fewer.")

    secret_checked_values = [name, instructions]
    if task.delivery_channel == "email":
        secret_checked_values.append(task.email_subject)
    elif task.delivery_channel in {"whatsapp", "telegram"}:
        secret_checked_values.append(task.delivery_destination)
    if _contains_secret("\n".join(secret_checked_values)):
        errors.append(
            "Remove passwords, API keys, tokens, or credentials. "
            "Scheduled tasks must not contain secrets."
        )
    job_type = catalog.job_type(task.job_type)
    if job_type is None:
        errors.append("Choose a configured task type.")
    if task.analysis_level not in ANALYSIS_LEVEL_LABELS:
        errors.append("Choose a supported FurnaceMind analysis level.")

    if task.furnace.strip() != "BF2":
        errors.append("This application can schedule tasks for BF2 only.")
    if catalog.data_source(task.data_source) is None:
        errors.append("Choose a configured furnace data source.")
    if task.output_format not in OUTPUT_FORMAT_LABELS:
        errors.append("Choose a supported result format.")

    target = catalog.target_device(task.target_device_id)
    if target is None:
        errors.append("Choose a configured target device.")
    elif task.target_device_type != target.device_type:
        errors.append("The target device type does not match its configured device ID.")

    if job_type is not None and job_type.input_profile == "eta_co":
        signal = catalog.eta_co_signal(task.eta_co_signal)
        aggregation = catalog.aggregation_interval(task.aggregation_interval)
        if signal is None:
            errors.append("Choose the configured ETA CO signal.")
        if not 5 <= task.report_duration_minutes <= 1440:
            errors.append("Choose an ETA CO data window from 5 to 1,440 minutes.")
        if aggregation is None:
            errors.append("Choose a configured ETA CO aggregation interval.")
        elif (
            aggregation.minutes is not None
            and aggregation.minutes > task.report_duration_minutes
        ):
            errors.append("The aggregation interval cannot exceed the data window.")
        if not (
            math.isfinite(task.critical_threshold)
            and math.isfinite(task.warning_threshold)
            and 0 <= task.critical_threshold <= 100
            and 0 <= task.warning_threshold <= 100
        ):
            errors.append("Keep ETA CO thresholds between 0 and 100 percent.")
        elif task.critical_threshold >= task.warning_threshold:
            errors.append(
                "ETA CO critical threshold must be lower than the warning threshold."
            )
        if task.include_ai_summary and task.analysis_level == "none":
            errors.append("Choose an analysis level or turn off the ETA CO AI summary.")
    else:
        fixed_data_period = fixed_data_period_for_job_type(task.job_type)
        if fixed_data_period is not None and task.data_period != fixed_data_period:
            task_type_label = job_type.label if job_type is not None else task.job_type
            errors.append(
                f"{task_type_label} always uses "
                f"{DATA_PERIOD_LABELS[fixed_data_period].lower()}."
            )
        elif task.data_period not in DATA_PERIOD_LABELS:
            errors.append("Choose a supported furnace-data period.")
        elif task.data_period == "custom_lookback":
            value = task.custom_lookback_value
            unit = task.custom_lookback_unit
            if value is None or value < 1:
                errors.append("Enter a custom lookback of at least 1.")
            elif unit == "hours" and value > 720:
                errors.append("Keep an hourly lookback to 720 hours or fewer.")
            elif unit == "days" and value > 90:
                errors.append("Keep a daily lookback to 90 days or fewer.")
            if unit not in {"hours", "days"}:
                errors.append("Choose hours or days for the custom lookback.")

    errors.extend(_schedule_validation_errors(task, now=now))

    destination = task.delivery_destination.strip()
    if task.delivery_channel not in DELIVERY_CHANNEL_LABELS:
        errors.append("Choose a supported delivery channel.")
    elif task.delivery_channel == "email":
        recipients = _normalized_email_recipients(task.email_recipients)
        if not recipients:
            errors.append("Enter at least one email recipient.")
        elif len(recipients) > 20:
            errors.append("Use no more than 20 email recipients.")
        if len(recipients) != len(
            [item for item in task.email_recipients if item.strip()]
        ):
            errors.append("Remove duplicate email recipients.")
        if any(not _EMAIL_PATTERN.fullmatch(item) for item in recipients):
            errors.append("Enter valid email recipients separated by commas.")
        if any(len(item) > 254 for item in recipients):
            errors.append("Keep each email recipient to 254 characters or fewer.")
        subject = task.email_subject.strip()
        if not subject:
            errors.append("Enter an email subject.")
        elif len(subject) > 160:
            errors.append("Keep the email subject to 160 characters or fewer.")
        if len(set(task.email_attachments)) != len(task.email_attachments):
            errors.append("Choose each email attachment only once.")
        if any(item not in EMAIL_ATTACHMENT_LABELS for item in task.email_attachments):
            errors.append("Choose only supported email attachments.")
        if (
            job_type is not None
            and job_type.input_profile == "eta_co"
            and not task.include_graph
            and "png" in task.email_attachments
        ):
            errors.append(
                "Turn on the ETA CO trend graph before attaching a PNG graph."
            )
    elif task.delivery_channel == "whatsapp":
        digits = re.sub(r"\D", "", destination)
        if not destination:
            errors.append("Enter the WhatsApp number that should receive the result.")
        elif not _WHATSAPP_PATTERN.fullmatch(destination) or not 8 <= len(digits) <= 15:
            errors.append(
                "Enter a valid WhatsApp number beginning with + and country code."
            )
    elif task.delivery_channel == "telegram":
        if not destination:
            errors.append("Enter the Telegram username or chat ID.")
        elif not (
            _TELEGRAM_USERNAME_PATTERN.fullmatch(destination)
            or _TELEGRAM_CHAT_ID_PATTERN.fullmatch(destination)
        ):
            errors.append("Enter a valid Telegram username or numeric chat ID.")

    if not 1 <= task.maximum_attempts <= 10:
        errors.append("Choose 1 to 10 maximum attempts.")
    if not 1 <= task.retry_interval_seconds <= 3600:
        errors.append("Choose a retry interval from 1 to 3,600 seconds.")
    if not 60 <= task.timeout_seconds <= 86400:
        errors.append("Choose a timeout from 60 to 86,400 seconds.")

    return tuple(dict.fromkeys(errors))


def _ordered_days(days: tuple[str, ...]) -> tuple[str, ...]:
    """Deduplicate weekday names and return them in calendar order."""

    return tuple(sorted(set(days), key=_WEEKDAY_INDEX.__getitem__))


def _timezone_label(timezone_name: str) -> str:
    """Return the concise timezone label used in schedule descriptions."""

    return "IST" if timezone_name == "Asia/Kolkata" else timezone_name


def describe_schedule(task: ScheduledTaskInput) -> str:
    """Return the operator-facing sentence represented by schedule fields."""

    time_label = task.run_time.strftime("%H:%M") if task.run_time else "--:--"
    zone_label = _timezone_label(task.timezone_name)

    if task.schedule_kind == "once":
        date_label = (
            task.run_date.strftime("%d %b %Y") if task.run_date else "date not set"
        )
        return f"Once on {date_label} at {time_label} {zone_label}"
    if task.schedule_kind == "hourly":
        return f"Every hour at minute {task.hourly_minute:02d} {zone_label}"
    if task.schedule_kind == "daily":
        return f"Every day at {time_label} {zone_label}"
    if task.schedule_kind == "weekdays":
        return f"Monday to Friday at {time_label} {zone_label}"
    if task.schedule_kind in {"selected_days", "weekly"}:
        days = _ordered_days(task.days_of_week)
        day_label = ", ".join(days) if days else "no days selected"
        return f"Every {day_label} at {time_label} {zone_label}"
    if task.schedule_kind == "monthly":
        day_label = task.day_of_month if task.day_of_month is not None else "--"
        return f"Every month on day {day_label} at {time_label} {zone_label}"
    if task.schedule_kind == "interval_hours":
        interval = task.interval_hours if task.interval_hours is not None else "--"
        anchor = task.run_date.strftime("%d %b %Y") if task.run_date else "date not set"
        return f"Every {interval} hours, starting {anchor} at {time_label} {zone_label}"
    if task.schedule_kind == "shift_end":
        labels = ", ".join(task.shift_labels) or "no shifts selected"
        delay = task.shift_delay_minutes
        delay_label = (
            "at shift end" if delay == 0 else f"{delay} minutes after shift end"
        )
        return f"After BF2 shifts {labels}, {delay_label}"
    return "Schedule not configured"


def _cron_expression(task: ScheduledTaskInput) -> str:
    """Compile a validated calendar schedule into a five-field cron value."""

    if task.schedule_kind == "hourly":
        return f"{task.hourly_minute} * * * *"
    if task.run_time is None:
        raise ValueError("A run time is required for this schedule.")
    minute = task.run_time.minute
    hour = task.run_time.hour

    if task.schedule_kind == "daily":
        return f"{minute} {hour} * * *"
    if task.schedule_kind == "weekdays":
        return f"{minute} {hour} * * 1-5"
    if task.schedule_kind in {"selected_days", "weekly"}:
        cron_days = sorted({_WEEKDAY_TO_CRON[day] for day in task.days_of_week})
        return f"{minute} {hour} * * {','.join(str(day) for day in cron_days)}"
    if task.schedule_kind == "monthly":
        return f"{minute} {hour} {task.day_of_month} * *"
    raise ValueError("This schedule does not use cron.")


def _shift_end_cron_expression(task: ScheduledTaskInput) -> str:
    """Compile selected shift ends and delay into a five-field cron value."""

    run_minutes = {
        ((_SHIFT_END_HOURS[label] * 60) + task.shift_delay_minutes) % (24 * 60)
        for label in task.shift_labels
    }
    minutes = {run_minute % 60 for run_minute in run_minutes}
    if len(minutes) != 1:
        raise ValueError("Configured shift endings cannot share one cron trigger.")
    minute = minutes.pop()
    hours = sorted(run_minute // 60 for run_minute in run_minutes)
    return f"{minute} {','.join(str(hour) for hour in hours)} * * *"


def _build_trigger(task: ScheduledTaskInput) -> dict[str, object]:
    """Build the executable trigger object for a validated task."""

    if task.schedule_kind == "once":
        if task.run_date is None or task.run_time is None:
            raise ValueError("A date and time are required for a one-time task.")
        return {
            "type": "once",
            "run_at": _local_datetime(
                task.run_date, task.run_time, task.timezone_name
            ).isoformat(timespec="minutes"),
        }

    if task.schedule_kind in {
        "hourly",
        "daily",
        "weekdays",
        "selected_days",
        "weekly",
        "monthly",
    }:
        return {
            "type": "cron",
            "expression": _cron_expression(task),
        }

    if task.schedule_kind == "interval_hours":
        if (
            task.interval_hours is None
            or task.run_date is None
            or task.run_time is None
        ):
            raise ValueError("An interval, start date, and start time are required.")
        return {
            "type": "interval",
            "interval_seconds": task.interval_hours * 60 * 60,
            "anchor_at": _local_datetime(
                task.run_date, task.run_time, task.timezone_name
            ).isoformat(timespec="minutes"),
        }

    if task.schedule_kind == "shift_end":
        shifts = [
            {
                "label": label,
                "end_time": f"{_SHIFT_END_HOURS[label]:02d}:00",
            }
            for label in task.shift_labels
        ]
        return {
            "type": "cron",
            "expression": _shift_end_cron_expression(task),
            "derived_from": {
                "type": "plant_shift_end",
                "shifts": shifts,
                "delay_minutes": task.shift_delay_minutes,
            },
        }

    raise ValueError("Choose a supported repeat option.")


def _build_inputs(task: ScheduledTaskInput) -> dict[str, object]:
    """Build the task-type-specific data input object."""

    inputs: dict[str, object] = {
        "furnace": "BF2",
        "data_source": task.data_source,
        "output_format": task.output_format,
    }
    catalog = get_scheduled_job_catalog()
    job_type = catalog.job_type(task.job_type)
    if job_type is not None and job_type.input_profile == "eta_co":
        inputs.update(
            {
                "signal": task.eta_co_signal,
                "report_duration_minutes": task.report_duration_minutes,
                "aggregation_interval": task.aggregation_interval,
                "warning_threshold": task.warning_threshold,
                "critical_threshold": task.critical_threshold,
                "include_graph": task.include_graph,
                "include_ai_summary": task.include_ai_summary,
            }
        )
    else:
        inputs.update(
            {
                "data_period": task.data_period,
                "data_period_anchor": "scheduled_for",
            }
        )
        if task.data_period == "custom_lookback":
            inputs["lookback"] = {
                "value": task.custom_lookback_value,
                "unit": task.custom_lookback_unit,
            }
    return inputs


def _build_delivery(task: ScheduledTaskInput) -> dict[str, object]:
    """Build a normalized delivery object for the selected channel."""

    delivery: dict[str, object] = {
        "channel": task.delivery_channel,
        "notify_on_failure": task.notify_on_failure,
    }
    if task.delivery_channel == "in_app":
        delivery["destination_ref"] = "task_owner"
    elif task.delivery_channel == "email":
        delivery.update(
            {
                "recipients": list(_normalized_email_recipients(task.email_recipients)),
                "subject": task.email_subject.strip(),
                "attachments": list(task.email_attachments),
            }
        )
    elif task.delivery_channel == "whatsapp":
        delivery["destination"] = f"+{re.sub(r'\D', '', task.delivery_destination)}"
    else:
        delivery["destination"] = task.delivery_destination.strip()
    return delivery


def build_task_definition(
    task: ScheduledTaskInput,
    *,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Build and validate a portable scheduled-task definition.

    ``generated_at`` is used only as the validation clock; it is not written to
    the generated JSON. Invalid form values raise
    :class:`ScheduledTaskValidationError`, while an internal contract mismatch
    raises :class:`ValueError`.
    """

    current = generated_at or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("generated_at must include a timezone.")

    errors = validate_task_input(task, now=current)
    if errors:
        raise ScheduledTaskValidationError(errors)

    catalog = get_scheduled_job_catalog()
    reliability = catalog.defaults["reliability"]
    definition: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "job_name": task.name.strip(),
        "instructions": task.instructions.strip(),
        "job_type": task.job_type,
        "analysis_level": task.analysis_level,
        "schedule": {
            "frequency": task.schedule_kind,
            "timezone": task.timezone_name,
            "overlap_policy": reliability["overlap_policy"],
            "misfire_policy": reliability["misfire_policy"],
            "trigger": _build_trigger(task),
        },
        "target_device": {
            "device_id": task.target_device_id,
            "device_type": task.target_device_type,
        },
        "inputs": _build_inputs(task),
        "delivery": _build_delivery(task),
        "retry": {
            "maximum_attempts": task.maximum_attempts,
            "retry_interval_seconds": task.retry_interval_seconds,
            "timeout_seconds": task.timeout_seconds,
        },
        "policy_profile": POLICY_PROFILE,
    }

    schema_errors = validate_task_definition(definition)
    if schema_errors:
        raise ValueError(
            "Generated job definition does not match its JSON Schema: "
            + " ".join(schema_errors)
        )
    return definition


@lru_cache(maxsize=1)
def _definition_validator() -> Draft202012Validator:
    """Load and cache the Draft 2020-12 validator for task definitions."""

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "scheduled_job_definition.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _cron_number(value: str, minimum: int, maximum: int) -> bool:
    """Return whether a cron field is one integer within inclusive bounds."""

    return value.isascii() and value.isdigit() and minimum <= int(value) <= maximum


def _cron_weekday_list(value: str, *, exactly_one: bool = False) -> bool:
    """Validate a sorted, duplicate-free list of numeric cron weekdays."""

    parts = value.split(",")
    if exactly_one and len(parts) != 1:
        return False
    if not parts or any(not _cron_number(part, 0, 6) for part in parts):
        return False
    numbers = [int(part) for part in parts]
    return numbers == sorted(set(numbers))


def _expected_shift_cron(trigger: dict[object, object]) -> str | None:
    """Rebuild the expected cron expression from a shift derivation."""

    derived = trigger.get("derived_from")
    if not isinstance(derived, dict):
        return None
    shifts = derived.get("shifts")
    delay = derived.get("delay_minutes")
    if (
        not isinstance(shifts, list)
        or not shifts
        or not isinstance(delay, int)
        or isinstance(delay, bool)
        or not 0 <= delay <= 180
    ):
        return None

    run_minutes: set[int] = set()
    for shift in shifts:
        if not isinstance(shift, dict):
            return None
        label = shift.get("label")
        if not isinstance(label, str) or label not in _SHIFT_END_HOURS:
            return None
        expected_end = f"{_SHIFT_END_HOURS[label]:02d}:00"
        if shift.get("end_time") != expected_end:
            return None
        run_minutes.add(((_SHIFT_END_HOURS[label] * 60) + delay) % (24 * 60))

    minutes = {value % 60 for value in run_minutes}
    if len(minutes) != 1:
        return None
    minute = minutes.pop()
    hours = sorted(value // 60 for value in run_minutes)
    return f"{minute} {','.join(str(hour) for hour in hours)} * * *"


def _schedule_definition_semantic_errors(
    definition: dict[str, object],
) -> tuple[str, ...]:
    """Check that a cron trigger agrees with its declared frequency."""

    schedule = definition.get("schedule")
    if not isinstance(schedule, dict):
        return ()
    frequency = schedule.get("frequency")
    trigger = schedule.get("trigger")
    if not isinstance(frequency, str) or not isinstance(trigger, dict):
        return ()
    if trigger.get("type") != "cron":
        return ()
    expression = trigger.get("expression")
    if not isinstance(expression, str):
        return ()

    fields = expression.split()
    matches = False
    if len(fields) == 5:
        minute, hour, day_of_month, month, day_of_week = fields
        clock_is_valid = _cron_number(minute, 0, 59) and _cron_number(hour, 0, 23)
        if frequency == "hourly":
            matches = (
                _cron_number(minute, 0, 59)
                and hour == day_of_month == month == day_of_week == "*"
            )
        elif frequency == "daily":
            matches = clock_is_valid and day_of_month == month == day_of_week == "*"
        elif frequency == "weekdays":
            matches = (
                clock_is_valid and day_of_month == month == "*" and day_of_week == "1-5"
            )
        elif frequency == "selected_days":
            matches = (
                clock_is_valid
                and day_of_month == month == "*"
                and _cron_weekday_list(day_of_week)
            )
        elif frequency == "weekly":
            matches = (
                clock_is_valid
                and day_of_month == month == "*"
                and _cron_weekday_list(day_of_week, exactly_one=True)
            )
        elif frequency == "monthly":
            matches = (
                clock_is_valid
                and _cron_number(day_of_month, 1, 28)
                and month == day_of_week == "*"
            )
        elif frequency == "shift_end":
            matches = expression == _expected_shift_cron(trigger)

    if matches:
        return ()
    return (
        "schedule.trigger.expression: cron expression does not match the "
        "selected frequency",
    )


def _schedule_definition_timezone_errors(
    definition: dict[str, object],
) -> tuple[str, ...]:
    """Check that a definition uses a timezone allowed by the catalog."""

    schedule = definition.get("schedule")
    if not isinstance(schedule, dict):
        return ()
    timezone_name = schedule.get("timezone")
    if not isinstance(timezone_name, str):
        return ()
    if get_scheduled_job_catalog().timezone(timezone_name) is None:
        return ("schedule.timezone: timezone is not configured",)
    return ()


def _task_type_definition_semantic_errors(
    definition: dict[str, object],
) -> tuple[str, ...]:
    """Check fixed data-period rules imposed by particular task types."""

    job_type = definition.get("job_type")
    inputs = definition.get("inputs")
    if not isinstance(job_type, str) or not isinstance(inputs, dict):
        return ()

    fixed_data_period = fixed_data_period_for_job_type(job_type)
    actual_data_period = inputs.get("data_period")
    if (
        fixed_data_period is not None
        and isinstance(actual_data_period, str)
        and actual_data_period != fixed_data_period
    ):
        return (f"inputs.data_period: must be {fixed_data_period} for {job_type}",)
    return ()


def _eta_definition_semantic_errors(
    definition: dict[str, object],
) -> tuple[str, ...]:
    """Check cross-field ETA CO threshold and aggregation rules."""

    if definition.get("job_type") != "eta_co_report":
        return ()
    inputs = definition.get("inputs")
    if not isinstance(inputs, dict):
        return ()

    errors: list[str] = []
    critical = inputs.get("critical_threshold")
    warning = inputs.get("warning_threshold")
    if (
        isinstance(critical, (int, float))
        and not isinstance(critical, bool)
        and isinstance(warning, (int, float))
        and not isinstance(warning, bool)
        and math.isfinite(critical)
        and math.isfinite(warning)
        and critical >= warning
    ):
        errors.append("inputs.critical_threshold: must be lower than warning_threshold")

    aggregation_value = inputs.get("aggregation_interval")
    duration = inputs.get("report_duration_minutes")
    aggregation = (
        get_scheduled_job_catalog().aggregation_interval(aggregation_value)
        if isinstance(aggregation_value, str)
        else None
    )
    if (
        aggregation is not None
        and aggregation.minutes is not None
        and isinstance(duration, int)
        and not isinstance(duration, bool)
        and aggregation.minutes > duration
    ):
        errors.append(
            "inputs.aggregation_interval: cannot exceed report_duration_minutes"
        )
    return tuple(errors)


def _delivery_definition_semantic_errors(
    definition: dict[str, object],
) -> tuple[str, ...]:
    """Check delivery rules that JSON Schema cannot express reliably."""

    delivery = definition.get("delivery")
    if not isinstance(delivery, dict) or delivery.get("channel") != "email":
        return ()
    recipients = delivery.get("recipients")
    if not isinstance(recipients, list) or not all(
        isinstance(recipient, str) for recipient in recipients
    ):
        return ()
    normalized = [recipient.strip().lower() for recipient in recipients]
    if len(normalized) != len(set(normalized)):
        return ("delivery.recipients: duplicate email recipients are not allowed",)
    return ()


def validate_task_definition(definition: object) -> tuple[str, ...]:
    """Return schema, safety, and cross-field errors for a definition."""

    if any(_contains_secret(value) for value in _string_values(definition)):
        return ("$: scheduled tasks must not contain credentials",)

    errors: list[str] = []
    for error in sorted(
        _definition_validator().iter_errors(definition),
        key=lambda item: (
            tuple(str(part) for part in item.absolute_path),
            item.message,
        ),
    ):
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{path}: {error.message}")
    if not isinstance(definition, dict):
        return tuple(dict.fromkeys(errors))
    errors.extend(_schedule_definition_timezone_errors(definition))
    errors.extend(_schedule_definition_semantic_errors(definition))
    errors.extend(_task_type_definition_semantic_errors(definition))
    errors.extend(_eta_definition_semantic_errors(definition))
    errors.extend(_delivery_definition_semantic_errors(definition))
    return tuple(dict.fromkeys(errors))


def task_definition_json(definition: dict[str, object]) -> str:
    """Serialize a definition as stable, human-readable JSON."""

    return json.dumps(definition, indent=2, ensure_ascii=False) + "\n"


def task_definition_filename(task_name: str) -> str:
    """Return a traversal-safe download filename derived from a job name."""

    ascii_name = (
        unicodedata.normalize("NFKD", task_name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = _SAFE_FILENAME_PATTERN.sub("-", ascii_name).strip("-")[:64]
    return f"{slug or 'scheduled-job'}.scheduled-job.json"


def task_input_fingerprint(task: ScheduledTaskInput) -> str:
    """Return a stable fingerprint used to detect a stale generated preview."""

    source = json.dumps(asdict(task), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def upcoming_run_times(
    task: ScheduledTaskInput,
    *,
    count: int = 3,
    now: datetime | None = None,
) -> tuple[datetime, ...]:
    """Return upcoming local occurrences for a valid schedule.

    Invalid schedules and counts below one return an empty tuple. A naive
    ``now`` value is interpreted in the task timezone.
    """

    if count < 1:
        return ()
    schedule_errors = _schedule_validation_errors(task, now=now)
    if schedule_errors:
        return ()

    current = _now_local(now, task.timezone_name)
    results: list[datetime] = []

    if task.schedule_kind == "once":
        assert task.run_date is not None and task.run_time is not None
        candidate = _local_datetime(task.run_date, task.run_time, task.timezone_name)
        return (candidate,) if candidate > current else ()

    if task.schedule_kind == "hourly":
        candidate = current.replace(
            minute=task.hourly_minute,
            second=0,
            microsecond=0,
        )
        if candidate <= current:
            candidate += timedelta(hours=1)
        return tuple(candidate + timedelta(hours=index) for index in range(count))

    if task.schedule_kind == "interval_hours":
        assert task.run_date is not None and task.run_time is not None
        assert task.interval_hours is not None
        anchor = _local_datetime(task.run_date, task.run_time, task.timezone_name)
        interval = timedelta(hours=task.interval_hours)
        if anchor <= current:
            elapsed = (current - anchor).total_seconds()
            steps = math.floor(elapsed / interval.total_seconds()) + 1
            anchor += interval * steps
        return tuple(anchor + (interval * index) for index in range(count))

    if task.schedule_kind == "shift_end":
        offset = 0
        while len(results) < count:
            candidate_date = current.date() + timedelta(days=offset)
            for label in task.shift_labels:
                end_hour = _SHIFT_END_HOURS[label]
                candidate = _local_datetime(
                    candidate_date,
                    time(end_hour),
                    task.timezone_name,
                ) + timedelta(minutes=task.shift_delay_minutes)
                if candidate > current:
                    results.append(candidate)
            offset += 1
        return tuple(sorted(results)[:count])

    assert task.run_time is not None
    if task.schedule_kind == "monthly":
        assert task.day_of_month is not None
        year = current.year
        month = current.month
        while len(results) < count:
            candidate = _local_datetime(
                date(year, month, task.day_of_month),
                task.run_time,
                task.timezone_name,
            )
            if candidate > current:
                results.append(candidate)
            month += 1
            if month == 13:
                month = 1
                year += 1
        return tuple(results)

    allowed_weekdays: set[int] | None = None
    if task.schedule_kind == "weekdays":
        allowed_weekdays = {0, 1, 2, 3, 4}
    elif task.schedule_kind in {"selected_days", "weekly"}:
        allowed_weekdays = {_WEEKDAY_INDEX[day] for day in task.days_of_week}

    offset = 0
    while len(results) < count:
        candidate_date = current.date() + timedelta(days=offset)
        offset += 1
        if (
            allowed_weekdays is not None
            and candidate_date.weekday() not in allowed_weekdays
        ):
            continue
        candidate = _local_datetime(
            candidate_date,
            task.run_time,
            task.timezone_name,
        )
        if candidate > current:
            results.append(candidate)

    return tuple(results)


__all__ = [
    "ANALYSIS_LEVEL_LABELS",
    "DATA_PERIOD_LABELS",
    "DEFAULT_TIMEZONE",
    "DELIVERY_CHANNEL_LABELS",
    "EMAIL_ATTACHMENT_LABELS",
    "OUTPUT_FORMAT_LABELS",
    "POLICY_PROFILE",
    "SCHEMA_VERSION",
    "SCHEDULE_KIND_LABELS",
    "WEEKDAY_NAMES",
    "ScheduledTaskInput",
    "ScheduledTaskValidationError",
    "build_task_definition",
    "describe_schedule",
    "fixed_data_period_for_job_type",
    "parse_email_recipients",
    "task_definition_filename",
    "task_definition_json",
    "task_input_fingerprint",
    "upcoming_run_times",
    "validate_task_definition",
    "validate_task_input",
]
