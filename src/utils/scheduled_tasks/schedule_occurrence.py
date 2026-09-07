"""Resolve the logical occurrence represented by a scheduled-job definition.

Systemd starts the same one-shot service for every timer event but does not pass
an occurrence identifier. This module derives a stable UTC ``scheduled_for``
value from the validated once, interval, or generated cron trigger. The database
uses that value with ``job_id`` to prevent duplicate execution.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from utils.scheduled_tasks.scheduled_task_definition import validate_task_definition

_CRON_LOOKBACK_MINUTES = 32 * 24 * 60


class ScheduledOccurrenceError(ValueError):
    """Raised when a definition has no due occurrence at the reference time."""


class ScheduledOccurrenceNotDueError(ScheduledOccurrenceError):
    """Raised when a valid timer probe has no runnable occurrence yet."""


def _aware_datetime(value: object, *, field_name: str) -> datetime:
    """Parse one ISO timestamp and require an explicit UTC offset."""

    if not isinstance(value, str):
        raise ScheduledOccurrenceError(f"{field_name} must be an ISO timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ScheduledOccurrenceError(
            f"{field_name} must be a valid ISO timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ScheduledOccurrenceError(f"{field_name} must include a UTC offset.")
    return parsed


def _reference_time(value: datetime | None) -> datetime:
    """Return an aware reference time, defaulting to the current UTC instant."""

    reference = value or datetime.now(timezone.utc)
    if reference.tzinfo is None or reference.utcoffset() is None:
        raise ScheduledOccurrenceError("Reference time must include a timezone.")
    return reference


def _cron_values(field: str, minimum: int, maximum: int) -> set[int] | None:
    """Expand a supported cron wildcard, integer, list, or inclusive range."""

    if field == "*":
        return None
    values: set[int] = set()
    for item in field.split(","):
        bounds = item.split("-", 1)
        try:
            if len(bounds) == 1:
                start = end = int(bounds[0])
            else:
                start, end = (int(value) for value in bounds)
        except ValueError as exc:
            raise ScheduledOccurrenceError(
                "Cron trigger contains an invalid field."
            ) from exc
        if start > end or start < minimum or end > maximum:
            raise ScheduledOccurrenceError("Cron trigger contains an invalid range.")
        values.update(range(start, end + 1))
    return values


def _cron_matches(candidate: datetime, expression: str) -> bool:
    """Return whether a local minute matches the supported five-field cron."""

    fields = expression.split()
    if len(fields) != 5:
        raise ScheduledOccurrenceError("Cron trigger must contain five fields.")
    minute_field, hour_field, day_field, month_field, weekday_field = fields
    minutes = _cron_values(minute_field, 0, 59)
    hours = _cron_values(hour_field, 0, 23)
    days = _cron_values(day_field, 1, 31)
    months = _cron_values(month_field, 1, 12)
    weekdays = _cron_values(weekday_field, 0, 6)

    if minutes is not None and candidate.minute not in minutes:
        return False
    if hours is not None and candidate.hour not in hours:
        return False
    if months is not None and candidate.month not in months:
        return False

    day_matches = days is None or candidate.day in days
    cron_weekday = (candidate.weekday() + 1) % 7
    weekday_matches = weekdays is None or cron_weekday in weekdays
    if days is not None and weekdays is not None:
        return day_matches or weekday_matches
    return day_matches and weekday_matches


def _latest_cron_occurrence(
    *,
    expression: str,
    timezone_name: str,
    reference: datetime,
) -> datetime:
    """Search backward to the latest matching generated cron minute."""

    local_candidate = reference.astimezone(ZoneInfo(timezone_name)).replace(
        second=0,
        microsecond=0,
    )
    for _ in range(_CRON_LOOKBACK_MINUTES + 1):
        if _cron_matches(local_candidate, expression):
            return local_candidate.astimezone(timezone.utc)
        local_candidate -= timedelta(minutes=1)
    raise ScheduledOccurrenceError(
        "Cron trigger has no occurrence within the supported lookback window."
    )


def latest_scheduled_occurrence(
    definition: dict[str, object],
    *,
    at: datetime | None = None,
) -> datetime:
    """Return the latest due occurrence as a stable UTC timestamp.

    The definition is validated before schedule fields are trusted. For a
    one-time or interval trigger whose first run is still in the future, the
    function raises :class:`ScheduledOccurrenceNotDueError` instead of
    fabricating an occurrence.
    """

    if not isinstance(definition, dict):
        raise ScheduledOccurrenceError(
            "Scheduled-job definition must be a JSON object."
        )
    errors = validate_task_definition(definition)
    if errors:
        raise ScheduledOccurrenceError(
            "Scheduled-job definition is invalid: " + " ".join(errors)
        )
    reference = _reference_time(at)
    schedule = definition["schedule"]
    assert isinstance(schedule, dict)
    timezone_name = schedule["timezone"]
    trigger = schedule["trigger"]
    assert isinstance(timezone_name, str)
    assert isinstance(trigger, dict)

    trigger_type = trigger["type"]
    if trigger_type == "once":
        occurrence = _aware_datetime(
            trigger.get("run_at"), field_name="schedule.trigger.run_at"
        )
        if occurrence > reference:
            raise ScheduledOccurrenceNotDueError(
                "The one-time occurrence is not due yet."
            )
        return occurrence.astimezone(timezone.utc)

    if trigger_type == "interval":
        anchor = _aware_datetime(
            trigger.get("anchor_at"), field_name="schedule.trigger.anchor_at"
        )
        if anchor > reference:
            raise ScheduledOccurrenceNotDueError(
                "The interval occurrence is not due yet."
            )
        interval_seconds = trigger["interval_seconds"]
        assert isinstance(interval_seconds, int)
        elapsed_seconds = (reference - anchor).total_seconds()
        intervals = int(elapsed_seconds // interval_seconds)
        return (anchor + timedelta(seconds=intervals * interval_seconds)).astimezone(
            timezone.utc
        )

    if trigger_type == "cron":
        expression = trigger["expression"]
        assert isinstance(expression, str)
        return _latest_cron_occurrence(
            expression=expression,
            timezone_name=timezone_name,
            reference=reference,
        )

    raise ScheduledOccurrenceError(f"Unsupported trigger type: {trigger_type}")
