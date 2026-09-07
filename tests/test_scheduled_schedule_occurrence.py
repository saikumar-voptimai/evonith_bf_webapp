"""Tests for deterministic scheduled occurrence resolution."""

from __future__ import annotations

from datetime import date, datetime, time, timezone

import pytest

from utils.scheduled_tasks.schedule_occurrence import (
    ScheduledOccurrenceError,
    ScheduledOccurrenceNotDueError,
    latest_scheduled_occurrence,
)
from utils.scheduled_tasks.scheduled_task_definition import (
    ScheduledTaskInput,
    build_task_definition,
)


def _task(**overrides: object) -> ScheduledTaskInput:
    """Build a valid common scheduled-task input with selected overrides."""

    values: dict[str, object] = {
        "name": "BF2 scheduled report",
        "instructions": "Prepare a concise BF2 operator report.",
        "furnace": "BF2",
        "data_period": "last_24_hours",
        "output_format": "operator_summary",
        "schedule_kind": "hourly",
        "delivery_channel": "in_app",
        "job_type": "furnace_summary",
        "hourly_minute": 5,
    }
    values.update(overrides)
    return ScheduledTaskInput(**values)


def _definition(task: ScheduledTaskInput) -> dict[str, object]:
    """Build a definition using a clock before every test schedule."""

    return build_task_definition(
        task,
        generated_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def test_hourly_occurrence_respects_definition_timezone() -> None:
    """Cron minutes should be interpreted in plant local time and returned in UTC."""

    occurrence = latest_scheduled_occurrence(
        _definition(_task()),
        at=datetime(2026, 9, 5, 10, 7, 42, tzinfo=timezone.utc),
    )

    assert occurrence == datetime(2026, 9, 5, 9, 35, tzinfo=timezone.utc)


def test_once_occurrence_returns_the_configured_instant_when_due() -> None:
    """A due one-time trigger should retain its exact configured instant."""

    definition = _definition(
        _task(
            schedule_kind="once",
            run_date=date(2026, 9, 5),
            run_time=time(14, 30),
        )
    )

    occurrence = latest_scheduled_occurrence(
        definition,
        at=datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc),
    )

    assert occurrence == datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc)


def test_once_occurrence_rejects_a_trigger_that_is_not_due() -> None:
    """A runner must not fabricate an occurrence before a one-time trigger."""

    definition = _definition(
        _task(
            schedule_kind="once",
            run_date=date(2026, 9, 5),
            run_time=time(14, 30),
        )
    )

    with pytest.raises(ScheduledOccurrenceNotDueError, match="not due"):
        latest_scheduled_occurrence(
            definition,
            at=datetime(2026, 9, 5, 8, 59, tzinfo=timezone.utc),
        )


def test_interval_probe_before_anchor_is_a_non_failure_not_due_condition() -> None:
    """Early systemd probes should be distinguishable from invalid schedules."""

    definition = _definition(
        _task(
            schedule_kind="interval_hours",
            interval_hours=5,
            run_date=date(2026, 9, 5),
            run_time=time(14, 30),
        )
    )

    with pytest.raises(ScheduledOccurrenceNotDueError, match="not due"):
        latest_scheduled_occurrence(
            definition,
            at=datetime(2026, 9, 5, 8, 59, tzinfo=timezone.utc),
        )


def test_interval_occurrence_uses_the_latest_anchor_multiple() -> None:
    """Missed interval triggers should collapse to the latest due occurrence."""

    definition = _definition(
        _task(
            schedule_kind="interval_hours",
            interval_hours=6,
            run_date=date(2026, 9, 2),
            run_time=time(6, 0),
        )
    )

    occurrence = latest_scheduled_occurrence(
        definition,
        at=datetime(2026, 9, 3, 8, 45, tzinfo=timezone.utc),
    )

    assert occurrence == datetime(2026, 9, 3, 6, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("overrides", "reference", "expected"),
    [
        (
            {"schedule_kind": "daily", "run_time": time(6, 30)},
            datetime(2026, 9, 5, 4, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 5, 1, 0, tzinfo=timezone.utc),
        ),
        (
            {"schedule_kind": "weekdays", "run_time": time(6, 30)},
            datetime(2026, 9, 5, 4, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 4, 1, 0, tzinfo=timezone.utc),
        ),
        (
            {
                "schedule_kind": "selected_days",
                "run_time": time(6, 30),
                "days_of_week": ("Tuesday", "Thursday"),
            },
            datetime(2026, 9, 5, 4, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 3, 1, 0, tzinfo=timezone.utc),
        ),
        (
            {
                "schedule_kind": "weekly",
                "run_time": time(6, 30),
                "days_of_week": ("Monday",),
            },
            datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 7, 1, 0, tzinfo=timezone.utc),
        ),
        (
            {
                "schedule_kind": "monthly",
                "run_time": time(6, 30),
                "day_of_month": 1,
            },
            datetime(2026, 9, 5, 4, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc),
        ),
        (
            {
                "schedule_kind": "shift_end",
                "shift_labels": ("A", "C"),
            },
            datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 5, 8, 30, tzinfo=timezone.utc),
        ),
    ],
)
def test_generated_calendar_schedules_resolve_latest_occurrence(
    overrides: dict[str, object],
    reference: datetime,
    expected: datetime,
) -> None:
    """Every calendar schedule offered by the UI should resolve correctly."""

    occurrence = latest_scheduled_occurrence(
        _definition(_task(**overrides)),
        at=reference,
    )

    assert occurrence == expected


def test_invalid_definition_is_rejected_before_schedule_fields_are_used() -> None:
    """Occurrence resolution must not trust malformed database JSON."""

    definition = _definition(_task())
    definition["schema_version"] = "unsupported"

    with pytest.raises(ScheduledOccurrenceError, match="definition is invalid"):
        latest_scheduled_occurrence(definition)
