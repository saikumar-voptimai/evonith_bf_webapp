"""Fixed/custom trend ranges and aggregation-window selection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from data.furnace_status.catalogue import SETTINGS

IST = SETTINGS.timezone
FIXED_INTERVALS = dict(SETTINGS.fixed_intervals)
CUSTOM_INTERVAL = SETTINGS.custom_interval
INTERVAL_OPTIONS = (*FIXED_INTERVALS, CUSTOM_INTERVAL)
DEFAULT_INTERVAL = SETTINGS.default_interval
MAX_CUSTOM_RANGE = SETTINGS.max_custom_range


class RangeError(ValueError):
    """A requested trend range is invalid; its message is UI-safe."""


def floor_to_minute(moment: datetime) -> datetime:
    return moment.replace(second=0, microsecond=0)


def resolve_fixed_range(interval: str, now: datetime) -> tuple[datetime, datetime]:
    if interval not in FIXED_INTERVALS:
        raise RangeError(f"Unknown interval: {interval!r}")
    end = floor_to_minute(now.astimezone(timezone.utc))
    return end - FIXED_INTERVALS[interval], end


def resolve_custom_range(
    start_local: datetime, end_local: datetime, now: datetime
) -> tuple[datetime, datetime]:
    def as_utc(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=IST)
        return moment.astimezone(timezone.utc)

    start = as_utc(start_local)
    end = min(as_utc(end_local), floor_to_minute(now.astimezone(timezone.utc)))
    if start >= end:
        raise RangeError("Start must be earlier than end (and not in the future).")
    if end - start > MAX_CUSTOM_RANGE:
        raise RangeError(
            f"Range is too large; choose at most {MAX_CUSTOM_RANGE.days} days."
        )
    return start, end


def choose_window(duration: timedelta) -> str:
    for item in SETTINGS.aggregation_windows:
        if duration <= item.max_duration:
            return item.window
    return SETTINGS.widest_window


def default_custom_range(now: datetime) -> tuple[datetime, datetime]:
    end = floor_to_minute(now.astimezone(IST))
    return end - FIXED_INTERVALS[DEFAULT_INTERVAL], end
