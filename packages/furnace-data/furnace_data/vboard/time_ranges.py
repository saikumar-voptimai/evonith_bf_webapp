"""Shared V-Board time-range resolution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from furnace_data.vboard.catalog import preset_by_id
from furnace_data.vboard.models import ResolvedRange


class VBoardTimeRangeError(ValueError):
    """Validation error carrying a stable API-facing code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def parse_aware_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise VBoardTimeRangeError(
            "INVALID_TIME_RANGE",
            "Absolute timestamps must include a timezone offset.",
        )
    return parsed


def resolve_time_range(
    request: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    max_absolute_range_days: int = 31,
    clock_skew_seconds: int = 300,
) -> ResolvedRange:
    """Resolve a typed V-Board time-range request to UTC datetimes."""

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    kind = str(request.get("kind") or "").strip().lower()
    if kind == "preset":
        preset_id = str(request.get("preset_id") or "").strip()
        preset = preset_by_id().get(preset_id)
        if preset is None:
            raise VBoardTimeRangeError("INVALID_TIME_PRESET", "Unknown V-Board time preset.")
        duration = timedelta(seconds=int(preset["duration_seconds"]))
        end = now
        start = end - duration
        return ResolvedRange(start=start, end=end, requested_kind="preset", preset_id=preset_id)

    if kind == "absolute":
        start = parse_aware_datetime(request.get("start")).astimezone(timezone.utc)
        end = parse_aware_datetime(request.get("end")).astimezone(timezone.utc)
        if start >= end:
            raise VBoardTimeRangeError("INVALID_TIME_RANGE", "Start must be before end.")
        max_duration = timedelta(days=max(1, int(max_absolute_range_days)))
        if end - start > max_duration:
            raise VBoardTimeRangeError(
                "VBOARD_RANGE_TOO_LARGE",
                "The requested V-Board range exceeds the configured limit.",
            )
        if end > now + timedelta(seconds=max(0, int(clock_skew_seconds))):
            raise VBoardTimeRangeError(
                "INVALID_TIME_RANGE",
                "The requested V-Board range ends too far in the future.",
            )
        return ResolvedRange(start=start, end=end, requested_kind="absolute")

    raise VBoardTimeRangeError("INVALID_TIME_RANGE", "Unknown V-Board time range kind.")
