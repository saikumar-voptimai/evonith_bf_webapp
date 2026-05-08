"""Config-backed helpers for production shift windows."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import pytz

from config.config_loader import load_config

_FURNACEMIND_CONFIG = load_config("furnacemind.yml") or {}
LOCAL_TIMEZONE_NAME = str(_FURNACEMIND_CONFIG.get("timezone", "Asia/Kolkata"))
LOCAL_TIMEZONE = pytz.timezone(LOCAL_TIMEZONE_NAME)
SHIFT_WINDOWS = tuple(_FURNACEMIND_CONFIG.get("shift_windows") or ())
SHIFT_LABELS = tuple(str(window["label"]) for window in SHIFT_WINDOWS)
_SHIFT_BY_LABEL = {str(window["label"]): window for window in SHIFT_WINDOWS}


def shift_window_config(shift_label: str) -> dict[str, Any]:
    """
    Return configured shift window metadata by label.

    Args:
         - shift_label: str - Shift label to look up.

    Returns:
         - return: dict[str, Any] - Configured shift window metadata.
    """
    try:
        return _SHIFT_BY_LABEL[shift_label]
    except KeyError as exc:
        valid_labels = ", ".join(SHIFT_LABELS)
        raise ValueError(
            f"Unknown shift label '{shift_label}'. Expected one of: {valid_labels}"
        ) from exc


def shift_window(shift_date: date, shift_label: str) -> tuple[datetime, datetime]:
    """
    Return timezone-aware start and end datetimes for a configured shift.

    Args:
         - shift_date: date - Calendar date assigned to the shift.
         - shift_label: str - Shift label to resolve.

    Returns:
         - return: tuple[datetime, datetime] - Shift start and end datetimes.
    """
    window = shift_window_config(shift_label)
    start_hour = int(window["start_hour"])
    end_hour = int(window["end_hour"])
    end_date = shift_date + timedelta(days=1) if end_hour <= start_hour else shift_date

    start = LOCAL_TIMEZONE.localize(datetime.combine(shift_date, time(start_hour)))
    end = LOCAL_TIMEZONE.localize(datetime.combine(end_date, time(end_hour)))
    return start, end


def shift_window_naive(shift_date: date, shift_label: str) -> tuple[datetime, datetime]:
    """
    Return timezone-naive local start and end datetimes for a configured shift.

    Args:
         - shift_date: date - Calendar date assigned to the shift.
         - shift_label: str - Shift label to resolve.

    Returns:
         - return: tuple[datetime, datetime] - Local shift start and end datetimes.
    """
    start, end = shift_window(shift_date, shift_label)
    return start.replace(tzinfo=None), end.replace(tzinfo=None)


def last_completed_shift(now: datetime | None = None) -> tuple[date, str]:
    """
    Return the date and label for the configured FurnaceMind shift.

    Args:
         - now: datetime | None - Optional current time override.

    Returns:
         - return: tuple[date, str] - Shift date and label.
    """
    if now is None:
        current = datetime.now(LOCAL_TIMEZONE)
    elif now.tzinfo is None:
        current = LOCAL_TIMEZONE.localize(now)
    else:
        current = now.astimezone(LOCAL_TIMEZONE)

    hour = current.hour
    for window in SHIFT_WINDOWS:
        start_hour = int(window["start_hour"])
        end_hour = int(window["end_hour"])
        label = str(window["label"])

        if start_hour < end_hour and start_hour <= hour < end_hour:
            return current.date(), label
        if start_hour > end_hour:
            if hour >= start_hour:
                return current.date(), label
            if hour < end_hour:
                return current.date() - timedelta(days=1), label

    first_shift = SHIFT_WINDOWS[0]
    return current.date(), str(first_shift["label"])


def previous_shift(shift_date: date, shift_label: str) -> tuple[date, str]:
    """
    Return the previous configured shift date and label.

    Args:
         - shift_date: date - Current shift date.
         - shift_label: str - Current shift label.

    Returns:
         - return: tuple[date, str] - Previous shift date and label.
    """
    current_index = SHIFT_LABELS.index(shift_label)
    previous_index = current_index - 1
    previous_label = SHIFT_LABELS[previous_index]
    previous_date = shift_date - timedelta(days=1) if current_index == 0 else shift_date
    return previous_date, previous_label


def shift_windows_description() -> str:
    """
    Return a human-readable description of configured shift windows.

    Args:
         - None

    Returns:
         - return: str - Shift window description for tool metadata.
    """
    parts = []
    for window in SHIFT_WINDOWS:
        start_hour = int(window["start_hour"])
        end_hour = int(window["end_hour"])
        next_day = " next day" if end_hour <= start_hour else ""
        parts.append(
            f"{window['label']} ({start_hour:02d}:00-{end_hour:02d}:00{next_day})"
        )

    timezone_label = (
        "IST" if LOCAL_TIMEZONE_NAME == "Asia/Kolkata" else LOCAL_TIMEZONE_NAME
    )
    return f"Shift: {', '.join(parts)} {timezone_label}"
