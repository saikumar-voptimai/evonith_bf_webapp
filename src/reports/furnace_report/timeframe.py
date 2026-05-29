"""Report timeframe resolution for furnace report generation."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import cast

from reports.furnace_report.data import ReportTimeframe, ReportType, ShiftLabel
from utils.shift_windows import (
    LOCAL_TIMEZONE,
    previous_shift,
    shift_window,
    shift_window_config,
)

REPORT_TYPES: tuple[ReportType, ...] = ("Shift", "Day")


def normalize_report_type(report_type: str | ReportType) -> ReportType:
    value = str(report_type or "").strip().title()
    if value in REPORT_TYPES:
        return cast(ReportType, value)
    valid = ", ".join(REPORT_TYPES)
    raise ValueError(f"Unknown report type '{report_type}'. Expected one of: {valid}")


def _normalize_shift_label(shift_label: str | ShiftLabel | None) -> ShiftLabel:
    if shift_label is None:
        raise ValueError("shift_label is required for Shift reports.")
    label = str(shift_label).strip().upper()
    shift_window_config(label)
    return cast(ShiftLabel, label)


def get_report_timeframe(
    report_type: str | ReportType,
    selected_date: date,
    shift: str | ShiftLabel | None = None,
) -> ReportTimeframe:
    resolved_type = normalize_report_type(report_type)
    if resolved_type == "Shift":
        label = _normalize_shift_label(shift)
        start_ist, end_ist = shift_window(selected_date, label)
        return ReportTimeframe(
            report_type="Shift",
            selected_date=selected_date,
            shift_label=label,
            start_ist=start_ist,
            end_ist=end_ist,
        )

    start_ist = LOCAL_TIMEZONE.localize(datetime.combine(selected_date, time.min))
    end_ist = LOCAL_TIMEZONE.localize(
        datetime.combine(selected_date + timedelta(days=1), time.min)
    )
    return ReportTimeframe(
        report_type="Day",
        selected_date=selected_date,
        start_ist=start_ist,
        end_ist=end_ist,
    )


def previous_report_timeframe(timeframe: ReportTimeframe) -> ReportTimeframe:
    if timeframe.report_type == "Shift":
        prev_date, prev_label = previous_shift(
            timeframe.selected_date,
            _normalize_shift_label(timeframe.shift_label),
        )
        return get_report_timeframe("Shift", prev_date, prev_label)

    return get_report_timeframe(
        timeframe.report_type,
        timeframe.selected_date - timedelta(days=1),
    )
