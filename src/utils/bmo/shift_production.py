"""Current daily hot-metal production from the last complete plant shift.

The plant reports production as charges per hour x hot metal per charge: on the
live ``process_params`` tags ``production_per_hour`` equals that product on
every hour. So the last shift's charging rate at its HM per charge, times 24,
is the plant's own current daily production, and the natural baseline against
which an optimizer blend's production is judged.

Shifts are the plant's fixed 8-hour windows: A 06:00-14:00, B 14:00-22:00 and
C 22:00-06:00 IST. Only a COMPLETE shift is used, so a shift that has just
started cannot read as a low-production day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

PLANT_TZ = "Asia/Kolkata"
SHIFT_HOURS = 8
_SHIFT_LABELS = {6: "A", 14: "B", 22: "C"}
# A shift with fewer usable hours than this is not a shift-level figure.
MIN_SHIFT_HOURS = 6.0


@dataclass(frozen=True)
class ShiftProduction:
    """Production rate of one complete shift.

    Attributes:
        shift_label: Plant shift letter, A, B or C.
        start_ist: Shift start (IST, inclusive).
        end_ist: Shift end (IST, exclusive).
        charges_per_hour: Mean charging rate over the shift.
        hm_per_charge_mt: Hot metal per charge over the shift.
        daily_production_mt: Mean hourly production x 24.
        hours: Usable hours of data inside the shift.
        source: Where the figures came from.
    """

    shift_label: str
    start_ist: pd.Timestamp
    end_ist: pd.Timestamp
    charges_per_hour: float | None
    hm_per_charge_mt: float | None
    daily_production_mt: float | None
    hours: float
    source: str

    @property
    def usable(self) -> bool:
        return self.daily_production_mt is not None and self.hours >= MIN_SHIFT_HOURS

    def describe(self) -> str:
        """'Shift B, 29 Sep 14:00-22:00 IST'."""

        return (
            f"Shift {self.shift_label}, {self.start_ist:%d %b} "
            f"{self.start_ist:%H:%M}-{self.end_ist:%H:%M} IST"
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["start_ist"] = self.start_ist.isoformat()
        values["end_ist"] = self.end_ist.isoformat()
        values["usable"] = self.usable
        values["description"] = self.describe()
        return values


def _as_ist(value: datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(PLANT_TZ)


def last_complete_shift(now: datetime | pd.Timestamp) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    """The most recent plant shift that has fully ended by ``now``.

    Args:
        now: Current time; a naive value is taken as UTC.

    Returns:
        (shift letter, start, end), both IST-aware.
    """

    now_ist = _as_ist(now)
    day = now_ist.normalize()
    candidates = [
        day - pd.Timedelta(days=1) + pd.Timedelta(hours=hour)
        for hour in sorted(_SHIFT_LABELS)
    ] + [day + pd.Timedelta(hours=hour) for hour in sorted(_SHIFT_LABELS)]
    finished = [
        start
        for start in candidates
        if start + pd.Timedelta(hours=SHIFT_HOURS) <= now_ist
    ]
    start = max(finished)
    return _SHIFT_LABELS[start.hour], start, start + pd.Timedelta(hours=SHIFT_HOURS)


def shift_production(
    samples: pd.DataFrame,
    *,
    now: datetime | pd.Timestamp,
    charges_col: str = "charges_per_hour",
    hm_col: str = "hm_per_charge",
    timestamps: Literal["start", "stop"] = "stop",
    source: str = "",
) -> ShiftProduction:
    """Daily production of the last complete shift from charging samples.

    Args:
        samples: Time-indexed rows with a charges-per-hour and an HM-per-charge
            column. A naive index is read as UTC.
        now: Current time; picks which shift is the last complete one.
        charges_col: Charges-per-hour column.
        hm_col: Hot-metal-per-charge column (t).
        timestamps: Whether each row is stamped at the start or the end of the
            interval it averages; decides which rows fall inside the shift.
        source: Label carried into the result.

    Returns:
        The shift's charging rate, HM per charge and daily production.
    """

    label, start, end = last_complete_shift(now)
    empty = ShiftProduction(label, start, end, None, None, None, 0.0, source)
    if samples is None or samples.empty or charges_col not in samples or hm_col not in samples:
        return empty
    index = pd.DatetimeIndex(samples.index)
    index = index.tz_localize("UTC") if index.tz is None else index
    index = index.tz_convert(PLANT_TZ)
    inside = (index > start) & (index <= end) if timestamps == "stop" else (
        (index >= start) & (index < end)
    )
    charges = pd.to_numeric(samples[charges_col], errors="coerce").to_numpy()[inside]
    hm = pd.to_numeric(samples[hm_col], errors="coerce").to_numpy()[inside]
    valid = np.isfinite(charges) & np.isfinite(hm) & (charges > 0) & (hm > 0)
    if not valid.any():
        return empty
    spacing = _sample_hours(index)
    hourly_production = charges[valid] * hm[valid]
    return ShiftProduction(
        shift_label=label,
        start_ist=start,
        end_ist=end,
        charges_per_hour=float(np.mean(charges[valid])),
        hm_per_charge_mt=float(np.sum(hourly_production) / np.sum(charges[valid])),
        daily_production_mt=float(np.mean(hourly_production) * 24.0),
        hours=float(valid.sum() * spacing),
        source=source,
    )


def _sample_hours(index: pd.DatetimeIndex) -> float:
    """Hours each row represents, from the typical spacing of the index."""

    if len(index) < 2:
        return 1.0
    step = pd.Series(index).diff().median()
    return float(step / pd.Timedelta(hours=1)) if pd.notna(step) else 1.0


def production_change_pct(expected_mt: Any, baseline_mt: Any) -> float | None:
    """Percent change of an expected daily production over the baseline."""

    try:
        expected, baseline = float(expected_mt), float(baseline_mt)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(expected) and np.isfinite(baseline)) or baseline <= 0:
        return None
    return (expected / baseline - 1.0) * 100.0


__all__ = [
    "MIN_SHIFT_HOURS",
    "PLANT_TZ",
    "ShiftProduction",
    "last_complete_shift",
    "production_change_pct",
    "shift_production",
]
