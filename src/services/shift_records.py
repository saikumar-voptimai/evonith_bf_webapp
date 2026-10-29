"""Shift handover records (SV-2.5, #297): ``shift.close`` freezes a closed shift into ``twin.shift_record``.

At 06:10, 14:10 and 22:10 IST the shift that ended 10 min earlier (C, A, B) is frozen once:

- twin values: the stored hourly snapshots whose window ends in the shift, ``(start, end]`` (8 windows), as they are
  then (``twin_views.snapshot_data``); the record's coke rates (kg/tHM dry) and implied shaft efficiency are those of
  the shift's own window [start, end) (the twin's window on a completed shift equals page 10's shift);
- plan decision: the latest decision taken in [start, end) (``twin.plan_decision``) with its run's headline;
- report link: the Shift Reports view of the shift (``/reports?date=…&shift=…``), which opens the shift's newest
  report. ``report.shift`` writes the closed report at HH:20, after the record, so the link names the shift, not a row.

``twin.evaluate`` stores the shift's window from HH:10 (retried to HH:25); while it is missing the job tries again
(``max_tries`` of the kind, 5 min apart, to HH:30) and the last try freezes what there is, the missing windows named.
A record exists once per shift: a second job for the shift returns the stored record (idempotent), and the table
refuses changes (trigger ``reject_change`` on PostgreSQL; the API has no write route). The job also freezes the
``backfill_shifts`` shifts before its target that have no record (a worker that was down at their slot).

Building blocks for the worker (``closed_shift``, ``bounds``, ``shift_window_ready``, ``record_values``, ``freeze``,
``due_shifts``) and views of stored rows for the API (``summary_data``, ``detail_data``, ``link_data``). Times UTC.
No Streamlit or ``ui`` import.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from data.twin.repositories import AlreadyRecorded, TwinStore
from services import twin_views
from services.reports import Shift, closed_shift
from utils.shift_windows import previous_shift

IST = ZoneInfo("Asia/Kolkata")
HOUR = timedelta(hours=1)
#: Hourly windows that end in a shift, count.
WINDOWS_PER_SHIFT = 8
#: Shifts before the target frozen when they have no record (the 24 h before), count.
BACKFILL_SHIFTS = 3
#: Record fields of the list view.
COLUMNS = ("k_trace", "k_forge", "k_strata", "k_charged", "omega_imp")


def bounds(shift: Shift) -> tuple[datetime, datetime]:
    """[start, end) of the shift in UTC."""
    frame = shift.timeframe()
    return frame.start_ist.astimezone(UTC), frame.end_ist.astimezone(UTC)


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """[00:00, 24:00) IST of ``day`` in UTC: the shifts A, B and C dated ``day`` start in it."""
    start = datetime.combine(day, time.min, tzinfo=IST).astimezone(UTC)
    return start, start + timedelta(days=1)


def report_ref(shift: Shift) -> str:
    """The Shift Reports view of the shift: the SPA opens its newest report."""
    return f"/reports?date={shift.date:%Y-%m-%d}&shift={shift.letter}"


def api_href(shift_key: str) -> str:
    return f"/api/shift-records/{shift_key}"


def previous(shift: Shift) -> Shift:
    day, letter = previous_shift(shift.date, shift.letter)
    return Shift(day, letter)


def _json(value: Any) -> Any:
    """JSON values: times as ISO text, non-finite floats as None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json(v) for v in value]
    return value


def shift_window_ready(store: TwinStore, shift: Shift) -> bool:
    """Whether the snapshot of the shift's own window [start, end) is stored."""
    return store.snapshots.get(bounds(shift)[1]) is not None


def record_values(
    store: TwinStore,
    shift: Shift,
    *,
    closed_at: datetime | None = None,
    job_id: str | None = None,
    submitted_by: str | None = None,
) -> dict[str, Any]:
    """The ``twin.shift_record`` row of ``shift`` from the store as it is now."""
    start, end = bounds(shift)
    hourly = [twin_views.snapshot_data(r) for r in store.snapshots.series(start + HOUR, end)]  # ends in (start, end]
    window = next((h for h in hourly if h["window_end"] == end), None)
    stored = {h["window_end"] for h in hourly}
    missing = [start + HOUR * i for i in range(1, WINDOWS_PER_SHIFT + 1) if start + HOUR * i not in stored]
    decision = store.optimiser.latest_decision_between(start, end)
    run = store.optimiser.get(decision.run_id) if decision is not None else None
    plan = (
        {
            "decision": twin_views.decision_data(decision),
            "run": twin_views.run_data(run) if run is not None else None,
        }
        if decision is not None
        else None
    )
    twin = {
        "hourly": hourly,
        "windows_expected": WINDOWS_PER_SHIFT,
        "missing_windows": missing,
        "shift_window_end": end if window is not None else None,
        "plan": plan,
        "frozen_by": {"job_id": job_id, "submitted_by": submitted_by},
    }
    return {
        "shift_key": shift.key,
        "letter": shift.letter,
        "shift_start": start,
        "shift_end": end,
        "closed_at": closed_at or datetime.now(UTC),
        **{k: (window or {}).get(k) for k in COLUMNS},
        "twin": _json(twin),
        "plan_run_id": decision.run_id if decision is not None else None,
        "plan_decision": decision.decision if decision is not None else None,
        "report_ref": report_ref(shift),
    }


def freeze(
    store: TwinStore,
    shift: Shift,
    *,
    closed_at: datetime | None = None,
    job_id: str | None = None,
    submitted_by: str | None = None,
) -> tuple[Any, bool]:
    """Write the record of ``shift`` once: ``(row, True)`` when written now, ``(stored row, False)`` when it exists."""
    existing = store.shifts.get(shift.key)
    if existing is not None:
        return existing, False
    values = record_values(store, shift, closed_at=closed_at, job_id=job_id, submitted_by=submitted_by)
    try:
        return store.shifts.freeze(values), True
    except AlreadyRecorded:  # another job froze it meanwhile
        return store.shifts.get(shift.key), False


def due_shifts(store: TwinStore, target: Shift, backfill: int = BACKFILL_SHIFTS) -> list[Shift]:
    """``target`` and the ``backfill`` shifts before it that have no record, oldest first."""
    shifts = [target]
    for _ in range(backfill):
        shifts.insert(0, previous(shifts[0]))
    return [s for s in shifts if store.shifts.get(s.key) is None]


# ------------------------------------------------------------------------------------------------- stored rows


def _get(row: Any, key: str) -> Any:
    return row.get(key) if isinstance(row, Mapping) else getattr(row, key, None)


def _twin(row: Any) -> dict[str, Any]:
    return dict(_get(row, "twin") or {})


def summary_data(row: Any) -> dict[str, Any]:
    """A record without its hourly snapshots (list view)."""
    key = Shift.parse(_get(row, "shift_key"))
    twin = _twin(row)
    hourly = list(twin.get("hourly") or [])
    run_id = _get(row, "plan_run_id")
    return {
        "shift_key": key.key,
        "shift": key.letter,
        "shift_date": key.date,
        "shift_start": twin_views.utc(_get(row, "shift_start")),
        "shift_end": twin_views.utc(_get(row, "shift_end")),
        "closed_at": twin_views.utc(_get(row, "closed_at")),
        **{k: twin_views.number(_get(row, k)) for k in COLUMNS},
        "windows_stored": len(hourly),
        "windows_expected": int(twin.get("windows_expected") or WINDOWS_PER_SHIFT),
        "shift_window": twin.get("shift_window_end") is not None,
        "plan_run_id": str(run_id) if run_id is not None else None,
        "plan_decision": _get(row, "plan_decision"),
        "report_ref": _get(row, "report_ref"),
        "href": api_href(key.key),
    }


def detail_data(row: Any, *, report_id: str | None = None) -> dict[str, Any]:
    """A record with its frozen hourly snapshots, the shift's window, the plan and its decision, and the id of the
    shift's newest closed report now (``report_id``, null before ``report.shift`` has written it)."""
    twin = _twin(row)
    hourly = list(twin.get("hourly") or [])
    end = twin.get("shift_window_end")
    window = (
        next((h for h in hourly if twin_views.utc(h.get("window_end")) == twin_views.utc(end)), None) if end else None
    )
    plan = twin.get("plan") or None
    frozen_by = dict(twin.get("frozen_by") or {})
    return summary_data(row) | {
        "hourly": hourly,
        "shift_window_snapshot": window,
        "missing_windows": [twin_views.utc(t) for t in twin.get("missing_windows") or []],
        "plan": plan,
        "frozen_by_job": frozen_by.get("job_id"),
        "frozen_by": frozen_by.get("submitted_by"),
        "report_id": report_id,
    }


def link_data(row: Any) -> dict[str, Any]:
    """The record as a shift report links it (``GET /api/reports/{id}``)."""
    return {
        "shift_key": _get(row, "shift_key"),
        "closed_at": twin_views.utc(_get(row, "closed_at")),
        "href": api_href(_get(row, "shift_key")),
    }


def job_result(target: Shift, frozen: list[tuple[Any, bool]]) -> dict[str, Any]:
    """The result of a ``shift.close`` job: the target's record and what this run wrote."""
    rows = {_get(r, "shift_key"): (r, created) for r, created in frozen if r is not None}
    out: dict[str, Any] = {
        "shift_key": target.key,
        "frozen": [k for k, (_, created) in rows.items() if created],
        "existing": [k for k, (_, created) in rows.items() if not created],
        "records": [_json(summary_data(r)) for r, _ in rows.values()],
    }
    target_row = rows.get(target.key, (None, False))[0]
    out["shift_window"] = bool(target_row is not None and summary_data(target_row)["shift_window"])
    return out


__all__ = [
    "BACKFILL_SHIFTS",
    "WINDOWS_PER_SHIFT",
    "Shift",
    "api_href",
    "bounds",
    "closed_shift",
    "day_bounds",
    "detail_data",
    "due_shifts",
    "freeze",
    "job_result",
    "link_data",
    "previous",
    "record_values",
    "report_ref",
    "shift_window_ready",
    "summary_data",
]
