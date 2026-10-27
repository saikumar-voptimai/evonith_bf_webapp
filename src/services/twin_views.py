"""Views of the stored twin for the API (SV-1.6, #286): freshness of the latest evaluation, hourly and per-shift
series, alarms, optimiser runs and decisions as plain values.

The twin evaluates the window [t - 8 h, t) at HH:05 for t = HH:00 (plan section 4): at a time ``now`` the newest
window that should exist ends at the full hour of ``now - 5 min``. The windows ending at 06:00, 14:00 and 22:00 IST
are the shifts C, A and B, so the per-shift series is the hourly series at those ends (the twin's window on a
completed shift equals page 10's shift, ``tests/test_twin_window.py``). Stored times are UTC: aware on PostgreSQL,
naive on SQLite; both are returned aware. Pure: rows in, plain values out.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd

PLANT_TZ = "Asia/Kolkata"
EVALUATION_MINUTE = 5
SHIFT_ENDS = {6: "C", 14: "A", 22: "B"}  # IST hour at which a shift window ends -> shift letter
SNAPSHOT_FIELDS = (
    "k_trace",
    "k_forge",
    "k_strata",
    "k_strata_sd",
    "k_charged",
    "omega_imp",
    "omega_strata",
    "omega_strata_sd",
    "omega_cal",
)


def utc(value: datetime | pd.Timestamp | str | None) -> datetime | None:
    """A stored or given time as an aware UTC datetime (naive values are UTC)."""
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def _get(row: Any, key: str, default: Any = None) -> Any:
    return row.get(key, default) if isinstance(row, Mapping) else getattr(row, key, default)


def number(value: Any) -> float | None:
    """A finite float, else None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def expected_window_end(now: datetime | None = None) -> datetime:
    """End of the newest window that should be evaluated at ``now`` (UTC)."""
    at = pd.Timestamp(now or datetime.now(UTC))
    at = at.tz_localize("UTC") if at.tzinfo is None else at
    ist = at.tz_convert(PLANT_TZ) - pd.Timedelta(minutes=EVALUATION_MINUTE)
    return ist.floor("h").tz_convert("UTC").to_pydatetime()


def freshness(latest: Any | None, now: datetime | None = None, stale_after_h: float = 1.0) -> dict[str, Any]:
    """Age of the latest evaluation and how many hourly windows it lags the expected one; stale from one missed
    evaluation (``stale_after_h``)."""
    now = utc(now or datetime.now(UTC))
    expected = expected_window_end(now)
    if latest is None:
        return dict(
            now=now, expected_window_end=expected, age_s=None, lag_h=None, stale=True, stale_after_h=stale_after_h
        )
    end, evaluated = utc(_get(latest, "window_end")), utc(_get(latest, "evaluated_at"))
    lag = max(0.0, (expected - end).total_seconds() / 3600.0)
    return dict(
        now=now,
        expected_window_end=expected,
        age_s=(now - evaluated).total_seconds() if evaluated else None,
        lag_h=lag,
        stale=lag >= stale_after_h,
        stale_after_h=stale_after_h,
    )


def snapshot_data(row: Any) -> dict[str, Any]:
    """The fields of ``services.events.TwinSnapshotData`` from a ``twin.snapshot`` row."""
    out = {
        "window_start": utc(_get(row, "window_start")),
        "window_end": utc(_get(row, "window_end")),
        "evaluated_at": utc(_get(row, "evaluated_at")),
        "job_id": _get(row, "job_id"),
        "status": _get(row, "status"),
        "quantities": {k: v for k, v in dict(_get(row, "quantities") or {}).items() if number(v) is not None},
        "flags": dict(_get(row, "flags") or {}),
        "versions": dict(_get(row, "versions") or {}),
    }
    out.update({k: number(_get(row, k)) for k in SNAPSHOT_FIELDS})
    return out


def shift_key(window_end: datetime) -> str | None:
    """``YYYY-MM-DD_A`` when the window [end - 8 h, end) is a shift (date of the shift start in IST), else None."""
    end = pd.Timestamp(utc(window_end)).tz_convert(PLANT_TZ)
    letter = SHIFT_ENDS.get(end.hour) if end.minute == 0 and end.second == 0 else None
    if letter is None:
        return None
    return f"{(end - pd.Timedelta(hours=8)):%Y-%m-%d}_{letter}"


def series_rows(
    rows: Iterable[Any], *, resolution: str = "hour", quantities: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Rows of the series, oldest first: window end, shift key (shift resolution), status, coke rates and shaft
    efficiencies, and the requested quantities (missing ones omitted)."""
    out = []
    for row in rows:
        end = utc(_get(row, "window_end"))
        key = shift_key(end)
        if resolution == "shift" and key is None:
            continue
        q = dict(_get(row, "quantities") or {})
        item = {
            "window_end": end,
            "shift_key": key,
            "status": _get(row, "status"),
            **{k: number(_get(row, k)) for k in SNAPSHOT_FIELDS},
            "quantities": {k: number(q[k]) for k in quantities if k in q and number(q[k]) is not None},
        }
        out.append(item)
    return out


def alarm_data(row: Any) -> dict[str, Any]:
    return {
        "id": _get(row, "id"),
        "kind": _get(row, "kind"),
        "subject": _get(row, "subject") or "",
        "severity": _get(row, "severity"),
        "message": _get(row, "message"),
        "details": dict(_get(row, "details") or {}),
        "raised_at": utc(_get(row, "raised_at")),
        "cleared_at": utc(_get(row, "cleared_at")),
        "acknowledged_by": _get(row, "acknowledged_by"),
        "acknowledged_at": utc(_get(row, "acknowledged_at")),
    }


RUN_HEADLINE = (
    "production_t_d",
    "cost_rs_thm",
    "saving_rs_thm",
    "coke_kg_thm",
    "coke_sd_kg_thm",
    "fuel_rate_kg_thm",
)


def decision_data(row: Any | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "run_id": str(_get(row, "run_id")),
        "decision": _get(row, "decision"),
        "decided_by": _get(row, "decided_by"),
        "decided_at": utc(_get(row, "decided_at")),
        "shift_key": _get(row, "shift_key"),
        "comment": _get(row, "comment"),
    }


def run_data(row: Any, *, detail: bool = False) -> dict[str, Any]:
    """An optimiser run: headline (and with ``detail`` its inputs, result and versions)."""
    out = {
        "id": str(_get(row, "id")),
        "job_id": _get(row, "job_id"),
        "trigger": _get(row, "trigger"),
        "requested_by": _get(row, "requested_by"),
        "mode": _get(row, "mode"),
        "omega_mode": _get(row, "omega_mode"),
        "status": _get(row, "status"),
        "started_at": utc(_get(row, "started_at")),
        "finished_at": utc(_get(row, "finished_at")),
        "certainty": _get(row, "certainty"),
        **{k: number(_get(row, k)) for k in RUN_HEADLINE},
    }
    if detail:
        out.update(
            inputs=dict(_get(row, "inputs") or {}),
            result=dict(_get(row, "result") or {}),
            versions=dict(_get(row, "versions") or {}),
        )
    return out


def current_shift_key(now: datetime | None = None) -> str:
    """Shift in which ``now`` falls (A 06-14, B 14-22, C 22-06 IST), ``YYYY-MM-DD_X`` with the date of its start."""
    at = pd.Timestamp(utc(now or datetime.now(UTC))).tz_convert(PLANT_TZ)
    if 6 <= at.hour < 14:
        return f"{at:%Y-%m-%d}_A"
    if 14 <= at.hour < 22:
        return f"{at:%Y-%m-%d}_B"
    start = at if at.hour >= 22 else at - pd.Timedelta(days=1)
    return f"{start:%Y-%m-%d}_C"


__all__ = [
    "EVALUATION_MINUTE",
    "number",
    "SHIFT_ENDS",
    "alarm_data",
    "current_shift_key",
    "decision_data",
    "expected_window_end",
    "freshness",
    "run_data",
    "series_rows",
    "shift_key",
    "snapshot_data",
    "utc",
]
