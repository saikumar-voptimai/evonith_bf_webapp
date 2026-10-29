"""Views of the stored twin for the API (SV-1.6, #286): freshness of the latest evaluation, hourly and per-shift
series, alarms, optimiser runs and decisions as plain values; for the Live Twin (SV-1.9, #289) the data age of each
source and the model gaps against TRACE.

The twin evaluates the window [t - 8 h, t) for t = HH:00 at HH:10 (``twin.evaluate``, SV-1.7: the window's last
hourly row is in the published build of about HH:06), normally within a few minutes: at a time ``now`` the newest
window that should exist ends at the full hour of ``now - 15 min``. The windows ending at 06:00, 14:00 and 22:00 IST
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

from services import source_health

PLANT_TZ = "Asia/Kolkata"
EVALUATION_MINUTE = 15  # the window ending HH:00 is expected from HH:15 (run at HH:10, retried to HH:25)
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


# ------------------------------------------------------------------------------------- Live Twin (SV-1.9, #289)
SOURCE_LABELS = source_health.SOURCE_LABELS
STALE_AFTER_H = {"influxdb": 0.5, "hourly": 2.5, "hm_slag": 6.0, "dpr": 48.0, "open_meteo": 36.0}


def freshness_limits(cfg: Mapping[str, Any] | None) -> dict[str, float]:
    """Age from which each source is stale, h (``config/coke_energy.yml -> twin.freshness.stale_after_h``)."""
    block = dict((((cfg or {}).get("twin") or {}).get("freshness") or {}).get("stale_after_h") or {})
    return {k: float(block.get(k, v)) for k, v in STALE_AFTER_H.items()}


def source_freshness(
    latest: Any | None,
    now: datetime | None = None,
    limits: Mapping[str, float] | None = None,
    health: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Data age of each source as the latest evaluation recorded it (snapshot ``flags.sources``): newest time, age at
    ``now`` and stale from ``limits`` (h); the DPR carries the evaluation's provisional flag (a date carried forward).
    A source without a recorded time is stale. ``health`` (``source_health.SourceHealth.view()``, SV-6.5) brings the
    live fetches: a newer time they recorded replaces the evaluation's, and a source whose last fetch failed is
    ``down`` and stale at once, whatever its data age."""
    now = utc(now or datetime.now(UTC))
    lim = {**STALE_AFTER_H, **dict(limits or {})}
    flags = dict(_get(latest, "flags") or {}) if latest is not None else {}
    ends = dict(flags.get("sources") or {})
    recorded = utc(_get(latest, "evaluated_at")) if latest is not None else None
    live = {str(h.get("key")): dict(h) for h in (health or [])}
    out = []
    for key, label in SOURCE_LABELS.items():
        end = utc(ends[key]) if ends.get(key) else None
        at = recorded
        seen = live.get(key) or {}
        live_end = utc(seen.get("data_end"))
        if live_end is not None and (end is None or live_end > end):
            end, at = live_end, utc(seen.get("last_ok_at"))
        age = (now - end).total_seconds() / 3600.0 if end is not None else None
        down = bool(seen.get("failed"))
        out.append(
            dict(
                key=key,
                label=label,
                data_end=end,
                age_h=age,
                stale_after_h=lim[key],
                stale=down or age is None or age >= lim[key],
                provisional=bool(flags.get("provisional_dpr")) if key == "dpr" else False,
                recorded_at=at,
                down=down,
                error=seen.get("last_error") if down else None,
                failed_at=utc(seen.get("last_failure_at")) if down else None,
            )
        )
    return out


def model_gaps(
    rows: Iterable[Any], now: datetime | None = None, *, days: float = 7.0, ratio: float = 2.0
) -> dict[str, Any]:
    """FORGE - TRACE and STRATA - TRACE of the newest window that has them and their RMSE over the hourly windows of
    the last ``days`` (kg/tHM dry), and the run of newest consecutive hourly windows with |STRATA - TRACE| above
    ``ratio`` x that RMSE (a missing hour ends the run)."""
    now = utc(now or datetime.now(UTC))
    since = now - pd.Timedelta(days=days)
    seen: list[tuple[datetime, float | None, float | None]] = []
    for row in rows:
        end = utc(_get(row, "window_end"))
        if end is None or end <= since or end > now:
            continue
        trace, forge, strata = (number(_get(row, k)) for k in ("k_trace", "k_forge", "k_strata"))
        if trace is None:
            continue
        seen.append((end, None if forge is None else forge - trace, None if strata is None else strata - trace))
    seen.sort(key=lambda r: r[0])

    def rmse(values: list[float]) -> float | None:
        return math.sqrt(sum(v * v for v in values) / len(values)) if values else None

    d_forge = [r[1] for r in seen if r[1] is not None]
    d_strata = [r[2] for r in seen if r[2] is not None]
    last_forge = next((r for r in reversed(seen) if r[1] is not None), None)
    last_strata = next((r for r in reversed(seen) if r[2] is not None), None)
    rmse_strata = rmse(d_strata)
    run: list[datetime] = []
    if rmse_strata:
        for end, _, d in reversed(seen):
            if d is None or abs(d) <= ratio * rmse_strata:
                break
            if run and (run[-1] - end) > pd.Timedelta(hours=1):
                break
            run.append(end)
    return dict(
        days=days,
        ratio=ratio,
        forge_at=last_forge[0] if last_forge else None,
        forge_trace_kg_thm=last_forge[1] if last_forge else None,
        strata_at=last_strata[0] if last_strata else None,
        strata_trace_kg_thm=last_strata[2] if last_strata else None,
        forge_trace_rmse_kg_thm=rmse(d_forge),
        strata_trace_rmse_kg_thm=rmse_strata,
        n_forge=len(d_forge),
        n_strata=len(d_strata),
        strata_excess_h=len(run),
        strata_excess_since=run[-1] if run else None,
    )


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
    "SOURCE_LABELS",
    "STALE_AFTER_H",
    "freshness_limits",
    "model_gaps",
    "source_freshness",
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
