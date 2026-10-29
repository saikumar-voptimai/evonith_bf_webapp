"""Live Twin alarms besides the implied-omega drift (SV-1.9, #289), raised and cleared in the twin store and published
as ``alarm.raised`` / ``alarm.cleared``:

strata_trace_gap   |K_STRATA - K_TRACE| above ``ratio`` x its RMSE over the hourly windows of the last ``rmse_days``
                   in at least ``min_hours`` consecutive windows (``twin_views.model_gaps``; at least ``min_windows``
                   windows for the RMSE). Checked by ``twin.evaluate`` after each window.
source_stale       one alarm per source (subject = influxdb, hourly, hm_slag, dpr, open_meteo) whose newest time, as
                   the latest evaluation recorded it, is older than its limit (``twin_views.source_freshness``); also
                   raised when a job skips because the source is down (``source_down``, SV-6.5), and cleared by the
                   next evaluation that finds the source within its limit.
implied_lag        the implied transit time, charging to burning (``services.transit.implied_lag`` on the last 14
                   days), outside [low_h, high_h]; an evaluation without a lag leaves the alarm as it is.
job_failed         one alarm per scheduled job kind (subject = the kind) whose last run failed; the next run that
                   finishes clears it. Raised by the worker when the job ends (``worker.runtime``).

Settings: ``config/coke_energy.yml -> twin.alarms`` and ``twin.freshness``. A condition that cannot be evaluated
(``active`` None) leaves its alarm unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from services import twin_views
from services.events import AlarmData, EventBus

log = logging.getLogger(__name__)

CODES = {
    "strata_trace_gap": "STRATA_TRACE_GAP",
    "source_stale": "SOURCE_STALE",
    "implied_lag": "IMPLIED_LAG",
    "job_failed": "JOB_FAILED",
}


@dataclass(frozen=True)
class AlarmSettings:
    rmse_days: float = 7.0
    ratio: float = 2.0
    min_hours: int = 3
    min_windows: int = 24
    lag_low_h: float = 5.0
    lag_high_h: float = 8.0
    stale_after_h: Mapping[str, float] = field(default_factory=lambda: dict(twin_views.STALE_AFTER_H))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> AlarmSettings:
        block = dict(((cfg or {}).get("twin") or {}).get("alarms") or {})
        gap, lag = dict(block.get("strata_trace_gap") or {}), dict(block.get("implied_lag") or {})
        base = cls()
        return cls(
            rmse_days=float(gap.get("rmse_days", base.rmse_days)),
            ratio=float(gap.get("ratio", base.ratio)),
            min_hours=int(gap.get("min_hours", base.min_hours)),
            min_windows=int(gap.get("min_windows", base.min_windows)),
            lag_low_h=float(lag.get("low_h", base.lag_low_h)),
            lag_high_h=float(lag.get("high_h", base.lag_high_h)),
            stale_after_h=twin_views.freshness_limits(cfg),
        )


@dataclass(frozen=True)
class Condition:
    kind: str
    active: bool | None  # None: not evaluable, the alarm stays as it is
    subject: str = ""
    severity: str = "warning"
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    value: float | None = None


def gap_condition(gaps: Mapping[str, Any], settings: AlarmSettings) -> Condition:
    """STRATA - TRACE outside ``ratio`` x its RMSE for ``min_hours`` consecutive windows (``twin_views.model_gaps``)."""
    rmse, n, hours = gaps.get("strata_trace_rmse_kg_thm"), int(gaps.get("n_strata") or 0), gaps.get("strata_excess_h")
    if rmse is None or n < settings.min_windows:
        return Condition("strata_trace_gap", None)
    gap = gaps.get("strata_trace_kg_thm")
    active = int(hours or 0) >= settings.min_hours
    return Condition(
        "strata_trace_gap",
        active,
        message=(
            f"STRATA − TRACE {gap:+.1f} kg/tHM for {hours} h, above {settings.ratio:g} × the 7-day RMSE "
            f"({rmse:.1f} kg/tHM)"
            if active and gap is not None
            else ""
        ),
        details=dict(
            gap_kg_thm=gap,
            rmse_kg_thm=rmse,
            ratio=settings.ratio,
            hours=hours,
            since=gaps["strata_excess_since"].isoformat() if gaps.get("strata_excess_since") else None,
            windows=n,
        ),
        value=gap,
    )


def source_conditions(fresh: Iterable[Mapping[str, Any]]) -> list[Condition]:
    """One condition per source of ``twin_views.source_freshness``."""
    out = []
    for f in fresh:
        age = f.get("age_h")
        text = "no data recorded" if age is None else f"newest data {age:.1f} h old"
        out.append(
            Condition(
                "source_stale",
                bool(f["stale"]),
                subject=str(f["key"]),
                message=f"{f['label']}: {text} (stale from {f['stale_after_h']:g} h)",
                details=dict(
                    data_end=f["data_end"].isoformat() if f.get("data_end") else None,
                    age_h=age,
                    stale_after_h=f["stale_after_h"],
                ),
                value=age,
            )
        )
    return out


def lag_condition(lag_h: float | None, ci: tuple[float | None, float | None], settings: AlarmSettings) -> Condition:
    if lag_h is None:
        return Condition("implied_lag", None)
    active = not settings.lag_low_h <= lag_h <= settings.lag_high_h
    return Condition(
        "implied_lag",
        active,
        message=(
            f"Implied transit time {lag_h:.1f} h, outside {settings.lag_low_h:g}–{settings.lag_high_h:g} h"
            if active
            else ""
        ),
        details=dict(
            lag_h=lag_h, ci_low_h=ci[0], ci_high_h=ci[1], low_h=settings.lag_low_h, high_h=settings.lag_high_h
        ),
        value=lag_h,
    )


def job_condition(kind: str, status: str, message: str | None) -> Condition:
    """A scheduled job's outcome: failed raises, done clears, cancelled leaves the alarm."""
    if status not in ("done", "failed"):
        return Condition("job_failed", None, subject=kind)
    return Condition(
        "job_failed",
        status == "failed",
        subject=kind,
        message=f"{kind} failed: {message or 'no reason given'}" if status == "failed" else "",
        details=dict(kind=kind, error=message),
    )


async def apply(
    conditions: Iterable[Condition], alarms: Any, bus: EventBus | None, now: datetime | None = None
) -> list[tuple[str, str, str]]:
    """Raise (or update) and clear the alarms to match ``conditions``; publish raises and clears. Returns the changes
    as (event type, kind, subject)."""
    when = now or datetime.now(UTC)
    changes: list[tuple[str, str, str]] = []
    for c in conditions:
        if c.active is None:
            continue
        if c.active:
            row, new = await asyncio.to_thread(
                alarms.raise_alarm,
                c.kind,
                severity=c.severity,
                message=c.message,
                subject=c.subject,
                details=dict(c.details),
                at=when,
            )
            if new:
                changes.append(("alarm.raised", c.kind, c.subject))
                if bus is not None:
                    await bus.publish("alarm.raised", _event(row, c.value))
        else:
            row = await asyncio.to_thread(alarms.clear_alarm, c.kind, subject=c.subject, at=when)
            if row is not None:
                changes.append(("alarm.cleared", c.kind, c.subject))
                if bus is not None:
                    await bus.publish("alarm.cleared", _event(row, c.value))
    return changes


def _event(row: Any, value: float | None) -> AlarmData:
    return AlarmData(
        alarm_id=str(row.id), code=CODES.get(row.kind, row.kind.upper()), severity=row.severity,
        message=row.message, value=value,
    )  # fmt: skip


def implied_lag(store: Any, cfg: Mapping[str, Any], end: datetime) -> tuple[float | None, tuple]:
    """Implied lag at ``end`` from the hourly dataset and the stored K_TRACE (as ``GET /api/twin/transit?lag=true``)."""
    import pandas as pd

    from services import coke_energy as ce
    from services import transit

    keys = ce.data_keys(cfg, 0)
    fd = ce.CACHED_LOADERS.hourly(keys.hourly_path, keys.hourly_mtime_ns)
    end_ist = pd.Timestamp(end).tz_convert(twin_views.PLANT_TZ).tz_localize(None)
    res = transit.implied_lag(fd, store.snapshots.series(end - timedelta(days=16), end), end_ist, cfg)
    return twin_views.number(res.lag_h), (twin_views.number(res.ci[0]), twin_views.number(res.ci[1]))


async def run(
    store: Any, bus: EventBus | None, cfg: Mapping[str, Any], now: datetime | None = None, *, lag: bool = True
) -> list[tuple[str, str, str]]:
    """Evaluate the STRATA - TRACE gap, the source ages and (``lag``) the implied lag on the store and apply them."""
    settings = AlarmSettings.from_config(cfg)
    end = twin_views.utc(now or datetime.now(UTC))
    rows = await asyncio.to_thread(store.snapshots.series, end - timedelta(days=settings.rmse_days), end)
    gaps = twin_views.model_gaps(rows, end, days=settings.rmse_days, ratio=settings.ratio)
    latest = await asyncio.to_thread(store.snapshots.latest)
    conditions = [gap_condition(gaps, settings)]
    conditions += source_conditions(twin_views.source_freshness(latest, end, settings.stale_after_h))
    if lag:
        try:
            lag_h, ci = await asyncio.to_thread(implied_lag, store, cfg, end)
        except Exception as exc:  # noqa: BLE001 - no lag (dataset unreadable, too few windows): the alarm stays
            log.warning("implied lag not evaluated (%s)", type(exc).__name__)
            lag_h, ci = None, (None, None)
        conditions.append(lag_condition(lag_h, ci, settings))
    return await apply(conditions, store.alarms, bus, end)


async def job_outcome(bus: EventBus | None, kind: str, status: str, message: str | None) -> list[tuple[str, str, str]]:
    """``job_failed`` for a scheduled job kind that ended (``worker.runtime``); nothing without a twin store."""
    from services.jobs import CATALOG
    from services.store import twin_store

    spec = CATALOG.get(kind)
    condition = job_condition(kind, status, message)
    if spec is None or spec.schedule is None or condition.active is None:
        return []
    store = await asyncio.to_thread(twin_store, "twin-worker")
    if store is None:
        return []
    return await apply([condition], store.alarms, bus)


async def source_down(bus: EventBus | None, key: str, reason: str) -> list[tuple[str, str, str]]:
    """``source_stale`` for a source a job found down (``worker.runtime`` on a skip, SV-6.5); nothing without a twin
    store."""
    from services.store import twin_store

    store = await asyncio.to_thread(twin_store, "twin-worker")
    if store is None:
        return []
    condition = Condition("source_stale", True, subject=key, message=reason, details=dict(down=True, reason=reason))
    return await apply([condition], store.alarms, bus)


__all__ = [
    "CODES",
    "AlarmSettings",
    "Condition",
    "apply",
    "gap_condition",
    "job_condition",
    "job_outcome",
    "lag_condition",
    "run",
    "source_conditions",
    "source_down",
]
