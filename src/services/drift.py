"""Implied-omega drift alarm of the live twin (SV-1.2, #282).

``evaluate(snapshots, settings, now)`` takes the stored hourly evaluations (``twin.snapshot`` rows: window end,
omega_imp, omega_STRATA), builds r = omega_imp - omega_STRATA and runs ``utils.coke_energy.drift`` (AR(1) innovations
between non-overlapping windows, two-sided CUSUM k 0.5, h 5, parameters re-estimated at every evaluation).
``apply(report, alarms, bus)`` keeps the alarm table in step with the report: one alarm per direction (kind
``omega_drift``, subject ``high`` or ``low``), raised at the first signal and cleared after ``clear_after`` without a
signal, each change published as ``alarm.raised`` / ``alarm.cleared`` (code ``OMEGA_DRIFT``) on the event bus, which
the API forwards as server-sent events. ``run(store, bus, now=...)`` does both on the twin store's history.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from services.events import AlarmData, EventBus
from utils.coke_energy import drift

KIND = "omega_drift"
CODE = "OMEGA_DRIFT"
SEVERITY = "warning"
PLANT_TZ = "Asia/Kolkata"
WORDS = {"high": "above", "low": "below"}


@dataclass(frozen=True)
class DriftConfig:
    settings: drift.DriftSettings
    history: pd.Timedelta

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> DriftConfig:
        """From the ``twin.drift`` block of ``config/coke_energy.yml`` (``services.coke_energy.config()``)."""
        block = dict(((cfg or {}).get("twin") or {}).get("drift") or {})
        return cls(drift.DriftSettings.from_config(block), pd.Timedelta(days=float(block.get("history_days", 56))))


def _utc(value: Any) -> pd.Timestamp:
    """A stored time as UTC (aware on PostgreSQL, naive UTC on SQLite)."""
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def residual_series(rows: Iterable[Any]) -> pd.Series:
    """r by window end (UTC) from snapshot rows (ORM objects or mappings with window_end, omega_imp, omega_strata)."""
    t, imp, st = [], [], []
    for row in rows:
        get = row.get if isinstance(row, Mapping) else (lambda k, row=row: getattr(row, k, None))
        if get("omega_imp") is None or get("omega_strata") is None:
            continue
        t.append(_utc(get("window_end")))
        imp.append(float(get("omega_imp")))
        st.append(float(get("omega_strata")))
    if not t:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    return drift.residuals(t, imp, st)


def evaluate(rows: Iterable[Any], config: DriftConfig, now: datetime | None = None) -> drift.DriftReport:
    return drift.evaluate(residual_series(rows), config.settings, until=pd.Timestamp(now) if now else None)


def _ist(ts: Any) -> str:
    return pd.Timestamp(ts).tz_convert(PLANT_TZ).strftime("%d %b %H:%M")


def message(direction: str, episode: drift.Episode, report: drift.DriftReport) -> str:
    latest = report.latest()
    r = latest.get("r")
    stat = episode.peak
    return (
        f"Implied omega {WORDS[direction]} STRATA since {_ist(episode.raised_at)} IST: CUSUM {stat:.1f} > "
        f"{report.settings.h:g} ({episode.signals} signal{'s' if episode.signals != 1 else ''}"
        + (f"; latest r {r:+.3f}" if r is not None else "")
        + ")"
    )


def details(direction: str, episode: drift.Episode, report: drift.DriftReport) -> dict[str, Any]:
    ar = report.ar1
    return {
        "direction": direction,
        "raised_at": episode.raised_at.isoformat(),
        "last_signal": episode.last_signal.isoformat(),
        "signals": episode.signals,
        "peak": round(episode.peak, 3),
        "latest": {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in report.latest().items()},
        "ar1": None
        if ar is None
        else {"rho": ar.rho, "sigma_e": ar.sigma_e, "n_pairs": ar.n_pairs, "source": ar.source},
        "k": report.settings.k,
        "h": report.settings.h,
        "lag_hours": report.settings.lag / pd.Timedelta(hours=1),
    }


def alarm_data(row: Any, value: float | None = None) -> AlarmData:
    return AlarmData(alarm_id=str(row.id), code=CODE, severity=row.severity, message=row.message, value=value)


async def apply(
    report: drift.DriftReport, alarms: Any, bus: EventBus | None = None, now: datetime | None = None
) -> list[tuple[str, str]]:
    """Raise, update or clear the two drift alarms to match ``report``; publish the raises and clears. Returns the
    changes as (event type, direction)."""
    changes: list[tuple[str, str]] = []
    if report.ar1 is None:  # too little history: leave the alarms as they are
        return changes
    current = {a.subject: a for a in alarms.open_alarms() if a.kind == KIND}
    episodes = report.open_episodes()
    when = pd.Timestamp(now or datetime.now(UTC))
    for direction in drift.DIRECTIONS:
        ep, row = episodes.get(direction), current.get(direction)
        if ep is not None:
            raised, new = alarms.raise_alarm(
                KIND,
                severity=SEVERITY,
                message=message(direction, ep, report),
                subject=direction,
                details=details(direction, ep, report),
                at=ep.raised_at.to_pydatetime(),
            )
            if new:
                changes.append(("alarm.raised", direction))
                if bus is not None:
                    await bus.publish("alarm.raised", alarm_data(raised, ep.peak))
        elif row is not None:
            closed = [e for e in report.episodes if e.direction == direction and e.cleared_at is not None]
            at = closed[-1].cleared_at if closed else when
            cleared = alarms.clear_alarm(KIND, subject=direction, at=pd.Timestamp(at).to_pydatetime())
            if cleared is not None:
                changes.append(("alarm.cleared", direction))
                if bus is not None:
                    await bus.publish("alarm.cleared", alarm_data(cleared))
    return changes


async def run(store: Any, bus: EventBus | None, config: DriftConfig, now: datetime | None = None) -> drift.DriftReport:
    """Evaluate the drift on the store's snapshots of the last ``config.history`` and apply it to the alarm table."""
    end = now or datetime.now(UTC)
    rows = store.snapshots.series(end - config.history.to_pytimedelta(), end)
    report = evaluate(rows, config, end)
    await apply(report, store.alarms, bus, end)
    return report
