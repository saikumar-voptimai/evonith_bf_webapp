"""Reads of the stored twin, composed once for the twin API and FurnaceMind's twin tools (SV-5.3, #314).

``now``: the latest evaluation, its freshness, the open alarms, the data age of each source and the model gaps (the
body of GET /api/twin/now); ``gaps``: FORGE − TRACE and STRATA − TRACE of the newest window up to a time and their
RMSE over the alarm settings' days (``coke_energy.yml -> twin.alarms.strata_trace_gap``); ``series``: the stored
windows between two window ends (GET /api/twin/series); ``run_summary``: an optimiser run with the decision that
applies (GET /api/optimiser/runs, /latest). The values come from ``services.twin_views``; coke rates kg/tHM dry,
times UTC.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from data.twin.repositories import TwinStore
from services import source_health, twin_alarms, twin_views

MAX_SERIES_DAYS = 92  # longest range of one series request, days
DEFAULT_SERIES_DAYS = 7


class InvalidRange(ValueError):
    """A series range that cannot be served (start after end, or longer than ``MAX_SERIES_DAYS``)."""


def aware(value: datetime | None, default: datetime) -> datetime:
    """``value`` as an aware UTC datetime (a naive value is UTC), ``default`` when None."""
    if value is None:
        return default
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def config() -> dict:
    """``config/coke_energy.yml`` (the twin's alarm and freshness settings)."""
    from services import coke_energy as ce

    return ce.config()


def gaps(store: TwinStore, at: datetime, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """FORGE − TRACE and STRATA − TRACE of the newest window ending at or before ``at`` and their RMSE over the
    alarm settings' days before it (``twin_views.model_gaps``)."""
    settings = twin_alarms.AlarmSettings.from_config(config() if cfg is None else cfg)
    days = settings.rmse_days
    recent = store.snapshots.series(at - timedelta(days=days), at)
    return twin_views.model_gaps(recent, at, days=days, ratio=settings.ratio)


def now(
    store: TwinStore,
    *,
    cfg: Mapping[str, Any] | None = None,
    health: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The fields of the API's ``TwinNow``: the latest evaluation (``snapshot``, None before the first), its freshness,
    the open alarms (count), the data age of each source with the live fetches (``health``, default the process's
    ``source_health.HEALTH``) and the model gaps (None before the first evaluation)."""
    latest = store.snapshots.latest()
    fresh = twin_views.freshness(latest)
    cfg = config() if cfg is None else cfg
    at = fresh["now"]
    if health is None:
        health = source_health.HEALTH.view(at)
    return dict(
        snapshot=twin_views.snapshot_data(latest) if latest is not None else None,
        alarms_open=len(store.alarms.open_alarms()),
        sources=twin_views.source_freshness(latest, at, twin_views.freshness_limits(cfg), health),
        gaps=gaps(store, at, cfg) if latest is not None else None,
        **fresh,
    )


def series(
    store: TwinStore,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    resolution: str = "hour",
    quantities: Sequence[str] = (),
) -> dict[str, Any]:
    """The fields of the API's ``TwinSeries``: stored windows with start <= window end <= end, oldest first (end
    default now, start default end − 7 days; naive times are UTC). Raises ``InvalidRange``."""
    stop = aware(end, datetime.now(UTC))
    first = aware(start, stop - timedelta(days=DEFAULT_SERIES_DAYS))
    if first > stop:
        raise InvalidRange("start lies after end.")
    if stop - first > timedelta(days=MAX_SERIES_DAYS):
        raise InvalidRange(f"At most {MAX_SERIES_DAYS} days per request.")
    rows = twin_views.series_rows(store.snapshots.series(first, stop), resolution=resolution, quantities=quantities)
    return dict(resolution=resolution, start=first, end=stop, count=len(rows), rows=rows)


def run_summary(store: TwinStore, row: Any, *, detail: bool = False) -> dict[str, Any]:
    """An optimiser run (``twin_views.run_data``) with the decision that applies; with ``detail`` its inputs, result,
    versions and every decision, oldest first."""
    data = twin_views.run_data(row, detail=detail)
    data["decision"] = twin_views.decision_data(store.optimiser.decision(row.id))
    if detail:
        data["decisions"] = [twin_views.decision_data(d) for d in store.optimiser.decisions(row.id)]
    return data


__all__ = [
    "DEFAULT_SERIES_DAYS",
    "InvalidRange",
    "MAX_SERIES_DAYS",
    "aware",
    "config",
    "gaps",
    "now",
    "run_summary",
    "series",
]
