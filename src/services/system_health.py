"""Sources and services of the admin health view (SV-4.8, #311).

``source_rows``: every source the twin reads with its last timestamp and age at ``now``: the five data sources of
``twin_views.source_freshness`` (InfluxDB, hourly dataset, HM/slag, DPR, Open-Meteo: the latest evaluation's data
times merged with the live fetches of ``source_health``; stale past ``coke_energy.yml -> twin.freshness``, down when
the last fetch failed) and the twin evaluation (its window end; stale from one missed hourly window,
``twin_views.freshness``). ``service_rows``: the API process, Redis, the twin store and the worker (heartbeat
``twin:worker:alive``, every 10 s; down past ``WORKER_STALE_S``), each with its last timestamp and age where it has
one. States: ok, stale, down, missing (no time recorded), off (not configured). Times UTC, ages s. No Streamlit
(tests/test_services_architecture.py).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import text

from data.twin.repositories import TwinStore
from services import live_status, source_health, twin_views
from services.jobs import QUEUE_NAME

log = logging.getLogger("services.health")

ALIVE_KEY = "twin:worker:alive"  # worker.runtime.ALIVE_KEY: the worker's heartbeat time (ISO, UTC)
WORKER_STALE_S = 60.0  # six missed heartbeats
PING_TIMEOUT_S = 2.0
EVALUATION_LABEL = "Twin evaluation"


def _age_s(at: datetime | None, now: datetime) -> float | None:
    return None if at is None else (now - at).total_seconds()


def _hours_s(value: Any) -> float | None:
    return None if value is None else float(value) * 3600.0


def source_rows(
    latest: Any | None,
    *,
    now: datetime,
    cfg: Mapping[str, Any] | None,
    health: Iterable[Mapping[str, Any]] | None,
    store_configured: bool = True,
) -> list[dict[str, Any]]:
    """The data sources and the twin evaluation, in that order. ``latest`` is the newest stored evaluation (None
    before the first, or without a twin store: ``store_configured`` False marks the evaluation ``off``)."""
    now = twin_views.utc(now)
    rows = []
    for s in twin_views.source_freshness(latest, now, twin_views.freshness_limits(cfg), health):
        last = s["data_end"]
        state = "down" if s["down"] else "missing" if last is None else "stale" if s["stale"] else "ok"
        note = None
        if s["provisional"]:
            note = "DPR date carried forward (provisional)"
        rows.append(
            dict(
                key=s["key"],
                label=s["label"],
                group="data",
                last_at=last,
                age_s=_age_s(last, now),
                stale_after_s=_hours_s(s["stale_after_h"]),
                state=state,
                checked_at=s["recorded_at"],
                error=s["error"],
                failed_at=s["failed_at"],
                note=note,
            )
        )
    fresh = twin_views.freshness(latest, now)
    end = twin_views.utc(latest.window_end) if latest is not None else None
    if not store_configured:
        state, note = "off", "The twin store is not configured: no evaluation to read."
    elif latest is None:
        state, note = "missing", "No evaluation stored yet."
    else:
        state = "stale" if fresh["stale"] else "ok"
        lag = fresh["lag_h"] or 0.0
        note = f"{lag:.0f} h behind the expected hourly window" if lag >= 1 else "At the expected hourly window"
    rows.append(
        dict(
            key="evaluation",
            label=EVALUATION_LABEL,
            group="twin",
            last_at=end,
            age_s=_age_s(end, now),
            stale_after_s=None,
            state=state,
            checked_at=twin_views.utc(latest.evaluated_at) if latest is not None and latest.evaluated_at else None,
            error=None,
            failed_at=None,
            note=note,
        )
    )
    return rows


def _row(key: str, label: str, state: str, **values: Any) -> dict[str, Any]:
    base = dict(last_at=None, age_s=None, latency_ms=None, detail=None)
    return dict(key=key, label=label, state=state, **{**base, **values})


async def follow_worker(redis: Any) -> None:
    """Merge the worker's source registry and its latest status snapshot from Redis into this process's
    ``source_health`` (as ``GET /api/twin/now``); Redis failing changes nothing."""
    if redis is None:
        return
    try:
        await source_health.follow(redis)
        await live_status.read_latest(redis)
    except Exception as exc:  # noqa: BLE001 - the registry keeps what this process saw
        log.debug("source health not read from Redis (%s)", type(exc).__name__)


async def redis_row(redis: Any, *, configured: bool) -> dict[str, Any]:
    if not configured:
        return _row("redis", "Redis", "off", detail="BF_REDIS_URL is not set: no job queue, no event stream.")
    if redis is None:
        return _row("redis", "Redis", "down", detail="Not connected; the API retries in the background.")
    started = time.perf_counter()
    try:
        await asyncio.wait_for(redis.ping(), PING_TIMEOUT_S)
        queued = int(await redis.zcard(QUEUE_NAME))
    except Exception as exc:  # noqa: BLE001 - shown as the row's detail
        return _row("redis", "Redis", "down", detail=f"No answer ({type(exc).__name__}).")
    return _row(
        "redis",
        "Redis",
        "ok",
        latency_ms=(time.perf_counter() - started) * 1000.0,
        detail=f"{queued} job{'' if queued == 1 else 's'} waiting in the queue",
    )


async def worker_row(redis: Any, *, now: datetime) -> dict[str, Any]:
    if redis is None:
        return _row("worker", "Worker", "missing", detail="Unknown: Redis is not connected.")
    try:
        raw = await asyncio.wait_for(redis.get(ALIVE_KEY), PING_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        return _row("worker", "Worker", "missing", detail=f"Heartbeat not read ({type(exc).__name__}).")
    if not raw:
        return _row("worker", "Worker", "missing", detail="No heartbeat recorded: the worker has not run.")
    try:
        at = twin_views.utc(datetime.fromisoformat(raw.decode() if isinstance(raw, bytes) else str(raw)))
    except ValueError:
        return _row("worker", "Worker", "missing", detail=f"{ALIVE_KEY} does not hold a time.")
    age = _age_s(at, now)
    state = "ok" if age is not None and age < WORKER_STALE_S else "down"
    detail = "Heartbeat every 10 s" if state == "ok" else f"No heartbeat for more than {WORKER_STALE_S:.0f} s"
    return _row("worker", "Worker", state, last_at=at, age_s=age, detail=detail)


def store_row(store: TwinStore | None) -> dict[str, Any]:
    """Reachability and schema revision of the twin store (blocking: run it in a thread)."""
    if store is None:
        return _row(
            "twin_store",
            "Twin store",
            "off",
            detail="BF_TWIN_DB_URL / DATABASE_URL not set: no job archive, no audit log.",
        )
    started = time.perf_counter()
    try:
        with store.sessions() as session:
            session.execute(text("SELECT 1"))
        revision = store.revision()
    except Exception as exc:  # noqa: BLE001 - shown as the row's detail
        return _row("twin_store", "Twin store", "down", detail=f"No answer ({type(exc).__name__}).")
    return _row(
        "twin_store",
        "Twin store",
        "ok" if revision else "down",
        latency_ms=(time.perf_counter() - started) * 1000.0,
        detail=f"Schema revision {revision}" if revision else "Schema not migrated (alembic -n twin upgrade head)",
    )


def api_row(*, started_at: datetime, now: datetime, version: str, release: str | None) -> dict[str, Any]:
    return _row(
        "api",
        "twin-api",
        "ok",
        last_at=started_at,
        age_s=_age_s(started_at, now),
        detail=f"Version {version}, release {release or 'development'}; the age is the time since the process start",
    )


def summary(sources: Iterable[Mapping[str, Any]], services: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Rows per state over the sources and services."""
    out: dict[str, int] = {}
    for row in [*sources, *services]:
        out[row["state"]] = out.get(row["state"], 0) + 1
    return out


__all__ = [
    "ALIVE_KEY",
    "EVALUATION_LABEL",
    "WORKER_STALE_S",
    "api_row",
    "follow_worker",
    "redis_row",
    "source_rows",
    "store_row",
    "summary",
    "worker_row",
]
