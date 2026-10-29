"""Job history of the admin view (SV-4.8, #311): the jobs of a period, newest first, with their durations, failures
and the controls that apply (cancel, retry).

``read`` takes the records of ``twin.jobs`` (the archive, up to 92 days) and those of Redis (the live records of the
last 7 days: progress, the cancel flag, jobs the archive has not received yet); a job present in both shows its Redis
record. The filters (kind, status, period) apply to the merged records. ``history_row`` gives the durations: waiting
(start − submission; now − submission while queued), execution (``run_s``, over the tries) and elapsed (now − start
while running). Times UTC, durations s. No Streamlit (tests/test_services_architecture.py).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from data.twin.repositories import TwinStore
from services import schedule
from services.job_audit import ACTIVE, RETRYABLE
from services.jobs import CATALOG, JobStore

log = logging.getLogger("services.jobs")

MAX_DAYS = 92
MAX_ROWS = 500
#: Redis records read for one request (a week holds about 2,500 jobs, mostly the forecasts).
LIVE_SCAN = 20_000


class Unavailable(RuntimeError):
    """Neither the twin store nor Redis can be read."""


def _utc(value: Any) -> datetime | None:
    return None if value is None else schedule.utc(value)


def _seconds(a: datetime | None, b: datetime | None) -> float | None:
    return None if a is None or b is None else max(0.0, (b - a).total_seconds())


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN -> None


def history_row(record: Any, *, now: datetime, live: bool) -> dict[str, Any]:
    """One job (a Redis ``JobRecord`` or a ``twin.jobs`` row) for the history table; ``live`` = read from Redis."""
    kind = CATALOG.get(record.kind)
    status = str(record.status)
    submitted, started, finished = (
        _utc(getattr(record, k, None)) for k in ("submitted_at", "started_at", "finished_at")
    )
    cancel_requested = bool(getattr(record, "cancel_requested", False))
    submitted_by = str(record.submitted_by or "")
    return {
        "id": record.id,
        "kind": record.kind,
        "title": kind.title if kind is not None else record.kind,
        "status": status,
        "submitted_by": submitted_by,
        "scheduled": schedule.is_scheduler(record),
        "skipped": submitted_by == schedule.SKIPPED_BY,
        "scheduled_for": _utc(getattr(record, "scheduled_for", None)),
        "submitted_at": submitted,
        "started_at": started,
        "finished_at": finished,
        "wait_s": _seconds(submitted, started if started is not None else now if status == "queued" else None),
        "run_s": _number(getattr(record, "run_s", None)),
        "elapsed_s": _seconds(started, now) if status == "running" else None,
        "tries": int(getattr(record, "tries", 0) or 0),
        "max_tries": kind.max_tries if kind is not None else None,
        "timeout_s": float(kind.timeout_s) if kind is not None else None,
        "progress": _number(getattr(record, "progress", None)),
        "message": getattr(record, "message", None),
        "error": getattr(record, "error", None),
        "params": dict(getattr(record, "params", None) or {}),
        "cancel_requested": cancel_requested,
        "can_cancel": live and status in ACTIVE and not cancel_requested,
        "can_retry": status in RETRYABLE and kind is not None,
        "live": live,
    }


def _matches(record: Any, kind: str | None, status: str | None, since: datetime, until: datetime) -> bool:
    if kind and record.kind != kind:
        return False
    if status and record.status != status:
        return False
    submitted = _utc(record.submitted_at)
    return submitted is not None and since <= submitted < until


def merge(
    archived: Iterable[Any],
    live: Iterable[Any],
    *,
    kind: str | None,
    status: str | None,
    since: datetime,
    until: datetime,
    limit: int,
    now: datetime,
) -> list[dict[str, Any]]:
    """The merged records that pass the filters, newest first, at most ``limit``; a Redis record replaces the
    archive's row of the same job."""
    merged: dict[str, tuple[Any, bool]] = {r.id: (r, False) for r in archived}
    for r in live:
        merged[r.id] = (r, True)
    kept = [(r, is_live) for r, is_live in merged.values() if _matches(r, kind, status, since, until)]
    kept.sort(key=lambda item: _utc(item[0].submitted_at) or since, reverse=True)
    return [history_row(r, now=now, live=is_live) for r, is_live in kept[:limit]]


async def read(
    jobs: JobStore | None,
    twin: TwinStore | None,
    *,
    since: datetime,
    until: datetime,
    kind: str | None = None,
    status: str | None = None,
    limit: int = 200,
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The rows of [since, until) and the stores read (``twin.jobs``, ``redis``). Redis failing next to a twin store
    leaves the archive's rows; ``Unavailable`` when neither can be read."""
    now = now or datetime.now(UTC)
    archived: list[Any] = []
    live: list[Any] = []
    sources: list[str] = []
    if twin is not None:
        # The filters are applied again after the merge: the archive may hold an older state than Redis.
        archived = await asyncio.to_thread(
            twin.jobs.recent, kind=kind, status=status, since=since, until=until, limit=limit
        )
        sources.append("twin.jobs")
    if jobs is not None:
        try:
            live = await jobs.list(LIVE_SCAN)
            sources.append("redis")
        except Exception as exc:
            if twin is None:
                raise Unavailable(
                    f"Redis cannot be read ({type(exc).__name__}) and no twin store is configured."
                ) from exc
            log.warning("job history without the Redis records (%s)", type(exc).__name__)
    if not sources:
        raise Unavailable("Neither the twin store nor Redis is available: no job records to read.")
    rows = merge(archived, live, kind=kind, status=status, since=since, until=until, limit=limit, now=now)
    return rows, sources


__all__ = ["LIVE_SCAN", "MAX_DAYS", "MAX_ROWS", "Unavailable", "history_row", "merge", "read"]
