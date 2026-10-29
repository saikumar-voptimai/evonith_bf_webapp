"""Execution of jobs in the worker: status, progress events, time limits, retries and the model-solve process pool.

arq delivers a job id to ``run_job``; the record in Redis (``services.jobs``) says which kind to run and with which
parameters. A job that raises ``TransientJobError`` is queued again after the kind's ``retry_delay_s`` until
``max_tries``; one that exceeds ``timeout_s`` fails with a time-limit message. A job interrupted by a worker stop is
run again by arq when the next worker starts (the record shows the try count).

Schedule (SV-1.7): the cron entry of a scheduled kind (``scheduled``) submits its job with the slot it serves; when the
previous job of the kind is still queued or running, the slot is recorded as skipped. The heartbeat writes the time of
every beat to ``twin:worker:alive``; at start, ``catch_up`` reads the last one and runs or records the slots missed
since (``services.schedule.catch_up_plan``).
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import socket
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import Any

from arq import Retry
from pydantic import BaseModel

from services import outbox, schedule, twin_alarms
from services.events import EventBus, JobDoneData, JobProgressData, LiveStatusData, WorkerStatus
from services.jobs import CATALOG, JobKind, JobStore

log = logging.getLogger("worker")

HEARTBEAT_S = 10.0
PROGRESS_MIN_INTERVAL_S = 0.5
ALIVE_KEY = "twin:worker:alive"
CATCH_UP_DELAY_S = 3.0  # after arq has set its first cron times (its first poll, 0.5 s after start)

Handler = Callable[["JobContext", Any], Awaitable[dict[str, Any] | None]]
HANDLERS: dict[str, Handler] = {}


class TransientJobError(RuntimeError):
    """A failure worth another try (data source timeout, Neon or InfluxDB unavailable)."""


class SkipJob(RuntimeError):
    """The job cannot run now and another try would not help (a source it needs is down, SV-6.5): the record ends
    ``cancelled`` with the reason as its message, logged once, and no ``job_failed`` alarm is raised."""


def handles(kind_name: str) -> Callable[[Handler], Handler]:
    """Register the worker implementation of a catalogue kind."""

    def decorate(fn: Handler) -> Handler:
        if kind_name not in CATALOG:
            raise KeyError(f"{kind_name} is not in services.jobs.CATALOG")
        HANDLERS[kind_name] = fn
        return fn

    return decorate


def process_pool(size: int) -> ProcessPoolExecutor:
    """Pool for CPU-bound model solves (STRATA-Flow network, LP iterations); forkserver on POSIX, spawn on Windows."""
    method = "forkserver" if os.name == "posix" else "spawn"
    return ProcessPoolExecutor(max_workers=size, mp_context=multiprocessing.get_context(method))


@dataclass
class JobContext:
    job_id: str
    kind: JobKind
    attempt: int
    store: JobStore
    bus: EventBus
    pool: ProcessPoolExecutor | None
    _last_progress: float = field(default=0.0, repr=False)

    async def progress(self, fraction: float | None, message: str | None = None, *, force: bool = False) -> None:
        """Record and publish progress; calls closer than 0.5 s are dropped unless ``force``."""
        now = time.monotonic()
        if not force and now - self._last_progress < PROGRESS_MIN_INTERVAL_S:
            return
        self._last_progress = now
        await self.store.update(self.job_id, progress=fraction, message=message)
        await self.bus.publish(
            "job.progress",
            JobProgressData(
                job_id=self.job_id, kind=self.kind.name, status="running", progress=fraction, message=message
            ),
        )

    async def run_cpu(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a picklable top-level function in the process pool."""
        if self.pool is None:
            return fn(*args)
        return await asyncio.get_running_loop().run_in_executor(self.pool, partial(fn, *args))

    async def cancel_requested(self) -> bool:
        record = await self.store.get(self.job_id)
        return bool(record and record.cancel_requested)


def _short(exc: BaseException, limit: int = 400) -> str:
    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _publish_done(ctx: dict[str, Any], job_id: str, kind: str, status: str, message: str | None, t0: float):
    await ctx["bus"].publish(
        "job.done",
        JobDoneData(
            job_id=job_id, kind=kind, status=status, message=message, duration_s=round(time.monotonic() - t0, 3)
        ),
    )
    try:  # a scheduled kind that failed raises job_failed, its next finished run clears it (SV-1.9)
        await twin_alarms.job_outcome(ctx["bus"], kind, status, message)
    except Exception as exc:  # noqa: BLE001 - the alarm is a by-product; the job's end stands
        log.warning("job_failed alarm of %s not updated: %s", kind, _short(exc))


async def run_job(ctx: dict[str, Any], job_id: str) -> None:
    """arq entry point for every job kind."""
    store: JobStore = ctx["store"]
    bus: EventBus = ctx["bus"]
    record = await store.get(job_id)
    if record is None:
        log.warning("job %s has no record (expired); skipped", job_id)
        return
    if record.status in ("done", "failed", "cancelled"):
        return
    t0 = time.monotonic()
    if record.cancel_requested:
        await store.finish(job_id, "cancelled", error="Cancelled before start.")
        await _publish_done(ctx, job_id, record.kind, "cancelled", "Cancelled before start.", t0)
        return
    kind = CATALOG.get(record.kind)
    handler = HANDLERS.get(record.kind)
    if kind is None or handler is None:
        await store.finish(job_id, "failed", error=f"No worker implementation for {record.kind}.")
        await _publish_done(ctx, job_id, record.kind, "failed", "No worker implementation.", t0)
        return

    attempt = int(ctx.get("job_try", 1))
    await store.update(
        job_id, status="running", started_at=record.started_at or datetime.now(UTC), tries=attempt, message=None
    )
    await bus.publish("job.progress", JobProgressData(job_id=job_id, kind=kind.name, status="running", progress=0.0))
    log.info("job %s %s started (try %d)", job_id, kind.name, attempt)
    job = JobContext(job_id=job_id, kind=kind, attempt=attempt, store=store, bus=bus, pool=ctx.get("pool"))
    ctx["running"].add(job_id)
    t_try = time.monotonic()

    def run_s() -> float:
        """Execution time of the earlier tries and of this one, s."""
        return round((record.run_s or 0.0) + time.monotonic() - t_try, 3)

    try:
        params: BaseModel = kind.params.model_validate(record.params)
        async with asyncio.timeout(kind.timeout_s) as limit:
            try:
                result = await handler(job, params)
            except TimeoutError:
                if limit.expired():
                    raise
                raise RuntimeError("an operation of the job timed out") from None
    except asyncio.CancelledError:
        if await job.cancel_requested():
            await store.finish(job_id, "cancelled", error="Cancelled while running.", run_s=run_s())
            await _publish_done(ctx, job_id, kind.name, "cancelled", "Cancelled while running.", t0)
        else:
            await store.update(
                job_id, status="queued", message="Interrupted by a worker stop; it runs again.", run_s=run_s()
            )
        raise
    except TimeoutError:
        message = f"Stopped at the time limit of {kind.timeout_s:g} s."
        await store.finish(job_id, "failed", error=message, run_s=run_s())
        await _publish_done(ctx, job_id, kind.name, "failed", message, t0)
        log.warning("job %s %s: %s", job_id, kind.name, message)
    except SkipJob as exc:
        message = f"Skipped: {str(exc)[:400]}"
        await store.finish(job_id, "cancelled", error=message, run_s=run_s())
        await _publish_done(ctx, job_id, kind.name, "cancelled", message, t0)
        log.warning("job %s %s skipped: %s", job_id, kind.name, exc)
    except TransientJobError as exc:
        if attempt < kind.max_tries:
            message = f"Try {attempt} failed ({_short(exc)}); next try in {kind.retry_delay_s:g} s."
            await store.update(job_id, status="queued", message=message, run_s=run_s())
            log.warning("job %s %s: %s", job_id, kind.name, message)
            raise Retry(defer=kind.retry_delay_s) from exc
        await store.finish(job_id, "failed", error=_short(exc), run_s=run_s())
        await _publish_done(ctx, job_id, kind.name, "failed", _short(exc), t0)
    except Exception as exc:
        log.exception("job %s %s failed", job_id, kind.name)
        await store.finish(job_id, "failed", error=_short(exc), run_s=run_s())
        await _publish_done(ctx, job_id, kind.name, "failed", _short(exc), t0)
    else:
        await store.finish(job_id, "done", result=result or {}, run_s=run_s())
        await _publish_done(ctx, job_id, kind.name, "done", None, t0)
        log.info("job %s %s done in %.1f s", job_id, kind.name, time.monotonic() - t0)
    finally:
        ctx["running"].discard(job_id)


def _ist(t: datetime) -> str:
    return f"{t.astimezone(schedule.IST):%d %b %H:%M} IST"


def scheduled(kind_name: str) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Cron entry that submits a recorded job of ``kind_name`` for its slot (so scheduled runs appear in the job list
    and in the schedule audit). arq runs the entry at the slot; ``ctx["score"]`` is that time in ms."""

    async def submit(ctx: dict[str, Any]) -> None:
        kind = CATALOG[kind_name]
        score = ctx.get("score")
        run_at = datetime.fromtimestamp(score / 1000, UTC) if score else datetime.now(UTC)
        slot = schedule.slot_of(kind.schedule, run_at) if kind.schedule else None
        store: JobStore = ctx["store"]
        record, duplicate = await store.submit(kind_name, {}, schedule.SCHEDULER, scheduled_for=slot)
        if duplicate and slot is not None and record.scheduled_for != slot:  # the active job serves another slot
            reason = f"Skipped: job {record.id} of {kind.title} was still {record.status} at {_ist(slot)}."
            await store.record_skipped(kind_name, slot, reason)
            log.warning("scheduled %s at %s skipped: job %s still %s", kind_name, _ist(slot), record.id, record.status)

    submit.__name__ = submit.__qualname__ = f"schedule_{kind_name.replace('.', '_')}"
    return submit


async def read_alive(redis: Any) -> datetime | None:
    """The time of the last heartbeat of a worker (None when no worker ever ran on this Redis)."""
    raw = await redis.get(ALIVE_KEY)
    if not raw:
        return None
    try:
        return schedule.utc(raw.decode() if isinstance(raw, bytes) else str(raw))
    except ValueError:
        return None


async def mark_alive(redis: Any, at: datetime | None = None) -> None:
    await redis.set(ALIVE_KEY, (at or datetime.now(UTC)).isoformat())


async def catch_up(
    ctx: dict[str, Any],
    kinds: list[JobKind],
    *,
    since: datetime | None,
    delay_s: float | None = None,
    now: datetime | None = None,
) -> list[schedule.Decision]:
    """Run the latest missed slot of each kind when it is at most 30 min old and record every other slot missed since
    ``since`` (the last heartbeat before this start) as skipped. Waits ``delay_s`` (default ``CATCH_UP_DELAY_S``)
    first, so that arq has set its first cron times: a slot after them is arq's, one before them is the catch-up's."""
    delay_s = CATCH_UP_DELAY_S if delay_s is None else delay_s
    if delay_s:
        await asyncio.sleep(delay_s)
    try:
        return await _catch_up(ctx["store"], kinds, since=since, now=now)
    except asyncio.CancelledError:
        raise
    except Exception:  # Redis unavailable at start: the cron entries still run the coming slots
        log.exception("catch-up at worker start failed; the missed slots stay unrecorded (the audit lists them)")
        return []


async def _catch_up(
    store: JobStore, kinds: list[JobKind], *, since: datetime | None, now: datetime | None
) -> list[schedule.Decision]:
    records = await store.list(limit=100_000)
    plan = schedule.catch_up_plan(kinds, records, now or datetime.now(UTC), since=since)
    skipped: dict[str, list[datetime]] = {}
    for d in plan:
        if d.action == "submit":
            record, duplicate = await store.submit(d.kind, {}, schedule.CATCH_UP_BY, scheduled_for=d.slot)
            if duplicate and record.scheduled_for == d.slot:  # the cron entry submitted this slot meanwhile
                continue
            if duplicate:  # a job of the kind from before the stop runs again (arq): the slot does not run
                reason = f"Skipped: job {record.id} of the kind was still {record.status} at the worker start."
                await store.record_skipped(d.kind, d.slot, reason)
                skipped.setdefault(d.kind, []).append(d.slot)
            else:
                log.info("catch-up: %s of %s submitted as job %s (%s)", d.kind, _ist(d.slot), record.id, d.reason)
        else:
            await store.record_skipped(d.kind, d.slot, d.reason)
            skipped.setdefault(d.kind, []).append(d.slot)
    for kind, missed in skipped.items():
        span = _ist(missed[0]) if len(missed) == 1 else f"{_ist(min(missed))} to {_ist(max(missed))}"
        log.warning(
            "catch-up: %s, %d slot%s recorded as skipped (%s); the job records give the reasons",
            kind,
            len(missed),
            "" if len(missed) == 1 else "s",
            span,
        )
    return plan


async def heartbeat_loop(ctx: dict[str, Any]) -> None:
    """Publish ``live.status`` every 10 s (on 10 s boundaries) and mark the time in ``twin:worker:alive`` (the next
    start catches up from it). The event carries this worker's state and, while API clients hold the event stream,
    the plant snapshot of the beat (``ctx["live_status"]``, a ``services.live_status.LiveStatusProducer``: one
    InfluxDB snapshot query per beat for all clients, SV-2.1)."""
    warned = 0.0
    replayed_at = 0.0
    while True:
        await asyncio.sleep(HEARTBEAT_S - (time.time() % HEARTBEAT_S))
        if time.monotonic() - replayed_at >= outbox.REPLAY_INTERVAL_S:  # buffered twin-store writes (SV-6.5)
            replayed_at = time.monotonic()
            try:
                await asyncio.to_thread(outbox.replay_pending)
            except Exception as exc:  # noqa: BLE001 - the replay is retried at the next interval
                log.warning("outbox replay failed: %s", _short(exc))
        try:
            redis = ctx["store"].redis
            await mark_alive(redis)
            status = WorkerStatus(
                name=ctx["name"],
                started_at=ctx["started_at"],
                running=len(ctx["running"]),
                queued=await ctx["store"].queued_count(),
                pool_size=ctx["pool_size"],
            )
            producer = ctx.get("live_status")
            plant = await producer.tick(redis) if producer is not None else {}
            await ctx["bus"].publish("live.status", LiveStatusData(worker=status, **plant))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # Redis briefly unavailable: keep the loop alive, warn once a minute
            if time.monotonic() - warned > 60:
                log.warning("heartbeat not published: %s", _short(exc))
                warned = time.monotonic()


def worker_name() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"
