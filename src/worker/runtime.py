"""Execution of jobs in the worker: status, progress events, time limits, retries and the model-solve process pool.

arq delivers a job id to ``run_job``; the record in Redis (``services.jobs``) says which kind to run and with which
parameters. A job that raises ``TransientJobError`` is queued again after the kind's ``retry_delay_s`` until
``max_tries``; one that exceeds ``timeout_s`` fails with a time-limit message. A job interrupted by a worker stop is
run again by arq when the next worker starts (the record shows the try count).
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

from services.events import EventBus, JobDoneData, JobProgressData, LiveStatusData, WorkerStatus
from services.jobs import CATALOG, JobKind, JobStore

log = logging.getLogger("worker")

HEARTBEAT_S = 10.0
PROGRESS_MIN_INTERVAL_S = 0.5

Handler = Callable[["JobContext", Any], Awaitable[dict[str, Any] | None]]
HANDLERS: dict[str, Handler] = {}


class TransientJobError(RuntimeError):
    """A failure worth another try (data source timeout, Neon or InfluxDB unavailable)."""


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
            await store.finish(job_id, "cancelled", error="Cancelled while running.")
            await _publish_done(ctx, job_id, kind.name, "cancelled", "Cancelled while running.", t0)
        else:
            await store.update(job_id, status="queued", message="Interrupted by a worker stop; it runs again.")
        raise
    except TimeoutError:
        message = f"Stopped at the time limit of {kind.timeout_s:g} s."
        await store.finish(job_id, "failed", error=message)
        await _publish_done(ctx, job_id, kind.name, "failed", message, t0)
        log.warning("job %s %s: %s", job_id, kind.name, message)
    except TransientJobError as exc:
        if attempt < kind.max_tries:
            message = f"Try {attempt} failed ({_short(exc)}); next try in {kind.retry_delay_s:g} s."
            await store.update(job_id, status="queued", message=message)
            log.warning("job %s %s: %s", job_id, kind.name, message)
            raise Retry(defer=kind.retry_delay_s) from exc
        await store.finish(job_id, "failed", error=_short(exc))
        await _publish_done(ctx, job_id, kind.name, "failed", _short(exc), t0)
    except Exception as exc:
        log.exception("job %s %s failed", job_id, kind.name)
        await store.finish(job_id, "failed", error=_short(exc))
        await _publish_done(ctx, job_id, kind.name, "failed", _short(exc), t0)
    else:
        await store.finish(job_id, "done", result=result or {})
        await _publish_done(ctx, job_id, kind.name, "done", None, t0)
        log.info("job %s %s done in %.1f s", job_id, kind.name, time.monotonic() - t0)
    finally:
        ctx["running"].discard(job_id)


def scheduled(kind_name: str) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Cron entry that submits a recorded job of ``kind_name`` (so scheduled runs appear in the job list)."""

    async def submit(ctx: dict[str, Any]) -> None:
        record, duplicate = await ctx["store"].submit(kind_name, {}, "scheduler")
        if duplicate:
            log.info("scheduled %s skipped: job %s still %s", kind_name, record.id, record.status)

    submit.__name__ = submit.__qualname__ = f"schedule_{kind_name.replace('.', '_')}"
    return submit


async def heartbeat_loop(ctx: dict[str, Any]) -> None:
    """Publish ``live.status`` with this worker's state every 10 s (on 10 s boundaries)."""
    warned = 0.0
    while True:
        await asyncio.sleep(HEARTBEAT_S - (time.time() % HEARTBEAT_S))
        try:
            status = WorkerStatus(
                name=ctx["name"],
                started_at=ctx["started_at"],
                running=len(ctx["running"]),
                queued=await ctx["store"].queued_count(),
                pool_size=ctx["pool_size"],
            )
            await ctx["bus"].publish("live.status", LiveStatusData(worker=status))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # Redis briefly unavailable: keep the loop alive, warn once a minute
            if time.monotonic() - warned > 60:
                log.warning("heartbeat not published: %s", _short(exc))
                warned = time.monotonic()


def worker_name() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"
