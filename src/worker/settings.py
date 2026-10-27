"""arq settings of the twin-worker (SV-0.8, #277).

Environment: ``BF_REDIS_URL`` (default ``redis://127.0.0.1:6379/0``), ``BF_WORKER_POOL`` (model-solve processes,
default 8), ``BF_WORKER_MAX_JOBS`` (concurrent jobs, default 4), ``BF_WORKER_JOB_TIMEOUT`` (arq time limit of one
delivery, default the longest kind limit plus 60 s; a job interrupted by a worker stop runs again once this limit plus
10 s has passed), ``BF_WORKER_SKIP_SCHEDULES`` (comma-separated kinds, or prefixes ending in ".", that are not
scheduled: the test suite skips ``forecast.``, so a test worker never reads plant data; such kinds can still be
submitted). Job records are archived in ``twin.jobs`` when ``BF_TWIN_DB_URL`` or ``DATABASE_URL`` is set.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, ClassVar

from arq import cron, func
from arq.connections import RedisSettings

import worker.kinds  # noqa: F401  (registers the handlers)
from services.events import RedisEventBus
from services.job_archive import open_job_archive
from services.jobs import ARQ_FUNCTION, CATALOG, QUEUE_NAME, JobStore
from worker.runtime import heartbeat_loop, process_pool, run_job, scheduled, worker_name

log = logging.getLogger("worker")

IST = timezone(timedelta(hours=5, minutes=30), "IST")
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"


def redis_url() -> str:
    return os.environ.get("BF_REDIS_URL") or DEFAULT_REDIS_URL


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def job_timeout_s() -> int:
    default = int(max(k.timeout_s for k in CATALOG.values())) + 60
    return _int_env("BF_WORKER_JOB_TIMEOUT", default)


def skipped(kind_name: str) -> bool:
    """Whether ``BF_WORKER_SKIP_SCHEDULES`` leaves the kind unscheduled."""
    for entry in os.environ.get("BF_WORKER_SKIP_SCHEDULES", "").split(","):
        entry = entry.strip()
        if entry and (kind_name == entry or (entry.endswith(".") and kind_name.startswith(entry))):
            return True
    return False


def cron_jobs() -> list[Any]:
    """arq cron entries of the scheduled kinds, at second 0 of their IST hours and minutes."""
    jobs = []
    for kind in CATALOG.values():
        if kind.schedule is None or skipped(kind.name):
            continue
        hours, minutes = kind.schedule
        # The cron entry only submits the job (milliseconds); its 10 s limit keeps arq's in-progress mark short.
        jobs.append(cron(scheduled(kind.name), hour=hours, minute=minutes, second=0, unique=True, timeout=10))
    return jobs


async def on_startup(ctx: dict[str, Any]) -> None:
    redis = ctx["redis"]

    async def enqueue(job_id: str) -> None:
        await redis.enqueue_job(ARQ_FUNCTION, job_id, _job_id=job_id, _queue_name=QUEUE_NAME)

    ctx["archive"] = await open_job_archive()
    ctx["store"] = JobStore(redis, enqueue=enqueue, archive=ctx["archive"])
    ctx["bus"] = RedisEventBus(redis)
    ctx["pool_size"] = _int_env("BF_WORKER_POOL", 8)
    ctx["pool"] = process_pool(ctx["pool_size"]) if ctx["pool_size"] > 0 else None
    ctx["running"] = set()
    ctx["name"] = worker_name()
    ctx["started_at"] = datetime.now(UTC)
    ctx["heartbeat"] = asyncio.create_task(heartbeat_loop(ctx))
    log.info(
        "twin-worker %s started: queue %s, %d kinds, pool %d, Redis %s",
        ctx["name"],
        QUEUE_NAME,
        len(CATALOG),
        ctx["pool_size"],
        RedisSettings.from_dsn(redis_url()).host,
    )


async def on_shutdown(ctx: dict[str, Any]) -> None:
    task = ctx.get("heartbeat")
    if task is not None:
        task.cancel()
    pool = ctx.get("pool")
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    archive = ctx.get("archive")
    if archive is not None:
        await archive.close()
    log.info("twin-worker %s stopped", ctx.get("name"))


class WorkerSettings:
    """``arq worker.settings.WorkerSettings`` or ``python -m worker``."""

    functions: ClassVar[list[Any]] = [func(run_job, name=ARQ_FUNCTION, timeout=job_timeout_s(), max_tries=5)]
    cron_jobs: ClassVar[list[Any]] = cron_jobs()
    redis_settings = RedisSettings.from_dsn(redis_url())
    queue_name = QUEUE_NAME
    on_startup = on_startup
    on_shutdown = on_shutdown
    timezone = IST
    max_jobs = _int_env("BF_WORKER_MAX_JOBS", 4)
    job_timeout = job_timeout_s()
    keep_result = 3600
    poll_delay = 0.5
    allow_abort_jobs = True
    health_check_interval = 60
    max_tries = 5
