"""System jobs: self-test of the worker and its process pool, and nightly housekeeping."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from services import jobstats
from services.jobs import CATALOG, NoParams, SelfTestParams
from worker.runtime import JobContext, handles

STATS_DAYS = 14


def cpu_probe(n: int) -> tuple[int, int]:
    """Runs in a pool process: its pid and a sum of squares (proves the pool executes Python)."""
    return os.getpid(), sum(i * i for i in range(n))


@handles("system.selftest")
async def selftest(job: JobContext, params: SelfTestParams) -> dict[str, Any]:
    t0 = time.monotonic()
    for step in range(1, params.steps + 1):
        if params.step_s:
            await asyncio.sleep(params.step_s)
        await job.progress(step / params.steps, f"Step {step} of {params.steps}", force=True)
    pool: dict[str, Any] | None = None
    if params.cpu:
        t1 = time.monotonic()
        pid, _ = await job.run_cpu(cpu_probe, 200_000)
        pool = {"pid": pid, "worker_pid": os.getpid(), "seconds": round(time.monotonic() - t1, 3)}
    return {"steps": params.steps, "seconds": round(time.monotonic() - t0, 3), "pool": pool}


@handles("system.housekeeping")
async def housekeeping(job: JobContext, params: NoParams) -> dict[str, Any]:
    """Copy the job records still in Redis (7 days) to the archive, then trim the index, then compute the runtime
    statistics per kind of the last 14 days (``services.jobstats``) into the result, which ``twin.jobs`` keeps: one
    document per day. An archive failure is reported in the result and does not stop the trim: the records stay in
    Redis until their 7 days end; the statistics are then computed from Redis."""
    result: dict[str, Any] = {"records_archived": None}
    records = await job.store.list(limit=100_000)
    archive = job.store.archive
    if archive is not None:
        try:
            result["records_archived"] = await archive.sync(records)
        except Exception as exc:
            result["archive_error"] = f"{type(exc).__name__}: {exc}"[:400]
    result["index_entries_removed"] = await job.store.trim()
    end = datetime.now(UTC)
    start = end - timedelta(days=STATS_DAYS)
    rows: list[Any] = records
    source = "redis"
    if archive is not None and "archive_error" not in result:
        try:
            rows, source = await asyncio.to_thread(archive.store.jobs.window, start, end), "twin.jobs"
        except Exception as exc:
            result["stats_error"] = f"{type(exc).__name__}: {exc}"[:400]
    table = jobstats.stats(rows, CATALOG.values(), start=start, end=end)
    result["job_stats"] = jobstats.document(table, start=start, end=end, source=source)
    return result
