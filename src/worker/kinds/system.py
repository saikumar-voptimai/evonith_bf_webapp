"""System jobs: self-test of the worker and its process pool, and nightly housekeeping."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from services.jobs import NoParams, SelfTestParams
from worker.runtime import JobContext, handles


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
    removed = await job.store.trim()
    return {"index_entries_removed": removed}
