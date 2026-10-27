"""STRATA-Learn shadow run (SV-1.7, #287): ``learn.shadow`` at 03:00, in a thread (a few seconds; no network solves).
Outputs in ``src/storage/burden_engine/learn`` (``services.learn``); the job result is the run's summary."""

from __future__ import annotations

import asyncio
from typing import Any

from services import learn
from services.jobs import NoParams
from worker.runtime import JobContext, handles


@handles("learn.shadow")
async def shadow(job: JobContext, params: NoParams) -> dict[str, Any]:
    await job.progress(0.1, "STRATA-Learn: challenger, gate, drift and register proposal", force=True)
    return await asyncio.to_thread(learn.shadow)
