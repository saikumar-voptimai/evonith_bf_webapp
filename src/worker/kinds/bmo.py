"""Blend Mix Optimiser runs (SV-3.5, #302).

bmo.run     page 9's Balanced Optimizer (LP) at a fixed production (``POST /api/bmo/runs``): the case is composed from
            the live sources with page 9's functions (``services.bmo_case``) and the requested changes, the LP solved,
            and the plan returned as the job result (``services.bmo_views.BmoRunResult``). The blocking work (Neon,
            InfluxDB, the hourly dataset, the forecast ledgers, one LP) runs in a thread, so the event loop keeps
            publishing heartbeats and progress.
"""

from __future__ import annotations

import asyncio
from typing import Any

from services import bmo_views
from services.jobs import BmoRunParams
from worker.runtime import JobContext, handles


@handles("bmo.run")
async def run(job: JobContext, params: BmoRunParams) -> dict[str, Any]:
    await job.progress(0.05, "Reading the yard, the chemistry, the live tags and the forecasts", force=True)
    return await asyncio.to_thread(bmo_views.run_case, params.model_dump())
