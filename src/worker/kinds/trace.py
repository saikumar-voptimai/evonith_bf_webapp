"""TRACE view (SV-1.10, #290): ``twin.trace`` every hour at :20.

Builds the TRACE view (``services.trace.refresh``) from the Coke Energy page data and writes it to
``src/storage/coke_energy/trace_view.json``, which the API serves; a request never runs the pipeline. The page data are
memoised in the worker process (keyed on the sources' tokens), so after ``twin.evaluate`` at :10 the build reads them
from memory; a cold build (first run of the process) loads the sources and balances every shift. Runs in a thread.
"""

from __future__ import annotations

import asyncio
from typing import Any

from services import trace
from services.jobs import NoParams
from worker.runtime import JobContext, handles


@handles("twin.trace")
async def build(job: JobContext, params: NoParams) -> dict[str, Any]:
    await job.progress(0.05, "Coke Energy sources and pipeline", force=True)
    view = await asyncio.to_thread(trace.refresh)
    await job.progress(1.0, "TRACE view stored", force=True)
    return trace.headline(view)
