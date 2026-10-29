"""FORGE view (SV-1.11, #291): ``twin.forge`` at HH:20.

The Coke Energy pipeline result (memoised in the worker process, so the run after ``twin.evaluate`` at HH:10 reuses
it), the Rist geometry of the stable shifts of the last ``rist_days`` and every balanced shift's coke rates and
calibration are stored as the FORGE view (``services.forge``), which the /forge routes read. When the sources cannot
be read the job tries again 5 min later; the page keeps the previous view meanwhile.
"""

from __future__ import annotations

import asyncio
from typing import Any

from services import forge
from services.jobs import NoParams
from worker.runtime import JobContext, TransientJobError, handles


@handles("twin.forge")
async def build(job: JobContext, params: NoParams) -> dict[str, Any]:
    await job.progress(0.05, "Coke Energy pipeline and the Rist geometry of the stable shifts", force=True)
    try:
        view = await asyncio.to_thread(forge.refresh, job_id=job.job_id)
    except forge.PlantDataUnavailable as exc:
        if job.attempt < job.kind.max_tries:
            raise TransientJobError(f"the Coke Energy pipeline did not run ({exc})") from exc
        raise RuntimeError(f"the Coke Energy pipeline did not run ({exc}); the stored FORGE view is kept") from exc
    return forge.headline(view)
