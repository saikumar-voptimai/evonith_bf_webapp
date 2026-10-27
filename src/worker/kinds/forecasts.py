"""Live forecasts (SV-1.5, #285): the hot-metal Si path every 5 min and the charged-coke block at HH:07.

Both are issued by ``services.forecasts`` into the ledgers page 9 reads (format unchanged). The blocking work (source
reads, feature build, model) runs in a thread, so the event loop keeps publishing heartbeats and progress.

forecast.si             one try: the next run is 5 min later. Publishes ``forecast.si`` for every attempt at a new
                        origin (a usable path, or the reason the inputs were not usable).
forecast.charged_coke   retried every 5 min (at most 5 tries, to HH:27) while the build of the hour is not published;
                        publishes ``forecast.charged_coke`` when a new row is issued.
"""

from __future__ import annotations

import asyncio
from typing import Any

from services import forecasts
from services.jobs import NoParams
from worker.runtime import JobContext, TransientJobError, handles


@handles("forecast.si")
async def si(job: JobContext, params: NoParams) -> dict[str, Any]:
    out = await asyncio.to_thread(forecasts.issue_si)
    attempt, record = out["attempt"], out["record"]
    if record is not None and not attempt.get("existing"):
        await job.bus.publish("forecast.si", forecasts.si_event(attempt, record))
    return forecasts.summary(attempt)


@handles("forecast.charged_coke")
async def charged_coke(job: JobContext, params: NoParams) -> dict[str, Any]:
    out = await asyncio.to_thread(forecasts.issue_charged)
    attempt, record = out["attempt"], out["record"]
    if attempt.get("written"):
        event = forecasts.charged_event(record)
        if event is not None:
            await job.bus.publish("forecast.charged_coke", event)
    if attempt.get("behind") and job.attempt < job.kind.max_tries:
        raise TransientJobError(
            f"the hourly dataset of this hour is not published yet (latest build {attempt.get('built_at')}, "
            f"expected row {attempt.get('expected_row')})"
        )
    return forecasts.summary(attempt)
