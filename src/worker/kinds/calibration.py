"""Nightly calibration (SV-1.7, #287): ``calibration.nightly`` at 02:30.

The two-zone calibration on the stable shifts dated in the 14 days before today (``services.day_calibration``) is
saved as the day's calibration of the optimiser runs, then the STRATA-Flow network Jacobian at it is built in the
model-solve pool (17 network solves, about 19 s on the development host; written to the disk cache that every engine
on this calibration reads). When the pipeline cannot run, or gives too few stable shifts and the calibration falls
back to the shipped one, the job is tried again in 10 min (three tries); the day's runs then calibrate live.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from services import day_calibration, optimiser
from services.jobs import NoParams
from worker.runtime import JobContext, TransientJobError, handles


@handles("calibration.nightly")
async def nightly(job: JobContext, params: NoParams) -> dict[str, Any]:
    day = day_calibration.today()
    await job.progress(0.05, f"Calibrating on the 14 days before {day:%d %b %Y}", force=True)
    cfg = optimiser.page_config(optimiser.config_mtime("burden_optimiser.yml"))
    try:
        info = await asyncio.to_thread(day_calibration.compute, day, cfg)
    except day_calibration.PlantDataUnavailable as exc:
        raise TransientJobError(f"the Coke Energy pipeline did not run ({exc})") from exc
    if info.source != "live":
        message = f"no live calibration: {info.message}"
        if job.attempt < job.kind.max_tries:
            raise TransientJobError(message)
        raise RuntimeError(message + " The day's optimiser runs calibrate live.")
    await job.progress(0.4, "STRATA-Flow network Jacobian at the calibration (17 solves)", force=True)
    network = await job.run_cpu(day_calibration.network_jacobian, json.dumps(optimiser.jsonable(info.cal)))
    previous = await asyncio.to_thread(day_calibration.load, day.fromordinal(day.toordinal() - 1))
    path = await asyncio.to_thread(
        day_calibration.save, info, day, made_at=datetime.now(UTC), job_id=job.job_id, network=network
    )
    out = day_calibration.summary(info) | {"day": day.isoformat(), "file": path.name, "network": network}
    if previous is not None:
        prev = day_calibration.summary(previous)
        out["change"] = {
            k: round(out[k] - prev[k], 4)
            for k in ("omega", "q_lz_gj_h", "q_uz_gj_h", "r_h", "coke_kg_thm")
            if out.get(k) is not None and prev.get(k) is not None
        }
    return out
