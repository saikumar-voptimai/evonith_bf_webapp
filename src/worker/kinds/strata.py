"""STRATA live point (SV-1.12, #292): ``strata.live_point`` at 03:15 IST, and on request from the STRATA page.

The calibration is the day's (``calibration.nightly`` at 02:30; the live one while it is missing, the shipped one
when the plant data cannot be read, labelled). In the model-solve pool: the MOD facts at that calibration
(``utils.burden_engine.model_facts`` on the reference yard and blend) and the ring view of the thermal reserve zone at
the calibration state (``services.strata.compute_live_point``: network solves only, the Jacobian from the nightly
calibration's disk cache). The document replaces ``src/storage/burden_engine/strata/live_point.json``, which
``GET /api/strata/live-point`` serves; the job result is its headline.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from services import optimiser, strata
from services.jobs import StrataLivePointParams
from worker.runtime import JobContext, handles


@handles("strata.live_point")
async def live_point(job: JobContext, params: StrataLivePointParams) -> dict[str, Any]:
    await job.progress(0.05, "Calibration of the live point", force=True)
    cfg = optimiser.page_config(optimiser.config_mtime("burden_optimiser.yml"))
    info, chosen = await asyncio.to_thread(optimiser.case_calibration, cfg, 0, params.calibration)
    calibration = strata.calibration_view(info, chosen)
    await job.progress(0.15, "MOD facts and ring view at the calibration (STRATA-Flow network solves)", force=True)
    result = await job.run_cpu(strata.compute_live_point, json.dumps(optimiser.jsonable(info.cal)))
    doc = strata.live_point_document(result, calibration, made_at=datetime.now(UTC), job_id=job.job_id)
    path = await asyncio.to_thread(strata.save_live_point, doc)
    responses = {r["id"]: r["strata_omega"] for r in result["responses"]}
    return dict(
        file=path.name,
        calibration=dict(
            source=calibration.get("source"), basis=calibration.get("basis"), window=calibration["window"]
        ),
        responses_strata=responses,
        omega_calibration=result["rings"]["omega"],
        network=dict(solves=result["network"].get("solves_run"), seconds=result["network"].get("solve_seconds")),
        seconds=result["seconds"],
    )
