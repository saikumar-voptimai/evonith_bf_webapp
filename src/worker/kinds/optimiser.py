"""Optimiser runs (SV-1.6, #286): page 11's default case, solved in the model-solve pool, stored in
``twin.optimiser_run``.

The case (yard, calibration, today's controls) is built in a thread; the solve runs in the process pool. A case the
LP or a plausibility band refuses is stored as ``infeasible`` with the reason; any other failure as ``failed``, and
the job fails. Without a configured twin store the plan is returned in the job result only.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from services import optimiser
from services.jobs import OptimiserRunParams
from services.store import twin_store
from utils.burden_engine import BandError, Infeasible
from worker.runtime import JobContext, handles

LABELS = {"step1": "Step 1, cheapest production level", "step2": "Step 2, fixed production", "max_production":
          "maximum production"}  # fmt: skip
HEADLINE = ("status", "production_t_d", "cost_rs_thm", "saving_rs_thm", "coke_kg_thm", "coke_sd_kg_thm",
            "fuel_rate_kg_thm", "certainty")  # fmt: skip


def _store_run(row: dict[str, Any], job_record: dict[str, Any] | None) -> str | None:
    store = twin_store("twin-worker")
    if store is None:
        return None
    if job_record is not None:  # the run refers to its job; the archive may not have written it yet
        store.jobs.upsert(job_record)
    return str(store.optimiser.add(row).id)


@handles("optimiser.run")
async def run(job: JobContext, params: OptimiserRunParams) -> dict[str, Any]:
    started = datetime.now(UTC)
    await job.progress(0.05, "Building the case: yard, calibration and today's controls", force=True)
    case = await asyncio.to_thread(
        optimiser.default_case,
        params.mode,
        omega_mode=params.omega_mode,
        production_t_d=params.production_t_d,
        certainty=params.certainty,
        risk_k=params.risk_k,
        auto=params.auto,
    )
    omega = "STRATA-ω" if params.omega_mode == "strata" else "Fixed-ω"
    await job.progress(0.2, f"Optimising: {LABELS[params.mode]}, {omega}", force=True)
    failure: BaseException | None = None
    try:
        res = await job.run_cpu(optimiser.compute_mode, *case.args)
        values = optimiser.run_values(case, res)
    except (BandError, Infeasible) as exc:
        values = optimiser.run_values(case, None, error=f"{type(exc).__name__}: {exc}", status="infeasible")
    except Exception as exc:
        failure = exc
        values = optimiser.run_values(case, None, error=f"{type(exc).__name__}: {exc}"[:400])
    record = await job.store.get(job.job_id) if job.store is not None else None
    row = {
        **values,
        "job_id": job.job_id if record is not None else None,
        "trigger": "schedule" if record is not None and record.submitted_by == "scheduler" else "user",
        "requested_by": record.submitted_by if record is not None else "unknown",
        "started_at": started,
        "finished_at": datetime.now(UTC),
    }
    run_id = await asyncio.to_thread(_store_run, row, record.model_dump() if record is not None else None)
    if failure is not None:
        raise failure
    return {"run_id": run_id, "mode": params.mode, "omega_mode": params.omega_mode} | {
        k: values.get(k) for k in HEADLINE
    }
