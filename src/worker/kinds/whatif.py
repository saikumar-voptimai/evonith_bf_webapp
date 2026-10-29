"""What-if runs of the admin space (SV-4.3, #306), on request (``POST /api/admin/whatif/runs``).

whatif.forge   up to four FORGE cases at the calibration point of the what-if day (or the shipped calibration): the
               two-zone balance (ω held) and the oxygen-balance model (ηCO held), milliseconds per case
whatif.strata  up to four Burden & Fuel Optimiser plans on the calibration of the what-if day: forced blends or the
               cheapest blend, STRATA-ω (network solves in the model-solve pool) or Fixed-ω

Each case's result, with the basis and the fact-register version it used, is stored with the saved case of the user
who started the run (``services.whatif.store_results``: only when the case is still theirs and unchanged); the job
result is the whole run.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from services import optimiser, whatif
from services.jobs import WhatIfForgeParams, WhatIfStrataParams
from services.store import twin_store
from utils.burden_engine.facts import load_register
from worker.runtime import JobContext, handles


async def _owner(job: JobContext) -> str:
    record = await job.store.get(job.job_id) if job.store is not None else None
    return record.submitted_by if record is not None else "unknown"


def _json(doc: Any) -> Any:
    return json.loads(json.dumps(doc, default=str))


async def _store(kind: str, owner: str, run: dict[str, Any], results: list, specs: list) -> list[str]:
    def write() -> list[str]:
        return whatif.store_results(twin_store("twin-worker"), kind, owner, run, results, specs)

    try:
        return await asyncio.to_thread(write)
    except Exception as exc:  # the run is still the job's result; the cases keep their previous result
        raise RuntimeError(f"the results could not be stored with the cases ({type(exc).__name__}: {exc})") from exc


@handles("whatif.forge")
async def forge(job: JobContext, params: WhatIfForgeParams) -> dict[str, Any]:
    await job.progress(0.05, "Calibration of the what-if day", force=True)
    cal, basis = await asyncio.to_thread(whatif.resolve_calibration, params.calibration, params.day)
    specs = [c.model_dump(mode="json") for c in params.cases]
    await job.progress(0.3, f"Two-zone and oxygen-balance solves of {len(specs)} case(s)", force=True)
    out = await job.run_cpu(whatif.forge_compute_json, json.dumps(optimiser.jsonable(cal)), specs)
    run = _json(
        dict(
            kind="forge",
            computed_at=datetime.now(UTC),
            job_id=job.job_id,
            facts_version=out["facts_version"],
            basis=basis,
            base=out["base"],
        )
    )
    stored = await _store("forge", await _owner(job), run, out["cases"], specs)
    return run | {"cases": out["cases"], "notes": out["notes"], "stored": stored}


def _strata_case(params: WhatIfStrataParams, cal: dict[str, Any], basis: dict[str, Any]) -> optimiser.Case:
    """Page 11's Step 2 case on the what-if basis: the day's calibration and blast moisture (the shipped calibration
    at its own moisture), the requested yard, production and shaft-efficiency model."""
    from data.burden_engine.calibration import CalibrationInfo
    from services import day_calibration

    reference = params.calibration == "reference"
    info = None if reference else day_calibration.load(basis["day"])  # with the day's regime (production, controls)
    if info is None:
        note = "The shipped Jul-Oct 2026 calibration (what-if)." if reference else ""
        info = CalibrationInfo(cal, None, "fallback" if reference else "live", ("", ""), 0, note)
    chosen = dict(basis="reference") if reference else dict(basis="day", day=basis["day"].isoformat())
    settings = {"blast_moisture_g_nm3": basis["blast_moisture_g_nm3"]} if reference else None
    return optimiser.default_case(
        "step2",
        omega_mode=params.omega_mode,
        production_t_d=params.production_t_d,
        production_source="requested" if params.production_t_d is not None else None,
        yard_basis=params.yard,
        settings=settings,
        plan_day=None if reference else basis["day"],
        calibration_info=(info, chosen),
    )


@handles("whatif.strata")
async def strata(job: JobContext, params: WhatIfStrataParams) -> dict[str, Any]:
    await job.progress(0.05, "Calibration of the what-if day, yard and controls", force=True)
    cal, basis = await asyncio.to_thread(whatif.resolve_calibration, params.calibration, params.day)
    case = await asyncio.to_thread(_strata_case, params, cal, basis)
    _, inputs_json, cal_json, settings, limits, target = case.args[:6]
    specs = [c.model_dump(mode="json") for c in params.cases]
    label = "STRATA-ω (network solves)" if params.omega_mode == "strata" else "Fixed-ω"
    await job.progress(0.15, f"Solving {len(specs)} plan(s) at {target:,.0f} t/d with {label}", force=True)
    results = await job.run_cpu(
        whatif.strata_compute, inputs_json, cal_json, settings, limits, target, specs, case.names
    )
    versions = sorted({r.get("facts_version") for r in results if r.get("facts_version")})
    run = _json(
        dict(
            kind="strata",
            computed_at=datetime.now(UTC),
            job_id=job.job_id,
            facts_version=versions[0] if len(versions) == 1 else (", ".join(versions) or load_register().label),
            basis=basis
            | dict(
                omega_mode=params.omega_mode,
                yard=case.yard_source,
                yard_message=case.yard_message,
                production_source=case.production_source,
            ),
        )
    )
    stored = await _store("strata", await _owner(job), run, results, specs)
    return run | {"cases": results, "stored": stored}
