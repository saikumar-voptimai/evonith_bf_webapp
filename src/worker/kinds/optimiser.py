"""Optimiser runs: on request (SV-1.6, #286) and on the schedule (SV-1.7, #287).

optimiser.run     one run with the requested mode and settings (``POST /api/optimiser/runs``)
optimiser.step2   06:15, 14:15, 22:15: Step 2 with STRATA-ω and Fixed-ω in parallel, two runs
optimiser.step1   06:30, 14:30, 22:30: Step 1 with STRATA-ω

Every case is page 11's default case (``services.optimiser.default_case``: live yard and prices, today's controls)
on the day's calibration (``services.day_calibration``, made at 02:30; the live one while it is missing) unless a
request asks for the live one. Step 2 runs at the current production: the latest twin window's (at most 3 h old),
else the median of the regime window. The cases are built in a thread, solved in the model-solve pool and stored in
``twin.optimiser_run``: ok, infeasible (the LP or a plausibility band refused the case, with the reason) or failed.
A scheduled run whose yard or calibration fell back to the shipped reference (Neon or the plant data unavailable) is
tried again after 5 min; the last try runs on the reference, labelled, as page 11 does. Without a twin store the
plans are returned in the job result only.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from services import optimiser
from services.jobs import NoParams, OptimiserRunParams
from services.store import twin_store
from utils.burden_engine import BandError, Infeasible
from worker.runtime import JobContext, TransientJobError, handles

LABELS = {"step1": "Step 1, cheapest production level", "step2": "Step 2, fixed production", "max_production":
          "maximum production"}  # fmt: skip
OMEGA_LABELS = {"strata": "STRATA-ω", "fixed": "Fixed-ω"}
HEADLINE = ("status", "production_t_d", "cost_rs_thm", "saving_rs_thm", "coke_kg_thm", "coke_sd_kg_thm",
            "fuel_rate_kg_thm", "certainty")  # fmt: skip


def _store_runs(rows: list[dict[str, Any]], job_record: dict[str, Any] | None) -> list[str | None]:
    store = twin_store("twin-worker")
    if store is None:
        return [None] * len(rows)
    if job_record is not None:  # the runs refer to their job; the archive may not have written it yet
        store.jobs.upsert(job_record)
    return [str(store.optimiser.add(row).id) for row in rows]


def _production(production_t_d: float | None) -> tuple[float | None, str | None, str | None]:
    """(target, its source, why there is no current one): the requested production, else the latest twin window's,
    else None (``default_case`` then takes the median of the regime window)."""
    if production_t_d is not None:
        return production_t_d, None, None
    try:
        value, label = optimiser.current_production(twin_store("twin-worker"))
    except Exception as exc:  # the twin store unreachable: the run still takes page 11's production
        return None, None, f"the twin store could not be read ({type(exc).__name__})"
    if value is None:
        return None, None, label
    return value, label, None


def _case(mode: str, omega_mode: str, production: tuple, **kw: Any) -> optimiser.Case:
    target, source, missing = production
    case = optimiser.default_case(mode, omega_mode=omega_mode, production_t_d=target, production_source=source, **kw)
    if missing and mode == "step2":
        case = replace(case, production_source=f"{case.production_source}; no current production: {missing}")
    return case


def _fallbacks(case: optimiser.Case) -> list[str]:
    out = []
    if case.yard_source != "live":
        out.append(f"yard: {case.yard_message or 'the reference yard of 16 Oct 2026'}")
    if case.calibration.get("source") != "live":
        out.append(f"calibration: {case.calibration.get('message') or 'the shipped calibration'}")
    return out


async def _solve(job: JobContext, case: optimiser.Case) -> tuple[dict[str, Any], BaseException | None]:
    """Solve one case in the pool; the run's row values and the failure that is not a refusal of the case."""
    try:
        res = await job.run_cpu(optimiser.solve_case, *case.args)
        return optimiser.run_values(case, res), None
    except (BandError, Infeasible) as exc:
        return optimiser.run_values(case, None, error=f"{type(exc).__name__}: {exc}", status="infeasible"), None
    except Exception as exc:
        return optimiser.run_values(case, None, error=f"{type(exc).__name__}: {exc}"[:400]), exc


async def _run_cases(job: JobContext, cases: list[optimiser.Case], *, message: str) -> dict[str, Any]:
    """Solve the cases in parallel in the pool and store them as runs of this job."""
    started = datetime.now(UTC)
    await job.progress(0.2, message, force=True)
    solved = await asyncio.gather(*(_solve(job, case) for case in cases))
    record = await job.store.get(job.job_id) if job.store is not None else None
    trigger = "schedule" if record is not None and record.submitted_by.startswith("scheduler") else "user"
    finished = datetime.now(UTC)
    rows = [
        {
            **values,
            "job_id": job.job_id if record is not None else None,
            "trigger": trigger,
            "requested_by": record.submitted_by if record is not None else "unknown",
            "started_at": started,
            "finished_at": finished,
        }
        for values, _ in solved
    ]
    run_ids = await asyncio.to_thread(_store_runs, rows, record.model_dump() if record is not None else None)
    failure = next((exc for _, exc in solved if exc is not None), None)
    if failure is not None:
        raise failure
    return {
        "runs": [
            {"run_id": run_id, "mode": case.mode, "omega_mode": case.omega_mode} | {k: values.get(k) for k in HEADLINE}
            for run_id, case, (values, _) in zip(run_ids, cases, solved, strict=True)
        ],
        "seconds": round((finished - started).total_seconds(), 1),
    }


@handles("optimiser.run")
async def run(job: JobContext, params: OptimiserRunParams) -> dict[str, Any]:
    await job.progress(0.05, "Building the case: yard, calibration and today's controls", force=True)
    production = (None, None, None)
    if params.mode == "step2":
        production = await asyncio.to_thread(_production, params.production_t_d)
    case = await asyncio.to_thread(
        _case,
        params.mode,
        params.omega_mode,
        production,
        certainty=params.certainty,
        risk_k=params.risk_k,
        auto=params.auto,
        calibration_basis=params.calibration,
    )
    out = await _run_cases(job, [case], message=f"Optimising: {LABELS[params.mode]}, {OMEGA_LABELS[params.omega_mode]}")
    (first,) = out["runs"]
    return first | {"seconds": out["seconds"]}


async def _scheduled_cases(job: JobContext, mode: str, omega_modes: tuple[str, ...]) -> list[optimiser.Case]:
    await job.progress(0.05, "Building the cases: yard, the day's calibration and today's controls", force=True)
    production = (None, None, None)
    if mode == "step2":
        production = await asyncio.to_thread(_production, None)
    cases = [
        await asyncio.to_thread(_case, mode, omega_mode, production, calibration_basis="day")
        for omega_mode in omega_modes
    ]
    reasons = sorted({r for case in cases for r in _fallbacks(case)})
    if reasons and job.attempt < job.kind.max_tries:
        raise TransientJobError(
            "plant data unavailable, the case fell back to the reference (" + "; ".join(reasons) + ")"
        )
    return cases


@handles("optimiser.step2")
async def step2(job: JobContext, params: NoParams) -> dict[str, Any]:
    cases = await _scheduled_cases(job, "step2", ("strata", "fixed"))
    target = cases[0].summary()["production_target_t_d"]
    out = await _run_cases(
        job, cases, message=f"Optimising Step 2 at {target:.0f} t/d with STRATA-ω and Fixed-ω in parallel"
    )
    return out | {"production_t_d": target, "production_source": cases[0].production_source}


@handles("optimiser.step1")
async def step1(job: JobContext, params: NoParams) -> dict[str, Any]:
    cases = await _scheduled_cases(job, "step1", ("strata",))
    return await _run_cases(job, cases, message="Optimising Step 1 (cheapest production level) with STRATA-ω")
