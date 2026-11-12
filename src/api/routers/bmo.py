"""Blend Mix Optimiser (SV-3.5, #302): the forecast ledgers, the saved runs and the plan board.

The LP runs of the Svelte app (``POST /api/bmo/runs``, job ``bmo.run``) were removed on 12 Nov 2026 (user decision:
page 9's LP stays in Streamlit only; ``services.bmo_views.run_case`` remains for page 9's parity tests). The forecasts
are read from the worker's ledgers (SV-1.5) as page 9 reads them; nothing is issued here. Snapshots are the runs saved
on page 9, without the plant data frozen with them. Units are in the schemas (t/day, kg/tHM, Rs/tHM, Rs/t, mass %);
times UTC unless IST.

SV-7.4 (#327) adds the plan board: ``GET /api/bmo/plan`` names the burden engine's Step 2 runs the page shows (the
shift's scheduled STRATA-ω run, the latest run, the implied production of the latest Step 1 run) and
``POST /api/bmo/plan/runs`` queues ``optimiser.run`` with Fixed-ω from the page's constraints (``services.bmo_plan``);
the plan itself is read from ``GET /api/optimiser/runs/{run_id}/result``.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from api.auth import require_role
from api.errors import ApiError
from api.live import job_store
from api.store import twin_store
from data.twin.repositories import TwinStore
from services import bmo_plan, bmo_views
from services.auth import Principal
from services.bmo_plan import BmoPlanBoard, BmoPlanRunParams
from services.bmo_views import BmoChargedCokeView, BmoSiForecastView, BmoSnapshotDetail, BmoSnapshotList
from services.jobs import JobRecord, JobStore

router = APIRouter(prefix="/api/bmo", tags=["bmo"], dependencies=[Depends(require_role("user"))])

PLAN_RUN_KIND = "optimiser.run"


class BmoRunSubmitted(BaseModel):
    job: JobRecord
    duplicate: bool = Field(..., description="True when an identical run was already queued or running")


# ---------------------------------------------------------------------------------------- plan (SV-7.4, #327)
@router.get("/plan", response_model=BmoPlanBoard, summary="The Blend Mix Optimiser plan board")
async def plan_board(store: Annotated[TwinStore, Depends(twin_store)]) -> BmoPlanBoard:
    """The Step 2 runs the page opens on: the implied production (latest ok Step 1 run), the shift's scheduled STRATA-ω
    run of the last day, the latest ok Step 2 run of any kind and the last day's Step 2 runs, each with the decision
    that applies. The plan of a run is GET /api/optimiser/runs/{run_id}/result; a decision is
    POST /api/optimiser/runs/{run_id}/decision (supervisor)."""
    return await asyncio.to_thread(bmo_plan.board, store)


@router.post(
    "/plan/runs", response_model=BmoRunSubmitted, status_code=202, summary="Run the plan from the page's constraints"
)
async def start_plan_run(
    request: Request,
    jobs: Annotated[JobStore, Depends(job_store)],
    body: Annotated[BmoPlanRunParams | None, Body()] = None,
) -> BmoRunSubmitted:
    """Queue ``optimiser.run`` with Fixed-ω on the day's calibration and the live yard: Step 2 at the given production
    (default: the current one) with page 9's slag window and charging limits, or Step 1 to recompute the implied
    production (the shift's scheduled run refines it). Follow the job with GET /api/optimiser/runs/{job id} or the
    ``job.done`` event. A request equal to a queued or running one returns it."""
    principal: Principal = request.state.principal
    params = bmo_plan.run_params(body or BmoPlanRunParams())
    try:
        record, duplicate = await jobs.submit(PLAN_RUN_KIND, params.model_dump(), principal.username)
    except ValidationError as exc:
        errors = [{**e, "loc": ("body", *e["loc"])} for e in exc.errors(include_url=False)]
        raise RequestValidationError(jsonable_encoder(errors)) from exc
    return BmoRunSubmitted(job=record, duplicate=duplicate)


@router.get("/forecasts/si", response_model=BmoSiForecastView, summary="Hot-metal Si forecast")
async def si_forecast() -> BmoSiForecastView:
    """The Si path (Now, +1, +2, +3 h) of the current origin from the worker's ledger; the previous origin's while the
    current one is being issued; an out-of-date record without values otherwise (``note`` says since when)."""
    return await asyncio.to_thread(bmo_views.si_forecast)


@router.get("/forecasts/charged-coke", response_model=BmoChargedCokeView, summary="Charged-coke forecast")
async def charged_coke_forecast() -> BmoChargedCokeView:
    """The 4-h charged-coke block for the row of the latest hourly build from the worker's ledger (the row before
    while it is being issued); without a forecast when the ledger is out of date (``reasons``)."""
    return await asyncio.to_thread(bmo_views.charged_coke_forecast)


@router.get("/snapshots", response_model=BmoSnapshotList, summary="Saved Blend Mix Optimiser runs")
async def list_snapshots(
    limit: Annotated[int, Query(ge=1, le=500, description="Snapshots, count")] = 50,
) -> BmoSnapshotList:
    """Page 9's snapshots, newest first: when, source, label and the recommended result's headline."""
    return await asyncio.to_thread(bmo_views.snapshot_list, limit=limit)


@router.get("/snapshots/{snapshot_id}", response_model=BmoSnapshotDetail, summary="A saved Blend Mix Optimiser run")
async def get_snapshot(snapshot_id: str) -> BmoSnapshotDetail:
    """Inputs (tables as row lists), the LP and DE results and the run's provenance; the frozen plant data stay in the
    file (Sandbox replay on page 9)."""
    detail = await asyncio.to_thread(bmo_views.snapshot_detail, snapshot_id)
    if detail is None:
        raise ApiError(404, "No Blend Mix Optimiser snapshot with this id.")
    return detail
