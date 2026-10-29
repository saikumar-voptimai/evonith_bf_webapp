"""Blend Mix Optimiser (SV-3.5, #302): LP runs at a fixed production, the forecast ledgers and the saved runs.

``POST /api/bmo/runs`` queues ``bmo.run`` (page 9's Balanced Optimizer with page 9's inputs and the requested changes;
composing the case reads Neon, InfluxDB and the hourly dataset, so it is a worker job) and answers 202 with the job;
``GET /api/bmo/runs/{job id}`` returns the job and, once it is done, its plan. The forecasts are read from the worker's
ledgers (SV-1.5) as page 9 reads them; nothing is issued here. Snapshots are the runs saved on page 9, without the
plant data frozen with them. Units are in the schemas (t/day, kg/tHM, Rs/tHM, Rs/t, mass %); times UTC unless IST.
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
from services import bmo_views
from services.auth import Principal
from services.bmo_views import (
    BmoChargedCokeView,
    BmoRunResult,
    BmoSiForecastView,
    BmoSnapshotDetail,
    BmoSnapshotList,
)
from services.jobs import BmoRunParams, JobRecord, JobStore

router = APIRouter(prefix="/api/bmo", tags=["bmo"], dependencies=[Depends(require_role("user"))])

RUN_KIND = "bmo.run"


class BmoRunSubmitted(BaseModel):
    job: JobRecord
    duplicate: bool = Field(..., description="True when an identical run was already queued or running")


class BmoRunLookup(BaseModel):
    job: JobRecord
    result: BmoRunResult | None = Field(None, description="The plan, once the job is done")


@router.post("/runs", response_model=BmoRunSubmitted, status_code=202, summary="Start a Blend Mix Optimiser run")
async def start_run(
    request: Request,
    jobs: Annotated[JobStore, Depends(job_store)],
    body: Annotated[BmoRunParams | None, Body()] = None,
) -> BmoRunSubmitted:
    """Queue ``bmo.run``: page 9's LP at a fixed production. Fields left out take page 9's defaults. Follow the job
    with GET /api/bmo/runs/{job id} or the ``job.done`` event. A request equal to a queued or running one returns it."""
    principal: Principal = request.state.principal
    try:
        record, duplicate = await jobs.submit(RUN_KIND, (body or BmoRunParams()).model_dump(), principal.username)
    except ValidationError as exc:
        errors = [{**e, "loc": ("body", *e["loc"])} for e in exc.errors(include_url=False)]
        raise RequestValidationError(jsonable_encoder(errors)) from exc
    return BmoRunSubmitted(job=record, duplicate=duplicate)


@router.get("/runs/{job_id}", response_model=BmoRunLookup, summary="A Blend Mix Optimiser run")
async def get_run(job_id: str, jobs: Annotated[JobStore, Depends(job_store)]) -> BmoRunLookup:
    """The job (records are kept 7 days) and its plan once it is done."""
    job = await jobs.get(job_id)
    if job is None or job.kind != RUN_KIND:
        raise ApiError(404, "No Blend Mix Optimiser run with this id.")
    result = BmoRunResult.model_validate(job.result) if job.status == "done" and job.result else None
    return BmoRunLookup(job=job, result=result)


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
