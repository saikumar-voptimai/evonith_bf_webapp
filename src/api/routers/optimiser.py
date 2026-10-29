"""Optimiser runs for the Live Twin plan card and the Burden & Fuel Optimiser: start a run, read runs and their typed
results, accept or reject a plan (SV-1.6, #286), write and read the furnace commentary (SV-3.2, #299); the plant
bands, facts and assumptions the Burden & Fuel Optimiser page shows beside a result (SV-3.3, #300).

A run is the job ``optimiser.run`` (page 11's default case, ``services.optimiser.default_case``); the worker stores
its plan in ``twin.optimiser_run``. A decision is appended to ``twin.plan_decision`` with its audit row in the same
transaction (``audit_log``, action ``plan.decide``); a later decision on the same run supersedes the earlier one.
Costs are Rs/tHM (ore + flux + fuel − slag credit), coke kg/tHM dry, production t/d, times UTC.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from api.auth import auth_state, client_address, require_role
from api.errors import ApiError
from api.live import job_store, live_state
from api.store import twin_store
from data.twin.repositories import Actor, NotFound, TwinStore
from services import optimiser_commentary, twin_reads, twin_views
from services.auth import Principal
from services.jobs import JobRecord, JobStore, OptimiserRunParams
from services.optimiser_reference import OptimiserReference, reference_view
from services.optimiser_results import OptimiserResult, result_view

router = APIRouter(prefix="/api/optimiser", tags=["optimiser"], dependencies=[Depends(require_role("user"))])

Mode = Literal["step1", "step2", "max_production"]


class PlanDecisionView(BaseModel):
    run_id: str = Field(..., description="Run id (UUID)")
    decision: Literal["accepted", "rejected"]
    decided_by: str
    decided_at: datetime = Field(..., description="UTC")
    shift_key: str | None = Field(None, description="Shift of the decision (YYYY-MM-DD_A, IST date)")
    comment: str | None = None


class OptimiserRunSummary(BaseModel):
    id: str = Field(..., description="Run id (UUID)")
    job_id: str | None = Field(None, description="Job that produced the run")
    trigger: Literal["schedule", "user"]
    requested_by: str
    mode: Mode = Field(..., description="step1: cheapest production level; step2: fixed production; max_production")
    omega_mode: Literal["strata", "fixed"] = Field(..., description="STRATA-ω (blend-dependent) or Fixed-ω")
    status: Literal["ok", "infeasible", "failed"]
    started_at: datetime = Field(..., description="UTC")
    finished_at: datetime = Field(..., description="UTC")
    production_t_d: float | None = Field(None, description="Hot metal of the plan, t/d")
    cost_rs_thm: float | None = Field(None, description="Ore + flux + fuel − slag credit of the plan, Rs/tHM")
    saving_rs_thm: float | None = Field(
        None, description="Cost of today's charged blend minus the plan's, Rs/tHM (Step 2 with the live yard)"
    )
    coke_kg_thm: float | None = Field(None, description="Coke rate of the plan, kg/tHM dry")
    coke_sd_kg_thm: float | None = Field(None, description="1 sigma of the coke rate, kg/tHM dry (STRATA-ω)")
    fuel_rate_kg_thm: float | None = Field(None, description="Coke (wet) + nut coke + PCI of the plan, kg/tHM")
    certainty: str | None = Field(None, description="High, Medium, Low or Investigate (STRATA-ω)")
    decision: PlanDecisionView | None = Field(None, description="The decision that applies (the latest)")


class OptimiserRunDetail(OptimiserRunSummary):
    inputs: dict[str, Any] = Field(default_factory=dict, description="The case: settings, limits, target, sources")
    result: dict[str, Any] = Field(default_factory=dict, description="The engine's result (plans, frontier, notes)")
    versions: dict[str, Any] = Field(default_factory=dict, description="Calibration and yard source")
    decisions: list[PlanDecisionView] = Field(default_factory=list, description="Every decision, oldest first")


class RunSubmitted(BaseModel):
    job: JobRecord
    duplicate: bool = Field(..., description="True when an identical run was already queued or running")


class RunLookup(BaseModel):
    job: JobRecord | None = Field(None, description="The job (records are kept 7 days)")
    run: OptimiserRunDetail | None = Field(None, description="The stored run, once the job has finished")


class CommentaryView(BaseModel):
    id: str = Field(..., description="Commentary id (UUID)")
    run_id: str = Field(..., description="Run id (UUID)")
    job_id: str | None = Field(None, description="Job that wrote it")
    requested_by: str
    generated_at: datetime = Field(..., description="UTC")
    status: Literal["ok", "failed"] = Field(..., description="failed: no text after the LLM's retry")
    text: str | None = Field(None, description="The commentary (Markdown), generated text")
    error: str | None = Field(None, description="Error of each try (failed)")
    tries: int = Field(0, description="LLM calls made, count")
    model: str | None = Field(None, description="Model that wrote it")
    missing: list[str] = Field(default_factory=list, description="What the context lacked")
    notes_sha1: str | None = Field(None, description="SHA-1 of the optimiser notes in the context")
    facts_version: str | None = Field(None, description="Fact register version in the context")
    context_sha1: str = Field(..., description="SHA-1 of the context")
    context: str | None = Field(None, description="The full context the model was given (with context=true)")


class DecisionRequest(BaseModel):
    model_config = {"extra": "forbid"}

    decision: Literal["accepted", "rejected"]
    comment: str | None = Field(None, max_length=2000, description="Reason, shown with the decision")


_summary = twin_reads.run_summary  # also FurnaceMind's get_optimiser_result (SV-5.3)


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


@router.post(
    "/runs",
    response_model=RunSubmitted,
    status_code=202,
    summary="Start an optimiser run",
    dependencies=[Depends(require_role("supervisor"))],
)
async def start_run(
    body: OptimiserRunParams, request: Request, jobs: Annotated[JobStore, Depends(job_store)]
) -> RunSubmitted:
    """Queue ``optimiser.run`` with page 11's defaults for the mode; follow it with GET /api/optimiser/runs/{job id}
    or the ``job.progress`` / ``job.done`` events. A request equal to a queued or running run returns that run."""
    principal: Principal = request.state.principal
    try:
        record, duplicate = await jobs.submit("optimiser.run", body.model_dump(), principal.username)
    except ValidationError as exc:
        errors = [{**e, "loc": ("body", *e["loc"])} for e in exc.errors(include_url=False)]
        raise RequestValidationError(jsonable_encoder(errors)) from exc
    return RunSubmitted(job=record, duplicate=duplicate)


@router.get("/runs", response_model=list[OptimiserRunSummary], summary="Recent optimiser runs")
def list_runs(
    store: Annotated[TwinStore, Depends(twin_store)],
    mode: Annotated[Mode | None, Query(description="Only this mode")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[OptimiserRunSummary]:
    """Newest first, with the decision that applies to each."""
    return [_summary(store, r) for r in store.optimiser.recent(mode=mode, limit=limit)]


@router.get("/runs/{run_or_job_id}", response_model=RunLookup, summary="An optimiser run or its job")
async def get_run(run_or_job_id: str, request: Request, store: Annotated[TwinStore, Depends(twin_store)]) -> RunLookup:
    """By run id (UUID) or by the job id returned when the run was started."""
    live = live_state(request)
    run_id = _uuid(run_or_job_id)
    row = await asyncio.to_thread(store.optimiser.get, run_id) if run_id is not None else None
    job = None
    job_id = row.job_id if row is not None else run_or_job_id
    if job_id and live.store is not None:
        job = await live.store.get(job_id)
        if row is None and job is not None and job.kind != "optimiser.run":
            raise ApiError(404, "This job is not an optimiser run.")
        if row is None and job is not None and (job.result or {}).get("run_id"):
            row = await asyncio.to_thread(store.optimiser.get, job.result["run_id"])
    if row is None and job is None:
        raise ApiError(404, "No optimiser run or job with this id.")
    run = await asyncio.to_thread(_summary, store, row, detail=True) if row is not None else None
    return RunLookup(job=job, run=run)


def _run_row(store: TwinStore, run_id: str) -> Any:
    key = _uuid(run_id)
    if key is None:
        raise ApiError(422, "run_id must be a run id (UUID).")
    row = store.optimiser.get(key)
    if row is None:
        raise ApiError(404, "No optimiser run with this id.")
    return row


@router.get("/runs/{run_id}/result", response_model=OptimiserResult, summary="The typed result of a run")
def run_result(run_id: str, store: Annotated[TwinStore, Depends(twin_store)]) -> OptimiserResult:
    """Plan (blend, flux, cost split, coke ± σ, fuel, slag, charging, process, shaft efficiency and certainty, binding
    limits), Step 2's ore + flux-only plan and today's charged blend, Step 1's highest output and frontier, the value
    of limits, the critical assumptions and the network solves (``services.optimiser_results``)."""
    return result_view(_run_row(store, run_id))


@router.post(
    "/runs/{run_id}/commentary",
    response_model=RunSubmitted,
    status_code=202,
    summary="Write the furnace commentary on a run",
)
async def start_commentary(
    run_id: str,
    request: Request,
    store: Annotated[TwinStore, Depends(twin_store)],
    jobs: Annotated[JobStore, Depends(job_store)],
) -> RunSubmitted:
    """Queue ``optimiser.commentary`` (page 11's commentary: optimiser notes, fact register and the run as context,
    one LLM retry); read it with GET /api/optimiser/runs/{run_id}/commentary once the job is done."""
    row = await asyncio.to_thread(_run_row, store, run_id)
    if row.status != "ok":
        raise ApiError(409, f"The run has no plan to comment on (status {row.status}).")
    principal: Principal = request.state.principal
    record, duplicate = await jobs.submit("optimiser.commentary", {"run_id": str(row.id)}, principal.username)
    return RunSubmitted(job=record, duplicate=duplicate)


@router.get("/runs/{run_id}/commentary", response_model=CommentaryView, summary="The latest commentary on a run")
def get_commentary(
    run_id: str,
    store: Annotated[TwinStore, Depends(twin_store)],
    context: Annotated[bool, Query(description="Include the full context the model was given")] = False,
) -> CommentaryView:
    row = _run_row(store, run_id)
    found = store.optimiser.commentary(row.id)
    if found is None:
        raise ApiError(404, "No commentary on this run yet.")
    data = optimiser_commentary.view(found)
    if not context:
        data["context"] = None
    return CommentaryView.model_validate(data)


@router.get(
    "/reference", response_model=OptimiserReference, summary="Plant bands, facts and assumptions of the optimiser"
)
def reference() -> OptimiserReference:
    """The plant's slag basicity bands (``utils.slag_bands``), the register's plausibility bands (BAND-01..03), the
    facts the engine rests on with Fixed-ω and STRATA-ω side by side, and the assumptions (page 11's expanders)."""
    return reference_view()


@router.get("/latest", response_model=OptimiserRunDetail, summary="Latest optimiser plan")
def latest(
    store: Annotated[TwinStore, Depends(twin_store)],
    mode: Annotated[Mode | None, Query(description="Only this mode (default: any)")] = None,
) -> OptimiserRunDetail:
    """The latest run with status ok, with its decisions."""
    row = store.optimiser.latest(mode=mode)
    if row is None:
        raise ApiError(404, "No optimiser run has a plan yet.")
    return OptimiserRunDetail.model_validate(_summary(store, row, detail=True))


@router.post(
    "/runs/{run_id}/decision",
    response_model=PlanDecisionView,
    status_code=201,
    summary="Accept or reject a plan",
    dependencies=[Depends(require_role("supervisor"))],
)
def decide(
    run_id: str, body: DecisionRequest, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> PlanDecisionView:
    """Append the decision, with its audit row (``plan.decide``), in the current shift. A later decision supersedes."""
    key = _uuid(run_id)
    if key is None:
        raise ApiError(422, "run_id must be a run id (UUID).")
    principal: Principal = request.state.principal
    actor = Actor(
        name=principal.username,
        role=principal.role,
        client_ip=client_address(request, auth_state(request).settings.trusted_proxy_set),
    )
    run = store.optimiser.get(key)
    if run is None:
        raise ApiError(404, "No optimiser run with this id.")
    if run.status != "ok":
        raise ApiError(409, f"The run has no plan to decide on (status {run.status}).")
    try:
        row = store.optimiser.decide(
            key,
            body.decision,
            actor,
            shift_key=twin_views.current_shift_key(),
            comment=(body.comment or "").strip() or None,
        )
    except NotFound as exc:
        raise ApiError(404, "No optimiser run with this id.") from exc
    return PlanDecisionView.model_validate(twin_views.decision_data(row))
