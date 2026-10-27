"""Background jobs: catalogue, submission with de-duplication, status and cancellation (SV-0.8, #277)."""

from __future__ import annotations

import contextlib
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from api.auth import require_role
from api.errors import ApiError
from api.live import job_store, live_state
from services.auth import Principal, Role, has_role
from services.events import JobDoneData
from services.jobs import CATALOG, QUEUE_NAME, JobRecord, JobStore, UnknownJobKind

router = APIRouter(prefix="/api/jobs", tags=["jobs"], dependencies=[Depends(require_role("user"))])

FINISHED = ("done", "failed", "cancelled")


class JobKindInfo(BaseModel):
    name: str
    title: str
    role: Role = Field(..., description="Lowest role that may start the job")
    timeout_s: float
    max_tries: int
    schedule: str | None = Field(None, description="Scheduled times (IST), e.g. '04:00'")
    description: str
    params_schema: dict[str, Any] = Field(..., description="JSON schema of the parameters")


class SubmitRequest(BaseModel):
    kind: str = Field(..., description="Job kind, see /api/jobs/kinds")
    params: dict[str, Any] = Field(default_factory=dict)


class SubmitResponse(BaseModel):
    job: JobRecord
    duplicate: bool = Field(..., description="True when an identical job was already queued or running")


def _schedule_text(schedule: tuple[set[int] | None, set[int]] | None) -> str | None:
    if schedule is None:
        return None
    hours, minutes = schedule
    hh = sorted(hours) if hours is not None else list(range(24))
    return ", ".join(f"{h:02d}:{m:02d}" for h in hh for m in sorted(minutes))


@router.get("/kinds", response_model=list[JobKindInfo], summary="Job kinds")
def kinds() -> list[JobKindInfo]:
    return [
        JobKindInfo(
            name=k.name,
            title=k.title,
            role=k.role,
            timeout_s=k.timeout_s,
            max_tries=k.max_tries,
            schedule=_schedule_text(k.schedule),
            description=k.description,
            params_schema=k.params.model_json_schema(),
        )
        for k in CATALOG.values()
    ]


@router.get("", response_model=list[JobRecord], summary="Recent jobs")
async def list_jobs(
    store: Annotated[JobStore, Depends(job_store)], limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[JobRecord]:
    """Jobs of the last 7 days, newest first."""
    return await store.list(limit)


@router.post("", response_model=SubmitResponse, status_code=202, summary="Start a job")
async def submit(
    body: SubmitRequest,
    request: Request,
    store: Annotated[JobStore, Depends(job_store)],
) -> SubmitResponse:
    """Queue a job. A request equal to a queued or running job (same kind and parameters) returns that job."""
    principal: Principal = request.state.principal
    kind = CATALOG.get(body.kind)
    if kind is None:
        raise ApiError(404, f"No job kind {body.kind!r}.")
    if not has_role(principal.role, kind.role):
        raise ApiError(403, f"Starting {kind.title} needs the {kind.role} role; signed in as {principal.role}.")
    try:
        record, duplicate = await store.submit(kind.name, body.params, principal.username)
    except ValidationError as exc:
        errors = [{**e, "loc": ("body", "params", *e["loc"])} for e in exc.errors(include_url=False)]
        raise RequestValidationError(jsonable_encoder(errors)) from exc
    except UnknownJobKind as exc:
        raise ApiError(404, f"No job kind {body.kind!r}.") from exc
    return SubmitResponse(job=record, duplicate=duplicate)


@router.get("/{job_id}", response_model=JobRecord, summary="Job status")
async def get_job(job_id: str, store: Annotated[JobStore, Depends(job_store)]) -> JobRecord:
    record = await store.get(job_id)
    if record is None:
        raise ApiError(404, "No job with this id (records are kept for 7 days).")
    return record


@router.post(
    "/{job_id}/cancel",
    response_model=JobRecord,
    summary="Cancel a job",
    dependencies=[Depends(require_role("supervisor"))],
)
async def cancel(job_id: str, request: Request, store: Annotated[JobStore, Depends(job_store)]) -> JobRecord:
    """A queued job is not started; a running job is stopped. Both end with status ``cancelled``."""
    from arq.jobs import Job

    record = await store.get(job_id)
    if record is None:
        raise ApiError(404, "No job with this id (records are kept for 7 days).")
    if record.status in FINISHED:
        raise ApiError(409, f"The job has already ended ({record.status}).")
    live = live_state(request)
    await store.update(job_id, cancel_requested=True)
    if record.status == "queued":
        # Not started: end it here; arq's abort below keeps the worker from starting it.
        ended = await store.finish(job_id, "cancelled", error="Cancelled before start.")
        await live.bus.publish(
            "job.done",
            JobDoneData(job_id=job_id, kind=record.kind, status="cancelled", message="Cancelled before start."),
        )
        record = ended or record
    if live.redis is not None:
        with contextlib.suppress(TimeoutError):
            await Job(job_id, live.redis, _queue_name=QUEUE_NAME).abort(timeout=0.2)
    return await store.get(job_id) or record
