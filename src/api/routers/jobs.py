"""Background jobs: catalogue, submission with de-duplication, status and cancellation (SV-0.8, #277); runtime
statistics per kind and the audit of the schedule (SV-1.7, #287), from ``twin.jobs`` (the Redis records of the last
7 days without a twin store). A cancel is recorded in the audit log (``job.cancel``, SV-4.8 #311) and needs the twin
store."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from api.auth import request_actor, require_role
from api.errors import ApiError
from api.live import job_store, live_state
from api.store import optional_twin_store
from data.twin.repositories import TwinStore
from services import job_audit, jobstats, schedule
from services.auth import Principal, Role, has_role
from services.events import JobDoneData
from services.jobs import CATALOG, QUEUE_NAME, RECORD_TTL_S, JobRecord, JobStore, UnknownJobKind

router = APIRouter(prefix="/api/jobs", tags=["jobs"], dependencies=[Depends(require_role("user"))])

FINISHED = ("done", "failed", "cancelled")


class JobKindInfo(BaseModel):
    name: str
    title: str
    role: Role = Field(..., description="Lowest role that may start the job")
    timeout_s: float
    max_tries: int
    schedule: str | None = Field(
        None, description="Scheduled times (IST), e.g. '04:00', 'every hour at :07', 'every 5 min'"
    )
    next_slot: datetime | None = Field(None, description="Next scheduled time (UTC); null for kinds without a schedule")
    description: str
    params_schema: dict[str, Any] = Field(..., description="JSON schema of the parameters")
    audited: bool = Field(False, description="A user's start is recorded in the audit log (action job.start)")


class KindStats(BaseModel):
    kind: str
    title: str
    schedule: str | None = Field(None, description="Scheduled times (IST)")
    timeout_s: float | None = Field(None, description="Time limit of one try, s")
    runs: int = Field(..., description="Jobs submitted in the period (skipped slots excluded), count")
    done: int = Field(..., description="Runs that ended done, count")
    failed: int = Field(..., description="Runs that failed, count")
    cancelled: int = Field(..., description="Runs cancelled by a user, count")
    skipped: int = Field(..., description="Schedule slots recorded as skipped, count")
    running: int = Field(..., description="Runs still queued or running, count")
    runtime_p50_s: float | None = Field(
        None, description="Median execution time of the done runs (over their tries), s"
    )
    runtime_p95_s: float | None = Field(None, description="95th percentile of the execution time of the done runs, s")
    runtime_max_s: float | None = Field(None, description="Longest execution time of a done run, s")
    start_delay_p95_s: float | None = Field(
        None, description="95th percentile of start minus schedule slot (scheduler) or submission (others), s"
    )
    last_finished_at: datetime | None = Field(None, description="End of the latest run that ended (UTC)")
    last_status: str | None = None


class JobStatsResponse(BaseModel):
    start: datetime = Field(..., description="Start of the period (UTC)")
    end: datetime = Field(..., description="End of the period (UTC)")
    source: Literal["twin.jobs", "redis"] = Field(..., description="Where the records were read")
    kinds: list[KindStats]


class SlotProblem(BaseModel):
    slot: datetime = Field(..., description="Schedule slot (UTC)")
    state: Literal["failed", "cancelled", "running", "skipped", "missed"]
    job_id: str | None = None
    reason: str | None = None


class KindSchedule(BaseModel):
    kind: str
    title: str
    schedule: str = Field(..., description="Scheduled times (IST)")
    expected: int = Field(..., description="Slots in the period, count")
    done: int = Field(..., description="Slots whose job ended done, count")
    failed: int = Field(..., description="Slots whose job failed, count")
    cancelled: int = Field(..., description="Slots whose job a user cancelled, count")
    running: int = Field(..., description="Slots whose job is still queued or running, count")
    skipped: int = Field(..., description="Slots recorded as skipped (worker not running, previous job active), count")
    missed: int = Field(..., description="Slots without any job record, count")
    clean: bool = Field(..., description="No slot missed or skipped")
    next_slot: datetime | None = Field(None, description="Next scheduled time (UTC)")
    problems: list[SlotProblem] = Field(..., description="The latest 50 slots not done, oldest first")


class ScheduleAuditResponse(BaseModel):
    start: datetime = Field(..., description="Start of the period (UTC)")
    end: datetime = Field(..., description="End of the period (UTC); slots of the last minute are not counted")
    source: Literal["twin.jobs", "redis"] = Field(..., description="Where the records were read")
    clean: bool = Field(..., description="No slot of any kind missed or skipped in the period")
    kinds: list[KindSchedule]


class SubmitRequest(BaseModel):
    kind: str = Field(..., description="Job kind, see /api/jobs/kinds")
    params: dict[str, Any] = Field(default_factory=dict)


class SubmitResponse(BaseModel):
    job: JobRecord
    duplicate: bool = Field(..., description="True when an identical job was already queued or running")


@router.get("/kinds", response_model=list[JobKindInfo], summary="Job kinds")
def kinds() -> list[JobKindInfo]:
    now = datetime.now(UTC)
    return [
        JobKindInfo(
            name=k.name,
            title=k.title,
            role=k.role,
            timeout_s=k.timeout_s,
            max_tries=k.max_tries,
            schedule=schedule.describe(k.schedule),
            next_slot=schedule.next_slot(k.schedule, now) if k.schedule else None,
            description=k.description,
            params_schema=k.params.model_json_schema(),
            audited=k.audited,
        )
        for k in CATALOG.values()
    ]


def optional_job_store(request: Request) -> JobStore | None:
    live = getattr(request.app.state, "live", None)
    return live.store if live is not None else None


async def _records(
    jobs: JobStore | None, twin: TwinStore | None, start: datetime, end: datetime
) -> tuple[list[Any], Literal["twin.jobs", "redis"]]:
    """Job records of [start, end): ``twin.jobs`` when a twin store is configured, else Redis (7 days)."""
    if twin is not None:
        return await asyncio.to_thread(twin.jobs.window, start, end), "twin.jobs"
    if jobs is None:
        raise ApiError(503, "Neither the twin store nor Redis is available: no job records to read.")
    return await jobs.list(limit=100_000), "redis"


def _period(days: float, end: datetime | None, source: str) -> tuple[datetime, datetime]:
    """[start, end) of the request; from Redis at most the 7 days its records are kept."""
    now = datetime.now(UTC)
    stop = min(schedule.utc(end), now) if end is not None else now
    start = stop - timedelta(days=days)
    if source == "redis":
        start = max(start, now - timedelta(seconds=RECORD_TTL_S))
    return start, stop


@router.get("/stats", response_model=JobStatsResponse, summary="Runtime statistics per job kind")
async def stats(
    jobs: Annotated[JobStore | None, Depends(optional_job_store)],
    twin: Annotated[TwinStore | None, Depends(optional_twin_store)],
    days: Annotated[float, Query(gt=0, le=92, description="Length of the period, d")] = 14,
    end: Annotated[datetime | None, Query(description="End of the period (default now)")] = None,
) -> JobStatsResponse:
    """Runs, outcomes, runtime percentiles (execution time over the tries) and start delay per kind for the jobs
    submitted in the period. Without a twin store only the Redis records of the last 7 days are available."""
    start, stop = _period(days, end, "twin.jobs" if twin is not None else "redis")
    rows, source = await _records(jobs, twin, start, stop)
    table = jobstats.stats(rows, CATALOG.values(), start=start, end=stop)
    return JobStatsResponse(start=start, end=stop, source=source, kinds=[KindStats(**vars(v)) for v in table.values()])


@router.get("/schedule", response_model=ScheduleAuditResponse, summary="Audit of the schedule")
async def schedule_audit(
    jobs: Annotated[JobStore | None, Depends(optional_job_store)],
    twin: Annotated[TwinStore | None, Depends(optional_twin_store)],
    days: Annotated[float, Query(gt=0, le=92, description="Length of the period, d")] = 14,
    end: Annotated[datetime | None, Query(description="End of the period (default now)")] = None,
) -> ScheduleAuditResponse:
    """For every scheduled kind, the slots of the period and what became of each: done, failed, cancelled, still
    running, skipped (with the reason) or missed (no record). Clean = no slot missed or skipped."""
    start, stop = _period(days, end, "twin.jobs" if twin is not None else "redis")
    rows, source = await _records(jobs, twin, start, stop)
    now = datetime.now(UTC)
    stop = min(stop, now - schedule.SETTLE)
    kinds_ = [k for k in CATALOG.values() if k.schedule is not None]
    table = schedule.audit(rows, kinds_, start, stop)
    out = [KindSchedule(**table[k.name].as_dict(), next_slot=schedule.next_slot(k.schedule, now)) for k in kinds_]
    return ScheduleAuditResponse(start=start, end=stop, source=source, clean=all(k.clean for k in out), kinds=out)


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
    twin: Annotated[TwinStore | None, Depends(optional_twin_store)],
) -> SubmitResponse:
    """Queue a job. A request equal to a queued or running job (same kind and parameters) returns that job. The
    start of an audited kind (``audited``) is recorded in the audit log and needs the twin store."""
    principal: Principal = request.state.principal
    kind = CATALOG.get(body.kind)
    if kind is None:
        raise ApiError(404, f"No job kind {body.kind!r}.")
    if not has_role(principal.role, kind.role):
        raise ApiError(403, f"Starting {kind.title} needs the {kind.role} role; signed in as {principal.role}.")
    if kind.audited and twin is None:
        raise ApiError(503, f"Starting {kind.title} is recorded in the audit log: the twin store is not configured.")
    try:
        if kind.audited and twin is not None:
            started = await job_audit.submit(store, twin, request_actor(request), kind.name, body.params, origin="jobs")
            record, duplicate = started.record, started.duplicate
        else:
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


async def audited_cancel(
    request: Request, store: JobStore, twin: TwinStore | None, job_id: str, *, origin: str
) -> job_audit.AuditedCancel:
    """Cancel through ``services.job_audit.cancel`` (the audit entry first), then announce a job ended before its
    start (``job.done``) and abort it in arq; 503 without the twin store, 404 without the record, 409 when ended."""
    from arq.jobs import Job

    if twin is None:
        raise ApiError(503, "Cancelling a job is recorded in the audit log: the twin store is not configured.")
    live = live_state(request)
    try:
        cancelled = await job_audit.cancel(store, twin, request_actor(request), job_id, origin=origin)
    except job_audit.JobNotFound as exc:
        raise ApiError(404, "No job with this id (records are kept for 7 days).") from exc
    except job_audit.JobConflict as exc:
        raise ApiError(409, str(exc)) from exc
    if cancelled.ended:
        await live.bus.publish(
            "job.done",
            JobDoneData(
                job_id=job_id, kind=cancelled.record.kind, status="cancelled", message=job_audit.CANCELLED_BEFORE_START
            ),
        )
    if live.redis is not None:
        # Not started: arq's abort keeps the worker from starting it; a running job is stopped.
        with contextlib.suppress(TimeoutError):
            await Job(job_id, live.redis, _queue_name=QUEUE_NAME).abort(timeout=0.2)
    return cancelled


@router.post(
    "/{job_id}/cancel",
    response_model=JobRecord,
    summary="Cancel a job (audited)",
    dependencies=[Depends(require_role("supervisor"))],
)
async def cancel(
    job_id: str,
    request: Request,
    store: Annotated[JobStore, Depends(job_store)],
    twin: Annotated[TwinStore | None, Depends(optional_twin_store)],
) -> JobRecord:
    """A queued job is not started; a running job is stopped. Both end with status ``cancelled``. The cancel is
    recorded in the audit log (``job.cancel``) before the job changes; without the twin store it is refused (503)."""
    cancelled = await audited_cancel(request, store, twin, job_id, origin="jobs")
    return await store.get(job_id) or cancelled.record
