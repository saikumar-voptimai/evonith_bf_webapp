"""Jobs and system health of the admin space (SV-4.8, #311). Administrators only.

``GET /api/admin/jobs``: the job history of a period (``services.job_history``: ``twin.jobs`` merged with the live
Redis records) with durations, failures and the controls that apply. ``POST /api/admin/jobs/{id}/cancel`` and
``/retry``: job control, each recorded in the audit log (``services.job_audit``: ``job.cancel``, ``job.retry``).
``GET /api/admin/health``: every source with its last timestamp and age, the services (API, Redis, twin store,
worker heartbeat) and the host (CPU, RAM, disks, temperatures: tegrastats on the Jetson, psutil elsewhere). Runtime
statistics and the schedule audit stay in ``/api/jobs/stats`` and ``/api/jobs/schedule``. Times UTC.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from api import __version__
from api.auth import request_actor, require_role
from api.errors import ApiError
from api.live import job_store
from api.routers.jobs import audited_cancel, optional_job_store
from api.routers.system import STARTED_AT
from api.settings import get_settings
from api.store import optional_twin_store, twin_store
from data.twin.repositories import TwinStore
from services import host_health, job_audit, job_history, source_health, system_health, twin_reads
from services.events import JobState
from services.jobs import JobRecord, JobStore, UnknownJobKind
from services.release import current_release

log = logging.getLogger("api.admin")

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_role("admin"))])

JobId = Annotated[str, Path(min_length=1, max_length=64, description="Job id")]
SourceState = Literal["ok", "stale", "down", "missing", "off"]
LimitState = Literal["ok", "warning", "critical"]


# --------------------------------------------------------------------------------------------------------- jobs
class JobHistoryRow(BaseModel):
    id: str = Field(..., description="Job id")
    kind: str = Field(..., description="Job kind, e.g. twin.evaluate")
    title: str = Field(..., description="Title of the kind (the kind itself when it left the catalogue)")
    status: JobState
    submitted_by: str = Field(..., description="User name, or scheduler / scheduler (catch-up) / scheduler (skipped)")
    scheduled: bool = Field(..., description="Started by the schedule rather than by a user")
    skipped: bool = Field(..., description="A schedule slot recorded as skipped (the job did not run)")
    scheduled_for: datetime | None = Field(None, description="Schedule slot the job serves (UTC)")
    submitted_at: datetime | None = Field(None, description="Submission (UTC)")
    started_at: datetime | None = Field(None, description="Start of the first try (UTC)")
    finished_at: datetime | None = Field(None, description="End (UTC)")
    wait_s: float | None = Field(None, description="Start − submission; now − submission while queued, s")
    run_s: float | None = Field(None, description="Execution time summed over the tries, s")
    elapsed_s: float | None = Field(None, description="now − start while running, s")
    tries: int = Field(..., description="Tries made, count")
    max_tries: int | None = Field(None, description="Tries the kind allows, count")
    timeout_s: float | None = Field(None, description="Time limit of one try, s")
    progress: float | None = Field(None, description="Progress of a running job, fraction 0-1 (Redis records only)")
    message: str | None = Field(None, description="Latest progress message or outcome")
    error: str | None = Field(None, description="Why the job failed or was cancelled")
    params: dict[str, Any] = Field(..., description="Parameters of the job (units in the kind's parameter schema)")
    cancel_requested: bool = Field(..., description="A cancel was requested (Redis records only)")
    can_cancel: bool = Field(..., description="Queued or running, no cancel requested yet, live record in Redis")
    can_retry: bool = Field(..., description="Failed or cancelled, and its kind is still in the catalogue")
    live: bool = Field(..., description="Read from Redis (the live record) rather than from twin.jobs")


class JobHistory(BaseModel):
    start: datetime = Field(..., description="Start of the period (UTC)")
    end: datetime = Field(..., description="End of the period (UTC)")
    sources: list[Literal["twin.jobs", "redis"]] = Field(..., description="Stores the rows were read from")
    limit: int = Field(..., description="Rows at most, count")
    rows: list[JobHistoryRow] = Field(..., description="Jobs submitted in the period, newest first")


class JobCancelled(BaseModel):
    job: JobRecord
    ended: bool = Field(..., description="The job was queued and is cancelled; a running job stops in the worker")
    audit_id: int = Field(..., description="Audit-log entry of the cancel (job.cancel)")


class JobRetried(BaseModel):
    job: JobRecord
    duplicate: bool = Field(..., description="True when an identical job was already queued or running")
    audit_id: int = Field(..., description="Audit-log entry of the retry (job.retry)")
    retry_of: str = Field(..., description="Id of the failed or cancelled job retried")


@router.get("/jobs", response_model=JobHistory, summary="Job history")
async def job_history_view(
    jobs: Annotated[JobStore | None, Depends(optional_job_store)],
    twin: Annotated[TwinStore | None, Depends(optional_twin_store)],
    kind: Annotated[str | None, Query(max_length=64, description="Only this kind, e.g. optimiser.step2")] = None,
    status: Annotated[JobState | None, Query(description="Only this status")] = None,
    days: Annotated[float, Query(gt=0, le=job_history.MAX_DAYS, description="Length of the period, d")] = 7,
    end: Annotated[datetime | None, Query(description="End of the period (default now; naive is UTC)")] = None,
    limit: Annotated[int, Query(ge=1, le=job_history.MAX_ROWS, description="Rows at most, count")] = 200,
) -> JobHistory:
    """Jobs submitted in the period, newest first, from ``twin.jobs`` (92 days) merged with the live Redis records
    (7 days; their progress and cancel flag): durations (waiting, execution over the tries, elapsed while running),
    tries, the error of a failed job and whether it can be cancelled or retried."""
    now = datetime.now(UTC)
    stop = min(twin_reads.aware(end, now), now)
    start = stop - timedelta(days=days)
    try:
        rows, sources = await job_history.read(
            jobs, twin, since=start, until=stop, kind=(kind or "").strip() or None, status=status, limit=limit, now=now
        )
    except job_history.Unavailable as exc:
        raise ApiError(503, str(exc)) from exc
    return JobHistory(start=start, end=stop, sources=sources, limit=limit, rows=rows)  # type: ignore[arg-type]


@router.post("/jobs/{job_id}/cancel", response_model=JobCancelled, summary="Cancel a job (audited)")
async def cancel_job(
    job_id: JobId,
    request: Request,
    jobs: Annotated[JobStore, Depends(job_store)],
    twin: Annotated[TwinStore, Depends(twin_store)],
) -> JobCancelled:
    """Cancel a queued or running job; the cancel is recorded in the audit log (``job.cancel``, with the job before
    and after) before the job changes. A queued job ends at once (``cancelled``), a running one when the worker stops
    it. 409 when the job has ended."""
    cancelled = await audited_cancel(request, jobs, twin, job_id, origin="admin jobs")
    return JobCancelled(job=cancelled.record, ended=cancelled.ended, audit_id=cancelled.audit_id)


@router.post("/jobs/{job_id}/retry", response_model=JobRetried, status_code=202, summary="Retry a job (audited)")
async def retry_job(
    job_id: JobId,
    request: Request,
    jobs: Annotated[JobStore, Depends(job_store)],
    twin: Annotated[TwinStore, Depends(twin_store)],
) -> JobRetried:
    """Queue the kind and parameters of a failed or cancelled job again (``job.retry`` in the audit log, with the job
    retried). A scheduler job's parameters take their defaults now (e.g. the current hour), not at its slot. An
    identical job already queued or running is returned instead (``duplicate``), and recorded as well. 409 when the
    job neither failed nor was cancelled; 422 when its parameters no longer fit the kind."""
    try:
        started = await job_audit.retry(jobs, twin, request_actor(request), job_id, origin="admin jobs")
    except job_audit.JobNotFound as exc:
        raise ApiError(404, "No job with this id in Redis (7 days) or twin.jobs.") from exc
    except job_audit.JobConflict as exc:
        raise ApiError(409, str(exc)) from exc
    except UnknownJobKind as exc:
        raise ApiError(409, f"The job kind {exc.args[0]!r} is no longer in the catalogue.") from exc
    except job_audit.RoleRefused as exc:
        raise ApiError(403, str(exc)) from exc
    except ValidationError as exc:
        errors = [{**e, "loc": ("body", "params", *e["loc"])} for e in exc.errors(include_url=False)]
        raise RequestValidationError(jsonable_encoder(errors)) from exc
    return JobRetried(job=started.record, duplicate=started.duplicate, audit_id=started.audit_id, retry_of=job_id)


# ------------------------------------------------------------------------------------------------------- health
class SourceRow(BaseModel):
    key: str = Field(..., description="influxdb, hourly, hm_slag, dpr, open_meteo, evaluation")
    label: str
    group: Literal["data", "twin"] = Field(..., description="data: a data source; twin: the twin's own evaluation")
    last_at: datetime | None = Field(
        None, description="Newest data time the source holds (UTC; the window end for the evaluation); null when none"
    )
    age_s: float | None = Field(None, description="Age of last_at at checked time, s; null without last_at")
    stale_after_s: float | None = Field(None, description="Age from which the source is stale, s")
    state: SourceState = Field(..., description="ok, stale, down (last fetch failed), missing (no time), off")
    checked_at: datetime | None = Field(None, description="Last successful fetch or evaluation (UTC)")
    error: str | None = Field(None, description="Error of the last fetch while the source is down")
    failed_at: datetime | None = Field(None, description="Time of the failed fetch (UTC)")
    note: str | None = None


class ServiceRow(BaseModel):
    key: str = Field(..., description="api, redis, twin_store, worker")
    label: str
    state: SourceState = Field(..., description="ok, stale, down, missing (not known), off (not configured)")
    last_at: datetime | None = Field(
        None, description="Process start (API) or last heartbeat (worker), UTC; null for the others"
    )
    age_s: float | None = Field(None, description="Age of last_at, s")
    latency_ms: float | None = Field(None, description="Time of the check (Redis ping, database query), ms")
    detail: str | None = None


class DiskRow(BaseModel):
    path: str = Field(..., description="A path on the file system")
    used_gb: float = Field(..., description="Used space, GB")
    total_gb: float = Field(..., description="Size, GB")
    pct: float = Field(..., description="Used share, %")
    state: LimitState


class TemperatureRow(BaseModel):
    sensor: str = Field(..., description="Thermal zone or sensor, e.g. cpu, gpu, tj")
    c: float = Field(..., description="Temperature, °C")
    state: LimitState


class HostLimit(BaseModel):
    warning: float = Field(..., description="Warning from this value, in the figure's unit (%, °C)")
    critical: float = Field(..., description="Critical from this value, in the figure's unit (%, °C)")


class HostHealth(BaseModel):
    collected_at: datetime = Field(..., description="Time of the reading (UTC)")
    hostname: str
    platform: str
    source: Literal["tegrastats", "psutil"] = Field(..., description="Where CPU, RAM and temperatures were read")
    state: LimitState = Field(..., description="Worst state of the figures")
    cpu_pct: float | None = Field(None, description="CPU load, mean of the online cores, %")
    cpu_count: int | None = Field(None, description="Logical CPUs, count")
    cores_pct: list[float | None] = Field(..., description="Load per core (tegrastats; null = core off), %")
    load_1m: float | None = Field(None, description="Load average over 1 min (not on Windows), runnable tasks")
    ram_used_mb: float | None = Field(None, description="RAM in use, MB")
    ram_total_mb: float | None = Field(None, description="RAM size, MB")
    ram_pct: float | None = Field(None, description="RAM in use, %")
    swap_used_mb: float | None = Field(None, description="Swap in use, MB")
    swap_total_mb: float | None = Field(None, description="Swap size, MB")
    disks: list[DiskRow]
    temperatures: list[TemperatureRow]
    uptime_s: float | None = Field(None, description="Time since boot, s")
    cpu_state: LimitState | None = None
    ram_state: LimitState | None = None
    limits: dict[str, HostLimit] = Field(
        ..., description="Warning and critical values: cpu_pct, ram_pct, disk_pct, temp_c"
    )
    notes: list[str]


class SystemHealth(BaseModel):
    checked_at: datetime = Field(..., description="Time of the check (UTC); the ages are taken at this time")
    sources: list[SourceRow] = Field(..., description="Every source the twin reads, with its last timestamp and age")
    services: list[ServiceRow] = Field(..., description="twin-api, Redis, the twin store and the worker")
    host: HostHealth | None = Field(None, description="The machine the API runs on; null when it could not be read")
    host_error: str | None = Field(None, description="Why the host could not be read")
    counts: dict[str, int] = Field(..., description="Sources and services per state, count")


@router.get("/health", response_model=SystemHealth, summary="Sources, services and host health")
async def health(request: Request, twin: Annotated[TwinStore | None, Depends(optional_twin_store)]) -> SystemHealth:
    """Every source with its last timestamp and age (the data sources with the live fetches of this process and the
    worker, and the latest twin evaluation), the services and the host's CPU, RAM, disks and temperatures."""
    now = datetime.now(UTC)
    live = getattr(request.app.state, "live", None)
    redis = getattr(live, "redis", None)
    settings = getattr(live, "settings", None) or get_settings()
    await system_health.follow_worker(redis)
    store = await asyncio.to_thread(system_health.store_row, twin)
    services = [
        system_health.api_row(started_at=STARTED_AT, now=now, version=__version__, release=current_release().tag),
        await system_health.redis_row(redis, configured=bool(settings.redis_url)),
        store,
        await system_health.worker_row(redis, now=now),
    ]
    latest = None
    if twin is not None and store["latency_ms"] is not None:  # the database answered
        try:
            latest = await asyncio.to_thread(twin.snapshots.latest)
        except Exception as exc:  # noqa: BLE001 - the evaluation row shows it as missing
            log.warning("latest evaluation not read (%s)", type(exc).__name__)
    cfg = await asyncio.to_thread(twin_reads.config)
    sources = system_health.source_rows(
        latest, now=now, cfg=cfg, health=source_health.HEALTH.view(now), store_configured=twin is not None
    )
    host, host_error = None, None
    try:
        host = await asyncio.to_thread(host_health.snapshot)
    except Exception as exc:  # noqa: BLE001 - the rest of the view stands
        host_error = f"{type(exc).__name__}: {exc}"
        log.warning("host health not read (%s)", host_error)
    return SystemHealth(
        checked_at=now,
        sources=sources,  # type: ignore[arg-type]
        services=services,  # type: ignore[arg-type]
        host=host,  # type: ignore[arg-type]
        host_error=host_error,
        counts=system_health.summary(sources, services),
    )
