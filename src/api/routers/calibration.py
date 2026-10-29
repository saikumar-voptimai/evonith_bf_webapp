"""Calibration console (SV-4.6, #309): the day's two-zone calibration, its stable-shift mask, the stored days, the
STRATA-Flow Jacobian cache and the nightly job; a manual recalibration. Administrators only.

``GET /api/admin/calibration`` reads what ``calibration.nightly`` stored (``services.calibration_console``; no pipeline
run, no balance): omega_cal, Q_LZ and Q_UZ (GJ/h), r_H and the window's charged coke as
``data.burden_engine.calibration.live_calibration`` returned them for the 14 days before the day, unrounded.
``POST /api/admin/calibration/run`` queues ``calibration.nightly`` now and records the start in the audit log
(``services.job_audit``, action ``job.start``): the calibration of today is replaced when the job ends, and the
optimiser runs that start after it use the new one. Days are IST calendar days, times UTC.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from api.auth import request_actor, require_role
from api.errors import ApiError
from api.live import job_store
from api.routers.jobs import optional_job_store
from api.store import optional_twin_store, twin_store
from data.twin.repositories import TwinStore
from services import calibration_console as console
from services import job_audit
from services.jobs import JobRecord, JobStore

router = APIRouter(prefix="/api/admin/calibration", tags=["admin"], dependencies=[Depends(require_role("admin"))])

RUNS_LISTED = 10
ShiftState = Literal["used", "stable", "unstable", "missing"]


class StabilityTest(BaseModel):
    name: str = Field(..., description="Test, named as the mask's value field (e.g. c_closure_pct)")
    label: str = Field(..., description="What the test checks")
    unit: str = Field(..., description="Unit of the value and the band")
    lo: float = Field(..., description="Lower end of the band (inclusive), unit of the test")
    hi: float = Field(..., description="Upper end of the band (inclusive), unit of the test")


class MaskShift(BaseModel):
    shift_key: str = Field(..., description="Shift key, YYYY-MM-DD_A|B|C (IST date of the shift start)")
    day: date = Field(..., description="IST date of the shift start")
    shift: Literal["A", "B", "C"] = Field(..., description="Shift: A 06:00, B 14:00, C 22:00 IST")
    state: ShiftState = Field(
        ...,
        description="used: averaged into the calibration; stable: passed the filter, not averaged; unstable: failed "
        "a test; missing: no balanced shift (data gap, or the C shift still running at the calibration)",
    )
    failed: list[str] = Field(..., description="Tests the shift failed (names of tests)")
    coke_kg_thm: float | None = Field(None, description="Charged coke, dry, kg/tHM")
    c_closure_pct: float | None = Field(None, description="Carbon closure of the shift balance, %")
    blast_meter_pct: float | None = Field(None, description="Blast meter against the reconciled blast, %")
    cooling_mj_thm: float | None = Field(None, description="Cooling losses of the shift balance, MJ/tHM")
    fe_closure_pct: float | None = Field(None, description="Iron closure of the shift balance, %")
    production_t_h: float | None = Field(None, description="Production, t/h")


class NetworkStats(BaseModel):
    network: str | None = Field(None, description="STRATA-Flow network version")
    solves: int | None = Field(None, description="Network solves of the job's process, count")
    seconds: float | None = Field(None, description="Time in network solves, s")
    seconds_total: float | None = Field(None, description="Time of the Jacobian step, s")
    anchors: int | None = Field(None, description="Network states solved exactly, count")
    jacobian: str | None = Field(None, description="Where the Jacobian came from: cache <file> or computed (...)")
    cache_file: str | None = Field(None, description="Jacobian cache file of the calibration state")


class DayCalibration(BaseModel):
    day: date | None = Field(None, description="IST day the calibration applies to")
    made_at: datetime | None = Field(None, description="When the job stored it (UTC)")
    job_id: str | None = Field(None, description="Job that made it")
    source: Literal["live", "fallback"] = Field(
        ..., description="live: calibrated on the window; fallback: too few stable shifts, the shipped calibration"
    )
    message: str = Field(..., description="The calibration's own account of the window")
    window_start: date | None = Field(None, description="First IST day of the window")
    window_end: date | None = Field(None, description="Last IST day of the window (the day before)")
    window_days: int | None = Field(None, description="Window length, days")
    min_shifts: int = Field(..., description="Stable shifts a live calibration needs, count")
    n_shifts: int = Field(..., description="Stable shifts in the window, count")
    complete: bool = Field(..., description="Live, on at least min_shifts stable shifts")
    omega_cal: float | None = Field(None, description="Shaft efficiency of the calibration, dimensionless")
    q_lz_gj_h: float | None = Field(None, description="Lower-zone heat loss, GJ/h")
    q_uz_gj_h: float | None = Field(None, description="Upper-zone heat loss, GJ/h")
    r_h: float | None = Field(None, description="Hydrogen reduction share, dimensionless")
    coke_kg_thm: float | None = Field(None, description="Charged coke of the window (the calibration basis), kg/tHM")
    coke_model_kg_thm: float | None = Field(
        None, description="Coke the model gives at the calibration burden (fixed omega), kg/tHM"
    )
    shifts: list[str] = Field(..., description="Shift keys averaged into the calibration")
    mask: list[MaskShift] | None = Field(
        None, description="Every shift of the window's days, oldest first; null for a calibration made before SV-4.6"
    )
    network: NetworkStats | None = Field(None, description="The job's Jacobian step; null when it recorded none")


class CalibrationDay(BaseModel):
    day: date | None = Field(None, description="IST day the calibration applies to")
    made_at: datetime | None = Field(None, description="When the job stored it (UTC)")
    job_id: str | None = Field(None, description="Job that made it")
    source: Literal["live", "fallback"] = Field(..., description="live or fallback (the shipped calibration)")
    n_shifts: int = Field(..., description="Stable shifts in the window, count")
    omega_cal: float | None = Field(None, description="Shaft efficiency, dimensionless")
    q_lz_gj_h: float | None = Field(None, description="Lower-zone heat loss, GJ/h")
    q_uz_gj_h: float | None = Field(None, description="Upper-zone heat loss, GJ/h")
    r_h: float | None = Field(None, description="Hydrogen reduction share, dimensionless")
    coke_kg_thm: float | None = Field(None, description="Charged coke of the window, kg/tHM")
    coke_model_kg_thm: float | None = Field(None, description="Model coke at the calibration burden, kg/tHM")
    jacobian_file: str | None = Field(None, description="Jacobian cache file the job named")
    jacobian_cached: bool | None = Field(None, description="The file is in the cache; null when the job named none")


class JacobianFile(BaseModel):
    name: str = Field(..., description="File name, jacobian_<key>.json")
    size_bytes: int = Field(..., description="File size, bytes")
    modified_at: datetime = Field(..., description="Last written (UTC)")
    network: str | None = Field(None, description="Network version the Jacobian was built for")
    current: bool = Field(..., description="Built for the current network version (a stale file is never read)")
    days: list[date] = Field(..., description="Stored calibration days whose job named the file, newest first")


class JacobianCache(BaseModel):
    directory: str = Field(..., description="Cache folder, relative to the repository")
    network_version: str = Field(..., description="Current STRATA-Flow network version")
    status: Literal["cached", "missing", "unknown", "none"] = Field(
        ...,
        description="The shown calibration's Jacobian: cached; missing (the next STRATA-ω run solves it again, 17 "
        "solves); unknown (its job named no file); none (no calibration shown)",
    )
    day_file: str | None = Field(None, description="Jacobian file of the shown calibration")
    n_files: int = Field(..., description="Files in the cache, count")
    total_bytes: int = Field(..., description="Size of the files, bytes")
    stale_files: int = Field(..., description="Files of another network version, count")
    files: list[JacobianFile] = Field(..., description="The newest 50 files, newest first")


class NightlyJob(BaseModel):
    kind: str = Field(..., description="Job kind")
    title: str
    role: str = Field(..., description="Lowest role that may start it")
    schedule: str | None = Field(None, description="Scheduled times (IST)")
    next_slot: datetime | None = Field(None, description="Next scheduled run (UTC)")
    timeout_s: float = Field(..., description="Time limit of one try, s")
    max_tries: int = Field(..., description="Tries before it fails, count")


class CalibrationRun(BaseModel):
    id: str = Field(..., description="Job id")
    status: str = Field(..., description="queued, running, done, failed or cancelled")
    submitted_by: str = Field(..., description="User name, or scheduler")
    scheduled: bool = Field(..., description="Started by the schedule (02:30 IST) rather than by a user")
    submitted_at: datetime | None = Field(None, description="Submission time (UTC)")
    started_at: datetime | None = Field(None, description="Start of the first try (UTC)")
    finished_at: datetime | None = Field(None, description="End (UTC)")
    run_s: float | None = Field(None, description="Execution time over the tries, s")
    tries: int = Field(..., description="Tries made, count")
    error: str | None = Field(None, description="Why it failed or was cancelled")
    day: date | None = Field(None, description="IST day of the calibration it stored")


class CalibrationConsole(BaseModel):
    today: date = Field(..., description="Today (IST)")
    day: date | None = Field(None, description="IST day shown; null when no calibration is stored")
    current: bool = Field(..., description="The shown calibration is today's")
    calibration: DayCalibration | None = Field(None, description="The shown day's calibration")
    days: list[CalibrationDay] = Field(..., description="Stored calibrations (the last 30 days), newest first")
    tests: list[StabilityTest] = Field(..., description="Tests of the stability filter with their bands")
    jacobian_cache: JacobianCache
    job: NightlyJob
    runs: list[CalibrationRun] = Field(..., description="Latest runs of the nightly job, newest first (at most 10)")
    runs_source: Literal["twin.jobs", "redis"] | None = Field(
        None, description="Where the runs were read; null when neither the twin store nor Redis is available"
    )


class CalibrationRunStarted(BaseModel):
    job: JobRecord
    duplicate: bool = Field(..., description="True when a calibration job was already queued or running")
    audit_id: int = Field(..., description="Audit-log entry of the start")


async def _runs(twin: TwinStore | None, jobs: JobStore | None) -> tuple[list[Any], str | None]:
    if twin is not None:
        return await asyncio.to_thread(twin.jobs.recent, kind=console.KIND, limit=RUNS_LISTED), "twin.jobs"
    if jobs is not None:
        return await jobs.list_kind(console.KIND, RUNS_LISTED), "redis"
    return [], None


@router.get("", response_model=CalibrationConsole, summary="Calibration console")
async def calibration_view(
    twin: Annotated[TwinStore | None, Depends(optional_twin_store)],
    jobs: Annotated[JobStore | None, Depends(optional_job_store)],
    day: Annotated[date | None, Query(description="IST day to show (default: the newest stored calibration)")] = None,
) -> CalibrationConsole:
    """The calibration ``calibration.nightly`` stored for the day: window, stable-shift mask, omega_cal, Q_LZ, Q_UZ,
    r_H; the stored days; the Jacobian cache; the nightly job and its latest runs. 404 when ``day`` has none."""
    try:
        data = await asyncio.to_thread(console.view, day)
    except console.DayNotFound as exc:
        raise ApiError(404, f"No {exc}.", title="Calibration not stored") from exc
    rows, source = await _runs(twin, jobs)
    return CalibrationConsole(**data, runs=console.run_rows(rows), runs_source=source)


@router.post("/run", response_model=CalibrationRunStarted, status_code=202, summary="Recalibrate now")
async def start_calibration(
    request: Request,
    store: Annotated[TwinStore, Depends(twin_store)],
    jobs: Annotated[JobStore, Depends(job_store)],
) -> CalibrationRunStarted:
    """Queue ``calibration.nightly`` now (the 14 days before today; today's calibration and its Jacobian) and record
    the start in the audit log (``job.start``). A calibration job already queued or running is returned instead
    (``duplicate``), and the request is recorded as well. Follow the job with GET /api/jobs/{job_id} or the
    ``job.progress`` / ``job.done`` events."""
    started = await job_audit.submit(
        jobs, store, request_actor(request), console.KIND, {}, origin="calibration console"
    )
    return CalibrationRunStarted(job=started.record, duplicate=started.duplicate, audit_id=started.audit_id)
