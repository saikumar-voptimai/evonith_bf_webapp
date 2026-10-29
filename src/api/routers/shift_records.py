"""Shift handover records (SV-2.5, #297): the records ``shift.close`` froze for the closed shifts (06:10, 14:10,
22:10 IST).

A record holds the shift's hourly twin snapshots as they were at the close, the coke rates (kg/tHM dry) and implied ω
of the shift's own window, the latest plan decision taken in the shift with its run's headline, and the link of the
shift's report. Records are read-only: these routes only read, and the table refuses UPDATE and DELETE (trigger
``reject_change``), so a PUT, PATCH, DELETE or POST answers 405. A shift report links its record (``record`` of
``GET /api/reports/{id}``). Times UTC.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, Field

from api.auth import require_role
from api.errors import ApiError
from api.store import twin_store
from data.twin.repositories import TwinStore
from services import shift_records
from services.events import TwinSnapshotData

router = APIRouter(prefix="/api/shift-records", tags=["shift records"], dependencies=[Depends(require_role("user"))])

Letter = Literal["A", "B", "C"]
SHIFT_KEY = r"^\d{4}-\d{2}-\d{2}_[ABC]$"


class ShiftRecordSummary(BaseModel):
    shift_key: str = Field(..., description="YYYY-MM-DD_A|B|C, IST date of the shift start")
    shift: Letter
    shift_date: date = Field(..., description="IST date of the shift start")
    shift_start: datetime = Field(..., description="UTC")
    shift_end: datetime = Field(..., description="UTC")
    closed_at: datetime = Field(..., description="Time the record was frozen, UTC")
    k_trace: float | None = Field(None, description="Burnt coke of the shift window (TRACE), kg/tHM dry")
    k_forge: float | None = Field(None, description="Two-zone balance at the 14-day medians (FORGE), kg/tHM dry")
    k_strata: float | None = Field(None, description="Two-zone balance at omega_STRATA, kg/tHM dry")
    k_charged: float | None = Field(None, description="Coke charged in the shift window, kg/tHM dry")
    omega_imp: float | None = Field(None, description="Implied shaft efficiency of the shift window, dimensionless")
    windows_stored: int = Field(..., description="Hourly twin windows ending in the shift that were stored, count")
    windows_expected: int = Field(..., description="Hourly twin windows ending in a shift, count (8)")
    shift_window: bool = Field(..., description="Whether the snapshot of the shift's own window was stored")
    plan_run_id: str | None = Field(None, description="Optimiser run of the latest decision in the shift (UUID)")
    plan_decision: Literal["accepted", "rejected"] | None = Field(
        None, description="The latest decision taken in the shift; null without one"
    )
    report_ref: str | None = Field(None, description="Shift Reports view of the shift (SPA path)")
    href: str = Field(..., description="This record, GET path")


class FrozenDecision(BaseModel):
    run_id: str = Field(..., description="Run id (UUID)")
    decision: Literal["accepted", "rejected"]
    decided_by: str
    decided_at: datetime = Field(..., description="UTC")
    shift_key: str | None = Field(None, description="Shift of the decision (YYYY-MM-DD_A, IST date)")
    comment: str | None = None


class FrozenRun(BaseModel):
    id: str = Field(..., description="Run id (UUID)")
    job_id: str | None = Field(None, description="Job that produced the run")
    trigger: Literal["schedule", "user"]
    requested_by: str
    mode: Literal["step1", "step2", "max_production"]
    omega_mode: Literal["strata", "fixed"]
    status: Literal["ok", "infeasible", "failed"]
    started_at: datetime = Field(..., description="UTC")
    finished_at: datetime = Field(..., description="UTC")
    certainty: str | None = Field(None, description="High, Medium, Low or Investigate (STRATA-ω)")
    production_t_d: float | None = Field(None, description="Hot metal of the plan, t/d")
    cost_rs_thm: float | None = Field(None, description="Ore + flux + fuel − slag credit of the plan, Rs/tHM")
    saving_rs_thm: float | None = Field(None, description="Cost of the charged blend minus the plan's, Rs/tHM")
    coke_kg_thm: float | None = Field(None, description="Coke rate of the plan, kg/tHM dry")
    coke_sd_kg_thm: float | None = Field(None, description="1 sigma of the coke rate, kg/tHM dry (STRATA-ω)")
    fuel_rate_kg_thm: float | None = Field(None, description="Coke (wet) + nut coke + PCI of the plan, kg/tHM")


class FrozenPlan(BaseModel):
    decision: FrozenDecision
    run: FrozenRun | None = Field(None, description="Headline of the decided run at the close")


class ShiftRecordDetail(ShiftRecordSummary):
    hourly: list[TwinSnapshotData] = Field(
        default_factory=list, description="Hourly twin snapshots ending in the shift as stored at the close"
    )
    shift_window_snapshot: TwinSnapshotData | None = Field(
        None, description="Snapshot of the shift's own window [shift_start, shift_end); null when it was missing"
    )
    missing_windows: list[datetime] = Field(
        default_factory=list, description="Ends of the hourly windows not stored at the close, UTC"
    )
    plan: FrozenPlan | None = Field(None, description="The latest plan decision of the shift and its run")
    frozen_by: str | None = Field(None, description="Who submitted the job that froze it (scheduler or a user)")
    frozen_by_job: str | None = Field(None, description="Job that froze the record")
    report_id: str | None = Field(
        None, description="The shift's newest closed report now (UUID); null before report.shift has written it"
    )


class ShiftRecordLink(BaseModel):
    """A shift report's link to the handover record of its shift."""

    shift_key: str = Field(..., description="YYYY-MM-DD_A|B|C")
    closed_at: datetime = Field(..., description="Time the record was frozen, UTC")
    href: str = Field(..., description="GET path of the record")


@router.get("", response_model=list[ShiftRecordSummary], summary="Shift handover records")
def list_records(
    store: Annotated[TwinStore, Depends(twin_store)],
    day: Annotated[date | None, Query(alias="date", description="IST date of the shift start")] = None,
    shift: Annotated[Letter | None, Query(description="Only this shift")] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Records, count (42 = 14 days)")] = 42,
) -> list[ShiftRecordSummary]:
    """Newest shift first."""
    start, end = shift_records.day_bounds(day) if day is not None else (None, None)
    rows = store.shifts.recent(start=start, end=end, letter=shift, limit=limit)
    return [ShiftRecordSummary.model_validate(shift_records.summary_data(r)) for r in rows]


@router.get("/{shift_key}", response_model=ShiftRecordDetail, summary="A shift handover record")
async def get_record(
    shift_key: Annotated[str, Path(pattern=SHIFT_KEY, description="YYYY-MM-DD_A|B|C, IST date of the shift start")],
    store: Annotated[TwinStore, Depends(twin_store)],
) -> ShiftRecordDetail:
    """The frozen record with its hourly snapshots, plan decision and the shift's newest closed report."""
    row = await asyncio.to_thread(store.shifts.get, shift_key)
    if row is None:
        raise ApiError(404, f"No handover record of shift {shift_key}: shift.close freezes it at 06:10, 14:10, 22:10.")
    report = await asyncio.to_thread(store.reports.latest, shift_key, kind="closed")
    data = shift_records.detail_data(row, report_id=str(report.id) if report is not None else None)
    return ShiftRecordDetail.model_validate(data)
