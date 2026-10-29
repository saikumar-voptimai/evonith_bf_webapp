"""Shift reports (SV-2.3, #295): the FurnaceMind Reports tab's report of a shift, its data, tables and LLM narrative.

Reports are written by the worker into ``twin.shift_report``: ``report.shift`` reports the shift that has just closed
at 06:20, 14:20 and 22:20 IST; ``POST /api/reports/live`` queues ``report.live`` for the running shift and answers 202
with the job, whose result names the report (``report_id``) once it is done. A report whose narrative failed after its
retry is stored without it (``status`` no_narrative, ``narrative_status`` failed). Times are UTC unless named ``_ist``.

SV-2.4 (#296), the Shift Reports page: ``GET /api/reports/thresholds`` serves the status thresholds of
``config/shift_report.yml`` (read at every request) and a report's ``kpis`` are its ETA CO, fuel rate, RAFT and
permeability against them (``services.reports.kpi_checks``, the builder's rule).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, time, timedelta
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Body, Depends, Query, Request
from pydantic import BaseModel, Field

from api.auth import require_role
from api.errors import ApiError
from api.live import job_store
from api.store import twin_store
from data.twin.repositories import TwinStore
from services import reports
from services.auth import Principal
from services.jobs import JobRecord, JobStore, LiveReportParams

router = APIRouter(prefix="/api/reports", tags=["reports"], dependencies=[Depends(require_role("user"))])

PLANT_TZ = "Asia/Kolkata"
REPORT_JOB_KINDS = ("report.shift", "report.live")
Letter = Literal["A", "B", "C"]
FurnaceStatus = Literal["STABLE", "ATTENTION REQUIRED", "UNSTABLE"]


class ParamStatsView(BaseModel):
    mean: float | None = Field(None, description="Mean over the report window, in the unit of the parameter")
    std: float | None = Field(None, description="Standard deviation over the window, in the unit of the parameter")


class TempRowView(BaseModel):
    q1: float | None = Field(None, description="Quadrant Q1, mean over the window, °C")
    q2: float | None = Field(None, description="Quadrant Q2, mean over the window, °C")
    q3: float | None = Field(None, description="Quadrant Q3, mean over the window, °C")
    q4: float | None = Field(None, description="Quadrant Q4, mean over the window, °C")
    spread_std: float | None = Field(None, description="Standard deviation across Q1 to Q4, °C")


def _stats(what: str) -> object:
    return Field(..., description=f"{what}: mean and standard deviation over the window")


def _temps(what: str) -> object:
    return Field(..., description=f"{what} temperatures by quadrant, °C")


class ShiftReportDataView(BaseModel):
    """The report's metrics (``reports.furnace_report.data.ShiftReportData``)."""

    shift_date: date = Field(..., description="IST date of the shift start")
    shift_label: Letter | None = Field(None, description="Shift letter (null for a day report)")
    shift_start_ist: datetime = Field(..., description="Start of the report window, IST")
    shift_end_ist: datetime = Field(..., description="End of the report window, IST")
    report_type: Literal["Shift", "Day"]
    window_id: str = Field(..., description="Window id of the Reports tab, e.g. 2026-10-27_SHIFT_A")
    duration_h: float = Field(..., description="Length of the report window, h")
    status: FurnaceStatus = Field(..., description="Rule-based status (thresholds of config/shift_report.yml)")
    status_flags: list[str] = Field(default_factory=list, description="The thresholds the window breached")
    production_rate: float | None = Field(None, description="Hot-metal production rate, mean, t/h")
    theoretical_production: float | None = Field(None, description="Production rate × window length, t")
    total_charges: int | None = Field(None, description="Charges in the window, count")
    coke_t: float | None = Field(None, description="Coke charged, t")
    nut_coke_t: float | None = Field(None, description="Nut coke charged, t")
    sinter_t: float | None = Field(None, description="Sinter charged, t")
    ore_t: float | None = Field(None, description="Lump ore charged, t")
    pellet_t: float | None = Field(None, description="Pellet charged, t")
    flux_t: float | None = Field(None, description="Flux charged, t")
    fuel_rate: float | None = Field(None, description="Coke + nut coke + PCI rate, kg/tHM")
    coke_rate: float | None = Field(None, description="Coke rate, mean, kg/tHM")
    nut_coke_rate: float | None = Field(None, description="Nut coke rate, mean, kg/tHM")
    pci_rate: float | None = Field(None, description="PCI rate, mean, kg/tHM")
    hm_si: float | None = Field(None, description="Hot-metal Si, mean of the samples, %")
    hm_s: float | None = Field(None, description="Hot-metal S, mean of the samples, %")
    hm_temp: float | None = Field(None, description="Hot-metal temperature, mean of the samples, °C")
    slag_basicity: float | None = Field(None, description="Slag CaO/SiO₂, dimensionless")
    total_taps: int | None = Field(None, description="Taps in the window, count")
    blast_volume: ParamStatsView = _stats("Hot blast volume, Nm³/h")
    blast_temp: ParamStatsView = _stats("Hot blast temperature, °C")
    blast_pressure: ParamStatsView = _stats("Hot blast pressure, bar")
    furnace_top_dp: ParamStatsView = _stats("Top differential pressure, bar")
    furnace_bottom_dp: ParamStatsView = _stats("Bottom differential pressure, bar")
    furnace_total_dp: ParamStatsView = _stats("Total differential pressure, bar")
    o2_flow: ParamStatsView = _stats("Oxygen flow, Nm³/h")
    total_o2_flow: float | None = Field(None, description="Day reports: the day's total oxygen flow, Nm³")
    o2_enrichment: ParamStatsView = _stats("Oxygen enrichment, %")
    permeability: ParamStatsView = _stats("Permeability index, dimensionless")
    etaco: ParamStatsView = _stats("ETA CO (gas utilisation), %")
    raft: ParamStatsView = _stats("Raceway adiabatic flame temperature, °C")
    uptake: TempRowView = _temps("Uptake (top gas)")
    lower_stack: TempRowView = _temps("Lower stack (18.66 m)")
    belly: TempRowView = _temps("Belly (15.16 m)")
    skin_flow: TempRowView = _temps("Skin flow")
    bosh: TempRowView = _temps("Bosh (12.98 m)")
    hearth_4_3_a: float | None = Field(None, description="Hearth pad 4.3 m, position A, mean, °C")
    hearth_5_4_c: float | None = Field(None, description="Hearth pad 5.4 m, position C, mean, °C")
    hearth_5_7_c: float | None = Field(None, description="Hearth pad 5.7 m, position C, mean, °C")
    hearth_6_1_b: float | None = Field(None, description="Hearth pad 6.1 m, position B, mean, °C")
    burden_moisture_input: float | None = Field(None, description="Moisture charged with the burden, kg/tHM")
    fines_input: float | None = Field(None, description="Fines charged with the burden, kg/tHM")
    burden_ratio: str | None = Field(None, description="Sinter : ore : pellet shares of the ferrous burden, % (text)")
    ibrm: float | None = Field(None, description="Iron-bearing raw material (dry ore + pellet, sinter), t/tHM")
    used_materials: dict[str, str] = Field(
        default_factory=dict, description="Ore and flux materials charged, text lines with t and burden share (%)"
    )


class ReportSectionView(BaseModel):
    title: str
    headers: list[str] = Field(..., description="Column headers with their units, as the Reports tab shows them")
    rows: list[list[str]] = Field(..., description="Formatted cells; '-' where a value is missing")
    placement: Literal["left", "right"]


class ReportNoteView(BaseModel):
    text: str
    placement: Literal["left", "right"]
    kind: str = Field(..., description="material_usage, analysis (the narrative) or note")


class ReportDocumentView(BaseModel):
    """The report as the Reports tab renders it."""

    title: str
    pre_blocks: list[str] = Field(default_factory=list, description="Heading text (markdown)")
    sections: list[ReportSectionView] = Field(default_factory=list)
    notes: list[ReportNoteView] = Field(default_factory=list)
    generated_at_ist: datetime | None = Field(None, description="Time the report was generated, IST")


class ShiftReportSummary(BaseModel):
    id: str = Field(..., description="Report id (UUID)")
    shift_key: str = Field(..., description="YYYY-MM-DD_A|B|C, IST date of the shift start")
    shift: Letter
    shift_date: date = Field(..., description="IST date of the shift start")
    shift_start: datetime = Field(..., description="UTC")
    shift_end: datetime = Field(..., description="UTC")
    kind: Literal["closed", "live"] = Field(
        ..., description="closed: the shift had ended; live: the running shift, data up to the request"
    )
    trigger: Literal["schedule", "user"]
    requested_by: str
    job_id: str | None = Field(None, description="Job that generated the report")
    generated_at: datetime = Field(..., description="UTC")
    status: Literal["complete", "no_narrative"] = Field(
        ...,
        description="complete: data and narrative; no_narrative: data only (LLM failed after its retry, or skipped)",
    )
    narrative_status: Literal["ok", "failed", "skipped"]
    furnace_status: FurnaceStatus = Field(..., description="Rule-based status of the shift")
    production_rate_t_h: float | None = Field(None, description="Hot-metal production rate, mean, t/h")
    fuel_rate_kg_thm: float | None = Field(None, description="Coke + nut coke + PCI rate, kg/tHM")
    coke_rate_kg_thm: float | None = Field(None, description="Coke rate, kg/tHM")
    pci_rate_kg_thm: float | None = Field(None, description="PCI rate, kg/tHM")
    etaco_pct: float | None = Field(None, description="ETA CO, mean, %")
    hm_si_pct: float | None = Field(None, description="Hot-metal Si, mean, %")


class ShiftReportDetail(ShiftReportSummary):
    narrative: str | None = Field(None, description="The LLM's handover analysis (null without narrative)")
    narrative_error: str | None = Field(None, description="Why there is no narrative: the error of each try")
    narrative_tries: int = Field(..., description="LLM calls made, count (2 = first call and the retry)")
    narrative_model: str | None = Field(None, description="Model that wrote the narrative")
    data: ShiftReportDataView
    document: ReportDocumentView
    kpis: list[KpiStatusView] = Field(
        default_factory=list, description="ETA CO, fuel rate, RAFT and permeability against the current thresholds"
    )


class KpiStatusView(BaseModel):
    """One KPI of the report against the current thresholds of config/shift_report.yml."""

    key: Literal["etaco", "fuel_rate", "raft", "permeability"]
    label: str
    unit: str = Field(..., description="Unit of value and thresholds ('' for the dimensionless permeability index)")
    value: float | None = Field(None, description="Mean over the report window, in unit")
    level: Literal["ok", "warning", "critical", "no_data"] = Field(
        ..., description="ok, warning or critical against the thresholds; no_data without a value"
    )
    rule: str = Field(..., description="The thresholds as text, e.g. 'warning below 42 %, critical below 40 %'")
    flag: str | None = Field(None, description="The Reports tab's flag text when the KPI breaches")
    warn: float | None = Field(None, description="Warning threshold, in unit (ETA CO: below, fuel rate: above)")
    crit: float | None = Field(None, description="Critical threshold, in unit (ETA CO: below, fuel rate: above)")
    lo: float | None = Field(None, description="Lower edge of the normal band, in unit (RAFT, permeability)")
    hi: float | None = Field(None, description="Upper edge of the normal band, in unit (RAFT, permeability)")


class ReportThresholds(BaseModel):
    """``config/shift_report.yml -> status_thresholds`` and the rule that turns flags into the report's status."""

    etaco_warn_pct: float = Field(..., description="ETA CO below this: warning, %")
    etaco_crit_pct: float = Field(..., description="ETA CO below this: critical, %")
    fuel_rate_warn_kg_thm: float = Field(..., description="Fuel rate above this: warning, kg/tHM")
    fuel_rate_crit_kg_thm: float = Field(..., description="Fuel rate above this: critical, kg/tHM")
    raft_lo_c: float = Field(..., description="RAFT normal band, lower edge, °C")
    raft_hi_c: float = Field(..., description="RAFT normal band, upper edge, °C")
    permeability_lo: float = Field(..., description="Permeability index normal band, lower edge, dimensionless")
    permeability_hi: float = Field(..., description="Permeability index normal band, upper edge, dimensionless")
    attention_from: int = Field(..., description="Flags from which the status is ATTENTION REQUIRED, count")
    unstable_from: int = Field(..., description="Flags from which the status is UNSTABLE, count")


class ReportSubmitted(BaseModel):
    job: JobRecord = Field(..., description="The report.live job; its result holds report_id once it is done")
    duplicate: bool = Field(..., description="True when a live report was already queued or running")


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    """[00:00, 24:00) IST of ``day`` in UTC: the shifts A, B and C dated ``day`` start in it."""
    start = pd.Timestamp(datetime.combine(day, time.min), tz=PLANT_TZ).tz_convert("UTC").to_pydatetime()
    return start, start + timedelta(days=1)


@router.get("", response_model=list[ShiftReportSummary], summary="Shift reports")
def list_reports(
    store: Annotated[TwinStore, Depends(twin_store)],
    day: Annotated[date | None, Query(alias="date", description="IST date of the shift start")] = None,
    shift: Annotated[Letter | None, Query(description="Only this shift")] = None,
    kind: Annotated[Literal["closed", "live"] | None, Query(description="Only closed or live reports")] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Reports, count")] = 50,
) -> list[ShiftReportSummary]:
    """Newest shift first and, within a shift, newest report first (a re-run or a live report adds a row)."""
    start, end = _day_bounds(day) if day is not None else (None, None)
    rows = store.reports.recent(start=start, end=end, letter=shift, kind=kind, limit=limit)
    return [ShiftReportSummary.model_validate(reports.summary_data(r)) for r in rows]


@router.post("/live", response_model=ReportSubmitted, status_code=202, summary="Report the running shift")
async def live_report(
    request: Request,
    jobs: Annotated[JobStore, Depends(job_store)],
    body: Annotated[LiveReportParams | None, Body()] = None,
) -> ReportSubmitted:
    """Queue ``report.live`` for the running shift; follow it with GET /api/jobs/{id} or the ``job.done`` event, then
    read the report at GET /api/reports/{report_id}. A request equal to a queued or running one returns that job."""
    principal: Principal = request.state.principal
    params = (body or LiveReportParams()).model_dump()
    record, duplicate = await jobs.submit("report.live", params, principal.username)
    return ReportSubmitted(job=record, duplicate=duplicate)


@router.get("/thresholds", response_model=ReportThresholds, summary="Status thresholds of the reports")
def thresholds() -> ReportThresholds:
    """The thresholds of config/shift_report.yml as they are now (read at every request)."""
    try:
        return ReportThresholds.model_validate(reports.thresholds_data())
    except ValueError as exc:
        raise ApiError(500, f"config/shift_report.yml: {exc}") from exc


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


@router.get("/{report_id}", response_model=ShiftReportDetail, summary="A shift report")
async def get_report(
    report_id: str, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> ShiftReportDetail:
    """Data, tables, narrative and status of a report, by report id or by the id of the job that wrote it."""
    key = _uuid(report_id)
    row = await asyncio.to_thread(store.reports.get, key) if key is not None else None
    if row is None:
        live = getattr(request.app.state, "live", None)
        job = await live.store.get(report_id) if live is not None and live.store is not None else None
        if job is not None and job.kind in REPORT_JOB_KINDS:
            found = (job.result or {}).get("report_id")
            if found:
                row = await asyncio.to_thread(store.reports.get, found)
            elif job.status in ("queued", "running"):
                raise ApiError(404, f"The report job is {job.status}; the report exists once it is done.")
            else:
                raise ApiError(404, f"The report job ended {job.status} without a stored report.")
    if row is None:
        raise ApiError(404, "No shift report with this id.")
    try:
        detail = await asyncio.to_thread(reports.detail_data, row)  # reads the thresholds file
    except ValueError as exc:  # thresholds missing from the configuration
        raise ApiError(500, f"config/shift_report.yml: {exc}") from exc
    return ShiftReportDetail.model_validate(detail)
