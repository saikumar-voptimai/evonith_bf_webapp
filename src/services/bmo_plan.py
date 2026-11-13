"""Blend Mix Optimiser plan board (SV-7.4, #327): what the /bmo page shows at the top and runs from its constraints.

The page's plan is the burden engine's Step 2 (``optimiser.run``, mode ``step2``; the page's own run with Fixed-ω, the
shift's scheduled run with STRATA-ω). ``board`` reads the runs the page needs from ``twin.optimiser_run`` and names
them: the implied production (the latest ok Step 1 run, its cheapest level), the shift run (the latest ok scheduled
STRATA-ω Step 2 run of the last day) and the latest ok Step 2 run of any kind, each with the decision that applies.
``run_params`` turns the page's request (production, page 9's slag window and charging limits) into the
``OptimiserRunParams`` the worker solves, so the page's run is page 11's Step 2 case with those limits and nothing
else changed. Nothing is composed here beyond reading the stored headline columns (``services.twin_views.run_data``).

Units: production t/d; cost Rs/tHM (ore + flux + fuel − slag credit); coke kg/tHM dry; fuel rate kg/tHM; times UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from services import twin_views
from services.bmo_views import BmoSlagBand, plant_slag_bands
from services.jobs import OptimiserLimits, OptimiserRunParams

#: A scheduled Step 2 run counts as the shift's while it is younger than this (three shifts).
SHIFT_RUN_MAX_AGE_H = 24.0
#: Step 2 runs read for the board, newest first.
RECENT_LIMIT = 60


class BmoPlanDecision(BaseModel):
    decision: Literal["accepted", "rejected"]
    decided_by: str
    decided_at: datetime = Field(..., description="UTC")
    shift_key: str | None = Field(None, description="Shift of the decision (YYYY-MM-DD_A, IST date)")
    comment: str | None = None


class BmoPlanRun(BaseModel):
    """A fixed-production run the page may show: its headline and the decision that applies."""

    id: str = Field(..., description="Run id (UUID)")
    job_id: str | None = Field(None, description="Job that produced the run")
    trigger: Literal["schedule", "user"]
    requested_by: str
    omega_mode: Literal["strata", "fixed"] = Field(
        ..., description="strata: the shift's scheduled run; fixed: a page run"
    )
    status: Literal["ok", "infeasible", "failed"]
    started_at: datetime = Field(..., description="UTC")
    finished_at: datetime = Field(..., description="UTC")
    production_t_d: float | None = Field(None, description="Hot metal of the plan, t/d")
    cost_rs_thm: float | None = Field(None, description="Ore + flux + fuel − slag credit of the plan, Rs/tHM")
    saving_rs_thm: float | None = Field(None, description="Cost of today's charged blend minus the plan's, Rs/tHM")
    coke_kg_thm: float | None = Field(None, description="Coke rate of the plan, kg/tHM dry")
    fuel_rate_kg_thm: float | None = Field(None, description="Coke (wet) + nut coke + PCI of the plan, kg/tHM")
    certainty: str | None = Field(None, description="High, Medium, Low or Investigate (STRATA-ω); null for Fixed-ω")
    decision: BmoPlanDecision | None = Field(None, description="The decision that applies (the latest)")


class BmoImpliedProduction(BaseModel):
    """The production implied by the latest cheapest-production run: the cheapest level under the day's limits."""

    production_t_d: float = Field(..., description="Hot metal of the cheapest level, t/d")
    run_id: str = Field(..., description="Run id (UUID)")
    omega_mode: Literal["strata", "fixed"] = Field(..., description="strata: the shift's run; fixed: a Recompute")
    trigger: Literal["schedule", "user"]
    finished_at: datetime = Field(..., description="UTC")


class BmoPlanBoard(BaseModel):
    """The runs the Blend Mix Optimiser page opens on."""

    read_at: datetime = Field(..., description="UTC")
    implied: BmoImpliedProduction | None = Field(None, description="Latest ok cheapest-production run; null while none")
    shift_run: BmoPlanRun | None = Field(
        None, description=f"Latest ok scheduled STRATA-ω fixed-production run of the last {SHIFT_RUN_MAX_AGE_H:g} h"
    )
    latest_run: BmoPlanRun | None = Field(None, description="Latest ok fixed-production run of any kind")
    recent: list[BmoPlanRun] = Field(
        default_factory=list, description="Fixed-production runs of the last day, newest first"
    )
    slag_bands: list[BmoSlagBand] = Field(
        default_factory=plant_slag_bands, description="Plant p5–p95 bands of B2, T-basicity and IB4"
    )


class BmoPlanRunParams(BaseModel):
    """A run from the page's constraints: fixed production at the chosen production (default), or the cheapest
    production level to recompute the implied production. Both run with Fixed-ω on the day's calibration and the live
    yard; the limits are page 9's slag window and charging capacity in the API's names; unset fields keep the saved
    operator inputs and the register."""

    model_config = {"extra": "forbid"}

    mode: Literal["step2", "step1"] = Field("step2", description="step2: the plan at a production; step1: implied")
    production_t_d: float | None = Field(
        None, ge=1500, le=3000, description="Fixed-production target, t/d (default: the current production)"
    )
    limits: OptimiserLimits | None = Field(
        None, description="Slag window and charging limits instead of the saved ones"
    )


def run_params(body: BmoPlanRunParams) -> OptimiserRunParams:
    """The ``optimiser.run`` request of a page run: Fixed-ω, the day's calibration, the live yard."""
    return OptimiserRunParams(
        mode=body.mode,
        omega_mode="fixed",
        production_t_d=body.production_t_d if body.mode == "step2" else None,
        calibration="day",
        yard="live",
        limits=body.limits,
    )


def _decision(store: Any, run_id: Any) -> BmoPlanDecision | None:
    data = twin_views.decision_data(store.optimiser.decision(run_id))
    return BmoPlanDecision.model_validate(data) if data is not None else None


def run_ref(store: Any, row: Any) -> BmoPlanRun:
    """One run's headline (``twin_views.run_data``) with its decision."""
    data = twin_views.run_data(row)
    return BmoPlanRun(
        id=data["id"],
        job_id=data["job_id"],
        trigger=data["trigger"],
        requested_by=data["requested_by"],
        omega_mode=data["omega_mode"],
        status=data["status"],
        started_at=data["started_at"],
        finished_at=data["finished_at"],
        production_t_d=data["production_t_d"],
        cost_rs_thm=data["cost_rs_thm"],
        saving_rs_thm=data["saving_rs_thm"],
        coke_kg_thm=data["coke_kg_thm"],
        fuel_rate_kg_thm=data["fuel_rate_kg_thm"],
        certainty=data["certainty"],
        decision=_decision(store, row.id),
    )


def implied_production(store: Any) -> BmoImpliedProduction | None:
    """The latest ok Step 1 run as the implied production; None while there is none or it has no production."""
    row = store.optimiser.latest(mode="step1")
    if row is None:
        return None
    data = twin_views.run_data(row)
    if data["production_t_d"] is None:
        return None
    return BmoImpliedProduction(
        production_t_d=data["production_t_d"],
        run_id=data["id"],
        omega_mode=data["omega_mode"],
        trigger=data["trigger"],
        finished_at=data["finished_at"],
    )


def board(store: Any, now: datetime | None = None) -> BmoPlanBoard:
    """The board: implied production, the shift's run, the latest run and the last day's Step 2 runs."""
    at = now or datetime.now(UTC)
    since = at - timedelta(hours=SHIFT_RUN_MAX_AGE_H)
    rows = store.optimiser.recent(mode="step2", since=since, limit=RECENT_LIMIT)
    recent = [run_ref(store, r) for r in rows]
    ok = [r for r in recent if r.status == "ok"]
    shift = next((r for r in ok if r.trigger == "schedule" and r.omega_mode == "strata"), None)
    latest = ok[0] if ok else None
    if latest is None:
        row = store.optimiser.latest(mode="step2")
        latest = run_ref(store, row) if row is not None else None
    return BmoPlanBoard(
        read_at=at, implied=implied_production(store), shift_run=shift, latest_run=latest, recent=recent
    )


__all__ = [
    "RECENT_LIMIT",
    "SHIFT_RUN_MAX_AGE_H",
    "BmoImpliedProduction",
    "BmoPlanBoard",
    "BmoPlanDecision",
    "BmoPlanRun",
    "BmoPlanRunParams",
    "board",
    "implied_production",
    "run_params",
    "run_ref",
]
