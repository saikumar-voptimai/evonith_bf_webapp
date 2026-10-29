"""Burnt against charged coke, admin view (SV-4.4, #307). Administrators only.

``GET /api/admin/burnt-vs-charged`` serves the analysis ``twin.trace`` builds with the TRACE view every hour at :20
(``services.burnt_vs_charged``): agreement per averaging window from one shift to one month (Lin's CCC, Bland-Altman),
the split SD(W)^2 = A/W^2 + B/W + C, the lag scan, the raw mass closure and the leak-water check, on every balanced
shift or on the stable shifts. A request never computes: before the first build the route answers 503. Coke rates
are kg/tHM dry, masses t, times UTC, days IST plant dates.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.auth import require_role
from api.errors import ApiError
from api.routers.trace import AgreementPoint, AgreementStats
from services import burnt_vs_charged as bvc
from services import trace

router = APIRouter(prefix="/api/admin/burnt-vs-charged", tags=["admin"], dependencies=[Depends(require_role("admin"))])

Basis = Literal["all", "stable"]
LeakState = Literal["leak", "moisture", "gap_only", "mixed", "h2_only", "none", "unknown"]


class BvcWindowPoints(BaseModel):
    window: str = Field(..., description="Averaging window, e.g. 1 day")
    shifts: int = Field(..., description="Shifts per window, count")
    points: list[AgreementPoint] = Field(..., description="Window pairs, oldest first")


class BvcSplitWindow(BaseModel):
    window: str = Field(..., description="Averaging window")
    shifts: int = Field(..., description="Shifts per window W, count")
    n: int = Field(..., description="Windows compared, count")
    observed_sd: float | None = Field(None, description="SD of burnt − charged over the windows, kg/tHM")
    fitted_sd: float | None = Field(
        None, description="SD of the fit √(A/W² + B/W + C) at the window (null for a window left out of the fit), "
        "kg/tHM",
    )  # fmt: skip
    transit_sd: float | None = Field(None, description="Transit part at the window, √A / W, kg/tHM")
    random_sd: float | None = Field(None, description="Random part at the window, √B / √W, kg/tHM")
    floor_sd: float | None = Field(None, description="Floor, √C (the same at every window), kg/tHM")
    in_fit: bool = Field(..., description="The window entered the fit (at least 5 windows with an SD)")


class BvcSplit(BaseModel):
    transit_sd: float | None = Field(None, description="√A: per-shift SD of coke in transit (falls as 1/W), kg/tHM")
    random_sd: float | None = Field(None, description="√B: per-shift SD of random error (falls as 1/√W), kg/tHM")
    floor_sd: float | None = Field(None, description="√C: SD that does not fall with the window, kg/tHM")
    windows: list[BvcSplitWindow] = Field(..., description="Observed and fitted SD per averaging window")


class BvcLagRow(BaseModel):
    lag_shifts: int = Field(..., description="Lag, shifts (burnt after charged is positive)")
    lag_h: float = Field(..., description="Lag, h")
    n: int = Field(..., description="Shift pairs compared, count")
    r: float | None = Field(None, description="Pearson correlation of the movements, dimensionless")
    ccc: float | None = Field(None, description="Lin's CCC of the movements, dimensionless")


class BvcLag(BaseModel):
    detrend_shifts: int = Field(..., description="Centred mean the movements are taken against, shifts")
    rows: list[BvcLagRow] = Field(..., description="One row per lag, ascending")
    peak_lag_h: float | None = Field(
        None, description="Lag of the correlation peak (parabola through the best lag and its neighbours), h"
    )


class BvcClosureMonth(BaseModel):
    month: str = Field(..., description="IST month of the shift dates, YYYY-MM")
    n: int = Field(..., description="Shifts with raw burnt and charged coke, count")
    ratio: float | None = Field(None, description="Total raw burnt over total charged coke, dimensionless")
    burnt_t: float | None = Field(None, description="Raw burnt coke, t")
    charged_t: float | None = Field(None, description="Charged coke, t")
    gap_t: float | None = Field(None, description="Charged − raw burnt coke, t")


class BvcClosurePoint(BaseModel):
    t: datetime = Field(..., description="End of the shift (UTC)")
    skey: str = Field(..., description="Shift key YYYY-MM-DD_A|B|C")
    cumulative_t: float | None = Field(None, description="Charged − raw burnt coke summed up to the shift, t")


class BvcClosure(BaseModel):
    n: int = Field(..., description="Shifts with raw burnt and charged coke, count")
    ratio: float | None = Field(None, description="Total raw burnt over total charged coke, dimensionless")
    burnt_t: float | None = Field(None, description="Raw burnt coke (before the 14-day correction), t")
    charged_t: float | None = Field(None, description="Charged coke, t")
    gap_t: float | None = Field(None, description="Charged − raw burnt coke, t")
    months: list[BvcClosureMonth] = Field(..., description="The same per IST month, oldest first")
    cumulative: list[BvcClosurePoint] = Field(..., description="Running charged − raw burnt, oldest first")


class LeakStep(BaseModel):
    key: Literal["gap", "h2", "raft", "moisture"] = Field(
        ..., description="gap: charged − raw burnt coke; h2: top-gas H₂; raft: RAFT; moisture: blast moisture"
    )
    label: str = Field(..., description="Name of the signal")
    unit: str = Field(..., description="Unit of recent, baseline, step and se: kg/tHM, %, °C or g/Nm³")
    n_recent: int = Field(..., description="Recent shifts with a value, count")
    n_baseline: int = Field(..., description="Baseline shifts with a value, count")
    recent: float | None = Field(None, description="Mean over the recent shifts, in the unit field's unit")
    baseline: float | None = Field(None, description="Mean over the baseline shifts, in the unit field's unit")
    step: float | None = Field(None, description="Recent − baseline, in the unit field's unit")
    se: float | None = Field(None, description="Standard error of the step (Welch), in the unit field's unit")
    z: float | None = Field(None, description="Step / standard error, dimensionless")


class LeakModel(BaseModel):
    leak_kg_h: float | None = Field(None, description="Water that would explain the coke-gap step, kg/h")
    water_kg_per_thm: float | None = Field(None, description="That water per tonne of hot metal, kg/tHM")
    h2_rise_pt: float | None = Field(None, description="Top-gas H₂ rise the model expects, percentage points")
    blast_moisture_equivalent_g_nm3: float | None = Field(
        None, description="The water as blast moisture at the reference blast, g/Nm³"
    )
    forge_coke_missing_kg_thm: float | None = Field(None, description="Coke FORGE would under-predict, kg/tHM")
    raw_burnt_low_by_kg_thm: float | None = Field(None, description="Raw burnt coke reads low by, kg/tHM")


class LeakShift(BaseModel):
    skey: str = Field(..., description="Shift key YYYY-MM-DD_A|B|C")
    start: datetime = Field(..., description="Shift start (UTC)")
    end: datetime = Field(..., description="Shift end (UTC)")
    period: Literal["recent", "baseline"] = Field(..., description="Side of the comparison")
    gap: float | None = Field(None, description="Charged − raw burnt coke, kg/tHM")
    h2: float | None = Field(None, description="Top-gas H₂, % of the dry gas")
    raft: float | None = Field(None, description="RAFT, °C")
    moisture: float | None = Field(None, description="Blast moisture, g/Nm³")


class LeakSettings(BaseModel):
    recent_days: int = Field(..., description="Plant days compared, d")
    baseline_days: int = Field(..., description="Plant days before them, d")
    min_recent_shifts: int = Field(..., description="Recent shifts needed, count")
    min_baseline_shifts: int = Field(..., description="Baseline shifts needed, count")
    z_min: float = Field(..., description="Step / standard error for a step, dimensionless")
    gap_min_kg_thm: float = Field(..., description="Smallest coke-gap rise, kg/tHM")
    h2_min_pt: float = Field(..., description="Smallest H₂ rise, percentage points")
    h2_share_min: float = Field(..., description="H₂ rise needed as a share of the model's, dimensionless")
    raft_max_c: float = Field(..., description="Largest RAFT step that counts as unchanged, °C")


class LeakCheck(BaseModel):
    state: LeakState = Field(
        ..., description="leak (the model's signature), moisture (H₂ up with a RAFT drop), gap_only, mixed, h2_only, "
        "none, or unknown (too few shifts)",
    )  # fmt: skip
    reason: str = Field(..., description="The state in words")
    recent_first: date | None = Field(None, description="First recent plant day (IST)")
    recent_last: date | None = Field(None, description="Last recent plant day (IST): the last day with a coke gap")
    baseline_first: date | None = Field(None, description="First baseline plant day (IST); it ends before recent")
    steps: list[LeakStep] = Field(..., description="Step of each signal")
    model: LeakModel | None = Field(None, description="Leak-detection model at the coke-gap step; null without a rise")
    shifts: list[LeakShift] = Field(..., description="The compared shifts, oldest first")
    settings: LeakSettings


class BurntVsCharged(BaseModel):
    job_kind: str = Field(trace.JOB_KIND, description="Job kind that builds the analysis")
    built_at: datetime = Field(..., description="Time of the build (UTC)")
    age_s: float | None = Field(None, description="Time since the build, s")
    stale: bool = Field(..., description="Older than stale_after_s")
    stale_after_s: float = Field(bvc.STALE_AFTER_S, description="Age from which the analysis is stale, s")
    basis: Basis = Field(..., description="all: every balanced shift; stable: the stable shifts only")
    shifts: int = Field(..., description="Shifts of the basis, count")
    stable_shifts: int = Field(..., description="Stable shifts among them, count")
    first_day: date | None = Field(None, description="First shift date (IST)")
    last_day: date | None = Field(None, description="Last shift date (IST)")
    windows: list[AgreementStats] = Field(..., description="Agreement per averaging window, one shift to one month")
    points: list[BvcWindowPoints] = Field(..., description="Pairs behind each window row")
    split: BvcSplit
    lag: BvcLag
    closure: BvcClosure | None = Field(None, description="Raw mass closure; null without a raw burnt coke")
    leak: LeakCheck


@router.get("", response_model=BurntVsCharged, summary="Burnt against charged coke")
def burnt_vs_charged(
    basis: Annotated[Basis, Query(description="all: every balanced shift; stable: the stable shifts only")] = "all",
) -> BurntVsCharged:
    """Agreement per averaging window (CCC, Bland-Altman), the SD(W)² split, the lag scan, the raw mass closure and the
    leak-water check of the latest build."""
    stored = bvc.load()
    if stored is None:
        raise ApiError(
            503, f"Burnt against charged coke has not been built yet: the {trace.JOB_KIND} job builds it every hour."
        )
    part = stored["bases"][basis]
    age = bvc.age_s(stored, datetime.now(UTC))
    labels = {w["window"]: w["shifts"] for w in stored["windows"]}
    return BurntVsCharged(
        built_at=stored["built_at"],
        age_s=age,
        stale=age is None or age > bvc.STALE_AFTER_S,
        basis=basis,
        shifts=part["shifts"],
        stable_shifts=part["stable_shifts"],
        first_day=part["first_day"],
        last_day=part["last_day"],
        windows=part["table"],
        points=[BvcWindowPoints(window=k, shifts=labels.get(k, 0), points=v) for k, v in part["points"].items()],
        split=part["split"],
        lag=part["lag"],
        closure=part["closure"],
        leak={**part["leak"], "settings": stored["leak_settings"]},
    )
