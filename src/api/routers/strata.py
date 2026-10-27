"""STRATA page (SV-1.12, #292): shaft efficiency of the STRATA-Flow ring model.

``/omega`` omega_STRATA against the implied omega per stored twin window, with the band omega_STRATA + mu ± 2 sigma_e
and the CUSUM of the drift alarm (``services.drift``); ``/summary`` the model's out-of-sample validation, the MOD
facts of the register at the reference point, the certainty tiers and the tier of the charged blend in the latest
window; ``/live-point`` the latest result of the job ``strata.live_point`` (MOD facts recomputed at the current
calibration and the ring view of the thermal reserve zone; network solves run in the worker only). Omega is
dimensionless, coke kg/tHM dry, times UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.auth import require_role
from api.store import optional_twin_store, twin_store
from data.twin.repositories import TwinStore
from services import coke_energy as ce
from services import drift, strata

router = APIRouter(prefix="/api/strata", tags=["strata"], dependencies=[Depends(require_role("user"))])

MAX_DAYS = 56
CURRENT_LOOKBACK = timedelta(days=2)


# --------------------------------------------------------------------------------------------------- omega panel
class OmegaRow(BaseModel):
    window_end: datetime = Field(..., description="End of the 8-h window (UTC)")
    status: str | None = Field(None, description="complete, partial or failed")
    omega_imp: float | None = Field(None, description="Implied shaft efficiency of the window, dimensionless")
    omega_strata: float | None = Field(None, description="STRATA shaft efficiency of the window, dimensionless")
    omega_strata_sd: float | None = Field(None, description="1 sigma of omega_strata (parameters), dimensionless")
    omega_cal: float | None = Field(None, description="14-day median shaft efficiency, dimensionless")
    band_low: float | None = Field(None, description="omega_strata + mu − 2 sigma_e, dimensionless")
    band_high: float | None = Field(None, description="omega_strata + mu + 2 sigma_e, dimensionless")
    r: float | None = Field(None, description="Residual omega_imp − omega_strata, dimensionless")
    z: float | None = Field(None, description="Standardised AR(1) innovation of r, sigma_e units (dimensionless)")
    s_pos: float | None = Field(None, description="CUSUM S+ (implied above STRATA), sigma_e units (dimensionless)")
    s_neg: float | None = Field(None, description="CUSUM S− (implied below STRATA), sigma_e units (dimensionless)")
    signal: Literal["high", "low"] | None = Field(None, description="CUSUM signal of the window (S > h)")


class CusumLatest(BaseModel):
    window_end: datetime = Field(..., description="Latest window in the CUSUM (UTC)")
    r: float | None = Field(None, description="Residual omega_imp − omega_strata, dimensionless")
    z: float | None = Field(None, description="Standardised innovation, sigma_e units (dimensionless)")
    s_pos: float | None = Field(None, description="CUSUM S+, sigma_e units (dimensionless)")
    s_neg: float | None = Field(None, description="CUSUM S−, sigma_e units (dimensionless)")


class DriftEpisode(BaseModel):
    direction: Literal["high", "low"] = Field(..., description="high: implied above STRATA; low: below")
    raised_at: datetime = Field(..., description="First signal (UTC)")
    last_signal: datetime = Field(..., description="Latest signal (UTC)")
    cleared_at: datetime | None = Field(None, description="Clear time (UTC); null while open")
    signals: int = Field(..., description="Signals in the episode, count")
    peak: float | None = Field(None, description="Largest CUSUM sum at a signal, sigma_e units (dimensionless)")


class CusumState(BaseModel):
    state: Literal["in_control", "high", "low", "both", "collecting"] = Field(
        ..., description="Open alarm direction, in_control, or collecting (too few pairs for the AR(1) fit)"
    )
    k: float = Field(..., description="CUSUM allowance, sigma_e units (dimensionless)")
    h: float = Field(..., description="CUSUM decision limit, sigma_e units (dimensionless)")
    mu: float = Field(..., description="In-control mean of r, dimensionless")
    lag_h: float = Field(..., description="Lag between the windows of one chain, h")
    clear_after_h: float = Field(..., description="An alarm clears this long after its last signal, h")
    history_days: float = Field(..., description="Snapshot history of the evaluation, days")
    rho: float | None = Field(None, description="AR(1) coefficient of r at the lag, dimensionless")
    sigma_e: float | None = Field(None, description="1 sigma of the AR(1) innovations of r, dimensionless")
    mu_hat: float | None = Field(None, description="Sample mean of r (bias diagnostic), dimensionless")
    n_pairs: int | None = Field(None, description="Pairs of windows lag_h apart in the fit, count")
    source: str | None = Field(None, description="estimated (on the history) or fixed (configured)")
    band_sigmas: float = Field(..., description="Half-width of the omega band in sigma_e, dimensionless")
    latest: CusumLatest | None = None
    episodes: list[DriftEpisode] = Field(default_factory=list, description="Open episodes and those in the period")
    note: str = Field("", description="Why the CUSUM is not evaluated (collecting)")


class StrataOmega(BaseModel):
    start: datetime = Field(..., description="First window end shown (UTC)")
    end: datetime = Field(..., description="Evaluation time (UTC)")
    count: int = Field(..., description="Windows, count")
    rows: list[OmegaRow]
    cusum: CusumState


@router.get("/omega", response_model=StrataOmega, summary="omega_STRATA against the implied omega, with the CUSUM")
def omega(
    store: Annotated[TwinStore, Depends(twin_store)],
    days: Annotated[float, Query(gt=0, le=MAX_DAYS, description="Windows of the last days shown, days")] = 7.0,
) -> StrataOmega:
    """The stored windows of the last ``days``; the CUSUM runs on the drift alarm's history (56 days by default)."""
    cfg = drift.DriftConfig.from_config(ce.config())
    now = datetime.now(UTC)
    start = now - timedelta(days=days)
    history = max(cfg.history.to_pytimedelta(), timedelta(days=days))
    rows = store.snapshots.series(now - history, now)
    return StrataOmega.model_validate(strata.omega_panel(rows, cfg, start=start, now=now))


# ------------------------------------------------------------------------------------------------------- summary
class ExpandingTest(BaseModel):
    rmse_strata: float | None = Field(None, description="RMSE of burnt coke, STRATA-ω, kg/tHM dry")
    rmse_fixed: float | None = Field(None, description="RMSE of burnt coke, Fixed-ω, kg/tHM dry")
    skill_coke_day: float | None = Field(None, description="1 − MSE/MSE(Fixed-ω), daily coke, dimensionless")
    skill_coke_7d: float | None = Field(None, description="Skill on 7-day means of coke, dimensionless")
    skill_omega_day: float | None = Field(None, description="Skill on daily omega, dimensionless")


class TimeSplit(BaseModel):
    split: str = Field(..., description="Training and test halves")
    skill_coke_day: float | None = Field(None, description="Skill on daily coke, dimensionless")
    skill_omega_day: float | None = Field(None, description="Skill on daily omega, dimensionless")


class StrictPeriod(BaseModel):
    period: str
    rmse_fixed: float | None = Field(None, description="RMSE of burnt coke, Fixed-ω, kg/tHM dry")
    rmse_strata: float | None = Field(None, description="RMSE of burnt coke, STRATA-ω, kg/tHM dry")


class StrictTest(BaseModel):
    method: str
    rmse_fixed: float | None = Field(None, description="RMSE over all days, Fixed-ω, kg/tHM dry")
    rmse_strata: float | None = Field(None, description="RMSE over all days, STRATA-ω, kg/tHM dry")
    periods: list[StrictPeriod]
    weekly_rmse_fixed: float | None = Field(None, description="RMSE of weekly means, Fixed-ω, kg/tHM dry")
    weekly_rmse_strata: float | None = Field(None, description="RMSE of weekly means, STRATA-ω, kg/tHM dry")
    skill_coke_day: float | None = Field(None, description="Skill on daily coke, dimensionless")
    skill_coke_7d: float | None = Field(None, description="Skill on 7-day means of coke, dimensionless")
    skill_omega_day: float | None = Field(None, description="Skill on daily omega, dimensionless")


class Validation(BaseModel):
    method: str = Field(..., description="Protocol of the 2026 tests")
    note: str
    verdict_rule: str = Field(..., description="Rule of the per-family verdicts")
    expanding: ExpandingTest = Field(..., description="2026 expanding-window test (parameters from earlier days)")
    time_splits: list[TimeSplit]
    strict: StrictTest = Field(..., description="Strict test on 2023-24 data with parameters frozen")


class FamilyRow(BaseModel):
    id: str = Field(..., description="Fact id (VAL-nn)")
    family: str = Field(..., description="Blend family")
    verdict: str = Field(..., description="Supported, Disputed, Neutral or Modelled only")
    grade: str
    n: int | None = Field(None, description="Days in the test, count")
    realised: float | None = Field(None, description="Mean realised burnt coke, kg/tHM dry")
    fixed_omega: float | None = Field(None, description="Mean Fixed-ω prediction, kg/tHM dry")
    strata_omega: float | None = Field(None, description="Mean STRATA-ω prediction, kg/tHM dry")
    rmse_fixed: float | None = Field(None, description="RMSE, Fixed-ω, kg/tHM dry")
    rmse_strata: float | None = Field(None, description="RMSE, STRATA-ω, kg/tHM dry")
    skill: float | None = Field(None, description="1 − MSE/MSE(Fixed-ω), dimensionless")
    bias: float | None = Field(None, description="Mean STRATA-ω error, kg/tHM dry")
    se_bias: float | None = Field(None, description="Standard error of the bias, kg/tHM dry")
    composition: dict[str, float] = Field(default_factory=dict, description="Mean share of ferrous by class, %")


class ReferenceResponse(BaseModel):
    id: str = Field(..., description="Fact id (MOD-nn)")
    statement: str
    unit: str = Field(..., description="Unit of the responses (kg/tHM dry coke per change)")
    fixed_omega: float | None = Field(None, description="Coke response, Fixed-ω, in unit")
    strata_omega: float | None = Field(None, description="Coke response, STRATA-ω, in unit")
    difference: float | None = Field(None, description="STRATA-ω minus Fixed-ω, in unit")
    sigma_strata: float | None = Field(None, description="1 sigma of the STRATA-ω response, in unit")
    d_omega: float | None = Field(None, description="Change of omega under STRATA-ω, dimensionless")
    version: str = Field("", description="Register version of the fact")
    date: str = Field("", description="Date of the fact")


class TierLevel(BaseModel):
    tier: Literal["High", "Medium", "Low", "Investigate"]
    sigma_max_kg_thm: float | None = Field(None, description="Largest sigma(coke) of the tier, kg/tHM wet coke")
    support: str = Field(..., description="Position of blend and production against the range BF-02 ran")
    rule: str


class Tiers(BaseModel):
    levels: list[TierLevel]
    investigate_sigma_kg_thm: float = Field(..., description="sigma(coke) above which a plan is Investigate, kg/tHM")
    investigate_classes_max_pct: dict[str, float] = Field(..., description="Class shares above this, % of ferrous")
    investigate_classes_min_pct: dict[str, float] = Field(..., description="Class shares below this, % of ferrous")
    investigate_note: str
    support_classes_pct: dict[str, list[float]] = Field(..., description="Range BF-02 ran by class, % of ferrous")
    support_production_t_h: list[float] = Field(..., description="Range BF-02 ran, t/h")
    margin_pt: float = Field(..., description="Margin of the class range, percentage points")
    margin_t_h: float = Field(..., description="Margin of the production range, t/h")


class CurrentBlend(BaseModel):
    window_end: datetime = Field(..., description="End of the window (UTC)")
    evaluated_at: datetime | None = Field(None, description="Evaluation time (UTC)")
    certainty: Literal["High", "Medium", "Low", "Investigate"]
    support: str | None = Field(None, description="inside, margin or outside the range BF-02 ran")
    outside: list[str] = Field(default_factory=list, description="Classes or production outside the range")
    investigate: list[str] = Field(default_factory=list, description="Reasons for Investigate")
    unmapped_ores: list[str] = Field(default_factory=list, description="Ore types taken as the default lump ore")
    k_strata: float | None = Field(None, description="Coke at omega_STRATA, kg/tHM dry")
    k_strata_sd: float | None = Field(None, description="1 sigma of k_strata, kg/tHM dry")
    omega_strata: float | None = Field(None, description="STRATA shaft efficiency, dimensionless")
    omega_strata_sd: float | None = Field(None, description="1 sigma of omega_strata, dimensionless")


class StrataSummary(BaseModel):
    version: str = Field(..., description="STRATA version of the asset")
    version_detail: str
    register_version: str | None = Field(None, description="Fact register version")
    validation: Validation
    families: list[FamilyRow] = Field(..., description="Register VAL group: out-of-sample coke by blend family")
    reference: list[ReferenceResponse] = Field(..., description="Register MOD group: responses at the reference point")
    tiers: Tiers
    current: CurrentBlend | None = Field(None, description="Tier of the charged blend in the latest STRATA window")
    current_note: str | None = Field(None, description="Why there is no current tier")


@router.get("/summary", response_model=StrataSummary, summary="Validation, reference responses and certainty tiers")
def summary(store: Annotated[TwinStore | None, Depends(optional_twin_store)] = None) -> StrataSummary:
    out: dict[str, Any] = strata.model_summary()
    note = None
    if store is None:
        note = "The twin store is not configured: set BF_TWIN_DB_URL or DATABASE_URL."
    else:
        now = datetime.now(UTC)
        out["current"] = strata.current_blend(store.snapshots.series(now - CURRENT_LOOKBACK, now))
        if out["current"] is None:
            note = "No twin window with STRATA in the last 2 days."
    return StrataSummary.model_validate(out | {"current_note": note})


# ---------------------------------------------------------------------------------------------------- live point
class LiveResponse(BaseModel):
    id: str = Field(..., description="Fact id (MOD-nn)")
    statement: str
    unit: str = Field(..., description="Unit of the responses (kg/tHM dry coke per change)")
    fixed_omega: float | None = Field(None, description="Coke response at the current calibration, Fixed-ω, in unit")
    strata_omega: float | None = Field(None, description="Coke response at the current calibration, STRATA-ω, in unit")
    difference: float | None = Field(None, description="STRATA-ω minus Fixed-ω, in unit")
    sigma_strata: float | None = Field(None, description="1 sigma of the STRATA-ω response, in unit")
    d_omega: float | None = Field(None, description="Change of omega under STRATA-ω, dimensionless")
    reference_fixed_omega: float | None = Field(None, description="Register value at the reference point, in unit")
    reference_strata_omega: float | None = Field(None, description="Register value at the reference point, in unit")
    reference_sigma_strata: float | None = Field(None, description="Register 1 sigma at the reference point, in unit")


class RingRow(BaseModel):
    ring: int = Field(..., description="Ring number, 1 centre to 6 wall (equal areas)")
    name: str
    gas_share: float = Field(..., description="Share of the gas at the top of the thermal reserve zone, fraction")
    fe_share: float = Field(..., description="Share of the burden's Fe, fraction")
    residence_h: float = Field(..., description="Time the burden spends in the thermal reserve zone, h")
    gas_capacity: float = Field(..., description="C_g = R × gas share, dimensionless")
    ore_oxygen: float = Field(..., description="C_s = Fe share (wustite oxygen of the ring), dimensionless")
    ntu: float = Field(..., description="Number of transfer units of the ring, dimensionless")
    efficiency: float = Field(..., description="Counter-current efficiency of the ring, fraction")
    omega_share: float = Field(..., description="The ring's part of omega (the parts sum to omega), dimensionless")
    limited_by: Literal["gas", "ore"] = Field(..., description="The smaller of C_g and C_s")


class RingInputs(BaseModel):
    blend_wet_pct: dict[str, float] = Field(..., description="Wet share of ferrous by model material, %")
    classes_wet_pct: dict[str, float] = Field(..., description="Wet share of ferrous by class, %")
    coke_kg_thm: float | None = Field(None, description="Coke, kg/tHM dry")
    nut_coke_kg_thm: float | None = Field(None, description="Nut coke, kg/tHM")
    pci_kg_thm: float | None = Field(None, description="PCI, kg/tHM")
    production_t_h: float | None = Field(None, description="Hot metal, t/h")
    blast_nm3_thm: float | None = Field(None, description="Hot blast incl. O2, Nm³/tHM")
    o2_pct: float | None = Field(None, description="O2 fraction of the hot blast, %")
    sinter_rdi_pct: float | None = Field(None, description="Sinter RDI, %")
    ferrous_kg_thm: float | None = Field(None, description="Ferrous charge, kg/tHM wet")
    blast_moisture_g_nm3: float | None = Field(None, description="Blast moisture, g/Nm³")


class RingView(BaseModel):
    state: str = Field(..., description="calibration: the network solve at the calibration state")
    distance: float = Field(..., description="Scaled distance from the nearest network solve (0 = solved there)")
    omega: float = Field(..., description="w of the ring model at the state, dimensionless")
    capacity_ratio: float = Field(..., description="R, gas oxygen capacity over wustite oxygen, dimensionless")
    k_blend: float = Field(..., description="Fe-weighted relative reduction rate of the blend, dimensionless")
    kappa: float = Field(..., description="Rate constant refitted so that w equals the calibrated omega, 1/h")
    gas_in_excess: float = Field(..., description="Gas share of the rings where C_g ≥ C_s, fraction")
    inputs: RingInputs
    rings: list[RingRow]


class LiveCalibration(BaseModel):
    source: str | None = Field(None, description="live, or fallback (the shipped calibration)")
    window: list[str] = Field(default_factory=list, description="First and last shift date of the calibration")
    n_shifts: int | None = Field(None, description="Stable shifts in the calibration, count")
    omega: float | None = Field(None, description="Calibrated shaft efficiency, dimensionless")
    q_lz_gj_h: float | None = Field(None, description="Lower-zone heat loss, GJ/h")
    q_uz_gj_h: float | None = Field(None, description="Upper-zone heat loss, GJ/h")
    r_h: float | None = Field(None, description="Hydrogen utilisation, dimensionless")
    coke_kg_thm: float | None = Field(None, description="Coke rate of the calibration, kg/tHM dry")
    message: str | None = None
    basis: str | None = Field(None, description="day (the nightly calibration) or live")
    day: str | None = Field(None, description="Day of the nightly calibration (IST)")
    note: str | None = None


class ReferencePoint(BaseModel):
    production_t_d: float = Field(..., description="Hot metal of the reference point, t/d")
    blend_wet_pct: dict[str, float] = Field(..., description="Base ferrous blend, wet %")
    yard: str


class LivePoint(BaseModel):
    kind: str = Field(..., description="Job kind that refreshes the live point")
    made_at: datetime | None = Field(None, description="When the job finished (UTC); null before the first run")
    job_id: str | None = None
    version: str | None = Field(None, description="STRATA version")
    calibration: LiveCalibration | None = None
    reference_point: ReferencePoint | None = None
    responses: list[LiveResponse] = Field(default_factory=list)
    rings: RingView | None = None
    network: dict[str, Any] = Field(default_factory=dict, description="Network solves and seconds of the job")
    seconds: float | None = Field(None, description="Run time of the computation, s")


@router.get("/live-point", response_model=LivePoint, summary="Coke responses and ring view at the calibration")
def live_point() -> LivePoint:
    """The latest result of the job ``strata.live_point`` (empty before its first run)."""
    return LivePoint.model_validate(strata.live_point_view(strata.read_live_point()))
