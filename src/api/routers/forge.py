"""FORGE page (SV-1.11, #291): FORGE against burnt coke per shift with page 10's statistics, the Rist diagram of the
stable shifts and the 14-day calibration.

Every route reads the view the ``twin.forge`` job stores hourly (``services.forge``); none runs the pipeline or a
balance. Coke rates are kg/tHM dry, heat losses GJ/h, Rist coordinates oxygen atoms per Fe atom (O/Fe) against the
gas oxidation (O/C), days IST calendar days, times UTC.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.auth import require_role
from api.errors import ApiError
from services import forge

router = APIRouter(prefix="/api/forge", tags=["forge"], dependencies=[Depends(require_role("user"))])

MAX_PERIOD_DAYS = 92  # the trailing statistics are computed per day of the period


class ForgeSources(BaseModel):
    hourly_end: datetime | None = Field(None, description="Latest hour of the hourly dataset (UTC)")
    online_end: datetime | None = Field(None, description="Latest 10-min process record (UTC)")
    hm_slag_end: datetime | None = Field(None, description="Latest hot-metal / slag analysis (UTC)")
    dpr_end: datetime | None = Field(None, description="Latest daily production report (UTC)")


class ForgeViewInfo(BaseModel):
    built_at: datetime = Field(..., description="When the twin.forge job stored the view (UTC)")
    job_id: str | None = Field(None, description="Job that stored it")
    model: str
    sources: ForgeSources
    shifts: int = Field(..., description="Balanced shifts in the view, count")
    pipeline_errors: int = Field(..., description="Shifts the balance could not solve, count")
    window_days: int = Field(..., description="Calibration window before a shift's date, days")
    min_shifts: int = Field(..., description="Stable shifts a calibration needs, count")
    stats_days: int = Field(..., description="Trailing window of the statistics, days")
    rist_days: int = Field(..., description="Stable shifts drawn in the Rist diagram: those of these last days, days")


class ForgeStats(BaseModel):
    n: int = Field(..., description="Shifts with FORGE and burnt coke, count")
    bias: float | None = Field(None, description="Mean of FORGE − burnt coke, kg/tHM")
    sd: float | None = Field(None, description="Standard deviation of FORGE − burnt coke (n − 1), kg/tHM")
    mae: float | None = Field(None, description="Mean absolute difference of FORGE and burnt coke, kg/tHM")
    r: float | None = Field(None, description="Pearson correlation of FORGE and burnt coke, dimensionless")
    within_5_pct: float | None = Field(None, description="Shifts within ±5 kg/tHM of burnt coke, %")
    within_10_pct: float | None = Field(None, description="Shifts within ±10 kg/tHM of burnt coke, %")


class ForgeShift(BaseModel):
    shift_key: str = Field(..., description="YYYY-MM-DD_A|B|C (IST date of the shift start)")
    day: date = Field(..., description="IST date of the shift")
    t_mid: datetime = Field(..., description="Middle of the shift (UTC)")
    stable: bool = Field(..., description="Passed the stability filter")
    provisional: bool = Field(..., description="Balanced on a DPR carried forward")
    k_forge: float | None = Field(None, description="FORGE: two-zone balance at the 14-day medians, kg/tHM")
    k_burnt: float | None = Field(None, description="Burnt coke (TRACE): metered blast less the 14-day offset, kg/tHM")
    k_obal: float | None = Field(None, description="Oxygen-balance model at the analyser's eta CO, kg/tHM")
    k_charged: float | None = Field(None, description="Coke charged, kg/tHM")
    error: float | None = Field(None, description="FORGE − burnt coke, kg/tHM")
    omega_fixed: float | None = Field(None, description="Shaft efficiency FORGE used (14-day median), dimensionless")
    omega_shift: float | None = Field(None, description="Shaft efficiency calibrated on the shift, dimensionless")


class ForgeRollingPoint(BaseModel):
    day: date = Field(..., description="Last IST day of the trailing window")
    t: datetime = Field(..., description="Middle of the day's last shift in the window (UTC)")
    n: int = Field(..., description="Shifts in the trailing window, count")
    bias: float | None = Field(None, description="Mean of FORGE − burnt coke over the window, kg/tHM")
    sd: float | None = Field(None, description="Standard deviation of FORGE − burnt coke over the window, kg/tHM")
    mae: float | None = Field(None, description="Mean absolute difference over the window, kg/tHM")
    r: float | None = Field(None, description="Correlation of FORGE and burnt coke over the window, dimensionless")
    within_5_pct: float | None = Field(None, description="Shifts within ±5 kg/tHM over the window, %")
    within_10_pct: float | None = Field(None, description="Shifts within ±10 kg/tHM over the window, %")


class ForgePrediction(BaseModel):
    view: ForgeViewInfo
    start: date | None = Field(None, description="First IST day of the period")
    end: date | None = Field(None, description="Last IST day of the period")
    first_day: date | None = Field(None, description="First IST day with FORGE and burnt coke")
    last_day: date | None = Field(None, description="Last IST day with FORGE and burnt coke")
    with_unstable: bool = Field(..., description="Shifts that failed the stability filter are included")
    stats_days: int = Field(..., description="Trailing window of the rolling statistics, days")
    stats: ForgeStats = Field(..., description="Statistics of the period (page 10's table, FORGE vs burnt)")
    record: ForgeStats = Field(..., description="Statistics of every stable shift in the view")
    shifts: list[ForgeShift] = Field(..., description="Shifts of the period, oldest first")
    rolling: list[ForgeRollingPoint] = Field(..., description="Per day of the period: statistics of its trailing days")


class RistLine(BaseModel):
    mu: float = Field(..., description="Slope: carbon gasified per Fe atom, O/Fe")
    y_a: float = Field(..., description="Y at X = 0: blast, PCI, Si/Mn/P/Ti and carbonate oxygen, O/Fe")
    y_d: float = Field(..., description="Y at X = 1: ore oxygen removed by carbon (direct reduction), O/Fe")
    omega: float = Field(..., description="Shaft efficiency the line was drawn at, dimensionless")
    eta_w: float = Field(..., description="CO utilisation at the wüstite–iron equilibrium W, dimensionless")
    y_wc: float = Field(..., description="Wüstite level, carbon's share, O/Fe")
    x_top: float = Field(..., description="Top gas: 1 + eta_f, O/C")
    y_top_c: float = Field(..., description="Carbon's share of the burden oxygen at the top, O/Fe")
    y_top: float = Field(..., description="All burden oxygen at the top (hydrogen's part included), O/Fe")


class RistIdeal(BaseModel):
    mu: float = Field(..., description="Slope of the ideal line (omega = 1), O/Fe")
    y_a: float = Field(..., description="Y at X = 0 of the ideal line, O/Fe")


class ForgeRistFrame(BaseModel):
    skey: str = Field(..., description="Shift key, YYYY-MM-DD_A|B|C")
    day: date = Field(..., description="IST date of the shift")
    t_mid: datetime = Field(..., description="Middle of the shift (UTC)")
    t_r: float = Field(..., description="Temperature of the thermal reserve zone, °C")
    y_w: float = Field(..., description="O/Fe of wüstite, O/Fe")
    actual: RistLine = Field(..., description="Shift line: calibrated on burnt coke, the reconciled blast, eta CO")
    model: RistLine | None = Field(None, description="FORGE line: omega, heat losses and r_H at the 14-day medians")
    p_actual: tuple[float, float] | None = Field(None, description="Pivot of the shift's lower-zone heat balance, O/C")
    p_model: tuple[float, float] | None = Field(None, description="Pivot of FORGE's lower-zone heat balance, O/C")
    ideal: RistIdeal | None = Field(None, description="Heat-balanced line through P and W (omega = 1)")
    k_burnt: float | None = Field(None, description="Burnt coke, kg/tHM")
    k_forge: float | None = Field(None, description="FORGE, kg/tHM")
    k_obal: float | None = Field(None, description="Oxygen-balance model, kg/tHM")
    k_charged: float | None = Field(None, description="Coke charged, kg/tHM")


class ForgeRist(BaseModel):
    view: ForgeViewInfo
    first_day: date | None = Field(None, description="IST date of the first frame")
    last_day: date | None = Field(None, description="IST date of the last frame")
    requested: int = Field(
        ..., description="Stable shifts the job drew, count (frames that did not solve are left out)"
    )
    frames: list[ForgeRistFrame] = Field(..., description="Rist diagram per stable shift, oldest first")


class CalibrationShift(BaseModel):
    shift_key: str = Field(..., description="Shift key, YYYY-MM-DD_A|B|C")
    t_mid: datetime = Field(..., description="Middle of the shift (UTC)")
    omega: float | None = Field(None, description="Shaft efficiency calibrated on the shift, dimensionless")
    q_lz_gj_h: float | None = Field(None, description="Lower-zone heat loss calibrated on the shift, GJ/h")
    q_uz_gj_h: float | None = Field(None, description="Upper-zone heat loss calibrated on the shift, GJ/h")
    r_h: float | None = Field(None, description="Hydrogen reduction share calibrated on the shift, dimensionless")
    k_burnt: float | None = Field(None, description="Burnt coke, kg/tHM")
    k_charged: float | None = Field(None, description="Coke charged (the calibration's basis), kg/tHM")


class ForgeCalibration(BaseModel):
    view: ForgeViewInfo
    day: date = Field(..., description="IST day the calibration applies to")
    window_start: date = Field(..., description="First IST day of the window")
    window_end: date = Field(..., description="Last IST day of the window (the day before)")
    window_days: int = Field(..., description="Window length, days")
    min_shifts: int = Field(..., description="Stable shifts the calibration needs, count")
    n_shifts: int = Field(..., description="Stable shifts in the window, count")
    n_calibrated: int = Field(..., description="Of them with a two-zone calibration, count")
    complete: bool = Field(..., description="The window holds at least min_shifts stable shifts")
    omega_cal: float | None = Field(None, description="Median shaft efficiency, dimensionless")
    q_lz_gj_h: float | None = Field(None, description="Median lower-zone heat loss, GJ/h")
    q_uz_gj_h: float | None = Field(None, description="Median upper-zone heat loss, GJ/h")
    r_h: float | None = Field(None, description="Median hydrogen reduction share, dimensionless")
    shifts: list[CalibrationShift] = Field(..., description="The stable shifts of the window, oldest first")


def _view() -> forge.StoredView:
    try:
        return forge.load()
    except forge.ViewUnavailable as exc:
        raise ApiError(404, str(exc), title="FORGE view not built") from exc


def _period(start: date | None, end: date | None) -> None:
    if start is not None and end is not None and start > end:
        raise ApiError(422, "start lies after end.")


@router.get("/prediction", response_model=ForgePrediction, summary="FORGE against burnt coke per shift")
def prediction(
    stored: Annotated[forge.StoredView, Depends(_view)],
    start: Annotated[
        date | None, Query(description="First IST day (default: end − stats_days; at most 92 days per request)")
    ] = None,
    end: Annotated[date | None, Query(description="Last IST day (default: the last day with both values)")] = None,
    unstable: Annotated[bool, Query(description="Include shifts that failed the stability filter")] = False,
) -> ForgePrediction:
    """FORGE and burnt coke per shift of the period with page 10's statistics, the whole record's statistics (stable
    shifts) and, per day, the statistics of its trailing ``stats_days``."""
    _period(start, end)
    first, last = forge.resolve_period(stored, start, end)
    if first is not None and last is not None and (last - first).days + 1 > MAX_PERIOD_DAYS:
        raise ApiError(422, f"At most {MAX_PERIOD_DAYS} days per request.")
    return ForgePrediction(view=stored.info(), **forge.cached_prediction(stored, start, end, unstable))


@router.get("/rist", response_model=ForgeRist, summary="Rist diagram of the stable shifts")
def rist(
    stored: Annotated[forge.StoredView, Depends(_view)],
    start: Annotated[date | None, Query(description="First IST day (default: the first frame)")] = None,
    end: Annotated[date | None, Query(description="Last IST day (default: the last frame)")] = None,
) -> ForgeRist:
    """The operating lines of the stable shifts of the last ``rist_days`` (``utils.coke_energy.rist_diagram``)."""
    _period(start, end)
    return ForgeRist(view=stored.info(), **forge.rist(stored, start, end))


@router.get("/calibration", response_model=ForgeCalibration, summary="FORGE calibration and its shifts")
def calibration(
    stored: Annotated[forge.StoredView, Depends(_view)],
    day: Annotated[
        date | None, Query(description="IST day it applies to (default: the day after the latest balanced shift)")
    ] = None,
) -> ForgeCalibration:
    """Medians of omega, Q_LZ, Q_UZ and r_H over the stable shifts of the 14 days before ``day`` and those shifts."""
    return ForgeCalibration(view=stored.info(), **forge.calibration(stored, day))
