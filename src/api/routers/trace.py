"""TRACE: burnt coke from the gas-side oxygen balance (SV-1.10, #290).

Per shift and per day from the TRACE view the worker builds every hour (``twin.trace``, ``services.trace``: the Coke
Energy pipeline's values, page 10's); per hour from the twin's 8-h windows in the twin store (``twin.evaluate``). A
request never runs the pipeline: before the first build the summary says so (``available`` false) and the other routes
answer 503. Coke rates are kg/tHM dry, times UTC.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.auth import require_role
from api.errors import ApiError
from api.store import optional_twin_store
from data.twin.repositories import TwinStore
from services import trace

log = logging.getLogger("api.trace")
router = APIRouter(prefix="/api/trace", tags=["trace"], dependencies=[Depends(require_role("user"))])

MAX_SERIES_DAYS = {"hour": 92, "shift": 400, "day": 800}
LATEST_HOUR_LOOKBACK = timedelta(hours=48)

K_BURNT = "Burnt coke, calibrated: raw + the 14-day correction (TRACE), kg/tHM"
K_RAW = "Burnt coke, raw: oxygen balance at the metered blast air and the analyser's ηCO (GBC), kg/tHM"
OFFSET = "14-day correction: median of charged − raw burnt over the stable shifts of the 14 days before, kg/tHM"
K_CHARGED = "Coke charged (dry), kg/tHM"


class TraceShift(BaseModel):
    skey: str = Field(..., description="Shift key YYYY-MM-DD_A|B|C (IST date of the shift start)")
    shift: str = Field(..., description="Shift letter: A 06-14, B 14-22, C 22-06 IST")
    start: datetime = Field(..., description="Shift start (UTC)")
    end: datetime = Field(..., description="Shift end (UTC)")
    stable: bool = Field(..., description="The shift passed the stability filter (calibration pool)")
    provisional: bool = Field(..., description="The DPR of the date was carried forward (not entered yet)")
    k_burnt: float | None = Field(None, description=K_BURNT)
    k_burnt_raw: float | None = Field(None, description=K_RAW)
    burnt_offset: float | None = Field(None, description=OFFSET)
    k_charged: float | None = Field(None, description=K_CHARGED)
    hm_t: float | None = Field(None, description="Hot metal of the shift, t")
    prod_th: float | None = Field(None, description="Production, t/h")


class TraceInput(BaseModel):
    key: str = Field(..., description="Input key, e.g. blast_air")
    label: str = Field(..., description="Name of the input")
    value: float | None = Field(None, description="Value in the shift, in the unit of the unit field")
    unit: str = Field(..., description="Unit of value, e.g. Nm³/tHM")
    digits: int = Field(..., description="Decimals to show, count")
    source: str = Field(
        ..., description="Source of the value: a freshness source (online_10min, hourly_dataset, hm_slag), "
        "open_meteo or plant_rule",
    )  # fmt: skip


class TraceFeed(BaseModel):
    source: str = Field(..., description="Freshness source, e.g. online_10min, hm_slag, rm_chemistry")
    material: str = Field(..., description="Material code ('' for a single-series source)")
    item: str = Field(..., description="What the row covers, e.g. Lloyds CLO chemistry")
    last: datetime | None = Field(None, description="Last valid data point at or before the reference (UTC)")
    age_s: float | None = Field(None, description="Age of the last valid data point at the reference, s")
    limit_s: float = Field(..., description="Freshness limit of the source, s")
    stale: bool = Field(..., description="No data, or older than the limit: the value is carried forward")
    used_by: str = Field(..., description="Which result rests on the source")
    feeds_models: bool = Field(..., description="A stale value changes the coke rates")


class TraceLatestShift(BaseModel):
    shift: TraceShift
    inputs: list[TraceInput] = Field(..., description="Inputs of the oxygen balance in the shift")
    feeds: list[TraceFeed] = Field(..., description="Freshness of every source at the end of the shift")


class TraceLatestHour(BaseModel):
    start: datetime = Field(..., description="Window start (UTC): the 8 h before end")
    end: datetime = Field(..., description="Window end, a full hour (UTC)")
    label: str = Field(..., description="Shift key when the window is a shift, else the IST end time")
    status: str | None = Field(None, description="complete, partial or failed")
    k_burnt: float | None = Field(None, description=K_BURNT)
    k_burnt_raw: float | None = Field(None, description=K_RAW)
    burnt_offset: float | None = Field(None, description=OFFSET)
    k_charged: float | None = Field(None, description=K_CHARGED)


class TraceCorrectionShift(BaseModel):
    skey: str = Field(..., description="Shift key of a stable shift in the window")
    start: datetime = Field(..., description="Shift start (UTC)")
    k_charged: float | None = Field(None, description=K_CHARGED)
    k_burnt_raw: float | None = Field(None, description=K_RAW)
    gap: float | None = Field(None, description="Charged − raw burnt coke, kg/tHM")


class TraceCorrection(BaseModel):
    skey: str = Field(..., description="Shift the correction belongs to (the latest with one)")
    offset: float | None = Field(None, description=OFFSET)
    window_start: date = Field(..., description="First shift date of the window (IST)")
    window_end: date = Field(..., description="Date the window ends before, the shift's date (IST)")
    window_days: int = Field(..., description="Length of the window, d")
    min_shifts: int = Field(..., description="Stable shifts the median needs, count")
    n_shifts: int = Field(..., description="Stable shifts in the window, count")
    pool: list[TraceCorrectionShift] = Field(..., description="The stable shifts of the window, oldest first")


class TraceSettings(BaseModel):
    window_days: int = Field(..., description="Window of the correction, d")
    min_shifts: int = Field(..., description="Stable shifts the correction needs, count")
    history_start: str = Field(..., description="First date the pipeline balances (IST)")


class TraceSources(BaseModel):
    hourly_end: datetime | None = Field(None, description="Last hour of the hourly dataset (UTC)")
    online_end: datetime | None = Field(None, description="Last 10-min bin of the process data (UTC)")
    hm_slag_end: datetime | None = Field(None, description="Last hot metal / slag analysis (UTC)")
    dpr_end: datetime | None = Field(None, description="Last DPR date entered (UTC, start of the IST day)")


class TraceSummary(BaseModel):
    available: bool = Field(..., description="A TRACE view has been built")
    job_kind: str = Field(trace.JOB_KIND, description="Job kind that builds the view")
    built_at: datetime | None = Field(None, description="Time of the build (UTC)")
    age_s: float | None = Field(None, description="Time since the build, s")
    stale: bool = Field(..., description="No view, or older than stale_after_s")
    stale_after_s: float = Field(trace.STALE_AFTER_S, description="Age from which the view is stale, s")
    settings: TraceSettings | None = None
    sources: TraceSources | None = None
    errors: int = Field(0, description="Shifts that could not be balanced, count")
    shifts: int = Field(0, description="Balanced shifts in the view, count")
    latest_shift: TraceLatestShift | None = Field(None, description="Latest shift with a burnt coke")
    previous_shift: TraceShift | None = Field(None, description="The shift before the latest")
    latest_hour: TraceLatestHour | None = Field(None, description="Latest twin window with a burnt coke")
    hour_available: bool = Field(..., description="The twin store is configured (hourly windows)")
    correction: TraceCorrection | None = None
    feeds: list[TraceFeed] = Field(default_factory=list, description="Freshness of every source, as of feeds_at")
    feeds_at: datetime = Field(..., description="Reference time of feeds: now (UTC)")


class TraceSeriesRow(BaseModel):
    start: datetime = Field(..., description="Window start (UTC)")
    end: datetime = Field(..., description="Window end (UTC)")
    label: str = Field(..., description="Shift key, IST date or IST window end")
    stable: bool | None = Field(None, description="Shift resolution: the shift passed the stability filter")
    status: str | None = Field(None, description="Hour resolution: complete, partial or failed")
    n_shifts: int | None = Field(None, description="Shifts behind the value, count")
    k_burnt: float | None = Field(None, description=K_BURNT)
    k_burnt_raw: float | None = Field(None, description=K_RAW)
    burnt_offset: float | None = Field(None, description=OFFSET)
    k_charged: float | None = Field(None, description=K_CHARGED)


class TraceSeries(BaseModel):
    resolution: Literal["hour", "shift", "day"]
    start: datetime = Field(..., description="First window end included (UTC)")
    end: datetime = Field(..., description="Last window end included (UTC)")
    count: int = Field(..., description="Rows, count")
    built_at: datetime | None = Field(None, description="Build of the view behind shift and day rows (UTC)")
    rows: list[TraceSeriesRow]


class AgreementStats(BaseModel):
    window: str = Field(..., description="Averaging window, e.g. Shift (8 h), 1 week")
    shifts: int = Field(..., description="Shifts per window, count")
    hours: float = Field(..., description="Length of the window, h")
    n: int = Field(..., description="Windows compared, count")
    ccc: float | None = Field(None, description="Lin's concordance correlation coefficient, dimensionless")
    r: float | None = Field(None, description="Pearson correlation, dimensionless")
    cb: float | None = Field(None, description="Bias correction factor C_b = CCC / r, dimensionless")
    bias: float | None = Field(None, description="Mean of burnt − charged, kg/tHM")
    sd: float | None = Field(None, description="Standard deviation of burnt − charged, kg/tHM")
    loa_lo: float | None = Field(None, description="Lower limit of agreement, bias − 1.96 SD, kg/tHM")
    loa_hi: float | None = Field(None, description="Upper limit of agreement, bias + 1.96 SD, kg/tHM")
    mean_charged: float | None = Field(None, description="Mean charged coke over the windows, kg/tHM")
    mean_burnt: float | None = Field(None, description="Mean burnt coke over the windows, kg/tHM")
    slope: float | None = Field(None, description="Deming slope of burnt on charged, dimensionless")


class AgreementPoint(BaseModel):
    t: datetime = Field(..., description="Mid-time of the window (UTC)")
    n: int = Field(..., description="Shifts in the window, count")
    charged: float | None = Field(None, description="Charged coke, production-weighted, kg/tHM")
    burnt: float | None = Field(None, description="Burnt coke (calibrated), production-weighted, kg/tHM")
    mean: float | None = Field(None, description="Mean of burnt and charged, kg/tHM")
    diff: float | None = Field(None, description="Burnt − charged, kg/tHM")


class AgreementSplit(BaseModel):
    transit_sd: float | None = Field(None, description="Per-shift SD from coke in transit (falls as 1/W), kg/tHM")
    random_sd: float | None = Field(None, description="Per-shift SD of random error (falls as 1/√W), kg/tHM")
    floor_sd: float | None = Field(None, description="SD that does not fall with the window, kg/tHM")


class AgreementClosure(BaseModel):
    n: int = Field(..., description="Shifts with both raw burnt and charged coke, count")
    ratio: float | None = Field(None, description="Total raw burnt over total charged coke, dimensionless")
    burnt_t: float | None = Field(None, description="Total raw burnt coke, t")
    charged_t: float | None = Field(None, description="Total charged coke, t")


class TraceAgreement(BaseModel):
    built_at: datetime | None = Field(None, description="Build of the view (UTC)")
    shift: AgreementStats | None = Field(None, description="Per shift (the window table's Shift (8 h) row)")
    week: AgreementStats | None = Field(None, description="Per week (the window table's 1 week row)")
    table: list[AgreementStats] = Field(..., description="Every averaging window of transit.summary")
    points_shift: list[AgreementPoint] = Field(..., description="Pairs per shift, oldest first")
    points_week: list[AgreementPoint] = Field(..., description="Pairs per week, oldest first")
    split: AgreementSplit
    peak_lag_h: float | None = Field(None, description="Lag of the correlation peak, burnt after charged, h")
    closure: AgreementClosure | None = None


# --------------------------------------------------------------------------------------------- helpers
def _aware(value: datetime | None, default: datetime) -> datetime:
    if value is None:
        return default
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _view() -> dict[str, Any]:
    view = trace.load()
    if view is None:
        raise ApiError(503, f"The TRACE view has not been built yet: the {trace.JOB_KIND} job builds it every hour.")
    return view


def _latest_hour(store: TwinStore | None, now: datetime) -> tuple[dict[str, Any] | None, bool]:
    """The latest stored window with a burnt coke, and whether the hourly windows can be read at all (the summary
    still answers when the twin store is down)."""
    if store is None:
        return None, False
    try:
        return trace.latest_hour(store.snapshots.series(now - LATEST_HOUR_LOOKBACK, now)), True
    except Exception:  # noqa: BLE001 - the shift view does not depend on the store
        log.warning("TRACE: the twin store could not be read for the latest hour", exc_info=True)
        return None, False


# --------------------------------------------------------------------------------------------- routes
@router.get("", response_model=TraceSummary, summary="TRACE: latest shift and hour, correction, feeds")
def summary(store: Annotated[TwinStore | None, Depends(optional_twin_store)] = None) -> TraceSummary:
    """The latest shift's burnt coke with the inputs of its oxygen balance, the latest hourly window, the 14-day
    correction and the age of every source as of now."""
    now = datetime.now(UTC)
    hour, hour_available = _latest_hour(store, now)
    view = trace.load()
    if view is None:
        return TraceSummary(available=False, stale=True, latest_hour=hour, hour_available=hour_available, feeds_at=now)
    shifts = view.get("shifts") or []
    latest = view.get("latest")
    latest_shift = None
    previous = None
    if latest is not None:
        index = next((i for i, r in enumerate(shifts) if r["skey"] == latest["skey"]), None)
        if index is not None:
            latest_shift = TraceLatestShift(shift=shifts[index], inputs=latest["inputs"], feeds=latest["feeds"])
            previous = shifts[index - 1] if index > 0 else None
    age = trace.age_s(view, now)
    return TraceSummary(
        available=True,
        built_at=view["built_at"],
        age_s=age,
        stale=age is None or age > trace.STALE_AFTER_S,
        settings=view["settings"],
        sources=view["sources"],
        errors=int(view.get("errors") or 0),
        shifts=len(shifts),
        latest_shift=latest_shift,
        previous_shift=previous,
        latest_hour=hour,
        hour_available=hour_available,
        correction=view.get("correction"),
        feeds=trace.feeds_now(view, now),
        feeds_at=now,
    )


@router.get("/series", response_model=TraceSeries, summary="Burnt coke per hour, shift or day")
def series(
    resolution: Annotated[Literal["hour", "shift", "day"], Query(description="hour (8-h windows), shift or day")],
    start: Annotated[datetime | None, Query(description="First window end (default: end - 14 days)")] = None,
    end: Annotated[datetime | None, Query(description="Last window end (default: now)")] = None,
    store: Annotated[TwinStore | None, Depends(optional_twin_store)] = None,
) -> TraceSeries:
    """Raw and calibrated burnt coke, the correction and charged coke for the windows ending in [start, end], oldest
    first."""
    stop = _aware(end, datetime.now(UTC))
    first = _aware(start, stop - timedelta(days=14))
    if first > stop:
        raise ApiError(422, "start lies after end.")
    if stop - first > timedelta(days=MAX_SERIES_DAYS[resolution]):
        raise ApiError(422, f"At most {MAX_SERIES_DAYS[resolution]} days per request at {resolution} resolution.")
    built_at = None
    if resolution == "hour":
        if store is None:
            raise ApiError(503, "Hourly TRACE needs the twin store: set BF_TWIN_DB_URL or DATABASE_URL.")
        rows = trace.hour_series(store.snapshots.series(first, stop))
    else:
        view = _view()
        built_at = view["built_at"]
        rows = (trace.shift_series if resolution == "shift" else trace.day_series)(view, first, stop)
    return TraceSeries(resolution=resolution, start=first, end=stop, count=len(rows), built_at=built_at, rows=rows)


@router.get("/agreement", response_model=TraceAgreement, summary="Burnt against charged coke")
def agreement() -> TraceAgreement:
    """Lin's CCC and the Bland-Altman bias and limits of agreement per averaging window (``transit.summary`` over every
    balanced shift), with the per-shift and per-week pairs."""
    view = _view()
    agree = view.get("agreement")
    if agree is None:
        raise ApiError(503, "The TRACE view has no balanced shifts to compare.")
    return TraceAgreement(
        built_at=view["built_at"],
        shift=trace.basis_stats(agree, "shift"),
        week=trace.basis_stats(agree, "week"),
        table=agree["table"],
        points_shift=agree["points"]["shift"],
        points_week=agree["points"]["week"],
        split=agree["split"],
        peak_lag_h=agree.get("peak_lag_h"),
        closure=agree.get("closure"),
    )
