"""Live twin: latest evaluation and its freshness, hourly and per-shift series, burden transit, alarms (SV-1.6, #286);
for the Live Twin page (SV-1.9, #289) the data age of each source, the model gaps against TRACE, one window in full and
the latest blend change in the column.

Snapshots and alarms come from the twin store (``twin.snapshot``, ``twin.alarm_event``; written by ``twin.evaluate``
and the drift alarm); the burden column is computed from the hourly dataset at the requested time. Coke rates are
kg/tHM dry, shaft efficiencies dimensionless, times UTC unless stated.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from api.auth import require_role
from api.errors import ApiError
from api.store import optional_twin_store, twin_store
from data.twin.repositories import TwinStore
from services import live_status, source_health, twin_alarms, twin_views
from services.events import TwinSnapshotData

log = logging.getLogger("api.twin")

router = APIRouter(prefix="/api/twin", tags=["twin"], dependencies=[Depends(require_role("user"))])

MAX_SERIES_DAYS = 92


class SourceAge(BaseModel):
    key: Literal["influxdb", "hourly", "hm_slag", "dpr", "open_meteo"]
    label: str = Field(..., description="Name of the source on the page")
    data_end: datetime | None = Field(None, description="Newest time the source held at the evaluation (UTC)")
    age_h: float | None = Field(None, description="Age of data_end now, h")
    stale_after_h: float = Field(..., description="Age from which the source is stale, h (coke_energy.yml)")
    stale: bool = Field(..., description="age_h >= stale_after_h, or no time recorded")
    provisional: bool = Field(False, description="DPR: the window's date is carried forward from an earlier report")
    recorded_at: datetime | None = Field(None, description="When the evaluation recorded data_end (UTC)")
    down: bool = Field(False, description="The last fetch of the source failed (SV-6.5); stale at once")
    error: str | None = Field(None, description="The error of the last fetch while down")
    failed_at: datetime | None = Field(None, description="Time of the failed fetch while down (UTC)")


class ModelGaps(BaseModel):
    days: float = Field(..., description="Period of the RMSE, days")
    ratio: float = Field(..., description="Multiple of the RMSE above which a window counts as excess, dimensionless")
    forge_at: datetime | None = Field(None, description="Window end of forge_trace_kg_thm (UTC)")
    forge_trace_kg_thm: float | None = Field(None, description="K_FORGE - K_TRACE of the newest window, kg/tHM dry")
    strata_at: datetime | None = Field(None, description="Window end of strata_trace_kg_thm (UTC)")
    strata_trace_kg_thm: float | None = Field(None, description="K_STRATA - K_TRACE of the newest window, kg/tHM dry")
    forge_trace_rmse_kg_thm: float | None = Field(None, description="RMSE of FORGE - TRACE over the days, kg/tHM dry")
    strata_trace_rmse_kg_thm: float | None = Field(None, description="RMSE of STRATA - TRACE over the days, kg/tHM dry")
    n_forge: int = Field(..., description="Hourly windows in the FORGE RMSE, count")
    n_strata: int = Field(..., description="Hourly windows in the STRATA RMSE, count")
    strata_excess_h: int = Field(
        ..., description="Newest consecutive hourly windows with |STRATA - TRACE| > ratio x RMSE, h"
    )
    strata_excess_since: datetime | None = Field(None, description="Window end where that run starts (UTC)")


class TwinNow(BaseModel):
    snapshot: TwinSnapshotData | None = Field(None, description="Latest stored evaluation (None before the first)")
    now: datetime = Field(..., description="Server time (UTC)")
    expected_window_end: datetime = Field(
        ..., description="End of the newest window that should be evaluated now (UTC; run at HH:10, due at HH:15)"
    )
    age_s: float | None = Field(None, description="Time since the latest evaluation, s")
    lag_h: float | None = Field(None, description="Expected window end minus the latest window end, h")
    stale: bool = Field(..., description="The latest window lags the expected one by stale_after_h or more")
    stale_after_h: float = Field(..., description="Lag from which the twin is stale, h (one missed evaluation)")
    alarms_open: int = Field(..., description="Open alarms, count")
    sources: list[SourceAge] = Field(default_factory=list, description="Data age of each source (SV-1.9)")
    gaps: ModelGaps | None = Field(None, description="FORGE and STRATA against TRACE, now and over 7 days (SV-1.9)")


class TwinSeriesRow(BaseModel):
    window_end: datetime = Field(..., description="End of the 8-h window (UTC)")
    shift_key: str | None = Field(None, description="Shift of the window when it is one (YYYY-MM-DD_A, IST date)")
    status: str = Field(..., description="complete, partial or failed")
    k_trace: float | None = Field(None, description="Burnt coke, oxygen balance + 14-day offset, kg/tHM dry")
    k_forge: float | None = Field(None, description="Two-zone balance at the 14-day medians, kg/tHM dry")
    k_strata: float | None = Field(None, description="Two-zone balance at omega_STRATA, kg/tHM dry")
    k_strata_sd: float | None = Field(None, description="1 sigma of k_strata, kg/tHM dry")
    k_charged: float | None = Field(None, description="Coke charged, kg/tHM dry")
    omega_imp: float | None = Field(None, description="Implied shaft efficiency, dimensionless")
    omega_strata: float | None = Field(None, description="STRATA 1.5 shaft efficiency, dimensionless")
    omega_strata_sd: float | None = Field(None, description="1 sigma of omega_strata, dimensionless")
    omega_cal: float | None = Field(None, description="14-day median shaft efficiency, dimensionless")
    quantities: dict[str, float] = Field(
        default_factory=dict, description="Requested quantities of the window, units in the keys (e.g. hmt_c, °C)"
    )


class TwinSeries(BaseModel):
    resolution: Literal["hour", "shift"]
    start: datetime = Field(..., description="First window end included (UTC)")
    end: datetime = Field(..., description="Last window end included (UTC)")
    count: int = Field(..., description="Rows, count")
    rows: list[TwinSeriesRow]


class TransitBand(BaseModel):
    zone: str = Field(..., description="above_cz, in_cz or below_cz")
    charges: float = Field(..., description="Charges in the zone, count")
    newest_h: float = Field(..., description="Age of the newest charge of the zone, h")
    oldest_h: float = Field(..., description="Age of the oldest charge (descent time to the zone's lower edge), h")
    coke_t_per_charge: float | None = Field(None, description="Coke per charge on the plant counter, t")
    coke_kg_per_t_ferrous: float | None = Field(None, description="Coke per tonne of ferrous of the zone, kg/t")
    coke_kg_thm: float | None = Field(None, description="Coke as charged per tonne of hot metal, kg/tHM")
    coke_volume_fraction: float | None = Field(None, description="Coke share of the charge volume, dimensionless")
    coke_inventory_t: float | None = Field(None, description="Coke in the zone, t")
    blend_wet_pct: dict[str, float] = Field(default_factory=dict, description="Wet share of the ferrous charge, %")
    tonnes: dict[str, float] = Field(default_factory=dict, description="Charged material over the zone, t")


class TransitLag(BaseModel):
    lag_h: float | None = Field(None, description="Implied transit time, charging to burning, h")
    ci_low_h: float | None = Field(None, description="Lower end of the 95 % bootstrap interval, h")
    ci_high_h: float | None = Field(None, description="Upper end of the 95 % bootstrap interval, h")
    r_peak: float | None = Field(None, description="Correlation at the lag, dimensionless")
    n_hours: int = Field(..., description="Hours in the comparison, count")


class BlendChangeView(BaseModel):
    changed_at: datetime = Field(..., description="Start of the first hour charged with the new blend (UTC)")
    step_pt: float = Field(..., description="Largest change of a ferrous share, percentage points")
    before_pct: dict[str, float] = Field(..., description="Wet share of sinter, pellet, ore in the ferrous before, %")
    after_pct: dict[str, float] = Field(..., description="Wet share of sinter, pellet, ore in the ferrous after, %")
    charges_since: float = Field(..., description="Charges from the change to the column time, count")
    cz_at: datetime = Field(..., description="Arrival of its first charge at the cohesive-zone top (UTC)")
    raceway_at: datetime = Field(..., description="Arrival of its first charge at the raceways (UTC)")
    zone: Literal["above_cz", "in_cz", "below_cz", "burnt"] = Field(..., description="Where it is at the column time")


class TwinTransit(BaseModel):
    at: datetime = Field(..., description="Time of the column state (UTC)")
    charges_per_hour: float | None = Field(None, description="Charging rate of the last hour, charges/h")
    tau_cz_h: float | None = Field(None, description="Descent time to the cohesive zone, h")
    tau_raceway_h: float | None = Field(None, description="Descent time to the raceways, h")
    tau_cz_rate_h: float | None = Field(None, description="Descent time to the cohesive zone at the current rate, h")
    tau_raceway_rate_h: float | None = Field(None, description="Descent time to the raceways at the current rate, h")
    coke_in_transit_t: float | None = Field(None, description="Coke in the column, t")
    complete: bool = Field(..., description="The hourly record reaches the oldest charge of the column")
    missing_hours: int = Field(..., description="Hours missing from the record inside the column, count")
    bands: list[TransitBand]
    lag: TransitLag | None = Field(None, description="Implied lag over the last days (with lag=true)")
    blend_change: BlendChangeView | None = Field(
        None, description="Latest change of the ferrous blend in the last 72 h and its arrival times (SV-1.9)"
    )


class AlarmView(BaseModel):
    id: int
    kind: str = Field(..., description="omega_drift, source_stale, job_failed, ...")
    subject: str = Field("", description="What the alarm is about (e.g. high or low for omega_drift)")
    severity: Literal["warning", "critical"]
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    raised_at: datetime = Field(..., description="UTC")
    cleared_at: datetime | None = Field(None, description="UTC; None while open")
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = Field(None, description="UTC")


class TwinAlarms(BaseModel):
    open: list[AlarmView] = Field(..., description="Open alarms, newest first")
    history: list[AlarmView] = Field(..., description="Alarms raised in the window, newest first")
    since: datetime = Field(..., description="Start of the history window (UTC)")


def _aware(value: datetime | None, default: datetime) -> datetime:
    if value is None:
        return default
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@router.get("/now", response_model=TwinNow, summary="Latest twin evaluation")
async def now(request: Request, store: Annotated[TwinStore, Depends(twin_store)]) -> TwinNow:
    """The latest stored evaluation, how far it lags the expected window and the number of open alarms. The sources
    carry the live fetches of this process and of the worker's status producer (SV-6.5): one that failed is down."""
    from services import coke_energy as ce

    await _follow_producer(request)
    latest = await asyncio.to_thread(store.snapshots.latest)
    fresh = twin_views.freshness(latest)
    cfg = ce.config()
    gaps = None
    if latest is not None:
        alarm_settings = twin_alarms.AlarmSettings.from_config(cfg)
        days, at = alarm_settings.rmse_days, fresh["now"]
        recent = await asyncio.to_thread(store.snapshots.series, at - timedelta(days=days), at)
        gaps = twin_views.model_gaps(recent, at, days=days, ratio=alarm_settings.ratio)
    health = source_health.HEALTH.view(fresh["now"])
    return TwinNow(
        snapshot=TwinSnapshotData.model_validate(twin_views.snapshot_data(latest)) if latest is not None else None,
        alarms_open=len(await asyncio.to_thread(store.alarms.open_alarms)),
        sources=twin_views.source_freshness(latest, fresh["now"], twin_views.freshness_limits(cfg), health),
        gaps=gaps,
        **fresh,
    )


async def _follow_producer(request: Request) -> None:
    """Record InfluxDB from the worker's latest status snapshot in Redis, when there is one (``read_latest`` feeds
    the registry); Redis down or absent changes nothing."""
    redis = getattr(getattr(request.app.state, "live", None), "redis", None)
    if redis is None:
        return
    try:
        await live_status.read_latest(redis)
    except Exception as exc:  # noqa: BLE001 - the registry keeps what this process saw
        log.debug("live snapshot not read from Redis (%s)", type(exc).__name__)


@router.get("/window", response_model=TwinSnapshotData, summary="One stored window")
def window(
    store: Annotated[TwinStore, Depends(twin_store)],
    end: Annotated[datetime, Query(description="Window end (a time without offset is UTC)")],
) -> TwinSnapshotData:
    """The stored evaluation of the window ending ``end``: every quantity, the flags (data counts, notes, network
    solves) and the model versions; the Live Twin's drill-down of an hourly point."""
    row = store.snapshots.get(_aware(end, end))
    if row is None:
        raise ApiError(404, "No stored evaluation for this window end.")
    return TwinSnapshotData.model_validate(twin_views.snapshot_data(row))


@router.get("/series", response_model=TwinSeries, summary="Twin series, hourly or per shift")
def series(
    store: Annotated[TwinStore, Depends(twin_store)],
    start: Annotated[datetime | None, Query(description="First window end (default: end - 7 days)")] = None,
    end: Annotated[datetime | None, Query(description="Last window end (default: now)")] = None,
    resolution: Annotated[Literal["hour", "shift"], Query(description="hour, or shift: the shift windows")] = "hour",
    quantities: Annotated[
        str | None, Query(description="Comma-separated quantity keys to include (e.g. hmt_c,tau_raceway_h)")
    ] = None,
) -> TwinSeries:
    """Stored windows with start <= window end <= end, oldest first (at most 92 days)."""
    stop = _aware(end, datetime.now(UTC))
    first = _aware(start, stop - timedelta(days=7))
    if first > stop:
        raise ApiError(422, "start lies after end.")
    if stop - first > timedelta(days=MAX_SERIES_DAYS):
        raise ApiError(422, f"At most {MAX_SERIES_DAYS} days per request.")
    keys = [k.strip() for k in (quantities or "").split(",") if k.strip()]
    rows = twin_views.series_rows(store.snapshots.series(first, stop), resolution=resolution, quantities=keys)
    return TwinSeries(resolution=resolution, start=first, end=stop, count=len(rows), rows=rows)


@router.get("/transit", response_model=TwinTransit, summary="Burden column and coke in transit")
def transit(
    at: Annotated[datetime | None, Query(description="Time of the column (default: now)")] = None,
    lag: Annotated[bool, Query(description="Also the implied lag against the stored K_TRACE (14 days)")] = False,
    store: Annotated[TwinStore | None, Depends(optional_twin_store)] = None,
) -> TwinTransit:
    """The charges in the column at ``at`` from the hourly dataset: descent times, coke by zone, the blend by zone."""
    from services import coke_energy as ce
    from services import transit as transit_service

    when = _aware(at, datetime.now(UTC))
    when_ist = pd.Timestamp(when).tz_convert(twin_views.PLANT_TZ).tz_localize(None)
    cfg = ce.config()
    try:
        keys = ce.data_keys(cfg, 0)
        fd = ce.CACHED_LOADERS.hourly(keys.hourly_path, keys.hourly_mtime_ns)
    except Exception as exc:  # noqa: BLE001 - the dataset is a file the worker refreshes
        raise ApiError(503, f"The hourly dataset is not available ({type(exc).__name__}).") from exc
    st = transit_service.state(fd, when_ist, cfg)
    q = transit_service.quantities(st)
    change = transit_service.blend_change(fd, when_ist, cfg)
    lag_view = None
    if lag:
        if store is None:
            raise ApiError(503, "The implied lag needs the twin store: set BF_TWIN_DB_URL or DATABASE_URL.")
        snapshots = store.snapshots.series(when - timedelta(days=16), when)
        res = transit_service.implied_lag(fd, snapshots, when_ist, cfg)
        lag_view = TransitLag(
            lag_h=twin_views.number(res.lag_h),
            ci_low_h=twin_views.number(res.ci[0]),
            ci_high_h=twin_views.number(res.ci[1]),
            r_peak=twin_views.number(res.r_peak),
            n_hours=int(res.n_hours),
        )
    return TwinTransit(
        at=when,
        charges_per_hour=q.get("transit_charges_per_hour"),
        tau_cz_h=q.get("tau_cz_h"),
        tau_raceway_h=q.get("tau_raceway_h"),
        tau_cz_rate_h=q.get("tau_cz_rate_h"),
        tau_raceway_rate_h=q.get("tau_raceway_rate_h"),
        coke_in_transit_t=q.get("coke_in_transit_t"),
        complete=bool(st.complete),
        missing_hours=int(st.missing_hours),
        bands=[
            {k: twin_views.number(v) if isinstance(v, float) else v for k, v in b.items()}
            for b in transit_service.bands(st)
        ],
        lag=lag_view,
        blend_change=None
        if change is None
        else BlendChangeView(
            changed_at=_ist_utc(change.changed_at),
            step_pt=change.step_pt,
            before_pct=change.before_pct,
            after_pct=change.after_pct,
            charges_since=change.charges_since,
            cz_at=_ist_utc(change.cz_at),
            raceway_at=_ist_utc(change.raceway_at),
            zone=change.zone,
        ),
    )


def _ist_utc(ts: pd.Timestamp) -> datetime:
    return pd.Timestamp(ts).round("s").tz_localize(twin_views.PLANT_TZ).tz_convert("UTC").to_pydatetime()


class HotMetalSample(BaseModel):
    at: datetime = Field(..., description="Sample time (UTC)")
    value: float = Field(..., description="HMT, °C, or Si, %")


class HotMetalSamples(BaseModel):
    start: datetime = Field(..., description="Start of the range (UTC)")
    end: datetime = Field(..., description="End of the range (UTC)")
    hmt: list[HotMetalSample] = Field(..., description="HMT per tap, °C (taps >= 1,350 °C), oldest first")
    si: list[HotMetalSample] = Field(..., description="Si per sample, % (finite, above zero), oldest first")


MAX_HOT_METAL_HOURS = 24 * 31


@router.get("/hot-metal", response_model=HotMetalSamples, summary="HMT per tap and Si per sample")
def hot_metal(
    hours: Annotated[
        float, Query(gt=0, le=MAX_HOT_METAL_HOURS, description="Range ending now, h (at most 31 days)")
    ] = 72.0,
) -> HotMetalSamples:
    """The hot-metal samples of Neon ``offline_feed.hot_metal_slag_analysis`` over the range: HMT of each tap by the
    pipeline's tap filter and Si of each sample (``services.status.hot_metal_samples``); the Live Twin's thermal
    state (SV-1.9)."""
    from services import status as status_service

    end = pd.Timestamp(status_service.clock()).tz_convert(twin_views.PLANT_TZ).tz_localize(None)
    start = end - pd.Timedelta(hours=hours)
    try:
        temps, si = status_service.hot_metal_samples(start, end + pd.Timedelta(minutes=1))
    except Exception as exc:  # noqa: BLE001 - Neon unreachable
        raise ApiError(503, f"The hot-metal samples are not available ({type(exc).__name__}).") from exc

    def rows(series: pd.Series) -> list[HotMetalSample]:
        return [HotMetalSample(at=_ist_utc(t), value=float(v)) for t, v in series.items()]

    return HotMetalSamples(start=_ist_utc(start), end=_ist_utc(end), hmt=rows(temps), si=rows(si))


@router.get("/alarms", response_model=TwinAlarms, summary="Open alarms and recent history")
def alarms(
    store: Annotated[TwinStore, Depends(twin_store)],
    days: Annotated[float, Query(gt=0, le=92, description="History window, days")] = 7.0,
    kind: Annotated[str | None, Query(description="Only this kind (e.g. omega_drift)")] = None,
) -> TwinAlarms:
    since = datetime.now(UTC) - timedelta(days=days)
    return TwinAlarms(
        open=[twin_views.alarm_data(a) for a in store.alarms.open_alarms() if kind is None or a.kind == kind],
        history=[twin_views.alarm_data(a) for a in store.alarms.history(since=since, kind=kind)],
        since=since,
    )
