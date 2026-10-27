"""Live twin: latest evaluation and its freshness, hourly and per-shift series, burden transit, alarms (SV-1.6, #286).

Snapshots and alarms come from the twin store (``twin.snapshot``, ``twin.alarm_event``; written by ``twin.evaluate``
and the drift alarm); the burden column is computed from the hourly dataset at the requested time. Coke rates are
kg/tHM dry, shaft efficiencies dimensionless, times UTC unless stated.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.auth import require_role
from api.errors import ApiError
from api.store import optional_twin_store, twin_store
from data.twin.repositories import TwinStore
from services import twin_views
from services.events import TwinSnapshotData

router = APIRouter(prefix="/api/twin", tags=["twin"], dependencies=[Depends(require_role("user"))])

MAX_SERIES_DAYS = 92


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
def now(store: Annotated[TwinStore, Depends(twin_store)]) -> TwinNow:
    """The latest stored evaluation, how far it lags the expected window and the number of open alarms."""
    latest = store.snapshots.latest()
    fresh = twin_views.freshness(latest)
    return TwinNow(
        snapshot=TwinSnapshotData.model_validate(twin_views.snapshot_data(latest)) if latest is not None else None,
        alarms_open=len(store.alarms.open_alarms()),
        **fresh,
    )


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
    )


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
