"""Burden transit of the live twin (SV-1.3, #283) for the API and the worker.

``state(fd, at)``: the burden column at ``at`` (IST): descent times, charges and coke by zone and the ledger of the
charges in the column, from the hourly furnace dataset that ``services.coke_energy`` loads (``utils.coke_energy.
burden_column``). ``quantities(state)``: its scalar quantities for the twin snapshot; ``bands(state)``: the ledger as
JSON rows. ``implied_lag(fd, snapshots, end)``: the lag between the coke content of the charges and K_TRACE of the
stored hourly windows over the days before ``end``. ``blend_change(fd, at)``: the latest change of the ferrous blend
and when its first charge reaches the cohesive zone and the raceways (SV-1.9). ``view(at, lag=, store=)``: all of it
at a time, composed once for GET /api/twin/transit and FurnaceMind's ``get_transit`` (SV-5.3). Settings:
``config/coke_energy.yml -> twin.transit``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from services import twin_views
from utils.coke_energy import burden_column as bc

PLANT_TZ = "Asia/Kolkata"


@dataclass(frozen=True)
class LagSettings:
    days: float = 14.0
    lags_h: tuple[float, ...] = tuple(float(h) for h in range(13))
    window_h: int = 8
    detrend_days: float = 7.0
    block_h: int = 24
    n_boot: int = 500
    basis: str = "ferrous"


def settings(cfg: Mapping[str, Any] | None) -> tuple[bc.TransitReference, LagSettings]:
    """Reference and implied-lag settings from the ``twin.transit`` block of ``config/coke_energy.yml``."""
    block = dict(((cfg or {}).get("twin") or {}).get("transit") or {})
    lag = dict(block.get("implied_lag") or {})
    base = LagSettings()
    lags = lag.get("lags_h")
    return bc.TransitReference.from_config(block.get("reference")), LagSettings(
        days=float(lag.get("days", base.days)),
        lags_h=tuple(float(h) for h in lags) if lags else base.lags_h,
        window_h=int(lag.get("window_h", base.window_h)),
        detrend_days=float(lag.get("detrend_days", base.detrend_days)),
        block_h=int(lag.get("block_h", base.block_h)),
        n_boot=int(lag.get("n_boot", base.n_boot)),
        basis=str(lag.get("basis", base.basis)),
    )


def state(fd: pd.DataFrame, at: Any, cfg: Mapping[str, Any] | None = None) -> bc.ColumnState:
    ref, _ = settings(cfg)
    return bc.column(fd, at, ref)


def quantities(st: bc.ColumnState) -> dict[str, float]:
    """Scalar transit quantities of a column state (``burden_column.quantities``, as in the twin snapshot)."""
    return bc.quantities(st)


def bands(st: bc.ColumnState) -> list[dict[str, Any]]:
    """The ledger as JSON rows, newest band first."""
    return [
        {
            "zone": b.zone,
            "charges": b.charges,
            "newest_h": b.newest_h,
            "oldest_h": b.oldest_h,
            "coke_t_per_charge": b.coke_t_per_charge,
            "coke_kg_per_t_ferrous": b.coke_kg_per_t_ferrous,
            "coke_kg_thm": b.coke_kg_thm,
            "coke_volume_fraction": b.coke_volume_fraction,
            "coke_inventory_t": b.coke_inventory_t,
            "blend_wet_pct": dict(b.blend_wet_pct),
            "tonnes": {k: v for k, v in b.tonnes.items() if v},
        }
        for b in st.bands
    ]


def blend_change(fd: pd.DataFrame, at: Any, cfg: Mapping[str, Any] | None = None) -> bc.BlendChange | None:
    """The latest change of the ferrous blend before ``at`` (IST) and its arrival at the cohesive zone and the raceways
    (``burden_column.blend_change``; threshold and span from ``twin.transit.blend_change``)."""
    ref, _ = settings(cfg)
    block = dict((((cfg or {}).get("twin") or {}).get("transit") or {}).get("blend_change") or {})
    return bc.blend_change(
        fd, at, ref, threshold_pt=float(block.get("threshold_pt", 5.0)), span_h=int(block.get("span_h", 2))
    )


def burnt_series(snapshots: Iterable[Any]) -> pd.Series:
    """K_TRACE of stored windows by window end, as IST wall-clock times (the hourly dataset's time base)."""
    t, k = [], []
    for row in snapshots:
        get = row.get if isinstance(row, Mapping) else (lambda key, row=row: getattr(row, key, None))
        if get("k_trace") is None:
            continue
        ts = pd.Timestamp(get("window_end"))
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        t.append(ts.tz_convert(PLANT_TZ).tz_localize(None))
        k.append(float(get("k_trace")))
    s = pd.Series(k, index=pd.DatetimeIndex(t), dtype=float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def implied_lag(
    fd: pd.DataFrame, snapshots: Iterable[Any], end: Any, cfg: Mapping[str, Any] | None = None
) -> bc.LagResult:
    """Implied transit time from the last ``days`` before ``end`` (IST): the charges' coke content (hourly dataset)
    against K_TRACE of the stored windows."""
    _, lag = settings(cfg)
    end = pd.Timestamp(end)
    start = end - pd.Timedelta(days=lag.days)
    burnt = burnt_series(snapshots)
    burnt = burnt[(burnt.index > start) & (burnt.index <= end)]
    charged = bc.charged_coke(fd, lag.basis)
    charged = charged[
        (charged.index >= start - pd.Timedelta(hours=max(lag.lags_h) + lag.window_h)) & (charged.index < end)
    ]
    return bc.implied_lag(
        charged,
        burnt,
        lags_h=lag.lags_h,
        window_h=lag.window_h,
        detrend_days=lag.detrend_days,
        block_h=lag.block_h,
        n_boot=lag.n_boot,
    )


class Unavailable(RuntimeError):
    """The hourly dataset (or, for the implied lag, the twin store) is not available."""


def view(
    at: datetime | None = None,
    *,
    lag: bool = False,
    store: Any | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The fields of the API's ``TwinTransit`` at ``at`` (UTC; naive is UTC; default now): descent times, coke in
    transit, the ledger by zone, the latest blend change and, with ``lag``, the implied lag against the stored K_TRACE
    of the 16 days before (``store``, a ``TwinStore``). Times UTC. Raises ``Unavailable``."""
    from services import coke_energy as ce

    when = datetime.now(UTC) if at is None else at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
    when_ist = pd.Timestamp(when).tz_convert(PLANT_TZ).tz_localize(None)
    cfg = ce.config() if cfg is None else cfg
    try:
        keys = ce.data_keys(cfg, 0)
        fd = ce.CACHED_LOADERS.hourly(keys.hourly_path, keys.hourly_mtime_ns)
    except Exception as exc:  # noqa: BLE001 - the dataset is a file the worker refreshes
        raise Unavailable(f"The hourly dataset is not available ({type(exc).__name__}).") from exc
    st = state(fd, when_ist, cfg)
    q = quantities(st)
    change = blend_change(fd, when_ist, cfg)
    lag_view = None
    if lag:
        if store is None:
            raise Unavailable("The implied lag needs the twin store: set BF_TWIN_DB_URL or DATABASE_URL.")
        snapshots = store.snapshots.series(when - timedelta(days=16), when)
        res = implied_lag(fd, snapshots, when_ist, cfg)
        lag_view = dict(
            lag_h=twin_views.number(res.lag_h),
            ci_low_h=twin_views.number(res.ci[0]),
            ci_high_h=twin_views.number(res.ci[1]),
            r_peak=twin_views.number(res.r_peak),
            n_hours=int(res.n_hours),
        )
    return dict(
        at=when,
        charges_per_hour=q.get("transit_charges_per_hour"),
        tau_cz_h=q.get("tau_cz_h"),
        tau_raceway_h=q.get("tau_raceway_h"),
        tau_cz_rate_h=q.get("tau_cz_rate_h"),
        tau_raceway_rate_h=q.get("tau_raceway_rate_h"),
        coke_in_transit_t=q.get("coke_in_transit_t"),
        complete=bool(st.complete),
        missing_hours=int(st.missing_hours),
        bands=[{k: twin_views.number(v) if isinstance(v, float) else v for k, v in b.items()} for b in bands(st)],
        lag=lag_view,
        blend_change=None
        if change is None
        else dict(
            changed_at=twin_views.ist_utc(change.changed_at),
            step_pt=change.step_pt,
            before_pct=change.before_pct,
            after_pct=change.after_pct,
            charges_since=change.charges_since,
            cz_at=twin_views.ist_utc(change.cz_at),
            raceway_at=twin_views.ist_utc(change.raceway_at),
            zone=change.zone,
        ),
    )
