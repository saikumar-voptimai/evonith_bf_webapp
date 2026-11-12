"""Live model view of the Blend & Fuel Optimiser (user decision 12 Nov 2026): the energy model against the furnace.

The operator wants to see, before trusting a plan, how the optimiser's model is doing on the furnace right now and
over the last days, and what it predicts next. This module composes that view from what the worker already stores;
a request never runs the pipeline, a balance or the network.

* ``hourly``: the twin's stored windows of the last ``hours`` (8 h ending at each full hour, ``twin.evaluate``):
  coke the blast burnt (TRACE), the energy balance (FORGE, Fixed-ω), STRATA-ω, charged coke; the window's hot-metal
  temperature, ηCO and top-gas temperature with the balance's ηCO and top-gas temperature; the HMT forecast one
  descent time ahead (``utils.coke_energy.hmt_forecast``).
* ``forecast``: the latest window's HMT forecast at t + τ_RW with its gate status and its live score over the last
  ``score_days``: the error of every stored forecast against the tap temperature at its horizon next to persistence
  (HMT at the forecast's origin), so the operator sees how good the forecast has been before the gate is run.
* ``taps``: the measured HMT per tap over the same hours (the measured line of the HMT chart).
* ``shifts``: the model check of page 11 on the FORGE view (``services.forge``, the Coke Energy pipeline's balanced
  shifts): burnt coke next to the energy- and oxygen-balance predictions per shift over the last ``days``, the
  verdict (``utils.burden_engine.model_check``: bias within ±8 and SD within ±12 kg/tHM over at least 6 stable
  shifts) and both models' statistics.

Times are UTC in the payload (the SPA formats IST). Pure composition; the loaders are arguments for the tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from services import forge, twin_views
from utils.burden_engine import model_check as mc
from utils.coke_energy import hmt_forecast
from utils.memo import memo

PLANT_TZ = twin_views.PLANT_TZ
DEFAULT_HOURS = 48
MAX_HOURS = 24 * 14
PERIOD_DAYS = (7, 14, 30)
SCORE_DAYS = 14
#: Window quantities of the view (snapshot ``quantities`` key -> view key).
QUANTITIES = {
    "hmt_c": "hmt_c",
    "hmt_now_c": "hmt_now_c",
    "hmt_forecast_c": "hmt_forecast_c",
    "hmt_forecast_horizon_h": "hmt_forecast_horizon_h",
    "tau_raceway_h": "tau_raceway_h",
    "k_arr_kg_thm": "k_arr_kg_thm",
    "eta_co_pct": "eta_co_pct",
    "eta_co_forge_pct": "eta_co_forge_pct",
    "top_gas_c": "top_gas_c",
    "top_gas_forge_c": "top_gas_forge_c",
    "hm_si_pct": "hm_si_pct",
    "production_t_h": "production_t_h",
}


# ------------------------------------------------------------------------------------------------ schemas
class LiveModelWindow(BaseModel):
    """One stored twin window (8 h ending at ``end``)."""

    end: datetime = Field(..., description="Window end, UTC")
    status: str = Field(..., description="complete | partial | failed")
    k_burnt: float | None = Field(None, description="Coke the blast burnt, TRACE, dry kg/tHM")
    k_forge: float | None = Field(None, description="Energy balance at fixed ω (FORGE), dry kg/tHM")
    k_strata: float | None = Field(None, description="Energy balance at ω_STRATA, dry kg/tHM")
    k_charged: float | None = Field(None, description="Coke charged, dry kg/tHM")
    hmt_c: float | None = Field(None, description="Hot-metal temperature of the window, °C")
    hmt_now_c: float | None = Field(None, description="Latest tap temperature before the window end, °C")
    hmt_forecast_c: float | None = Field(None, description="HMT forecast at end + horizon, °C")
    hmt_forecast_horizon_h: float | None = Field(None, description="Forecast horizon τ_RW, h")
    tau_raceway_h: float | None = Field(None, description="Descent time stockline to raceway, h")
    k_arr_kg_thm: float | None = Field(None, description="Coke of the charges of the last hour, dry kg/tHM")
    eta_co_pct: float | None = Field(None, description="ηCO, analyser, %")
    eta_co_forge_pct: float | None = Field(None, description="ηCO of the energy balance, %")
    top_gas_c: float | None = Field(None, description="Top-gas temperature, measured, °C")
    top_gas_forge_c: float | None = Field(None, description="Top-gas temperature of the energy balance, °C")
    hm_si_pct: float | None = Field(None, description="Hot-metal Si, %")
    production_t_h: float | None = Field(None, description="Production, t/h")


class HmtTap(BaseModel):
    at: datetime = Field(..., description="Tap time, UTC")
    value: float = Field(..., description="Hot-metal temperature, °C")


class ForecastScore(BaseModel):
    """The stored forecasts against the tap temperature at their horizon, last ``days``."""

    days: int = Field(..., description="Days scored")
    n: int = Field(..., description="Forecasts with a measured temperature at the horizon")
    rmse_model_c: float | None = Field(None, description="RMSE of the forecast, °C")
    rmse_persistence_c: float | None = Field(None, description="RMSE of persistence (HMT at the origin), °C")
    bias_model_c: float | None = Field(None, description="Mean forecast − measured, °C")
    ratio: float | None = Field(None, description="RMSE model / RMSE persistence (gate: at most 0.90)")
    verdict: Literal["better", "worse", "insufficient"] = Field(..., description="Against persistence")
    text: str = Field(..., description="One line for the page")


class ForecastPoint(BaseModel):
    origin: datetime = Field(..., description="The forecast's origin (its window end), UTC")
    at: datetime = Field(..., description="Instant the forecast is for, UTC")
    value: float = Field(..., description="Forecast HMT, °C")
    measured: float | None = Field(None, description="Measured HMT at the horizon, °C, once known")


class HmtForecastView(BaseModel):
    gate: Literal["pending", "passed", "failed"] = Field(..., description="Validation gate of the forecast (SV-1.4)")
    latest: ForecastPoint | None = Field(None, description="The newest window's forecast")
    points: list[ForecastPoint] = Field(default_factory=list, description="Every stored forecast of the hours shown")
    score: ForecastScore


class ShiftCheckRow(BaseModel):
    shift_key: str = Field(..., description="YYYY-MM-DD_A|B|C")
    t_mid: datetime = Field(..., description="Shift middle, UTC")
    stable: bool
    k_burnt: float | None = Field(None, description="Burnt coke, dry kg/tHM")
    k_energy: float | None = Field(None, description="Energy balance, dry kg/tHM")
    k_obal: float | None = Field(None, description="Oxygen balance at the analyser's ηCO, dry kg/tHM")
    k_charged: float | None = Field(None, description="Charged coke, dry kg/tHM")


class ModelStats(BaseModel):
    n: int = Field(..., description="Stable shifts scored")
    bias: float | None = Field(None, description="Mean model − burnt, kg/tHM")
    sd: float | None = Field(None, description="SD of model − burnt, kg/tHM")
    mae: float | None = Field(None, description="Mean absolute error, kg/tHM")


class ShiftCheck(BaseModel):
    """Page 11's model check: the record of the energy model over the period."""

    days: int = Field(..., description="Period, days")
    state: Literal["ok", "watch", "insufficient"] = Field(..., description="On track, check, or too few shifts")
    reason: str
    energy: ModelStats
    oxygen: ModelStats
    thresholds: dict[str, float] = Field(default_factory=dict, description="bias_kgthm, sd_kgthm, min_shifts")
    latest_shift: str | None = Field(None, description="Latest scored shift")
    age_days: int | None = Field(None, description="Days since the latest scored shift")
    rows: list[ShiftCheckRow]
    built_at: datetime | None = Field(None, description="When the FORGE view was built, UTC")


class LiveModelView(BaseModel):
    now: datetime = Field(..., description="Time of the request, UTC")
    hours: int = Field(..., description="Hours of hourly windows shown")
    latest: LiveModelWindow | None = Field(None, description="The newest stored window")
    hourly: list[LiveModelWindow]
    taps: list[HmtTap] = Field(default_factory=list, description="Measured HMT per tap over the hours")
    forecast: HmtForecastView
    shifts: ShiftCheck | None = Field(None, description="None until the FORGE view is built")
    notes: list[str] = Field(default_factory=list, description="What is missing and why")


# ------------------------------------------------------------------------------------------------ builders
def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def window_rows(snapshots: Sequence[Any]) -> list[dict[str, Any]]:
    """The stored windows as view rows, oldest first."""
    out = []
    for r in twin_views.series_rows(snapshots, resolution="hour", quantities=list(QUANTITIES)):
        q = r["quantities"]
        out.append(
            {
                "end": r["window_end"],
                "status": r["status"],
                "k_burnt": r["k_trace"],
                "k_forge": r["k_forge"],
                "k_strata": r["k_strata"],
                "k_charged": r["k_charged"],
                **{view: q.get(key) for key, view in QUANTITIES.items()},
            }
        )
    return out


def forecast_points(rows: Sequence[Mapping[str, Any]], taps: pd.Series) -> list[dict[str, Any]]:
    """Every row's forecast as (origin, at, value, measured at the horizon when the taps reach it)."""
    out = []
    for r in rows:
        value, horizon = _num(r.get("hmt_forecast_c")), _num(r.get("hmt_forecast_horizon_h"))
        if value is None or horizon is None:
            continue
        origin = pd.Timestamp(r["end"])
        at = origin + pd.Timedelta(hours=horizon)
        measured = hmt_forecast.target(taps, _ist_naive(at)) if len(taps) else None
        out.append({"origin": origin.to_pydatetime(), "at": at.to_pydatetime(), "value": value, "measured": measured})
    return out


def score(points: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], days: int) -> dict[str, Any]:
    """RMSE and bias of the forecasts with a measured temperature, next to persistence (``hmt_now_c`` of the origin)."""
    now_by_origin = {pd.Timestamp(r["end"]): _num(r.get("hmt_now_c")) for r in rows}
    model, persist = [], []
    for p in points:
        if p["measured"] is None:
            continue
        base = now_by_origin.get(pd.Timestamp(p["origin"]))
        if base is None:
            continue
        model.append(p["value"] - p["measured"])
        persist.append(base - p["measured"])
    n = len(model)
    if n < 3:
        return {
            "days": days, "n": n, "rmse_model_c": None, "rmse_persistence_c": None, "bias_model_c": None,
            "ratio": None, "verdict": "insufficient",
            "text": f"{n} forecasts with a measured temperature at the horizon in the last {days} days; "
                    "at least 3 are needed for a score.",
        }  # fmt: skip
    m = pd.Series(model)
    p = pd.Series(persist)
    rmse_m = float(math.sqrt((m**2).mean()))
    rmse_p = float(math.sqrt((p**2).mean()))
    bias = float(m.mean())
    ratio = rmse_m / rmse_p if rmse_p > 0 else None
    better = ratio is not None and ratio < 1.0
    text = (
        f"Over {n} forecasts in the last {days} days the forecast missed by ±{rmse_m:.0f} °C (bias {bias:+.0f}), "
        f"persistence by ±{rmse_p:.0f} °C: "
        + (f"{100 * (1 - ratio):.0f} % better than persistence." if better and ratio is not None
           else "not better than holding the present temperature.")
    )  # fmt: skip
    return {
        "days": days,
        "n": n,
        "rmse_model_c": rmse_m,
        "rmse_persistence_c": rmse_p,
        "bias_model_c": bias,
        "ratio": ratio,
        "verdict": "better" if better else "worse",
        "text": text,
    }


def shift_check(
    shifts: pd.DataFrame | None, days: int, *, thresholds: mc.Thresholds, today: pd.Timestamp
) -> dict[str, Any]:
    """Page 11's model check over the last ``days`` of the FORGE view's shifts."""
    scored = mc.scored_shifts(shifts)
    latest = mc.latest(scored)
    frame = mc.window(scored, days)
    e = mc.summary(frame, "K_energy")
    o = mc.summary(frame, "K_obal")
    state, reason = mc.verdict(e, thresholds)

    def stats(a: Mapping[str, float]) -> dict[str, Any]:
        return {
            "n": int(a.get("n") or 0),
            "bias": _num(a.get("bias")),
            "sd": _num(a.get("sd")),
            "mae": _num(a.get("mae")),
        }

    rows = []
    for skey, r in frame.iterrows():
        rows.append(
            {
                "shift_key": str(skey),
                "t_mid": pd.Timestamp(r["t_mid"]).tz_localize(PLANT_TZ).tz_convert("UTC").to_pydatetime(),
                "stable": bool(r.get("stable", False)),
                "k_burnt": _num(r.get("K_burnt")),
                "k_energy": _num(r.get("K_energy")),
                "k_obal": _num(r.get("K_obal")),
                "k_charged": _num(r.get("K_charged")),
            }
        )
    return {
        "days": days,
        "state": state,
        "reason": reason,
        "energy": stats(e),
        "oxygen": stats(o),
        "thresholds": {
            "bias_kgthm": thresholds.bias_kgthm,
            "sd_kgthm": thresholds.sd_kgthm,
            "min_shifts": float(thresholds.min_shifts),
        },
        "latest_shift": None if latest is None else str(latest.name),
        "age_days": mc.age_days(latest, today=today),
        "rows": rows,
    }


def _ist_naive(t: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return (t.tz_convert(PLANT_TZ) if t.tzinfo else t.tz_localize("UTC").tz_convert(PLANT_TZ)).tz_localize(None)


def _taps_utc(temps: pd.Series) -> pd.Series:
    """Tap temperatures indexed by naive IST (as the pipeline), from the status service's IST index."""
    if temps is None or temps.empty:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(temps.index)
    if idx.tz is not None:
        idx = idx.tz_convert(PLANT_TZ).tz_localize(None)
    return pd.Series(temps.to_numpy(dtype=float), index=idx).sort_index()


@memo(ttl_seconds=300, max_entries=8)
def _taps(start_key: str, end_key: str) -> pd.Series:
    """HMT per tap between two IST instants (memoised: the taps are read from Neon once per 5 min per range)."""
    from services import status as status_service

    temps, _ = status_service.hot_metal_samples(pd.Timestamp(start_key), pd.Timestamp(end_key))
    return _taps_utc(temps)


def view(
    store: Any,
    *,
    hours: int = DEFAULT_HOURS,
    days: int = SCORE_DAYS,
    now: datetime | None = None,
    cfg: Mapping[str, Any] | None = None,
    taps: Callable[[pd.Timestamp, pd.Timestamp], pd.Series] | None = None,
    shifts: pd.DataFrame | None | Literal["load"] = "load",
    built_at: datetime | None = None,
    gate: str | None = None,
) -> dict[str, Any]:
    """The fields of ``LiveModelView``. ``taps(start_ist, end_ist)`` and ``shifts`` are the loaders (tests)."""
    from services import twin as twin_service

    now = now or datetime.now(UTC)
    notes: list[str] = []
    score_days = SCORE_DAYS
    hourly = window_rows(store.snapshots.series(now - timedelta(hours=hours), now))
    scoring = hourly
    if hours < score_days * 24:
        scoring = window_rows(store.snapshots.series(now - timedelta(days=score_days), now))
    latest = hourly[-1] if hourly else None

    # The taps: the hours shown and, for the score, the forecasts' horizons of the last score_days.
    horizon_max = max((_num(r.get("hmt_forecast_horizon_h")) or 0.0 for r in scoring), default=0.0)
    start = _ist_naive(pd.Timestamp(now) - pd.Timedelta(hours=max(hours, score_days * 24)))
    end = _ist_naive(pd.Timestamp(now) + pd.Timedelta(hours=horizon_max + 1))
    temps = pd.Series(dtype=float)
    try:
        temps = (taps or (lambda a, b: _taps(a.isoformat(), b.isoformat())))(start, end)
        temps = _taps_utc(temps)
    except Exception as exc:  # noqa: BLE001 - Neon unreachable: the view stands without the taps
        notes.append(f"The tap temperatures are not available ({type(exc).__name__}): no measured HMT and no score.")
    shown_from = _ist_naive(pd.Timestamp(now) - pd.Timedelta(hours=hours))
    shown = temps[temps.index >= shown_from] if len(temps) else temps
    tap_rows = [
        {"at": t.tz_localize(PLANT_TZ).tz_convert("UTC").to_pydatetime(), "value": float(v)} for t, v in shown.items()
    ]

    points = forecast_points(hourly, temps)
    latest_point = None
    if latest is not None and points and pd.Timestamp(points[-1]["origin"]) == pd.Timestamp(latest["end"]):
        latest_point = points[-1]
    if gate is None:
        from services import twin_reads

        gate = twin_service.hmt_gate_status(twin_reads.config())
    forecast_view = {
        "gate": gate,
        "latest": latest_point,
        "points": points,
        "score": score(forecast_points(scoring, temps), scoring, score_days),
    }
    if not hourly:
        notes.append("No twin window is stored for these hours: the worker's twin.evaluate job writes one per hour.")

    # Page 11's model check on the FORGE view.
    check = None
    if isinstance(shifts, str):
        try:
            stored = forge.load()
            shifts, built_at = stored.shifts, stored.built_at
        except forge.ViewUnavailable as exc:
            shifts = None
            notes.append(str(exc))
    if shifts is not None:
        th = mc.Thresholds.from_config(_model_check_config(cfg))
        today = pd.Timestamp(now).tz_convert(PLANT_TZ).tz_localize(None).normalize()
        check = shift_check(shifts, days, thresholds=th, today=today) | {"built_at": built_at}
    return {
        "now": now,
        "hours": hours,
        "latest": latest,
        "hourly": hourly,
        "taps": tap_rows,
        "forecast": forecast_view,
        "shifts": check,
        "notes": notes,
    }


def _model_check_config(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if cfg is not None and "model_check" in cfg:
        return dict(cfg["model_check"] or {})
    try:
        from services.optimiser import config_mtime, page_config

        return dict((page_config(config_mtime("burden_optimiser.yml")) or {}).get("model_check") or {})
    except Exception:  # noqa: BLE001 - the defaults stand without the page config
        return {}
