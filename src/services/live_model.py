"""Live model view of the Blend & Fuel Optimiser (user decisions 12 Nov 2026): the energy model against the furnace.

The operator wants to see, before trusting a plan, how the optimiser's model is doing on the furnace right now and
over the last days, and what the hot-metal temperature will do next. This module composes that view from what the
worker stores and from three reads (the status snapshot, the per-charge record, the taps); a request never runs the
pipeline, a balance or the network.

* ``clock``: how old each data stream is. Process tags arrive every 10 s; the charge record is published once an
  hour and the taps after each cast, so every per-tonne figure follows the charge record. The notice states it.
* ``process``: fuel and blast now from the status snapshot (PCI in kg/tHM and t/h, hot blast, oxygen, RAFT, ηCO,
  top-gas temperature).
* ``coke``: charged coke of the last charges (``utils.coke_energy.charge_record``), the coke the blast burnt and the
  energy balance over the newest 8-hour window, model − burnt against the model check's bias limit.
* ``hmt``: the tap filter (``utils.coke_energy.hmt_filter``): estimate now, the next tap with the chance of a cold
  tap, the temperature ``horizon_h`` ahead, a band that widens with the lead, its score against the last tap held
  over the last ``SCORE_DAYS`` (every tap against the forecast made h hours before it with the taps published by
  then) and a check of the newest tap against the band.
* ``hourly``: the twin's stored windows of the last ``hours`` (8 h ending at each full hour, ``twin.evaluate``):
  burnt coke (TRACE), the energy balance (FORGE, Fixed-ω), STRATA-ω, charged coke, ηCO and top-gas temperature
  measured and of the balance. A window whose PCI lies outside the valid range carries no coke rates.
* ``shifts``: the model check of page 11 on the FORGE view (``services.forge``): burnt coke next to the energy- and
  oxygen-balance predictions per shift over the last ``days``, the verdict (``utils.burden_engine.model_check``)
  and both models' statistics.

Times are UTC in the payload (the SPA formats IST). Pure composition; the loaders are arguments for the tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from services import forge, twin_views
from utils.burden_engine import model_check as mc
from utils.coke_energy import charge_record, hmt_filter, hmt_forecast
from utils.memo import memo

PLANT_TZ = twin_views.PLANT_TZ
DEFAULT_HOURS = 48
MAX_HOURS = 24 * 14
PERIOD_DAYS = (7, 14, 30)
SCORE_DAYS = 14
#: Window quantities of the view (snapshot ``quantities`` key -> view key).
QUANTITIES = {
    "hmt_c": "hmt_c",
    "tau_cz_h": "tau_cz_h",
    "tau_raceway_h": "tau_raceway_h",
    "eta_co_pct": "eta_co_pct",
    "eta_co_forge_pct": "eta_co_forge_pct",
    "top_gas_c": "top_gas_c",
    "top_gas_forge_c": "top_gas_forge_c",
    "hm_si_pct": "hm_si_pct",
    "production_t_h": "production_t_h",
    "pci_kg_thm": "pci_kg_thm",
    "hot_blast_c": "hot_blast_c",
    "oxygen_nm3_thm": "oxygen_nm3_thm",
}
#: Status-catalogue parameters of "fuel and blast now" (``config/furnace_status.yml`` key -> label, unit, decimals).
PROCESS_TAGS = {
    "pci_rate": ("PCI rate", "kg/tHM", 0),
    "production_rate": ("Production", "t/h", 1),
    "hbt": ("Hot blast temperature", "°C", 0),
    "hot_blast_volume": ("Hot blast volume", "Nm³/h", 0),
    "oxygen_flow": ("Oxygen flow", "Nm³/h", 0),
    "o2_injection": ("Oxygen enrichment", "%", 2),
    "raft": ("RAFT", "°C", 0),
    "co_utilization": ("ηCO", "%", 1),
    "uptake_avg": ("Top-gas temperature", "°C", 0),
}
COKE_COLUMNS = ("k_burnt", "k_forge", "k_strata", "k_charged")
#: Cold-tap limit when neither the config nor the fact register gives one, °C (BAND-02 of register v1.8).
COLD_TAP_FALLBACK_C = 1440.0


@dataclass(frozen=True)
class Settings:
    """``config/coke_energy.yml -> twin.live_model``."""

    horizon_h: float = 6.0  # the forecast band reaches this far
    band_step_min: int = 30
    cold_tap_c: float | None = None  # a tap below it is cold; None: the lower edge of BAND-02 (fact register)
    score_horizons_h: tuple[float, ...] = (2.0, 6.0)
    score_min_taps: int = 10
    burn_in_days: int = 10  # taps before the scored period, for the filter to settle
    default_tap_delay_h: float = 1.0  # a tap without publication time counts as published this much later
    z_single: float = 2.0
    z_pair: float = 1.0
    stale_after_h: float = 12.0  # without a tap for this long the estimate is the level alone (three time constants)
    pci_valid_kg_thm: tuple[float, float] = (80.0, 320.0)
    charge_days: int = 4  # per-charge record read (72 hourly windows and their charges)
    filter: hmt_filter.Params = field(default_factory=hmt_filter.Params)
    charged: charge_record.Settings = field(default_factory=charge_record.Settings)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> Settings:
        block = dict((((cfg or {}).get("twin") or {}).get("live_model")) or {})
        hmt, charged, base = dict(block.get("hmt") or {}), dict(block.get("charged_now") or {}), cls()
        lo, hi = block.get("pci_valid_kg_thm", base.pci_valid_kg_thm)
        return cls(
            horizon_h=float(hmt.get("horizon_h", base.horizon_h)),
            band_step_min=int(hmt.get("band_step_min", base.band_step_min)),
            cold_tap_c=None if hmt.get("cold_tap_c") is None else float(hmt["cold_tap_c"]),
            score_horizons_h=tuple(float(h) for h in hmt.get("score_horizons_h", base.score_horizons_h)),
            score_min_taps=int(hmt.get("score_min_taps", base.score_min_taps)),
            burn_in_days=int(hmt.get("burn_in_days", base.burn_in_days)),
            default_tap_delay_h=float(hmt.get("default_tap_delay_h", base.default_tap_delay_h)),
            z_single=float(hmt.get("z_single", base.z_single)),
            z_pair=float(hmt.get("z_pair", base.z_pair)),
            stale_after_h=float(hmt.get("stale_after_h", base.stale_after_h)),
            pci_valid_kg_thm=(float(lo), float(hi)),
            charge_days=int(charged.get("read_days", base.charge_days)),
            filter=hmt_filter.Params.from_config(hmt.get("filter")),
            charged=charge_record.Settings.from_config(charged),
        )


# ------------------------------------------------------------------------------------------------ schemas
class LiveModelWindow(BaseModel):
    """One stored twin window (8 h ending at ``end``)."""

    end: datetime = Field(..., description="Window end, UTC")
    status: str = Field(..., description="complete | partial | failed")
    valid: bool = Field(True, description="False when the window's PCI is outside the valid range: no coke rates")
    k_burnt: float | None = Field(None, description="Coke the blast burnt, TRACE, dry kg/tHM")
    k_forge: float | None = Field(None, description="Energy balance at fixed ω (FORGE), dry kg/tHM")
    k_strata: float | None = Field(None, description="Energy balance at ω_STRATA, dry kg/tHM")
    k_charged: float | None = Field(None, description="Coke charged, dry kg/tHM")
    hmt_c: float | None = Field(None, description="Hot-metal temperature of the window, °C")
    tau_cz_h: float | None = Field(None, description="Descent time stockline to cohesive zone, h")
    tau_raceway_h: float | None = Field(None, description="Descent time stockline to raceway, h")
    eta_co_pct: float | None = Field(None, description="ηCO, analyser, %")
    eta_co_forge_pct: float | None = Field(None, description="ηCO of the energy balance, %")
    top_gas_c: float | None = Field(None, description="Top-gas temperature, measured, °C")
    top_gas_forge_c: float | None = Field(None, description="Top-gas temperature of the energy balance, °C")
    hm_si_pct: float | None = Field(None, description="Hot-metal Si, %")
    production_t_h: float | None = Field(None, description="Production, t/h")
    pci_kg_thm: float | None = Field(None, description="PCI of the window, dry kg/tHM")
    hot_blast_c: float | None = Field(None, description="Hot blast temperature of the window, °C")
    oxygen_nm3_thm: float | None = Field(None, description="Oxygen of the window, Nm³/tHM")


class HmtTap(BaseModel):
    at: datetime = Field(..., description="Tap time, UTC")
    value: float = Field(..., description="Hot-metal temperature, °C")
    received: datetime | None = Field(None, description="When the tap was published, UTC")


class ClockSource(BaseModel):
    """One data stream of the model with the age of its newest record."""

    key: Literal["process", "charges", "taps"]
    label: str
    cadence: str = Field(..., description="How often the stream is published")
    newest: datetime | None = Field(None, description="Time of the newest record, UTC")
    received: datetime | None = Field(None, description="When the newest record was published, UTC")
    age_min: float | None = Field(None, description="Now minus the newest record, min")
    delay_min: float | None = Field(None, description="Median publication delay of the stream's recent records, min")


class DataClock(BaseModel):
    sources: list[ClockSource]
    notice: str = Field(..., description="What the delay of the charge record means for the per-tonne figures")


class LiveReading(BaseModel):
    key: str
    label: str
    value: float | None = Field(None, description="Latest reading, in unit")
    unit: str
    digits: int = Field(..., description="Decimals shown, count")
    at: datetime | None = Field(None, description="Time of the reading, UTC")
    setpoint: float | None = Field(None, description="Setpoint, in unit, when the tag has one")


class ProcessNow(BaseModel):
    """Fuel and blast now, from the furnace-status snapshot (process tags, every 10 s)."""

    at: datetime | None = Field(None, description="Newest reading of the snapshot, UTC")
    telemetry: str = Field("offline", description="live | partial | offline (furnace-status telemetry)")
    readings: list[LiveReading] = Field(default_factory=list)


class CokeBalance(BaseModel):
    """Coke per tonne of hot metal, dry kg/tHM: charged now, burnt and modelled over the newest 8-hour window."""

    charged_now: float | None = Field(None, description="Coke of the last charges on the pipeline's basis, kg/tHM")
    charged_charges: int = Field(0, description="Charges of the ratio, count")
    charged_first: datetime | None = Field(None, description="First charge of the ratio, UTC")
    charged_last: datetime | None = Field(None, description="Last charge of the ratio, UTC")
    charged_received: datetime | None = Field(None, description="Publication of the newest charge, UTC")
    charged_basis: Literal["windows", "default", "none"] = Field(
        "none", description="windows: factor from the stored windows; default: the configured factor"
    )
    charged_factor: float | None = Field(None, description="K_charged per kg of coke per t of ferrous burden")
    window_end: datetime | None = Field(None, description="End of the 8-hour window of burnt and model, UTC")
    charged_window: float | None = Field(None, description="Charged coke of that window, kg/tHM")
    burnt: float | None = Field(None, description="Coke the blast burnt over the window, kg/tHM")
    model: float | None = Field(None, description="Energy balance over the window, kg/tHM")
    gap: float | None = Field(None, description="Model minus burnt, kg/tHM")
    limit: float = Field(..., description="Limit of |model − burnt| (the model check's bias limit), kg/tHM")
    state: Literal["ok", "watch", "unknown"]
    text: str


class HmtEstimate(BaseModel):
    at: datetime = Field(..., description="Instant of the estimate, UTC")
    value: float = Field(..., description="Hot-metal temperature, °C")
    sd: float | None = Field(None, description="1σ of the estimate, °C")


class HmtBandPoint(BaseModel):
    at: datetime = Field(..., description="Instant, UTC")
    value: float = Field(..., description="Forecast mean, °C")
    lo: float = Field(..., description="Mean − 1σ of a tap, °C")
    hi: float = Field(..., description="Mean + 1σ of a tap, °C")


class HmtScore(BaseModel):
    """The filter's forecast against the taps, next to the last tap held, over ``days``."""

    horizon_h: float = Field(..., description="Lead of the forecast, h")
    days: int
    n: int = Field(..., description="Taps scored")
    rmse_model_c: float | None = Field(None, description="RMSE of the forecast, °C")
    rmse_persistence_c: float | None = Field(None, description="RMSE of the last published tap held, °C")
    bias_model_c: float | None = Field(None, description="Mean forecast − measured, °C")
    ratio: float | None = Field(None, description="RMSE forecast / RMSE last tap held")
    within_1sd: float | None = Field(None, description="Share of the taps inside ±1σ, fraction")
    within_2sd: float | None = Field(None, description="Share of the taps inside ±2σ, fraction")
    verdict: Literal["better", "worse", "insufficient"]
    text: str


class HmtCheck(BaseModel):
    """The newest tap against what the filter expected before it."""

    state: Literal["ok", "watch", "unknown"]
    rule: Literal["single", "pair"] | None = Field(None, description="single: beyond 2σ; pair: two beyond 1σ")
    z: float | None = Field(None, description="(tap − expected) / σ")
    expected: float | None = Field(None, description="Expected tap temperature, °C")
    sd: float | None = Field(None, description="1σ of the expectation, °C")
    value: float | None = Field(None, description="The tap, °C")
    text: str


class HmtBand(BaseModel):
    lo: float = Field(..., description="Lower edge, °C")
    hi: float = Field(..., description="Upper edge, °C")
    fact: str = Field(..., description="Fact of the process fact register")


class HmtView(BaseModel):
    """Hot-metal temperature: taps, the filter's estimate and its forecast band."""

    available: bool = Field(..., description="False with fewer taps than the filter needs")
    taps_used: int = Field(0, description="Taps the filter ran over, count")
    last_tap: HmtTap | None = None
    now: HmtEstimate | None = Field(None, description="Thermal state now (σ of the state)")
    next_tap: HmtEstimate | None = Field(None, description="The next tap, at its expected time (σ of a tap)")
    cold_tap_c: float = Field(
        ..., description="A tap below it is cold, °C: the lower edge of BAND-02 unless configured"
    )
    p_cold_next: float | None = Field(None, description="Probability of the next tap below cold_tap_c, fraction")
    ahead: HmtEstimate | None = Field(None, description="A tap horizon_h from now (σ of a tap)")
    horizon_h: float
    tap_interval_h: float | None = Field(None, description="Median interval of the recent taps, h")
    tap_age_h: float | None = Field(None, description="Now minus the newest tap, h")
    stale: bool = Field(False, description="No tap within stale_after_h: the estimate is the level alone")
    estimate: list[HmtEstimate] = Field(default_factory=list, description="Estimate after each tap of the hours")
    band: list[HmtBandPoint] = Field(default_factory=list, description="Forecast from the newest tap to the horizon")
    scores: list[HmtScore] = Field(default_factory=list)
    check: HmtCheck
    plant_band: HmtBand | None = Field(None, description="Plausibility band of the HMT (BAND-02)")


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
    clock: DataClock
    process: ProcessNow
    coke: CokeBalance
    hmt: HmtView
    latest: LiveModelWindow | None = Field(None, description="The newest stored window")
    hourly: list[LiveModelWindow]
    taps: list[HmtTap] = Field(default_factory=list, description="Measured HMT per tap over the hours")
    shifts: ShiftCheck | None = Field(None, description="None until the FORGE view is built")
    notes: list[str] = Field(default_factory=list, description="What is missing and why")


# ------------------------------------------------------------------------------------------------ builders
def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _ist_naive(t: Any) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return (t.tz_convert(PLANT_TZ) if t.tzinfo else t.tz_localize("UTC").tz_convert(PLANT_TZ)).tz_localize(None)


def _utc(t: Any) -> datetime | None:
    """A tz-naive IST instant as UTC; None for NaT."""
    if t is None or pd.isna(t):
        return None
    return pd.Timestamp(t).floor("ms").tz_localize(PLANT_TZ).tz_convert("UTC").to_pydatetime()


def _signed(value: float, digits: int = 1) -> str:
    """A signed number with a true minus sign, for the texts of the view."""
    return f"{value:+.{digits}f}".replace("-", "−")


def _minutes(later: Any, earlier: Any) -> float | None:
    if later is None or earlier is None or pd.isna(later) or pd.isna(earlier):
        return None
    return float((pd.Timestamp(later) - pd.Timestamp(earlier)) / pd.Timedelta(minutes=1))


def window_rows(snapshots: Sequence[Any], pci_valid: tuple[float, float] | None = None) -> list[dict[str, Any]]:
    """The stored windows as view rows, oldest first. A window whose PCI lies outside ``pci_valid`` (a stuck or
    missing PCI tag) is marked not valid and carries no coke rates: its balances were solved at a wrong fuel input."""
    out = []
    for r in twin_views.series_rows(snapshots, resolution="hour", quantities=list(QUANTITIES)):
        q = r["quantities"]
        row = {
            "end": r["window_end"],
            "status": r["status"],
            "valid": True,
            "k_burnt": r["k_trace"],
            "k_forge": r["k_forge"],
            "k_strata": r["k_strata"],
            "k_charged": r["k_charged"],
            **{view: q.get(key) for key, view in QUANTITIES.items()},
        }
        pci = _num(row.get("pci_kg_thm"))
        if pci_valid is not None and pci is not None and not pci_valid[0] <= pci <= pci_valid[1]:
            row.update({"valid": False, **dict.fromkeys(COKE_COLUMNS)})
        out.append(row)
    return out


def tap_table(taps: Any) -> pd.DataFrame:
    """The taps as a table indexed by tap time (IST, tz-naive) with ``hmt`` (°C) and ``received`` (publication time,
    NaT when unknown). Accepts the hot-metal frame of ``data.coke_energy.sources.fetch_hm_slag(received=True)``, a
    series of temperatures by tap time, or a table already in this form."""
    empty = pd.DataFrame(
        {"hmt": pd.Series(dtype=float), "received": pd.Series(dtype="datetime64[ns]")}, index=pd.DatetimeIndex([])
    )
    if taps is None or len(taps) == 0:
        return empty
    if isinstance(taps, pd.Series):
        temps, received = taps, None
    elif "hmt" in taps.columns:
        temps, received = taps["hmt"], taps.get("received")
    else:
        temps = hmt_forecast.taps(taps)
        received = None
        if "received" in taps.columns:
            received = pd.Series(taps["received"].to_numpy(), index=pd.to_datetime(taps["t"]))
    idx = pd.DatetimeIndex(temps.index)
    if idx.tz is not None:
        idx = idx.tz_convert(PLANT_TZ).tz_localize(None)
    out = pd.DataFrame({"hmt": temps.to_numpy(dtype=float)}, index=idx)
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna(subset=["hmt"])
    if received is None:
        out["received"] = pd.NaT
    else:
        received = received[~received.index.duplicated(keep="last")]
        out["received"] = pd.to_datetime(received.reindex(out.index).to_numpy())
    return out


def process_now(status: Any) -> dict[str, Any]:
    """Fuel and blast now from a furnace-status snapshot view (``services.status.StatusSnapshotView``): the tags of
    ``PROCESS_TAGS`` and PCI in t/h (PCI rate x production, the plant's own derivation)."""
    if status is None:
        return {"at": None, "telemetry": "offline", "readings": []}
    by_key = {p.key: p for p in status.parameters}
    readings = []
    for key, (label, unit, digits) in PROCESS_TAGS.items():
        p = by_key.get(key)
        readings.append(
            {
                "key": key,
                "label": label,
                "value": _num(getattr(p, "value", None)),
                "unit": unit,
                "digits": digits,
                "at": getattr(p, "at", None),
                "setpoint": _num(getattr(p, "setpoint", None)),
            }
        )
    values = {r["key"]: r for r in readings}
    pci, prod = values["pci_rate"], values["production_rate"]
    injected = pci["value"] * prod["value"] / 1000.0 if pci["value"] is not None and prod["value"] is not None else None
    readings.append(
        {
            "key": "pci_t_h",
            "label": "PCI injected",
            "value": injected,
            "unit": "t/h",
            "digits": 1,
            "at": pci["at"],
            "setpoint": None,
        }  # fmt: skip
    )
    fresh = status.freshness
    return {"at": fresh.data_at, "telemetry": fresh.telemetry, "readings": readings}


def data_clock(now: datetime, process: Mapping[str, Any], charges: pd.DataFrame, taps: pd.DataFrame) -> dict[str, Any]:
    """Age of the newest record of each stream and the notice on the charge record's delay."""
    now_ist = _ist_naive(now)

    def delay(frame: pd.DataFrame) -> float | None:
        if frame.empty or frame["received"].isna().all():
            return None
        d = (frame["received"] - frame.index.to_series()) / pd.Timedelta(minutes=1)
        d = d[np.isfinite(d) & (d >= 0)]
        return float(d.median()) if len(d) else None

    def stream(key: str, label: str, cadence: str, frame: pd.DataFrame) -> dict[str, Any]:
        newest = frame.index[-1] if len(frame) else None
        return {
            "key": key,
            "label": label,
            "cadence": cadence,
            "newest": _utc(newest),
            "received": _utc(frame["received"].iloc[-1]) if len(frame) else None,
            "age_min": _minutes(now_ist, newest),
            "delay_min": delay(frame),
        }

    at = process.get("at")
    charge_delay = delay(charges)
    recent_taps = taps[taps.index >= now_ist - pd.Timedelta(days=SCORE_DAYS)]
    age = f" (median age of a charge at publication {charge_delay:.0f} min)" if charge_delay is not None else ""
    return {
        "sources": [
            {
                "key": "process",
                "label": "Process tags",
                "cadence": "every 10 s",
                "newest": at,
                "received": None,
                "age_min": None if at is None else max(0.0, (now - at).total_seconds() / 60.0),
                "delay_min": None,
            },
            stream("charges", "Charge record", "published once an hour", charges),
            stream("taps", "Hot-metal taps", "published after each cast", recent_taps),
        ],
        "notice": (
            f"Coke per tonne and the model check follow the charge record, which is published once an hour{age}. "
            "Published per charge, they would follow the furnace within 10 min."
        ),
    }


def coke_balance(
    latest: Mapping[str, Any] | None, charged: Mapping[str, Any], limit: float, window_h: float = 8.0
) -> dict[str, Any]:
    """Charged now next to burnt and modelled coke of the newest window; model − burnt against ``limit``."""
    burnt = _num((latest or {}).get("k_burnt"))
    model = _num((latest or {}).get("k_forge"))
    gap = model - burnt if burnt is not None and model is not None else None
    if gap is None:
        state = "unknown"
        text = (
            "No stored window with burnt and modelled coke."
            if latest is None or latest.get("valid", True)
            else "The newest window's PCI is outside the valid range: its coke rates are not shown."
        )
    else:
        state = "ok" if abs(gap) <= limit else "watch"
        text = (
            f"Model − burnt {_signed(gap)} kg/tHM over the last {window_h:g} h, "
            f"{'within' if state == 'ok' else 'outside'} ±{limit:g} kg/tHM."
        )
    return {
        "charged_now": _num(charged.get("value")),
        "charged_charges": int(charged.get("charges") or 0),
        "charged_first": _utc(charged.get("first")),
        "charged_last": _utc(charged.get("last")),
        "charged_received": _utc(charged.get("received")),
        "charged_basis": charged.get("basis", "none") if charged.get("value") is not None else "none",
        "charged_factor": _num(charged.get("factor")),
        "window_end": None if latest is None else latest["end"],
        "charged_window": _num((latest or {}).get("k_charged")),
        "burnt": burnt,
        "model": model,
        "gap": gap,
        "limit": limit,
        "state": state,
        "text": text,
    }


def _score_view(rows: pd.DataFrame, horizon: float, days: int, min_taps: int) -> dict[str, Any]:
    stats = hmt_filter.score(rows)
    n = stats["n"]
    if n < min_taps:
        return {
            **stats,
            "horizon_h": horizon,
            "days": days,
            "verdict": "insufficient",
            "text": f"{horizon:g} h ahead: {n} taps scored in the last {days} days; at least {min_taps} are needed.",
        }
    better = stats["ratio"] is not None and stats["ratio"] < 1.0
    text = (
        f"{horizon:g} h ahead: ±{stats['rmse_model_c']:.1f} °C against ±{stats['rmse_persistence_c']:.1f} °C for the "
        f"last tap held, over {n} taps of the last {days} days (bias {_signed(stats['bias_model_c'])} °C; "
        f"{100 * stats['within_1sd']:.0f} % of the taps inside ±1σ)."
    )
    return {**stats, "horizon_h": horizon, "days": days, "verdict": "better" if better else "worse", "text": text}


def _check_view(check: Mapping[str, Any]) -> dict[str, Any]:
    if check["state"] == "unknown":
        return {**check, "text": "No tap to check."}
    side = "above" if check["z"] > 0 else "below"
    expected = f"the expected {check['expected']:,.0f} ± {check['sd']:.0f} °C"
    if check["rule"] == "single":
        text = f"The newest tap, {check['value']:,.0f} °C, lies {abs(check['z']):.1f}σ {side} {expected}."
    elif check["rule"] == "pair":
        text = f"The last two taps both lie more than 1σ {side} their expected values (newest {_signed(check['z'])}σ)."
    else:
        text = f"The newest tap, {check['value']:,.0f} °C, lies within ±2σ of {expected}."
    return {**check, "text": text}


def hmt_view(
    taps: pd.DataFrame, now: datetime, hours: int, s: Settings, plant_band: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The fields of ``HmtView`` from the tap table (``tap_table``)."""
    p = s.filter
    now_ist = _ist_naive(now)
    cold = s.cold_tap_c
    if cold is None:  # user decision 12 Nov 2026: the plausibility band's lower edge, 1,440 °C
        cold = float(plant_band["lo"]) if plant_band else COLD_TAP_FALLBACK_C
    known = taps[taps.index <= now_ist]
    f = hmt_filter.filter_taps(known["hmt"], p)
    base = {
        "available": f.ready,
        "taps_used": len(f.steps),
        "cold_tap_c": cold,
        "horizon_h": s.horizon_h,
        "plant_band": plant_band,
        "check": _check_view(hmt_filter.tap_check(f.steps if f.ready else [], z_single=s.z_single, z_pair=s.z_pair)),
    }
    if not f.ready:
        return base

    def at(t_h: float) -> datetime:
        return _utc(f.origin + pd.Timedelta(hours=t_h))  # type: ignore[return-value]

    state = f.steps[-1].state
    t_now = max(float(hmt_filter.hours([now_ist], f.origin)[0]), state.t_h)
    interval = f.tap_interval_h()
    t_next = max(t_now, state.t_h + (interval or 2.0))
    est_now = hmt_filter.predict(state, t_now, p)
    est_next = hmt_filter.predict(state, t_next, p)
    t_end = t_now + s.horizon_h
    est_end = hmt_filter.predict(state, t_end, p)
    step = s.band_step_min / 60.0
    grid = sorted({state.t_h, t_now, t_end, *np.arange(math.ceil(state.t_h / step) * step, t_end, step).tolist()})
    band = []
    for t in grid:
        e = hmt_filter.predict(state, t, p)
        band.append({"at": at(t), "value": e.mean, "lo": e.mean - e.sd_tap, "hi": e.mean + e.sd_tap})
    shown_from = now_ist - pd.Timedelta(hours=hours)
    estimate = [
        {
            "at": _utc(t),
            "value": st.state.level + st.state.dev,
            "sd": math.sqrt(max(st.state.p_ll + 2.0 * st.state.p_ld + st.state.p_dd, 0.0)),
        }
        for t, st in zip(f.times, f.steps, strict=True)
        if t >= shown_from
    ]
    rows = hmt_filter.backtest(
        known["hmt"], known["received"], p, s.score_horizons_h,
        since=now_ist - pd.Timedelta(days=SCORE_DAYS), default_delay_h=s.default_tap_delay_h,
    )  # fmt: skip
    last = known.iloc[-1]
    return {
        **base,
        "last_tap": {"at": _utc(known.index[-1]), "value": float(last["hmt"]), "received": _utc(last["received"])},
        "now": {"at": at(t_now), "value": est_now.mean, "sd": est_now.sd_state},
        "next_tap": {"at": at(t_next), "value": est_next.mean, "sd": est_next.sd_tap},
        "p_cold_next": hmt_filter.p_below(est_next.mean, est_next.sd_tap, cold),
        "ahead": {"at": at(t_end), "value": est_end.mean, "sd": est_end.sd_tap},
        "tap_interval_h": interval,
        "tap_age_h": t_now - state.t_h,
        "stale": t_now - state.t_h > s.stale_after_h,
        "estimate": estimate,
        "band": band,
        "scores": [
            _score_view(rows[rows["horizon_h"] == h], h, SCORE_DAYS, s.score_min_taps) for h in s.score_horizons_h
        ],
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


# ------------------------------------------------------------------------------------------------ loaders
@memo(ttl_seconds=300, max_entries=8)
def _taps(start_key: str, end_key: str) -> pd.DataFrame:
    """The taps with their publication times between two IST instants (one Neon read per 5 min per range)."""
    from data.coke_energy.sources import fetch_hm_slag

    return tap_table(fetch_hm_slag(pd.Timestamp(start_key), pd.Timestamp(end_key), received=True))


@memo(ttl_seconds=300, max_entries=8)
def _charges(start_key: str, end_key: str) -> pd.DataFrame:
    """The per-charge record between two IST instants (one Neon read per 5 min per range)."""
    from data.coke_energy.sources import fetch_charges

    return fetch_charges(pd.Timestamp(start_key), pd.Timestamp(end_key))


def _status_view() -> Any:
    """The furnace-status snapshot of the current 10-s slot (``services.status``)."""
    from services import status as status_service

    return status_service.current_snapshot_view()


def _settings() -> Settings:
    try:
        from services import twin_reads

        return Settings.from_config(twin_reads.config())
    except Exception:  # noqa: BLE001 - the defaults stand without the config
        return Settings()


def _plant_band() -> dict[str, Any] | None:
    """BAND-02 of the process fact register (the plausibility band of the HMT)."""
    try:
        from utils.burden_engine.facts import load_register

        lo, hi = load_register().band("BAND-02")
        return {"lo": float(lo), "hi": float(hi), "fact": "BAND-02"}
    except Exception:  # noqa: BLE001 - the chart stands without the band
        return None


def _range_keys(start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, str]:
    """A read range on a 5-min grid, so that requests within 5 min share one read."""
    return start.floor("h").isoformat(), end.ceil("5min").isoformat()


def view(
    store: Any,
    *,
    hours: int = DEFAULT_HOURS,
    days: int = SCORE_DAYS,
    now: datetime | None = None,
    cfg: Mapping[str, Any] | None = None,
    settings: Settings | None = None,
    taps: Callable[[pd.Timestamp, pd.Timestamp], Any] | None = None,
    charges: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame] | None = None,
    process: Callable[[], Any] | None = None,
    shifts: pd.DataFrame | None | Literal["load"] = "load",
    built_at: datetime | None = None,
) -> dict[str, Any]:
    """The fields of ``LiveModelView``. ``taps(start_ist, end_ist)``, ``charges(start_ist, end_ist)``, ``process()``
    (a furnace-status snapshot view) and ``shifts`` are the loaders (tests)."""
    now = now or datetime.now(UTC)
    s = settings or _settings()
    notes: list[str] = []
    now_ist = _ist_naive(now)
    hourly = window_rows(store.snapshots.series(now - timedelta(hours=hours), now), s.pci_valid_kg_thm)
    recent = hourly
    if hours < SCORE_DAYS * 24:
        recent = window_rows(store.snapshots.series(now - timedelta(days=SCORE_DAYS), now), s.pci_valid_kg_thm)
    latest = hourly[-1] if hourly else None
    if not hourly:
        notes.append("No twin window is stored for these hours: the worker's twin.evaluate job writes one per hour.")
    withheld = sum(1 for r in hourly if not r["valid"])
    if withheld:
        lo, hi = s.pci_valid_kg_thm
        notes.append(
            f"{withheld} of the {len(hourly)} windows have a PCI outside {lo:g}–{hi:g} kg/tHM (PCI tag stuck or "
            "injection off): their coke rates are not shown."
        )

    # The taps: the hours shown, and the scored days with the filter's settling time before them.
    tap_hours = max(hours, (SCORE_DAYS + s.burn_in_days) * 24)
    table = tap_table(None)
    try:
        start, end = now_ist - pd.Timedelta(hours=tap_hours), now_ist + pd.Timedelta(minutes=1)
        table = tap_table((taps or (lambda a, b: _taps(*_range_keys(a, b))))(start, end))
    except Exception as exc:  # noqa: BLE001 - Neon unreachable: the view stands without the taps
        notes.append(f"The tap temperatures are not available ({type(exc).__name__}): no measured HMT and no forecast.")
    hmt = hmt_view(table, now, hours, s, _plant_band())
    if not hmt["available"] and len(table):
        notes.append(
            f"{len(table)} taps in the last {tap_hours // 24} days: the HMT estimate needs {s.filter.min_taps}."
        )
    if hmt.get("stale"):
        notes.append(
            f"No tap published for {hmt['tap_age_h']:.0f} h: the HMT estimate is the long-run level with its full band."
        )
    shown = table[(table.index >= now_ist - pd.Timedelta(hours=hours)) & (table.index <= now_ist)]
    tap_rows = [{"at": _utc(t), "value": float(r["hmt"]), "received": _utc(r["received"])} for t, r in shown.iterrows()]

    # The per-charge record: charged coke of the last charges on the basis of the stored windows.
    record = charge_record.charge_table(pd.DataFrame(), pd.DatetimeIndex([]))
    try:
        start, end = now_ist - pd.Timedelta(days=s.charge_days), now_ist + pd.Timedelta(minutes=1)
        record = (charges or (lambda a, b: _charges(*_range_keys(a, b))))(start, end)
    except Exception as exc:  # noqa: BLE001 - Neon unreachable: the view stands without the charge record
        notes.append(f"The charge record is not available ({type(exc).__name__}): no charged coke of the last charges.")
    windows = [(_ist_naive(r["end"]), _num(r.get("k_charged"))) for r in recent]
    charged = charge_record.charged_now(record, windows, s.charged, at=now_ist)

    # Fuel and blast now: the furnace-status snapshot.
    status = None
    try:
        status = (process or _status_view)()
    except Exception as exc:  # noqa: BLE001 - InfluxDB unreachable: the view stands without the process tags
        notes.append(f"The process tags are not available ({type(exc).__name__}).")
    tags = process_now(status)

    # Page 11's model check on the FORGE view.
    th = mc.Thresholds.from_config(_model_check_config(cfg))
    check = None
    if isinstance(shifts, str):
        try:
            stored = forge.load()
            shifts, built_at = stored.shifts, stored.built_at
        except forge.ViewUnavailable as exc:
            shifts = None
            notes.append(str(exc))
    if shifts is not None:
        today = pd.Timestamp(now).tz_convert(PLANT_TZ).tz_localize(None).normalize()
        check = shift_check(shifts, days, thresholds=th, today=today) | {"built_at": built_at}
    return {
        "now": now,
        "hours": hours,
        "clock": data_clock(now, tags, record, table),
        "process": tags,
        "coke": coke_balance(latest, charged, th.bias_kgthm, s.charged.window_h),
        "hmt": hmt,
        "latest": latest,
        "hourly": hourly,
        "taps": tap_rows,
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
