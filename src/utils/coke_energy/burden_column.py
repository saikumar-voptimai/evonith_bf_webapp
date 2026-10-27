"""Burden column of the live twin (SV-1.3, #283): descent times, charges and coke in the furnace by zone, the ledger
of the charges in the column, and the implied lag between charged and burnt coke.

Reference (layered gas-flow model v0.4 on the drawn BF-02 profile, momentum model notes 6.5; burnt-vs-charged study
1.9): at 5.9 charges/h (23.5 m^3 per charge at the stockline) a charge reaches the cohesive-zone top 2.2 h after
charging (13.0 charges above the zone, 61 t of coke), passes the zone in 0.9 h (14 t) and its coke reaches the
raceways 6.2 h after charging (3.1 h in the coke bed below the zone, 59 t): 133 t of coke in transit. The zones hold
fixed volumes, so fixed numbers of charges (``TransitReference.zone_charges``); the charging rate sets how fast
they pass:

    tau_zone = tau_zone,ref x n_ref / n_ch                       (from the current charging rate n_ch)

and, exactly for a varying rate, the age of the charge at each zone boundary in the charging record: the time over
which the last N charges were charged (``column``).

Coke in a zone. Above the cohesive zone and in it: reference tonnes x (coke share of the volume of the charges now in
the zone / that of the reference charge, 0.387; from the coke and ore-layer tonnes at the reference bulk densities,
coke 0.50 t/m^3, ore layer 1.84 t/m^3), since the zone volume is fixed; the share needs no charge count. Below the
zone the coke bed of fixed volume: the reference tonnes.

Ledger (``column``): from the hourly charge record (tonnes per material, charges per hour, hot metal per hour) the
charges in the column by band, newest first: above the cohesive zone, in it, below it to the raceways; per band the
hours it spans, the charges, tonnes per charge and wet blend of the ferrous charge, coke per charge and the coke rate
of those charges (kg/tHM, on the hot metal of the same hours). A band takes fractions of the hours at its edges.

Implied lag (``implied_lag``): hourly coke rate of the charges against K_TRACE of the 8-h windows ending each hour,
both as deviations from a centred 7-day mean; the charged series is averaged over the same 8-h box shifted back by L;
the lag of the largest correlation (parabola through the best hour and its neighbours) is the implied transit time to
the raceways; 95 % interval by moving-block bootstrap of the hours (blocks of 24 h).

Pure numerics: no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

ZONES = ("above_cz", "in_cz", "below_cz")
FERROUS = ("SINTER_CALC_MT", "TOTAL_PELLET_CALC_MT", "ORE_CALC_MT")
NONCOKE_EXTRA = ("NUTCOKE_CALC_MT", "FLUX_CALC_MT")  # charged in the ore layer
COKE = "COKE_CALC_MT"
CHARGES = "CHARGES/HRS."
HM = "PRODUCTIONTONNESPERHR"


@dataclass(frozen=True)
class TransitReference:
    """Descent times (h) and coke (t) of the zones at the reference charging rate and coke per charge."""

    charges_per_hour: float = 5.9
    hours: Mapping[str, float] = field(default_factory=lambda: {"above_cz": 2.2, "in_cz": 0.9, "below_cz": 3.1})
    coke_t: Mapping[str, float] = field(default_factory=lambda: {"above_cz": 61.0, "in_cz": 14.0, "below_cz": 59.0})
    coke_t_per_charge: float = 4.55  # 305.7 kg/tHM x 94.6 t/h / 6.35 charges/h (model reference charge)
    volume_per_charge_m3: float = 23.5  # at the stockline
    coke_bulk_t_m3: float = 0.50
    noncoke_t_per_charge: float = 26.46  # ferrous + nut coke + flux of the reference charge

    @property
    def noncoke_bulk_t_m3(self) -> float:
        """Bulk density of the ore layer that gives the reference charge its volume (1.84 t/m^3)."""
        coke_m3 = self.coke_t_per_charge / self.coke_bulk_t_m3
        return self.noncoke_t_per_charge / (self.volume_per_charge_m3 - coke_m3)

    def coke_volume_fraction(self, coke_t: float, noncoke_t: float) -> float:
        """Share of the charge volume taken by coke; the reference charge gives 0.387."""
        vc = coke_t / self.coke_bulk_t_m3
        vn = noncoke_t / self.noncoke_bulk_t_m3
        return vc / (vc + vn) if vc + vn > 0 else float("nan")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> TransitReference:
        c = dict(cfg or {})
        base = cls()
        return cls(
            charges_per_hour=float(c.get("charges_per_hour", base.charges_per_hour)),
            hours={**base.hours, **{k: float(v) for k, v in (c.get("hours") or {}).items()}},
            coke_t={**base.coke_t, **{k: float(v) for k, v in (c.get("coke_t") or {}).items()}},
            coke_t_per_charge=float(c.get("coke_t_per_charge", base.coke_t_per_charge)),
            volume_per_charge_m3=float(c.get("volume_per_charge_m3", base.volume_per_charge_m3)),
            coke_bulk_t_m3=float(c.get("coke_bulk_t_m3", base.coke_bulk_t_m3)),
            noncoke_t_per_charge=float(c.get("noncoke_t_per_charge", base.noncoke_t_per_charge)),
        )

    def zone_charges(self) -> dict[str, float]:
        """Charges each zone holds (fixed volume): reference hours x reference charging rate."""
        return {z: self.hours[z] * self.charges_per_hour for z in ZONES}

    @property
    def hours_to_cz(self) -> float:
        return self.hours["above_cz"]

    @property
    def hours_to_raceway(self) -> float:
        return sum(self.hours[z] for z in ZONES)


def descent_times(n_ch: float, ref: TransitReference | None = None) -> dict[str, float]:
    """Descent times (h) at a charging rate n_ch (charges/h): to the cohesive-zone top, through the zone, below it,
    to the raceways."""
    ref = ref or TransitReference()
    if not n_ch > 0:
        raise ValueError(f"charging rate {n_ch} must be positive")
    f = ref.charges_per_hour / float(n_ch)
    out = {f"hours_{z}": ref.hours[z] * f for z in ZONES}
    out["hours_to_cz"] = out["hours_above_cz"]
    out["hours_to_raceway"] = sum(out[f"hours_{z}"] for z in ZONES)
    return out


@dataclass
class Band:
    """The charges of one zone in the column (newest first: above the zone, in it, below it)."""

    zone: str
    charges: float
    newest_h: float  # age of its newest charge, h before ``at``
    oldest_h: float  # age of its oldest charge (the descent time to the zone's lower boundary)
    tonnes: dict[str, float]  # per material, over the band (fractions of the hours at its edges)
    hm_t: float  # hot metal of the same hours (same fractions)
    coke_t_per_charge: float  # on the plant's charge counter (CHARGES/HRS.)
    coke_kg_per_t_ferrous: float  # coke to ferrous of the same charges (no charge count needed)
    coke_kg_thm: float  # coke as charged per tonne of hot metal of the same hours
    blend_wet_pct: dict[str, float]  # wet share of the ferrous charge (sinter, pellet, ore types)
    coke_volume_fraction: float  # coke share of the charge volume (reference 0.387)
    coke_inventory_t: float  # coke in the zone (see ``column``)


@dataclass
class ColumnState:
    at: pd.Timestamp
    charges_per_hour: float  # the last hour of the record
    tau: dict[str, float]  # ages of the charges at the zone boundaries, h (to_cz, to_raceway, per zone)
    tau_rate: dict[str, float]  # descent_times() at the current charging rate
    bands: list[Band]
    complete: bool  # the record reaches back to the oldest charge of the column
    missing_hours: int  # hours absent from the record inside the column's span (counted as no charges)

    @property
    def coke_in_transit_t(self) -> float:
        return float(sum(b.coke_inventory_t for b in self.bands))


LOOKBACK = pd.Timedelta(hours=72)  # record read back from ``at`` (the column spans about 6 h of charging)


def _hourly(record: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """Hours that ended by ``at`` (an hour stamped t covers [t, t + 1 h)) within ``LOOKBACK``, newest first."""
    t = record.index if isinstance(record.index, pd.DatetimeIndex) else pd.DatetimeIndex(pd.to_datetime(record["time"]))
    keep = np.asarray((t + pd.Timedelta(hours=1) <= at) & (t >= at - LOOKBACK))
    h = record[keep].copy()
    h.index = t[keep]
    h = h[~h.index.duplicated(keep="last")]
    return h.sort_index(ascending=False)


def column(
    record: pd.DataFrame, at: Any, ref: TransitReference | None = None, materials: Sequence[str] | None = None
) -> ColumnState:
    """The charges in the column at ``at`` from the hourly charge record (``time`` index or column; tonnes per
    material ``*_CALC_MT``, charges in the hour ``CHARGES/HRS.``, hot metal ``PRODUCTIONTONNESPERHR``).

    Charges are counted back from ``at`` (uniformly spaced within an hour, the newest at its end): the first
    ``ref.zone_charges()["above_cz"]`` are above the cohesive zone, the next in it, the next below it to the
    raceways."""
    ref = ref or TransitReference()
    at = pd.Timestamp(at)
    rows = _hourly(record, at)
    mats = list(materials) if materials is not None else [c for c in record.columns if str(c).endswith("_CALC_MT")]
    n = pd.to_numeric(rows.get(CHARGES, pd.Series(0.0, index=rows.index)), errors="coerce").fillna(0.0).clip(lower=0)
    cum = n.cumsum().to_numpy()  # charges from ``at`` back to the start of each hour
    lo = np.r_[0.0, cum[:-1]]
    t_end = (rows.index + pd.Timedelta(hours=1)).to_numpy()
    need = ref.zone_charges()
    bounds = np.cumsum([0.0] + [need[z] for z in ZONES])
    phi_ref = ref.coke_volume_fraction(ref.coke_t_per_charge, ref.noncoke_t_per_charge)

    def age_at(c: float) -> float:
        """Age (h before ``at``) of the charge ``c`` charges back."""
        k = int(np.searchsorted(cum, c - 1e-12))
        if k >= len(cum):
            return float("nan")
        frac = (c - lo[k]) / n.iloc[k] if n.iloc[k] > 0 else 0.0
        return float((at - pd.Timestamp(t_end[k])) / pd.Timedelta(hours=1) + frac)

    bands = []
    for i, zone in enumerate(ZONES):
        a, b = bounds[i], bounds[i + 1]
        w = np.clip(np.minimum(cum, b) - np.maximum(lo, a), 0.0, None)  # charges of each hour in the band
        frac = np.divide(w, n.to_numpy(), out=np.zeros_like(w), where=n.to_numpy() > 0)
        tonnes = {m: float(np.nansum(pd.to_numeric(rows[m], errors="coerce").to_numpy() * frac)) for m in mats}
        hm = float(np.nansum(pd.to_numeric(rows.get(HM), errors="coerce").to_numpy() * frac)) if HM in rows else 0.0
        charges = float(w.sum())
        coke = tonnes.get(COKE, 0.0)
        fer = sum(tonnes.get(m, 0.0) for m in FERROUS)
        noncoke = fer + sum(tonnes.get(m, 0.0) for m in NONCOKE_EXTRA)
        parts = {m: tonnes[m] for m in mats if m in ("SINTER_CALC_MT", "TOTAL_PELLET_CALC_MT") or _is_ore_type(m)}
        phi = ref.coke_volume_fraction(coke, noncoke)
        if zone == "below_cz":  # coke-only bed of fixed volume below the zone: the reference tonnes
            inventory = ref.coke_t[zone]
        else:  # fixed zone volume: the coke share of the volume of the charges in it
            inventory = ref.coke_t[zone] * phi / phi_ref if math.isfinite(phi) else float("nan")
        bands.append(
            Band(
                zone=zone,
                charges=charges,
                newest_h=age_at(a),
                oldest_h=age_at(b),
                tonnes=tonnes,
                hm_t=hm,
                coke_t_per_charge=coke / charges if charges > 0 else float("nan"),
                coke_kg_per_t_ferrous=1000.0 * coke / fer if fer > 0 else float("nan"),
                coke_kg_thm=1000.0 * coke / hm if hm > 0 else float("nan"),
                blend_wet_pct={m: 100.0 * v / fer for m, v in parts.items() if fer > 0},
                coke_volume_fraction=phi,
                coke_inventory_t=inventory,
            )
        )
    span_h = bands[-1].oldest_h
    complete = bool(len(cum) and cum[-1] >= bounds[-1] - 1e-9)
    missing = 0
    if len(rows) and math.isfinite(span_h):
        hours = pd.date_range(at - pd.Timedelta(hours=math.ceil(span_h)), at - pd.Timedelta(hours=1), freq="h")
        missing = int(len(hours.difference(rows.index)))
    tau = {"to_cz": bands[0].oldest_h, "to_raceway": bands[-1].oldest_h}
    tau.update({f"in_{b.zone}": b.oldest_h - b.newest_h for b in bands})
    cph = float(n.iloc[0]) if len(n) else float("nan")
    return ColumnState(at, cph, tau, descent_times(cph, ref) if cph > 0 else {}, bands, complete, missing)


def quantities(st: ColumnState) -> dict[str, float]:
    """Scalar quantities of a column state, units in the keys: charging rate, descent times from the record and from
    the current rate, coke by zone and in transit, coke per tonne of ferrous by zone (the oldest band's is the coke
    now reaching the raceways)."""
    out: dict[str, Any] = {
        "transit_charges_per_hour": st.charges_per_hour,
        "tau_cz_h": st.tau.get("to_cz"),
        "tau_raceway_h": st.tau.get("to_raceway"),
        "tau_cz_rate_h": st.tau_rate.get("hours_to_cz"),
        "tau_raceway_rate_h": st.tau_rate.get("hours_to_raceway"),
        "coke_in_transit_t": st.coke_in_transit_t,
    }
    for b in st.bands:
        out[f"coke_{b.zone}_t"] = b.coke_inventory_t
        out[f"coke_kg_per_t_ferrous_{b.zone}"] = b.coke_kg_per_t_ferrous
    return {k: float(v) for k, v in out.items() if v is not None and math.isfinite(float(v))}


def _is_ore_type(name: str) -> bool:
    parts = str(name).split("_")
    return len(parts) == 4 and parts[0] == "ORE" and parts[1].isdigit()


# ---------------------------------------------------------------------------------------------- implied lag
def charged_coke(record: pd.DataFrame, basis: str = "ferrous") -> pd.Series:
    """Hourly coke content of the charges, by the start of the hour: ``ferrous`` (default) coke per tonne of ferrous
    charged in the hour (kg/t), ``charge`` coke per charge on the plant's counter (t). The hourly tonnes are whole
    charges while CHARGES/HRS. is a rate, so coke per charge alternates by about one charge in six from hour to hour
    (lag-1 autocorrelation of its deviations -0.40 on Sep - Oct 2026); coke to ferrous of the same charges has no
    count in it (CV 0.035, lag-1 +0.36)."""
    h = record.copy()
    if not isinstance(h.index, pd.DatetimeIndex):
        h = h.set_index(pd.to_datetime(h["time"]))
    coke = pd.to_numeric(h[COKE], errors="coerce")
    if basis == "charge":
        n = pd.to_numeric(h[CHARGES], errors="coerce")
        out = coke / n.where(n > 0)
    elif basis == "ferrous":
        fer = sum(pd.to_numeric(h[m], errors="coerce").fillna(0.0) for m in FERROUS if m in h)
        out = 1000.0 * coke / fer.where(fer > 0)
    else:
        raise ValueError(f"basis must be 'ferrous' or 'charge', not {basis!r}")
    return out[~out.index.duplicated(keep="last")].sort_index()


@dataclass
class LagResult:
    lag_h: float  # implied transit time, charging to burning (h)
    ci: tuple[float, float]  # 95 % moving-block bootstrap interval
    r_peak: float
    scan: pd.DataFrame  # lag_h, r, n
    n_hours: int
    boot: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _detrend(s: pd.Series, days: float) -> pd.Series:
    w = int(round(days * 24))
    return s - s.rolling(w, center=True, min_periods=w // 2).mean()


def _peak(r: np.ndarray, lags: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(r)
    if not ok.any():
        return float("nan"), float("nan")
    i = int(np.nanargmax(np.where(ok, r, -np.inf)))
    if 0 < i < len(r) - 1 and np.isfinite(r[i - 1]) and np.isfinite(r[i + 1]):
        a, b, c = r[i - 1], r[i], r[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den < 0 else 0.0
        return float(lags[i] + max(-0.5, min(0.5, off)) * (lags[1] - lags[0])), float(b)
    return float(lags[i]), float(r[i])


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    a, b = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def implied_lag(
    charged: pd.Series,
    burnt: pd.Series,
    lags_h: Sequence[float] = tuple(range(0, 13)),
    window_h: int = 8,
    detrend_days: float = 7.0,
    block_h: int = 24,
    n_boot: int = 500,
    seed: int = 0,
) -> LagResult:
    """Implied lag between the hourly coke rate of the charges (``charged``, hourly index) and K_TRACE of the windows
    of ``window_h`` hours ending at each hour (``burnt``, index = window end)."""
    idx = pd.date_range(
        min(charged.index.min(), burnt.index.min()), max(charged.index.max(), burnt.index.max()), freq="h"
    )
    c = charged.reindex(idx).astype(float)
    b = burnt.reindex(idx).astype(float)
    box = c.rolling(window_h, min_periods=window_h - 1).mean()  # mean over the hours [t - window_h, t)
    box = box.shift(1)  # an hour stamped t covers [t, t + 1); the window ending at t covers the hours t-8 .. t-1
    bd = _detrend(b, detrend_days).to_numpy()
    lags = np.asarray(lags_h, float)
    X = np.column_stack([_detrend(box.shift(int(L)), detrend_days).to_numpy() for L in lags])
    r = np.array([_corr(X[:, j], bd) for j in range(len(lags))])
    n = np.array([int((np.isfinite(X[:, j]) & np.isfinite(bd)).sum()) for j in range(len(lags))])
    lag, rp = _peak(r, lags)
    rng = np.random.default_rng(seed)
    T = len(idx)
    nb = max(1, int(math.ceil(T / block_h)))
    boot = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, T - block_h + 1), nb)
        rows = np.concatenate([np.arange(s, min(s + block_h, T)) for s in starts])[:T]
        rb = np.array([_corr(X[rows, j], bd[rows]) for j in range(len(lags))])
        boot.append(_peak(rb, lags)[0])
    boot_a = np.asarray(boot, float)
    boot_a = boot_a[np.isfinite(boot_a)]
    ci = (float(np.percentile(boot_a, 2.5)), float(np.percentile(boot_a, 97.5))) if len(boot_a) else (math.nan,) * 2
    scan = pd.DataFrame({"lag_h": lags, "r": r, "n": n})
    return LagResult(lag, ci, rp, scan, int(np.isfinite(bd).sum()), boot_a)
