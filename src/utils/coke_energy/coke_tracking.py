"""Coke tracked to the cohesive zone (user decision 13 Nov 2026; claude/coke_tracking_plan.md).

The charged coke of a window is the coke of the charges dumped in it. The tracked coke is the coke of the charges
whose ore melts in it, expressed on the window's own hot metal:

    K_tracked(W) = K_charged(W) x r_melt(W) / r_charged(W)

r = dry coke / ferrous burden (kg/t) of a set of charges: r_charged over the charges of the window's hourly rows,
r_melt over the charges weighted by the share of their melting inside the window. The window's production count, the
denominator of K_charged and of the burnt coke, is kept, so only the timing of the coke:ferrous ratio changes.

Path of a charge. The hourly charge record (row HH = charges dumped in [HH - 1 h, HH)) is split into parcels (6 per
hour). A parcel melts while it passes the cohesive zone, between the depths D1 = 305.5 m3 and D2 = 430.3 m3 below the
stockline (as-charged volume; SV-1.3 zones: 13.0 and 18.3 reference charges of 23.5 m3). Its ages there are
D / (charged volume per hour, trailing 14-day median of the charging hours), with the charge volume from the bulk
densities of the materials (coke 0.493, nut coke 0.541, sinter 2.00, pellet 2.10, lump ore 2.32, flux 1.55 t/m3:
literature apparent densities and the voidage correlations of Jiao, Shu & Yu, Metals 2025, 15, 1048, on the plant's
size analyses). The clock that carries the ages is the blast: cumulative blast oxygen, one hour of age being an hour
at the trailing 14-day median oxygen rate; with the blast off (below 60,000 Nm3/h) the column does not move. The
ore melts linearly between the two ages. Tags of row HH are the mean over [HH - 0.5 h, HH + 0.5 h), so the melting
window of the window [s, e) of hourly rows is [s - 0.5 h, e - 0.5 h).

Tests on the plant record (13 Nov 2026; claude/coke_tracking_plan.md section 7). Clock: the charge-volume clock
(cumulative charged volume) placed the ages with more scatter than the blast clock (independent error of the tracked
coke 8.0 against 7.3 kg/tHM per stable shift, 2025-26). Independent error per stable shift (three-cornered hat with raw
burnt coke and FORGE, detrended), charged -> tracked: 9.0 -> 8.7 kg/tHM (Aug 2025 - Nov 2026, 611 shifts), 7.5 -> 7.3
(2023-24, 1,194) and 8.0 -> 7.7 (Jan - Apr 2025, 204). Calibration on the tracked coke: FORGE - TRACE SD per stable
shift -0.4 % (Dec 2025 - Nov 2026), -0.2 % (Oct - Nov 2026), 0.0 % (2023-24, Jan - Apr 2025); shift-to-shift scatter
of the calibrated omega +3.5 %, -2.4 %, 0.0 %, +0.8 % (acceptance gate: -10 %). The tracker averages the basis over
about 17 stable shifts (effective sample at a 3-day half-life), which removes most of the timing error the tracked coke
corrects. A window whose melting weights cover less than 80 % of the parcels expected (record gaps) has no tracked
coke.

Pure numerics: no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BASES = ("charged", "tracked")
_HOUR_NS = 3_600_000_000_000
EXTEND_H = 12  # hours the blast clock runs on past the end of the record
# hourly dataset columns: material -> tonnes per hour
MATERIALS = {
    "coke": "COKE_CALC_MT",
    "nut": "NUTCOKE_CALC_MT",
    "sinter": "SINTER_CALC_MT",
    "pellet": "TOTAL_PELLET_CALC_MT",
    "ore": "ORE_CALC_MT",
    "flux": "FLUX_CALC_MT",
}
FERROUS = ("sinter", "pellet", "ore")
COKE_MOISTURE = "COKE_MOIST%"
BLAST = "HOT BLAST VOLUMENM3/HR."
O2_ENRICHMENT = "O2 ENRICHMENT %"
O2_FLOW = "OXYGENFLOWNM3/HR."


@dataclass(frozen=True)
class TrackingSettings:
    """Coke tracking: the basis of the per-shift calibration and of TRACE's offset, and the path of a charge."""

    basis: str = "charged"  # "charged" (coke of the window's charges) or "tracked" (coke of the charges melting in it)
    density_coke: float = 0.493  # bulk densities, t/m3
    density_nut: float = 0.541
    density_sinter: float = 2.00
    density_pellet: float = 2.10
    density_ore: float = 2.32
    density_flux: float = 1.55
    cz_top_m3: float = 305.5  # depth of the cohesive-zone top below the stockline, m3 as charged
    cz_bottom_m3: float = 430.3  # depth of the cohesive-zone bottom
    rate_days: float = 14.0  # trailing window of the volume and oxygen rates
    blast_on_nm3h: float = 60000.0  # below this blast volume the clock stops
    charge_offset_h: float = 1.0  # row HH of the hourly record holds the charges of [HH - 1 h, HH)
    tag_offset_h: float = 0.5  # row HH holds the tags' mean over [HH - 0.5 h, HH + 0.5 h)
    fill_h: int = 3  # gaps of the record up to this length are interpolated
    parcels_per_hour: int = 6
    min_cover: float = 0.8  # share of the expected melting weight a window needs

    def __post_init__(self) -> None:
        if self.basis not in BASES:
            raise ValueError(f"coke basis {self.basis!r}: one of {', '.join(BASES)}")
        if not 0 < self.cz_top_m3 < self.cz_bottom_m3:
            raise ValueError("cohesive-zone depths must satisfy 0 < top < bottom")

    @property
    def densities(self) -> dict[str, float]:
        return {
            "coke": self.density_coke,
            "nut": self.density_nut,
            "sinter": self.density_sinter,
            "pellet": self.density_pellet,
            "ore": self.density_ore,
            "flux": self.density_flux,
        }

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> TrackingSettings:
        """From the ``coke_tracking`` block of ``coke_energy.yml``."""
        c, d = dict(cfg or {}), cls()
        dens = dict(c.get("bulk_density_t_m3") or {})
        return cls(
            basis=str(c.get("basis", d.basis)),
            density_coke=float(dens.get("coke", d.density_coke)),
            density_nut=float(dens.get("nut_coke", d.density_nut)),
            density_sinter=float(dens.get("sinter", d.density_sinter)),
            density_pellet=float(dens.get("pellet", d.density_pellet)),
            density_ore=float(dens.get("lump_ore", d.density_ore)),
            density_flux=float(dens.get("flux", d.density_flux)),
            cz_top_m3=float(c.get("cz_top_m3", d.cz_top_m3)),
            cz_bottom_m3=float(c.get("cz_bottom_m3", d.cz_bottom_m3)),
            rate_days=float(c.get("rate_days", d.rate_days)),
            blast_on_nm3h=float(c.get("blast_on_nm3h", d.blast_on_nm3h)),
            charge_offset_h=float(c.get("charge_offset_h", d.charge_offset_h)),
            tag_offset_h=float(c.get("tag_offset_h", d.tag_offset_h)),
            fill_h=int(c.get("fill_h", d.fill_h)),
            parcels_per_hour=int(c.get("parcels_per_hour", d.parcels_per_hour)),
            min_cover=float(c.get("min_cover", d.min_cover)),
        )


def _ns(x: Any) -> np.ndarray:
    """int64 nanoseconds of datetimes of any resolution."""
    return np.asarray(pd.to_datetime(x)).astype("datetime64[ns]").astype(np.int64)


def _hourly(fd: pd.DataFrame) -> pd.DataFrame:
    """The hourly dataset indexed by its row label (IST, tz-naive), one row per hour of its span."""
    f = fd.set_index(pd.to_datetime(fd["time"])) if "time" in fd.columns else fd.copy()
    f.index = pd.DatetimeIndex(f.index)
    f = f[~f.index.duplicated(keep="last")].sort_index()
    if f.empty:
        return f
    return f.reindex(pd.date_range(f.index.min(), f.index.max(), freq="h"))


def _col(f: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(f[name], errors="coerce") if name in f.columns else pd.Series(np.nan, index=f.index)


def _fill_gaps(x: pd.Series, limit: int) -> pd.Series:
    """Linear interpolation of the interior gaps of at most ``limit`` rows; a longer gap stays empty (pandas'
    ``interpolate(limit=...)`` would fill its first ``limit`` rows)."""
    na = x.isna()
    run = na.groupby((~na).cumsum()).transform("sum")
    return x.interpolate(limit_area="inside").where(~na | (run <= limit))


class Tracker:
    """Parcels of one hourly record and their path through the cohesive zone on the blast clock."""

    def __init__(self, fd: pd.DataFrame, settings: TrackingSettings | None = None) -> None:
        self.settings = s = settings or TrackingSettings()
        f = _hourly(fd)
        n = len(f)
        self.empty = n == 0
        if self.empty:
            return
        sub = s.parcels_per_hour
        tons = pd.DataFrame({k: _col(f, c) for k, c in MATERIALS.items()})
        tons.index = f.index - pd.Timedelta(hours=s.charge_offset_h)  # start of the hour the charges were dumped in
        valid = tons["coke"].notna() & tons["sinter"].notna()
        tons = tons.apply(_fill_gaps, limit=s.fill_h)
        valid = valid | (tons["coke"].notna() & tons["sinter"].notna())
        tons[["nut", "pellet", "ore", "flux"]] = tons[["nut", "pellet", "ore", "flux"]].fillna(0.0)
        moist = _col(f, COKE_MOISTURE).ffill().fillna(0.0).to_numpy()
        coke_dry = (tons["coke"] * (1 - moist / 100)).to_numpy()
        ferrous = tons[list(FERROUS)].sum(axis=1, min_count=1).to_numpy()
        dens = s.densities
        volume = sum(tons[k].to_numpy() / dens[k] for k in MATERIALS)
        vol_rate = pd.Series(np.where(valid & (volume > 20), volume, np.nan), index=tons.index)
        vol_rate = vol_rate.rolling(f"{s.rate_days:g}D", min_periods=48).median().ffill().bfill()

        # blast oxygen, Nm3/h, on the tag hours; blast off -> 0 (the clock stops); short gaps interpolated
        blast = _col(f, BLAST)
        if O2_ENRICHMENT in f.columns:
            o2 = blast * (0.208 + _col(f, O2_ENRICHMENT) / 100)
        else:
            flow = _col(f, O2_FLOW).fillna(0.0)
            o2 = 0.208 * (blast - flow) + flow
        o2 = o2.where(~(blast < s.blast_on_nm3h), 0.0)
        o2.index = f.index - pd.Timedelta(hours=s.tag_offset_h)
        o2 = _fill_gaps(o2, s.fill_h)
        o2_ok = o2.notna().to_numpy()
        o2v = o2.fillna(0.0).to_numpy()
        rate = o2.where(o2 > 0).rolling(f"{s.rate_days:g}D", min_periods=48).median().ffill().bfill()
        # cumulative oxygen at the hour boundaries of the tag grid; past the end of the record the clock runs on at the
        # trailing median rate, which places the melting of the parcels that straddle the end
        t_tag = _ns(o2.index)
        end = t_tag[-1] + _HOUR_NS
        c = np.cumsum(o2v)
        last_rate = float(rate.iloc[-1]) if np.isfinite(rate.iloc[-1]) else 0.0
        self._clock_t = np.r_[t_tag, end, end + EXTEND_H * _HOUR_NS]
        self._clock_c = np.r_[0.0, c, c[-1] + EXTEND_H * last_rate]
        self._clock_ok = np.r_[o2_ok, True, True]

        # parcels
        rep = np.repeat(np.arange(n), sub)
        step = _HOUR_NS // sub
        self.t = _ns(tons.index)[rep] + np.tile(np.arange(sub) * step + step // 2, n)
        self.coke = np.nan_to_num(coke_dry[rep] / sub)
        self.ferrous = np.nan_to_num(ferrous[rep] / sub)
        self.valid = valid.to_numpy()[rep] & np.isfinite(ferrous[rep]) & (ferrous[rep] > 0)
        vr = vol_rate.to_numpy()[rep]
        r_at = np.interp(self.t, _ns(rate.index), rate.to_numpy())
        c_at = np.interp(self.t, self._clock_t, self._clock_c)
        self.age_top_h = s.cz_top_m3 / vr
        self.age_bottom_h = s.cz_bottom_m3 / vr
        self.t_top = np.interp(c_at + self.age_top_h * r_at, self._clock_c, self._clock_t, left=np.nan, right=np.nan)
        self.t_bottom = np.interp(
            c_at + self.age_bottom_h * r_at, self._clock_c, self._clock_t, left=np.nan, right=np.nan
        )
        # a parcel whose path starts before the blast record or crosses an hour without blast tags is not tracked
        bad = np.flatnonzero(~self._clock_ok)
        bad_t = self._clock_t[bad]
        i0 = np.searchsorted(bad_t, self.t)
        i1 = np.searchsorted(bad_t, np.nan_to_num(self.t_bottom, nan=np.inf))
        self.ok = (
            self.valid
            & (self.t >= self._clock_t[0])
            & np.isfinite(self.t_top)
            & np.isfinite(self.t_bottom)
            & (i0 == i1)
        )

    def _ratio(self, w: np.ndarray) -> float:
        fer = float((w * self.ferrous).sum())
        return 1000.0 * float((w * self.coke).sum()) / fer if fer > 0 else float("nan")

    def melting(self, start: Any, end: Any) -> tuple[float, float]:
        """(coke:ferrous of the parcels melting in [start, end), kg/t; share of the expected melting weight)."""
        if self.empty:
            return float("nan"), 0.0
        a, b = _ns([start])[0], _ns([end])[0]
        lo = np.searchsorted(self.t, a - 24 * _HOUR_NS)
        hi = np.searchsorted(self.t, b)
        t1, t2, ok = self.t_top[lo:hi], self.t_bottom[lo:hi], self.ok[lo:hi]
        span = np.maximum(np.nan_to_num(t2 - t1, nan=1.0), 1.0)
        w = (
            np.clip(
                np.minimum(np.nan_to_num(t2, nan=-np.inf), b) - np.maximum(np.nan_to_num(t1, nan=np.inf), a), 0, None
            )
            / span
        )
        w = np.where(ok, w, 0.0)
        fer = self.ferrous[lo:hi]
        tot = float((w * fer).sum())
        ratio = 1000.0 * float((w * self.coke[lo:hi]).sum()) / tot if tot > 0 else float("nan")
        cover = float(w.sum()) / (self.settings.parcels_per_hour * (b - a) / _HOUR_NS)
        return ratio, cover

    def charged(self, start: Any, end: Any) -> float:
        """Coke:ferrous (kg/t) of the parcels dumped in [start, end)."""
        if self.empty:
            return float("nan")
        a, b = _ns([start])[0], _ns([end])[0]
        lo, hi = np.searchsorted(self.t, a), np.searchsorted(self.t, b)
        w = self.valid[lo:hi].astype(float)
        fer = float((w * self.ferrous[lo:hi]).sum())
        return 1000.0 * float((w * self.coke[lo:hi]).sum()) / fer if fer > 0 else float("nan")

    def k_tracked(self, rows_start: Any, rows_end: Any, k_charged: float) -> float:
        """K_tracked of the window of hourly rows [rows_start, rows_end) with the window's K_charged; NaN when the
        record does not cover the melting of the window."""
        s = self.settings
        start, end = pd.Timestamp(rows_start), pd.Timestamp(rows_end)
        r_melt, cover = self.melting(
            start - pd.Timedelta(hours=s.tag_offset_h), end - pd.Timedelta(hours=s.tag_offset_h)
        )
        r_chg = self.charged(start - pd.Timedelta(hours=s.charge_offset_h), end - pd.Timedelta(hours=s.charge_offset_h))
        if not (np.isfinite(k_charged) and np.isfinite(r_melt) and np.isfinite(r_chg) and r_chg > 0):
            return float("nan")
        if cover < s.min_cover:
            return float("nan")
        return float(k_charged * r_melt / r_chg)

    def ages_h(self) -> dict[str, float]:
        """Median ages of the tracked parcels at the cohesive-zone top and bottom, h."""
        if self.empty or not self.ok.any():
            return {"cz_top": float("nan"), "cz_bottom": float("nan")}
        return {
            "cz_top": float(np.nanmedian(self.age_top_h[self.ok])),
            "cz_bottom": float(np.nanmedian(self.age_bottom_h[self.ok])),
        }


def tracked_for_windows(
    fd: pd.DataFrame,
    starts: pd.Series,
    k_charged: pd.Series,
    settings: TrackingSettings | None = None,
    hours: float = 8.0,
) -> pd.Series:
    """K_tracked per window: ``starts`` (index = key) gives the first hourly row of each window of ``hours``."""
    tr = Tracker(fd, settings)
    out = {}
    for key, start in starts.items():
        k = k_charged.get(key, np.nan)
        out[key] = tr.k_tracked(
            start, pd.Timestamp(start) + pd.Timedelta(hours=hours), float(k) if pd.notna(k) else np.nan
        )
    return pd.Series(out, dtype=float)
