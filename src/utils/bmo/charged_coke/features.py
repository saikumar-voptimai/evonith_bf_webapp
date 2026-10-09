"""Causal hourly features for the charged-coke model, as it was trained.

Line-for-line port of the research package's ``data.build`` and
``screening.bounds``, with the study's fixed dates replaced by parameters:

1. Reindex the source onto a complete hourly clock from ``start - warmup`` to
   ``end``. Missing hours stay missing; nothing is zero-filled.
2. Drop the reported coke rate, reported fuel rate and unit cost before any
   other step (``FORBIDDEN``); they cannot reach a feature, label or gate.
3. Screen every column against its plausibility range (``bounds``); values
   outside become missing.
4. Quality gates: observed / frozen feed / ETA and pressure identities /
   valid / core envelope / four-hour stability, and the charge-mass check.
5. Delay laboratory data: hot metal and slag 3 h (forward fill at most 2 h),
   raw-material assays 24 h (forward fill at most 12 h). Masses become
   four-hour means; nut coke a four-hour kg/THM ratio.
6. Expected slag per hour from the pinned repository physics.
7. The 24 model channels, the 4-h charged-ratio labels and the gates.

A row labelled t covers the hour t:00-t+1:00 (interval start), so it is
complete at t+1:00. Every rolling window and shift is in rows of that clock.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from utils.bmo.charged_coke.physics import expected_slag

ETA = "FURNACETOPGASANALYSISCO2ETACO"
WIND = "HOT BLAST VOLUMENM3/HR."
PROD = "PRODUCTIONTONNESPERHR"
PCI = "PCI_KG/THM"
TEMP = "HOT BLAST TEMP.OC"
CO = "FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%"
CO2 = "FURNACE TOP GAS ANALYSISCO2%"
REPORTED_COKE = "COKE RATE KG/THM"
FORBIDDEN = ("COKE RATE KG/THM", "ACT. FUEL RATEKG/THM.", "UNITCOST LAKHS/THM")
DEFAULT_WARMUP = pd.Timedelta(days=8)

ONLINE = (WIND, TEMP, PCI, ETA, "HOT BLAST PRESSUREBAR", "TOPPRESSUREBAR", "FTG_UPTAKE_TEMP_AVG", "STOCKRODLEVEL")
CHANNEL_MAP = {
    "eta": ETA, "wind": WIND, "blast": TEMP, "raft": "RAFTOC", "oxygen": "O2 ENRICHMENT %",
    "pressure": "DIFFERENTIAL PRESSURETOTALBAR", "top_temp": "FTG_UPTAKE_TEMP_AVG",
    "hm_si": "CHEM_PCT_SI", "hm_ti": "CHEM_PCT_TI", "hm_temp": "HMT_GT_1480C",
    "ore_fe": "ORE_FE(T)%", "sinter_fe": "SINTER_FE(T)%", "coke_ash": "COKE_ASH%",
    "pci_ash": "PCI_ASH%", "csr": "COKE_CSR", "cri": "COKE_CRI",
}
CHANNELS = ["charge4", "charge24", "pci", "nut", "slag", "energy", *CHANNEL_MAP]
_ESSENTIAL = [
    PROD, PCI, "nut", "ORE_FE(T)%", "SINTER_FE(T)%", "ORE_AL2O3%", "ORE_SIO2%",
    "SINTER_AL2O3%", "SINTER_SIO2%", "ORE_CALC_MT", "SINTER_CALC_MT",
]

_EXACT_BOUNDS: dict[str, tuple[float, float | None, str]] = {
    REPORTED_COKE: (150, 800, "kg/thm"), PROD: (20, 160, "t/h"), PCI: (0, 350, "kg/thm"),
    WIND: (10000, 180000, "Nm3/h"), TEMP: (500, 1350, "deg C"), ETA: (10, 65, "%"),
    "ACT. FUEL RATEKG/THM.": (250, 1200, "kg/thm"), "HOT BLAST PRESSUREBAR": (0, 5, "bar"),
    "TOPPRESSUREBAR": (0, 3, "bar"), "DIFFERENTIAL PRESSURETOTALBAR": (0, 3, "bar"),
    "BOTTOMBAR": (0, 3, "bar"), "TOPBAR": (0, 3, "bar"), "OXYGENFLOWNM3/HR.": (0, 15000, "Nm3/h"),
    "STEAMKGS/HR.": (0, 20000, "kg/h"), "RAFTOC": (1500, 2800, "deg C"),
    "PERMEABILITYKGS/HR.": (0, 5000, "unverified index; header says kg/h"),
    "TUYEREVELOCITYM/S": (50, 350, "m/s"), "O2 ENRICHMENT %": (0, 15, "%"),
    "CHARGES/HRS.": (0.5, 15, "charges/h"), "STOCKRODLEVEL": (0, 8, "m assumed"),
    "TOTAL HEAT LOAD": (0, None, "unverified"), "HMT_GT_1480C": (1250, 1600, "deg C; not binary"),
    "UNITCOST LAKHS/THM": (0, None, "unverified; excluded"),
    "CHEM_PCT_C": (2, 5.5, "%"), "CHEM_PCT_FE": (85, 99, "%"), "CHEM_PCT_SI": (0, 3, "%"),
    "CHEM_PCT_TI": (0, 1, "%"), "CHEM_PCT_S": (0, 0.5, "%"), "CHEM_PCT_P": (0, 1, "%"),
    "CHEM_PCT_MN": (0, 2, "%"), "CHEM_PCT_CR": (0, 1, "%"), "SINTER_BASICITY": (0.2, 4, "ratio"),
    "SLAG_BASICITY": (0.4, 2, "ratio"), "SLAG_T_BASICITY": (0.5, 2.5, "ratio"),
    "COKE_CALC_MT": (0, 100, "t/h assumed"), "NUTCOKE_CALC_MT": (0, 30, "t/h assumed"),
    "PCI_CALC_MT": (0, 45, "t/h assumed"), "SINTER_CALC_MT": (0, 250, "t/h assumed"),
    "ORE_CALC_MT": (0, 200, "t/h assumed"), "TOTAL_PELLET_CALC_MT": (0, 100, "t/h assumed"),
    "FLUX_CALC_MT": (0, 30, "t/h assumed"),
}


def bounds(column: str) -> tuple[float, float | None, str]:
    """Plausibility range of a source column (screening, not alarm limits)."""

    if column in _EXACT_BOUNDS:
        return _EXACT_BOUNDS[column]
    if re.match(r"(ORE|FLUX)_\d+_CALC_MT", column):
        return (0, 200, "t/h assumed")
    if column.startswith("HEARTH_TEMP"):
        return (10, 1200, "deg C; wall sensor")
    if any(k in column for k in ("BOSH_TEMP", "BELLY_TEMP", "LOWER_STACK_TEMP")):
        return (10, 700, "deg C; wall sensor")
    if column.startswith("FTG_UPTAKE_TEMP"):
        return (40, 700, "deg C")
    if "ANGLE" in column:
        return (0, 90, "degrees")
    if "DISCHARGE_TIME" in column:
        return (0, 600, "seconds assumed")
    if "PORTIONS" in column:
        return (0, 100, "count")
    if "%" in column or "PCT" in column or "STRENGTH" in column or column in (
        "COKE_M-40", "COKE_M-10", "COKE_CSR", "COKE_CRI",
    ):
        return (0, 100, "%")
    raise ValueError("Unclassified column " + column)


def _is_hot_metal(column: str) -> bool:
    return column.startswith(("CHEM_", "SLAG_")) or column == "HMT_GT_1480C"


def _is_material(column: str) -> bool:
    return (
        "%" in column or "PCT" in column or "STRENGTH" in column or column.startswith("COKE_M-")
        or column in ("COKE_CSR", "COKE_CRI", "SINTER_BASICITY")
    ) and column not in (ETA, CO, CO2, "O2 ENRICHMENT %", "FURNACE TOP GAS ANALYSISH2%")


@dataclass(frozen=True)
class FeatureSet:
    """Hourly frames on the complete clock ``start..end`` (warm-up removed).

    Attributes:
        raw: Source values, numeric, reindexed onto the complete clock,
            forbidden columns removed. Used only to tell a missing hour from
            a screened value.
        cleaned: Screened source without the forbidden columns.
        causal: Delayed laboratory data and four-hour masses.
        physics: Expected slag and its parts.
        channels: The 24 model channels in training order.
        labels: ``current4`` (anchor) and ``future4`` (target) charged ratios.
        gates: Quality and regime flags.
    """

    raw: pd.DataFrame
    cleaned: pd.DataFrame
    causal: pd.DataFrame
    physics: pd.DataFrame
    channels: pd.DataFrame
    labels: pd.DataFrame
    gates: pd.DataFrame


def build_features(
    source: pd.DataFrame,
    physics_config: Mapping[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup: pd.Timedelta = DEFAULT_WARMUP,
) -> FeatureSet:
    """Build the model's causal features for every hour in ``start..end``.

    Args:
        source: Hourly furnace dataset indexed by interval-start time (plant
            clock), unique timestamps.
        physics_config: Frozen physics configuration from the model bundle.
        start: First hour returned.
        end: Last hour returned (the latest complete hour for live use).
        warmup: History read before ``start`` for delays, forward fills and
            rolling windows; the research build used eight days.

    Returns:
        The feature set restricted to ``start..end``.
    """

    if source.index.has_duplicates:
        raise ValueError("Duplicate timestamps")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    clock = pd.date_range(start - warmup, end, freq="h", name="time")
    raw = source.sort_index().reindex(clock).apply(pd.to_numeric, errors="coerce")
    d = raw.drop(columns=list(FORBIDDEN), errors="ignore").copy()
    for column in raw:
        lo, hi, _unit = bounds(column)
        bad = raw[column].lt(lo) | raw[column].gt(hi if hi is not None else np.inf)
        if column in d:
            d.loc[bad, column] = np.nan

    online = list(ONLINE)
    observed = raw[online].notna().all(axis=1)
    frozen = raw[online].diff().abs().lt(1e-9).all(axis=1) & observed
    eta_bad = (d[ETA] - 100 * d[CO2] / (d[CO] + d[CO2])).abs().gt(1)
    dp_bad = (d["HOT BLAST PRESSUREBAR"] - d["TOPPRESSUREBAR"] - d["DIFFERENTIAL PRESSURETOTALBAR"]).abs().gt(0.15)
    valid = (
        observed & ~frozen & ~eta_bad & ~dp_bad & d[PROD].between(50, 140) & d[WIND].between(60000, 150000)
        & d[PCI].between(20, 350) & d[TEMP].between(800, 1350) & (d[CO] + d[CO2]).between(35, 50)
    )
    core = (
        valid & d[ETA].gt(41) & d[WIND].ge(105000) & d[PCI].between(150, 250) & d[TEMP].ge(1150)
        & d[PROD].between(80, 110) & d.RAFTOC.between(2100, 2450) & (d[CO] + d[CO2]).between(38, 47)
    )
    stable = core.rolling(4).sum().eq(4)
    for column, limit in ((WIND, 0.02), (PROD, 0.03), (PCI, 0.08)):
        stable &= d[column].rolling(4).std().div(d[column].rolling(4).mean()).le(limit)
    for column, limit in ((TEMP, 40), (ETA, 1.5)):
        stable &= d[column].rolling(4).max().sub(d[column].rolling(4).min()).le(limit)

    ratio = 1000 * d.COKE_CALC_MT / d[PROD]
    mass_ok = valid & ratio.between(150, 700) & d.COKE_CALC_MT.gt(0)
    coke = d.COKE_CALC_MT.where(mass_ok)
    prod = d[PROD].where(mass_ok)
    anchor = 1000 * coke.rolling(4).sum() / prod.rolling(4).sum()
    labels = pd.DataFrame({"current4": anchor, "future4": anchor.shift(-5)}, index=d.index)

    causal = d.copy()
    for column in d:
        hot_metal, material = _is_hot_metal(column), _is_material(column)
        if hot_metal or material:
            causal[column] = d[column].shift(3 if hot_metal else 24).ffill(limit=2 if hot_metal else 12)
    causal = causal.copy()  # defragment after the column-by-column writes
    causal["nut"] =1000 * d.NUTCOKE_CALC_MT.rolling(4).sum() / d[PROD].rolling(4).sum()
    for column in ("ORE_CALC_MT", "SINTER_CALC_MT", "TOTAL_PELLET_CALC_MT", "FLUX_CALC_MT", "CHARGES/HRS.", PROD):
        causal[column] = d[column].rolling(4).mean()

    rows = []
    for t, r in causal.iterrows():
        pellet_missing = (r["TOTAL_PELLET_CALC_MT"] > 0.01) and (
            not np.isfinite(r["PELLET_PCT_FE2O3"]) or r["PELLET_PCT_FE2O3"] < 10
        )
        if not valid.loc[t] or not np.isfinite(r[_ESSENTIAL].to_numpy(dtype=float)).all() or pellet_missing:
            rows.append({})
            continue
        rows.append(expected_slag(r, physics_config))
    phys = pd.DataFrame(rows, index=d.index).replace([np.inf, -np.inf], np.nan)
    if "slag" not in phys:
        phys["slag"] = np.nan
    phys["slag"] = phys.slag.where(phys.slag.between(150, 700))

    ch = pd.DataFrame(index=d.index)
    ch["charge4"] = anchor
    ch["charge24"] = 1000 * coke.rolling(24, min_periods=18).sum() / prod.rolling(24, min_periods=18).sum()
    ch["pci"] = d[PCI]
    ch["nut"] = causal.nut
    ch["slag"] = phys.slag
    # The research energy expert; not an input of the structured model.
    ch["energy"] = np.nan
    for name, column in CHANNEL_MAP.items():
        ch[name] = causal[column]
    burden = causal[["ORE_CALC_MT", "SINTER_CALC_MT", "TOTAL_PELLET_CALC_MT"]].sum(axis=1, min_count=3)
    ch["sinter_share"] = causal.SINTER_CALC_MT / burden
    ch["pellet_share"] = causal.TOTAL_PELLET_CALC_MT / burden

    g = pd.DataFrame(
        {"observed": observed, "frozen": frozen, "eta_bad": eta_bad, "dp_bad": dp_bad, "valid": valid,
         "mass_ok": mass_ok, "core": core, "stable": stable},
        index=d.index,
    )
    g["usable"] = (
        valid & valid.rolling(12).mean().ge(0.75) & anchor.notna() & phys.slag.notna()
        & ch.pci.notna() & ch.nut.notna()
    )
    g["regime"] = np.select([~g.usable, g.stable, g.core], ["invalid", "stable", "unsettled"], default="outside_core")

    window = slice(start, end)
    return FeatureSet(
        raw=raw.drop(columns=list(FORBIDDEN), errors="ignore").loc[window],
        cleaned=d.loc[window], causal=causal.loc[window],
        physics=phys.loc[window], channels=ch.loc[window], labels=labels.loc[window], gates=g.loc[window],
    )


__all__ = [
    "CHANNELS", "DEFAULT_WARMUP", "FORBIDDEN", "FeatureSet", "PCI", "PROD", "WIND", "bounds",
    "build_features",
]
