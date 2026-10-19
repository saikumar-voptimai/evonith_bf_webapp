"""How old is the data behind a number? Freshness limits per source and the check against them.

Lab results, charge reports and the DPR arrive late and irregularly, and the hourly dataset carries the last value
forward until the next one arrives. The plant's rule (15 Oct 2026): when a value rests on a data point older than its
source's limit, say so and name that last valid data point.

    hot metal & slag analysis   3 h       raw-material chemistry     1 day     (ore, sinter, fuel, flux; per material)
    DPR                         1 day     size analysis              1 day     (per material; pellets excluded - fixed size)
    raw-material strength       1 day     coke / nut coke / PCI ash  15 days
    burden distribution         1 month

Limits for the remaining sources (10-min process data, the hourly dataset, charge reports, dust analyses) are this
page's choice and live in ``coke_energy.yml`` with the others.

Ages are measured to a REFERENCE TIME: the end of the shift being shown (what the balance of that shift could use) or
now (the status line). Daily records (DPR, ash and dust analyses) are stamped at the end of their date, so a day's
report counts as current until the following day ends.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceSpec:
    label: str
    used_by: str                 # which part of the page rests on it
    feeds_models: bool           # True: a stale value changes the coke rates shown
    per_material: bool = False


SOURCES: dict[str, SourceSpec] = {
    "hourly_dataset": SourceSpec("Hourly charges & composition", "both models", True),
    "online_10min": SourceSpec("10-min process data", "both models", True),
    "hm_slag": SourceSpec("Hot metal & slag analysis", "both models", True),
    "dpr": SourceSpec("DPR (hot metal, dust)", "both models", True),
    "rm_chemistry": SourceSpec("Raw-material chemistry", "both models", True, per_material=True),
    "ash_chemistry": SourceSpec("Fuel ash chemistry", "both models", True, per_material=True),
    "dust_basic": SourceSpec("Dust Fe & carbon", "both models", True, per_material=True),
    "charge_data": SourceSpec("Charge reports", "materials in use", False),
    "fines": SourceSpec("Size analysis", "ω model (planned)", False, per_material=True),
    "rm_strength": SourceSpec("Coke & sinter strength", "ω model (planned)", False, per_material=True),
    "burden_distribution": SourceSpec("Burden distribution", "ω model (planned)", False),
    "dust_chemistry": SourceSpec("Dust oxides", "not used", False, per_material=True),
}

DEFAULT_LIMITS = {
    "hourly_dataset": "3h", "online_10min": "30min", "hm_slag": "3h", "dpr": "1D", "rm_chemistry": "1D",
    "ash_chemistry": "15D", "dust_basic": "1D", "charge_data": "2h", "fines": "1D", "rm_strength": "1D",
    "burden_distribution": "30D", "dust_chemistry": "15D",
}

# Which ash analysis covers which charged fuel.
ASH_TYPE = {"coke": "coke", "nut_coke": "nutcoke", "pci": "pci"}


def parse_limits(cfg: Mapping[str, str] | None) -> dict[str, pd.Timedelta]:
    """Limits from config strings such as '3h', '1D', '30min', '15D'; missing sources take the defaults."""

    merged = {**DEFAULT_LIMITS, **{k: v for k, v in (cfg or {}).items() if v}}
    return {k: pd.Timedelta(str(v).replace("d", "D")) for k, v in merged.items()}


def materials_for(source: str, in_use: Collection[str], known: Collection[str], pci_used: bool = True) -> set[str]:
    """The materials of ``source`` that matter, given the material codes charged in the window.

    ``known`` are the materials that source has ever reported (strength is tested for coke_1 and sinter only, for
    example). Pellets are left out of the size check: their size is fixed (5 mm screen)."""

    in_use = set(in_use) | ({"pci_1"} if pci_used else set())
    if source == "rm_chemistry":
        return in_use
    if source == "fines":
        return {m for m in in_use if not m.startswith("pellet") and m != "pci_1"} & set(known)
    if source == "rm_strength":
        return in_use & set(known)
    if source == "ash_chemistry":
        return {ash for prefix, ash in ASH_TYPE.items() if any(m.startswith(prefix) for m in in_use)}
    if source in ("dust_basic", "dust_chemistry"):
        return {"dust_1", "dust_2"}
    return {""}


def assess(samples: pd.DataFrame, reference: pd.Timestamp, limits: Mapping[str, pd.Timedelta],
           in_use: Collection[str], *, pci_used: bool = True, sources: Collection[str] | None = None) -> pd.DataFrame:
    """Last valid data point at or before ``reference`` for every source (and material), its age and status.

    ``samples``: columns source, material ('' for single-series sources), t (IST, tz-naive), one row per valid record.
    Returns one row per source/material with: source, label, material, last, age, limit, stale, used_by, feeds_models.
    """

    reference = pd.Timestamp(reference)
    rows = []
    groups = {key: g["t"].to_numpy() for key, g in samples.groupby(["source", "material"])}
    known = samples.groupby("source")["material"].agg(lambda s: set(s.unique())).to_dict()
    for source, spec in SOURCES.items():
        if sources is not None and source not in sources:
            continue
        if source not in known and not spec.per_material:
            mats = {""}
        else:
            mats = materials_for(source, in_use, known.get(source, set()), pci_used) if spec.per_material else {""}
        for material in sorted(mats):
            times = groups.get((source, material))
            last = None
            if times is not None and len(times):
                times = np.sort(times)
                i = np.searchsorted(times, np.datetime64(reference), side="right")
                if i > 0:
                    last = pd.Timestamp(times[i - 1])
            age = None if last is None else reference - last
            limit = limits[source]
            rows.append(dict(source=source, label=spec.label, material=material, last=last, age=age, limit=limit,
                             stale=last is None or age > limit, used_by=spec.used_by, feeds_models=spec.feeds_models))
    return pd.DataFrame(rows)


def in_use_between(charged: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> set[str]:
    """Material codes with charged tonnes in [start, end). ``charged`` is hourly (IST hour index), one column per code."""

    if charged.empty:
        return set()
    window = charged[(charged.index >= start) & (charged.index < end)].drop(columns=["charges"], errors="ignore")
    totals = window.sum(numeric_only=True)
    return {m for m, v in totals.items() if v > 0}


def format_age(age: pd.Timedelta | None) -> str:
    if age is None or pd.isna(age):
        return "no data"
    hours = age / pd.Timedelta(hours=1)
    if hours < 0:
        return "current"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def format_limit(limit: pd.Timedelta) -> str:
    hours = limit / pd.Timedelta(hours=1)
    return f"{hours:.0f} h" if hours < 24 else (f"{hours / 24:.0f} day" + ("s" if hours >= 48 else ""))
