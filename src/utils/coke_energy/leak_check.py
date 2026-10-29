"""Leak-water check on the burnt-coke record: a step in (charged − raw burnt) coke and in top-gas H₂, RAFT unchanged.

Cooling water from a failed stave that reaches the burden is gasified on coke (C + H₂O → CO + H₂). The oxygen balance
has no term for the water's oxygen, so the raw burnt coke (TRACE before the 14-day correction) reads low while the
charged coke follows the furnace's real need: (charged − raw burnt) steps up. Part of the water's hydrogen leaves as
H₂ in the top gas. More blast moisture would raise both as well, but it also lowers RAFT; a leak does not. The
leak-detection model is ``utils.strata_flow.plant_effects`` (``leak_water``, ``leak_for_gap``). A steady leak is
absorbed by the 14-day correction, so a new one shows as a step against the days before.

The check compares the shifts of the last ``recent_days`` plant days with those of the ``baseline_days`` days before
(step = difference of the means, z = step / its standard error, Welch) and asks the model which H₂ rise the coke step
implies. Pure functions on the Coke Energy pipeline's shift table with the top-gas H₂ (``h2_pct``, % dry) and RAFT
(``raft``, °C) of each shift added; no physics here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from utils.strata_flow.plant_effects import leak_for_gap

#: Signal -> (label, unit). ``gap`` is charged − raw burnt coke; ``moisture`` is context for the verdict only.
SIGNALS: dict[str, tuple[str, str]] = {
    "gap": ("Charged − raw burnt coke", "kg/tHM"),
    "h2": ("Top-gas H₂", "%"),
    "raft": ("RAFT", "°C"),
    "moisture": ("Blast moisture", "g/Nm³"),
}

#: Verdict -> wording. ``leak`` matches the model's signature; the others say what the steps point to instead.
VERDICTS: dict[str, str] = {
    "leak": "Leak signature: coke gap and H₂ rise as the model expects, RAFT unchanged",
    "moisture": "H₂ rise with a RAFT drop: blast moisture or steam, not a leak",
    "gap_only": "Coke gap without an H₂ rise: blast meter, analyser or charge data",
    "mixed": "Coke gap and H₂ rise that do not match a leak (H₂ below the model or RAFT changed)",
    "h2_only": "H₂ rise without a coke gap",
    "none": "No step in the coke gap or the top-gas H₂",
    "unknown": "Not enough shifts with raw burnt coke and top-gas H₂ to compare",
}


@dataclass(frozen=True)
class LeakCheckSettings:
    """Windows and thresholds of the check (``coke_energy.yml → twin.burnt_vs_charged.leak_check``)."""

    recent_days: int = 3
    baseline_days: int = 14
    min_recent_shifts: int = 6
    min_baseline_shifts: int = 21
    z_min: float = 3.0
    gap_min_kg_thm: float = 3.0
    h2_min_pt: float = 0.15
    h2_share_min: float = 0.5
    raft_max_c: float = 15.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> LeakCheckSettings:
        names = {f.name: f.type for f in fields(cls)}
        values = {k: v for k, v in (cfg or {}).items() if k in names and v is not None}
        return cls(**{k: (int(v) if names[k] == "int" else float(v)) for k, v in values.items()})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def signals(T: pd.DataFrame) -> pd.DataFrame:
    """The check's signals per shift (index = shift key): gap = K_charged − K_burnt_raw, h2 = h2_pct, raft, moisture =
    humidity_g, with ``sdate``, ``t_mid`` and the production and analyser ηCO the model is evaluated at."""

    def col(name: str) -> pd.Series:
        return T[name].astype(float) if name in T else pd.Series(np.nan, index=T.index)

    out = pd.DataFrame(index=T.index)
    out["sdate"] = pd.to_datetime(T["sdate"]).dt.normalize() if "sdate" in T else pd.NaT
    out["t_mid"] = T["t_mid"] if "t_mid" in T else pd.NaT
    out["gap"] = col("K_charged") - col("K_burnt_raw")
    out["h2"] = col("h2_pct")
    out["raft"] = col("raft")
    out["moisture"] = col("humidity_g")
    out["prod_th"] = col("prod_th")
    out["eta_co_pct"] = col("eta_meas")
    return out.sort_values("t_mid") if "t_mid" in T else out


def step(recent: pd.Series, baseline: pd.Series) -> dict[str, Any]:
    """Difference of the means (recent − baseline), its standard error (Welch) and z; NaN below two values a side."""

    r, b = recent.dropna().astype(float), baseline.dropna().astype(float)
    out = dict(n_recent=int(len(r)), n_baseline=int(len(b)), recent=np.nan, baseline=np.nan, step=np.nan,
               se=np.nan, z=np.nan)  # fmt: skip
    if len(r):
        out["recent"] = float(r.mean())
    if len(b):
        out["baseline"] = float(b.mean())
    if len(r) < 2 or len(b) < 2:
        return out
    out["step"] = out["recent"] - out["baseline"]
    se = float(np.sqrt(r.var(ddof=1) / len(r) + b.var(ddof=1) / len(b)))
    out["se"] = se
    out["z"] = out["step"] / se if se > 0 else np.nan
    return out


def periods(S: pd.DataFrame, settings: LeakCheckSettings) -> tuple[pd.Series, pd.Series, pd.Timestamp | None]:
    """Masks of the recent and the baseline shifts and the first recent day. The recent days end on the last plant day
    with a coke gap; the baseline is the ``baseline_days`` days before them."""

    have = S["gap"].notna() & S["sdate"].notna()
    if not have.any():
        none = pd.Series(False, index=S.index)
        return none, none, None
    last = S.loc[have, "sdate"].max()
    first = last - pd.Timedelta(days=settings.recent_days - 1)
    base_first = first - pd.Timedelta(days=settings.baseline_days)
    recent = (S["sdate"] >= first) & (S["sdate"] <= last)
    baseline = (S["sdate"] >= base_first) & (S["sdate"] < first)
    return recent, baseline, first


def _finite(x: Any) -> bool:
    return x is not None and np.isfinite(x)


def model_for(gap_step: float, S_recent: pd.DataFrame) -> dict[str, float] | None:
    """The leak that would explain a coke-gap step (``plant_effects.leak_for_gap``) at the recent production and
    analyser ηCO (model defaults where they are missing); None for a step that is not a rise."""

    if not _finite(gap_step) or gap_step <= 0:
        return None
    kw: dict[str, float] = {}
    prod = S_recent["prod_th"].mean()
    if _finite(prod) and prod > 0:
        kw["production_t_h"] = float(prod)
    eta = S_recent["eta_co_pct"].mean()
    if _finite(eta) and 20.0 < eta < 70.0:
        kw["eta_co"] = float(eta) / 100.0
    return {k: float(v) for k, v in leak_for_gap(float(gap_step), **kw).items()}


def verdict(steps: Mapping[str, Mapping[str, Any]], model: Mapping[str, float] | None,
            settings: LeakCheckSettings) -> str:  # fmt: skip
    """``leak``, ``moisture``, ``gap_only``, ``mixed``, ``h2_only``, ``none`` or ``unknown`` (see ``VERDICTS``)."""

    gap, h2, raft = steps["gap"], steps["h2"], steps["raft"]
    for s in (gap, h2):
        if s["n_recent"] < settings.min_recent_shifts or s["n_baseline"] < settings.min_baseline_shifts:
            return "unknown"
        if not _finite(s["z"]):
            return "unknown"
    gap_up = gap["step"] >= settings.gap_min_kg_thm and gap["z"] >= settings.z_min
    h2_up = h2["step"] >= settings.h2_min_pt and h2["z"] >= settings.z_min
    raft_step = raft["step"]
    raft_flat = None if not _finite(raft_step) else abs(raft_step) <= settings.raft_max_c
    raft_drop = _finite(raft_step) and raft_step <= -settings.raft_max_c
    h2_matches = model is not None and h2["step"] >= settings.h2_share_min * model["top_gas_h2_rise_pt"]
    if gap_up and h2_up and h2_matches and raft_flat is not False:
        return "leak"
    if h2_up and raft_drop:
        return "moisture"
    if gap_up and not h2_up:
        return "gap_only"
    if gap_up and h2_up:
        return "mixed"
    if h2_up:
        return "h2_only"
    return "none"


def check(T: pd.DataFrame, settings: LeakCheckSettings | None = None) -> dict[str, Any]:
    """The leak-water check on a shift table: the recent and baseline days, the step of every signal, the model's
    leak for the coke-gap step and the verdict; ``frame`` holds the signals of the compared shifts (``period`` recent
    or baseline), oldest first."""

    settings = settings or LeakCheckSettings()
    S = signals(T) if T is not None and not T.empty else signals(pd.DataFrame(columns=["sdate", "t_mid"]))
    recent, baseline, first = periods(S, settings)
    steps = {key: step(S.loc[recent, key], S.loc[baseline, key]) for key in SIGNALS}
    model = model_for(steps["gap"]["step"], S.loc[recent])
    state = verdict(steps, model, settings)
    frame = S.loc[recent | baseline].copy()
    frame["period"] = np.where(recent.loc[frame.index], "recent", "baseline")
    last = None if first is None else first + pd.Timedelta(days=settings.recent_days - 1)
    return dict(
        state=state,
        reason=VERDICTS[state],
        recent_first=first,
        recent_last=last,
        baseline_first=None if first is None else first - pd.Timedelta(days=settings.baseline_days),
        steps=steps,
        model=model,
        frame=frame,
    )
