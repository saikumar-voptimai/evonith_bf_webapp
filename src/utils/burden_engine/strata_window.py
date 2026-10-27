"""STRATA-omega of an observed window against its 14-day anchor (the twin's K_STRATA, SV-1.1, #281).

The STRATA 1.5 posterior was fitted on the change of each day's omega against its 14-day anchor, with the network
channels evaluated at four states (asset ``channel_scales.channels``, strict test protocol GAS-25):

    A   anchor: blend and operation of the stable shifts dated in [d - 14 d, d), omega_A = their median omega
    B   the window's blend at the anchor's operation (ferrous rate of the window)
    P   B at the window's production
    F   the window's blend and operation (coke as charged, nut coke, PCI, production, blast, O2, blast moisture)

    omega_STRATA = omega_A + [w(B) - w(A)] + sum_c (lam_g - 1) g_c (s_c,W - s_c,A)        blend channel (scaled)
                           + lam_rest [w(F) - w(P)] + 0 x [w(P) - w(B)] + a_n (nut_W - nut_A)

which is ``omega_model.PlanOmega`` with the anchor as its calibration and the window as its plan, evaluated at the
window's charged coke. The network context (charging matrix, top gas, hot-blast temperature) is the anchor's, the
sinter RDI the latest strength test before d for A and the window alike (as the engine holds it at its calibration),
the capacity ratio R the anchor's median (R = (1 - r_d) / omega of each calibrated shift). kappa is refitted so that
w(A, R_A) = omega_A.

The model runs in the exact mode of ``strata_network``: every state is a network solve (A once per anchor, B, P and
F once per window), no Jacobian. Ores are mapped to the model's materials by the asset's ``plant_map`` from the
charge of each ore type (``ORE_k_CALC_MT``); a row without that split takes the asset's reference lump-ore split.

Pure numerical code: takes the pipeline's shift frames (S inputs, R balances, Z two-zone calibrations); no I/O except
the omega model's JSON constants.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.burden_engine.calibration import network_context, sinter_strength
from utils.burden_engine.omega_model import OmegaModel, OmegaState, OmegaTerms
from utils.burden_engine.strata_network import NET_VERSION, solve_stats

ORE_COLUMN = re.compile(r"ORE_(\d+)_CALC_MT")


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _pos(x: Any) -> float:
    """A charged mass: the value when positive, else 0 (missing or negative entries)."""
    v = _num(x)
    return v if v > 0 else 0.0


def ore_split(row: Mapping[str, Any], model: OmegaModel) -> tuple[dict[str, float], list[str]]:
    """Share of the ore charge by model material, from the charge of each ore type (``ORE_k_CALC_MT``) and the asset's
    ``plant_map``; the reference lump-ore split when the row has no positive ore-type charge. Also returns the ore
    types charged that the plant map does not name (taken as the default lump ore)."""
    plant = model.spec.get("plant_map") or {}
    default = model.kind_default.get("ore", "lloyds_clo")
    kg: dict[str, float] = {}
    unmapped: list[str] = []
    for col, value in row.items():
        m = ORE_COLUMN.fullmatch(str(col))
        v = _pos(value)
        if m is None or v <= 0:
            continue
        name = str(plant.get(f"ore_{m.group(1)}", "")).split(" ")[0]
        if name not in model.materials:
            unmapped.append(f"ore_{m.group(1)}")
            name = default
        kg[name] = kg.get(name, 0.0) + v
    total = sum(kg.values())
    if total <= 0:
        return {k: float(v) for k, v in model.spec["reference"]["lump_split"].items()}, []
    return {k: v / total for k, v in kg.items()}, sorted(set(unmapped))


def omega_state(
    row: Mapping[str, Any], model: OmegaModel, *, humidity: float, rdi: float
) -> tuple[OmegaState, list[str]]:
    """The omega state of a row of shift inputs (``build_shift_inputs``: a window, or the mean of the anchor's shifts).

    Per tonne of hot metal (the row's production): wet charge of sinter, pellet and each ore as charged, dry coke
    (``COKE_DRY``), nut coke and PCI as charged; production in t/h; hot blast incl. O2 per tonne (meter basis) and the
    O2 fraction of the hot blast from the 10-min means. Returns the state and the unmapped ore types."""
    hm = _num(row.get("PRODUCTIONTONNESPERHR"))
    hours = _num(row.get("n_hours"))
    if not hm > 0 or not hours > 0:
        raise ValueError("no hot metal in the row")
    k = 1000.0 / hm
    wet = {"sinter": _pos(row.get("SINTER_CALC_MT")) * k, "pellet": _pos(row.get("TOTAL_PELLET_CALC_MT")) * k}
    split, unmapped = ore_split(row, model)
    ore = _pos(row.get("ORE_CALC_MT")) * k
    for name, f in split.items():
        wet[name] = wet.get(name, 0.0) + ore * f
    sh, fe, cls = model.blend_parts(wet)
    coke = _pos(row.get("COKE_DRY")) * k
    hbv, o2 = _num(row.get("hot_blast_vol_nm3h")), _num(row.get("oxygen_flow"))
    if coke <= 0:
        raise ValueError("no coke charged in the row")
    if not (hbv > 0 and o2 >= 0):
        raise ValueError("no hot-blast or oxygen flow in the row")
    prod = hm / hours
    state = OmegaState(
        shares=sh,
        fe_frac=fe,
        cls=cls,
        coke=coke,
        nut=_pos(row.get("NUTCOKE_CALC_MT")) * k,
        pci=_pos(row.get("PCI_CALC_MT")) * k,
        prod=prod,
        blast_t=hbv / prod,
        o2=o2 / hbv,
        rdi=float(rdi),
        fer=sum(wet.values()),
        humidity=float(humidity),
    )
    return state, unmapped


def anchor_keys(R: pd.DataFrame, stable: pd.Series, date: Any, window_days: int) -> list[str]:
    """Keys of the stable shifts dated in [date - window_days, date): the shifts of the pipeline's 14-day medians."""
    d = pd.to_datetime(R["sdate"])
    day = pd.Timestamp(date).normalize()
    sel = stable.reindex(R.index).fillna(False).astype(bool) & (d >= day - pd.Timedelta(days=window_days)) & (d < day)
    return list(R.index[sel])


@dataclass
class StrataAnchor:
    """The calibrated omega model of one 14-day anchor (state A) and what it was built from."""

    date: pd.Timestamp
    keys: tuple[str, ...]
    omega: float
    R: float
    state: OmegaState
    model: OmegaModel
    context: dict[str, float] = field(default_factory=dict)
    strength: dict[str, Any] = field(default_factory=dict)
    unmapped: tuple[str, ...] = ()

    @property
    def version(self) -> str:
        return self.model.version.split(" (")[0]


def build_anchor(
    S: pd.DataFrame,
    R: pd.DataFrame,
    Z: pd.DataFrame,
    keys: Iterable[str],
    omega: float,
    date: Any,
    model: OmegaModel | None = None,
) -> StrataAnchor:
    """Anchor A on the given shifts (the pool of the pipeline's 14-day medians) with omega_A = ``omega``.

    State: the mean of the shifts' input rows (production-weighted per tonne, as ``calibrate_from_shifts`` averages
    them); blast moisture the mean of their balances; capacity ratio the median of (1 - r_d) / omega of their two-zone
    calibrations. ``model`` (default: the shipped asset) is configured and calibrated in place."""
    keys = [k for k in keys if k in S.index and k in R.index]
    if not keys:
        raise ValueError("no anchor shifts")
    day = pd.Timestamp(date).normalize()
    m = model or OmegaModel.load()
    ctx = network_context(S, keys)
    strength = sinter_strength(S, day)
    m.set_network_context(ctx)
    m.set_sinter_strength(strength)
    row = S.loc[keys].select_dtypes("number").mean()
    humidity = float(pd.to_numeric(R.loc[keys, "humidity_g"], errors="coerce").mean())
    state, unmapped = omega_state(row, m, humidity=humidity, rdi=m.rdi)
    z = Z.reindex(keys)
    r_cap = (1.0 - pd.to_numeric(z.get("r_d"), errors="coerce")) / pd.to_numeric(z.get("omega"), errors="coerce")
    R_A = float(np.nanmedian(r_cap.to_numpy(dtype=float))) if r_cap.notna().any() else float("nan")
    if not (math.isfinite(R_A) and 0.3 < R_A < 1.5):
        raise ValueError(f"capacity ratio of the anchor {R_A:.3f} outside 0.3-1.5")
    m.calibrate(state, float(omega), R_A, jacobian=False)
    return StrataAnchor(day, tuple(keys), float(omega), R_A, state, m, ctx, strength, tuple(unmapped))


@dataclass
class StrataWindow:
    """omega_STRATA of one window, its 1 sigma (parameter uncertainty relative to the anchor, as page 11 reports it)
    and the parts; ``solves`` / ``seconds``: network solves made for the window's states (0 when they were solved
    before)."""

    omega: float
    sigma: float
    terms: OmegaTerms
    solves: int
    seconds: float
    unmapped: tuple[str, ...] = ()

    def channels(self) -> dict[str, float]:
        t = self.terms
        return dict(
            strata_blend_channel=float(t.d_blend + t.d_disc),
            strata_operation_channel=float(t.d_rest),
            strata_production_physics=float(t.d_prod_phys),
            strata_nut_coke_term=float(t.d_nut),
            strata_blend_physics=float(t.d_blend_phys),
        )


def evaluate_window(anchor: StrataAnchor, row: Mapping[str, Any], *, humidity: float) -> StrataWindow:
    """omega_STRATA of a window row (``build_shift_inputs`` row of the window) against ``anchor``."""
    m = anchor.model
    state, unmapped = omega_state(row, m, humidity=humidity, rdi=m.rdi)
    n0, s0 = solve_stats()
    plan = m.plan(state)
    terms = plan.terms(state.coke, anchor.R)
    n1, s1 = solve_stats()
    return StrataWindow(float(terms.omega), float(terms.sigma), terms, n1 - n0, s1 - s0, tuple(unmapped))


def versions(anchor: StrataAnchor) -> dict[str, str]:
    return dict(strata=anchor.version, strata_network=NET_VERSION)
