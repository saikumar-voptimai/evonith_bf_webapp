"""Pure-Python element-balance computation for the Material Balance page.

No Streamlit imports — every function in this module is unit-testable.
The single entry point is :func:`run_full_balance` which orchestrates
the four day-windowed fetches and returns a :class:`BalanceResult`
dataclass holding the inputs/outputs/closure-table that the plotters
consume.

Algorithm (per day):
    1. Fetch RM averages (3-shift), HM/slag day-avg, DPR row, online avgs.
    2. Determine material masses from DPR (with RM-sum fallback).
    3. For every material, walk its composition spec, applying the
       direct/oxide/H2O/ASH/LOI rules → element tonnes.
    4. Add gas-phase inputs: blast (O+N), O2 enrichment (O), steam (H+O).
    5. Outputs: HM elements (mass × wt%), slag elements (oxide rule),
       top-gas elements (CO/CO2/H2/N2 from bosh-vol formula).
    6. Build closure table In_t / Out_t / Closure% per element.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple

import pandas as pd

from utils.material_balance.constants import (
    AIR_N2_MASS_FRAC,
    AIR_O2_MASS_FRAC,
    ATOMIC_WEIGHTS,
    ELEMENTS,
    HM_COMPOSITION,
    MATERIAL_REGISTRY,
    MOLAR_VOLUME_NTP,
    OXIDE_TO_ELEMENT_MASS_FRAC,
    RHO_AIR_NTP,
    SLAG_COMPOSITION,
    MaterialSpec,
)
from utils.material_balance.dpr_mapping import (
    apply_dpr_mapping,
    load_dpr_mapping,
    load_full_config,
)

log = logging.getLogger("root")

# Production fallback assumes ~30 % slag/HM ratio when DPR is missing.
SLAG_TO_HM_FALLBACK_RATIO = 0.30

# Stream labels used in the inputs/outputs nested dicts.
GAS_INPUT_BLAST = "Hot Blast"
GAS_INPUT_O2 = "O2 Enrichment"
GAS_INPUT_STEAM = "Steam"
OUT_HM = "Hot Metal"
OUT_SLAG = "Slag"
OUT_TOPGAS = "Top Gas"
OUT_DUST = "Dust Catcher"
OUT_UNACCOUNTED = "Unaccounted"


@dataclass
class BalanceResult:
    """Bundle of everything the plotters and the page need.

    Attributes:
        day: The IST calendar date the balance was computed for.
        inputs: ``{element: {stream: tonnes}}`` for every input stream.
        outputs: ``{element: {stream: tonnes}}`` for every output stream.
        closure_table: pandas DataFrame with columns
            ``Element, In_t, Out_t, Closure_pct, Delta_t``.
        material_masses: ``{material_name: tonnes}`` actually used.
        gas_phase: dict of intermediate gas-phase quantities for the
            "Assumptions & limitations" expander on the page.
        warnings: List of human-readable issues raised during the run.
        used_dpr: True iff DPR mass mapping was complete and applied.
        n_rm_rows: How many hourly CSV rows were available for the day.
        rm_lag_hours: Lag applied to RM / composition window.
        blast_lag_hours: Lag applied to blast / process-param window.
        dust_catcher_t: Manually-entered dust catcher daily tonnage.
    """

    day: date
    inputs: Dict[str, Dict[str, float]]
    outputs: Dict[str, Dict[str, float]]
    closure_table: pd.DataFrame
    material_masses: Dict[str, float]
    gas_phase: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    used_dpr: bool = False
    n_rm_rows: int = 0
    rm_lag_hours: int = 0
    blast_lag_hours: int = 0
    dust_catcher_t: float = 0.0


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _get_pct(row: pd.Series, col: str) -> float:
    """Pull a percentage column safely; NaN/missing → 0.0."""
    if col not in row.index:
        return 0.0
    v = row[col]
    if v is None or pd.isna(v):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _ensure_element_dict() -> Dict[str, Dict[str, float]]:
    """Empty {element: {}} dict for every tracked element."""
    return {el: {} for el in ELEMENTS}


def _add_element(
    out: Dict[str, Dict[str, float]],
    element: str,
    stream: str,
    delta_t: float,
) -> None:
    """Accumulate ``delta_t`` tonnes of ``element`` into a stream bucket."""
    if element not in out or delta_t == 0.0:
        return
    out[element][stream] = out[element].get(stream, 0.0) + float(delta_t)


# ---------------------------------------------------------------------------
# Material → element conversion
# ---------------------------------------------------------------------------


def material_to_elements(
    mass_t: float,
    row: pd.Series,
    spec: MaterialSpec,
    ash_assumptions: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Convert one material's row + tonnage into element tonnes.

    Args:
        mass_t: Total mass of this material for the day [t].
        row: Row of composition pcts (column names match spec.composition keys).
        spec: MaterialSpec describing how each column maps to elements.
        ash_assumptions: Yml-loaded {assumption_key: {oxide: pct, ...}}.

    Returns:
        ``{element: tonnes_in_this_material}``.
    """
    elements: Dict[str, float] = {}
    if mass_t <= 0:
        return elements

    for col, (token, kind) in spec.composition.items():
        pct = _get_pct(row, col)
        if pct <= 0:
            continue
        contrib = mass_t * pct / 100.0  # tonnes of this composition slice

        if kind == "direct":
            elements[token] = elements.get(token, 0.0) + contrib

        elif kind == "oxide":
            mfrac_entry = OXIDE_TO_ELEMENT_MASS_FRAC.get(token)
            if mfrac_entry is None:
                continue
            elem, mfrac = mfrac_entry
            elements[elem] = elements.get(elem, 0.0) + contrib * mfrac
            elements["O"] = elements.get("O", 0.0) + contrib * (1.0 - mfrac)

        elif kind == "fe_as_fe2o3":
            # Column gives Fe wt%, but iron enters as Fe2O3.
            # Fe tonnes = mass_t * Fe% / 100 (= contrib).
            # Back-calculate Fe2O3 mass, then extract O.
            fe_t = contrib
            _, mfrac_fe = OXIDE_TO_ELEMENT_MASS_FRAC["Fe2O3"]
            fe2o3_t = fe_t / mfrac_fe
            o_t = fe2o3_t - fe_t
            elements["Fe"] = elements.get("Fe", 0.0) + fe_t
            elements["O"] = elements.get("O", 0.0) + o_t

        elif kind == "fe_as_fe2o3_minus_feo":
            # Column gives Total Fe wt%; iron enters as Fe2O3 + FeO.
            # FeO is handled separately as kind="oxide", so subtract
            # the Fe contributed by FeO to avoid double-counting.
            # Token format: "Fe2O3|<feo_col_name>"
            parts = token.split("|", 1)
            feo_col = parts[1] if len(parts) > 1 else ""
            feo_pct = _get_pct(row, feo_col) if feo_col else 0.0

            _, mfrac_fe_in_feo = OXIDE_TO_ELEMENT_MASS_FRAC["FeO"]
            fe_from_feo_t = mass_t * feo_pct / 100.0 * mfrac_fe_in_feo

            total_fe_t = contrib
            fe_as_fe2o3_t = max(total_fe_t - fe_from_feo_t, 0.0)

            _, mfrac_fe_in_fe2o3 = OXIDE_TO_ELEMENT_MASS_FRAC["Fe2O3"]
            fe2o3_t = fe_as_fe2o3_t / mfrac_fe_in_fe2o3
            o_from_fe2o3_t = fe2o3_t - fe_as_fe2o3_t

            elements["Fe"] = elements.get("Fe", 0.0) + fe_as_fe2o3_t
            elements["O"] = elements.get("O", 0.0) + o_from_fe2o3_t

        elif kind == "H2O":
            elements["H"] = elements.get("H", 0.0) + contrib * (
                2.0 * ATOMIC_WEIGHTS["H"] / (2.0 * ATOMIC_WEIGHTS["H"] + ATOMIC_WEIGHTS["O"])
            )
            elements["O"] = elements.get("O", 0.0) + contrib * (
                ATOMIC_WEIGHTS["O"] / (2.0 * ATOMIC_WEIGHTS["H"] + ATOMIC_WEIGHTS["O"])
            )

        elif kind == "ASH":
            assumption = ash_assumptions.get(spec.ash_assumption_key or "", {})
            if not assumption:
                continue
            # Distribute ash mass into species.
            # Species can be oxides (split into element + O) OR direct elements
            # (e.g. S, P reported as elemental wt% of ash).
            for species_key, species_pct in assumption.items():
                if species_key.lower() == "other" or (species_pct or 0.0) <= 0:
                    continue
                species_mass = contrib * float(species_pct) / 100.0
                mfrac_entry = OXIDE_TO_ELEMENT_MASS_FRAC.get(species_key)
                if mfrac_entry is not None:
                    # Oxide: split into metal element + oxygen
                    elem, mfrac = mfrac_entry
                    elements[elem] = elements.get(elem, 0.0) + species_mass * mfrac
                    elements["O"] = elements.get("O", 0.0) + species_mass * (1.0 - mfrac)
                elif species_key in ATOMIC_WEIGHTS:
                    # Direct element (e.g. S, P as wt% of ash)
                    elements[species_key] = elements.get(species_key, 0.0) + species_mass
                # else: unknown key → skip silently

        elif kind == "LOI":
            # TODO(material-balance): split LOI into CO2/H2O via per-material
            # loi_split_pct from the yml. Dropped in v1.
            continue

    return elements


# ---------------------------------------------------------------------------
# Mass resolution (DPR with RM-sum fallback)
# ---------------------------------------------------------------------------


def resolve_material_masses(
    rm_row: pd.Series,
    dpr_masses: Dict[str, float],
    online: Dict[str, float],
) -> Tuple[Dict[str, float], List[str], bool]:
    """Pick a tonnage for every material; prefer DPR, fall back to RM ``*_mt``.

    Args:
        rm_row (pd.Series): Day-averaged RM composition row.
        dpr_masses (dict): Canonical mass dict from
            :func:`~utils.material_balance.dpr_mapping.apply_dpr_mapping`.
        online (dict): Day-averaged ``process_params`` fields.

    Returns:
        tuple: ``(masses_by_material_name, warnings, used_dpr)`` where
        *masses_by_material_name* maps material display name to tonnes,
        *warnings* is a list of human-readable fallback notices, and
        *used_dpr* indicates whether any DPR mass was non-zero.
    """
    warnings: List[str] = []

    # Map MaterialSpec.name → DPR canonical key.
    dpr_keys = {
        "Coke": "coke_mass_t",
        "Nut Coke": "nut_coke_mass_t",
        "PCI": "pci_mass_t",
        "Ore": "ore_mass_t",
        "Sinter": "sinter_mass_t",
        "Pellet": "pellet_mass_t",
        "Flux": "flux_mass_t",
    }

    used_dpr = any(
        (dpr_masses.get(dpr_keys[s.name], 0.0) or 0.0) > 0
        for s in MATERIAL_REGISTRY
    )

    masses: Dict[str, float] = {}
    for spec in MATERIAL_REGISTRY:
        dpr_mass = dpr_masses.get(dpr_keys[spec.name], 0.0) or 0.0
        if dpr_mass > 0:
            masses[spec.name] = float(dpr_mass)
        else:
            rm_val = _get_pct(rm_row, spec.mass_field)
            if used_dpr and rm_val > 0:
                warnings.append(
                    f"DPR missing {dpr_keys[spec.name]} — falling back to "
                    f"RM column '{spec.mass_field}' ({rm_val:.0f} t)."
                )
            masses[spec.name] = float(rm_val)

    return masses, warnings, used_dpr


def resolve_hm_slag_masses(
    dpr_masses: Dict[str, float],
    online: Dict[str, float],
    rm_row: pd.Series,
    warnings: List[str],
) -> Tuple[float, float]:
    """Pick HM and slag tonnage for the day.

    Order of preference:
        1. ``total_hot_metal_mt`` / ``slag_generation_mt`` from DPR
           (``dpr_data`` measurement — no user mapping required)
        2. User-configured DPR mapping fields (``hm_mass_t`` / ``slag_mass_t``)
        3. ``production_per_hour * 24`` from static CSV (HM fallback)
        4. ``0.30 × HM`` (slag fallback)

    Args:
        dpr_masses (dict): Canonical mass dict from DPR mapping; already
            pre-populated with ``total_hot_metal_mt`` / ``slag_generation_mt``
            values by the caller before this function is invoked.
        online (dict): Day-averaged ``process_params`` fields.
        rm_row (pd.Series): Day-averaged RM composition row.
        warnings (list): Mutable list — fallback notices are appended.

    Returns:
        tuple: ``(hm_mass_t, slag_mass_t)`` in tonnes.
    """
    hm = float(dpr_masses.get("hm_mass_t", 0.0) or 0.0)
    slag = float(dpr_masses.get("slag_mass_t", 0.0) or 0.0)

    if hm <= 0:
        prod_per_hour = float(online.get("production_per_hour", 0.0) or 0.0)
        if prod_per_hour > 0:
            hm = prod_per_hour * 24.0
            warnings.append(
                f"DPR total_hot_metal_mt unavailable — "
                f"using production_per_hour×24 = {hm:.0f} t."
            )

    if slag <= 0 and hm > 0:
        slag = SLAG_TO_HM_FALLBACK_RATIO * hm
        warnings.append(
            f"DPR slag_generation_mt unavailable — "
            f"using fallback 0.30×HM = {slag:.0f} t."
        )

    return hm, slag


# ---------------------------------------------------------------------------
# Gas-phase math (blast / steam / top gas)
# ---------------------------------------------------------------------------


def compute_blast_elements(online: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute O + N tonnes from hot blast volume and O₂ enrichment.

    Mirrors ``oxygen_flow_from_enrichment`` locally to avoid the
    :class:`~utils.recommendations.dependencies.BFColumns` column-name
    dependency. Splits the wind into an air portion (carrying its own
    O + N) and a pure O₂ enrichment portion (carrying only O).

    Args:
        online (dict): Day-averaged ``process_params`` fields. Must
            contain ``hot_blast_vol_nm3h`` and ``oxygen_enrichment_pct``.

    Returns:
        tuple: ``(elements, debug)`` where *elements* has keys
        ``blast_O_t``, ``blast_N_t``, ``enrich_O_t`` (tonnes) and
        *debug* exposes intermediate quantities for the page expander.
    """
    wind = float(online.get("hot_blast_vol_nm3h", 0.0) or 0.0)
    enr = float(online.get("oxygen_enrichment_pct", 0.0) or 0.0)

    if wind <= 0:
        return {}, {"wind_nm3h": 0.0, "o2_flow_nm3h": 0.0}

    # OF = (wind*(20.8 + enr)/100 - 0.208*wind) / 0.792
    effective = 20.8 + enr
    o2_flow = (wind * (effective / 100.0) - 0.208 * wind) / 0.792
    if o2_flow < 0.0:
        o2_flow = 0.0

    air_only_nm3h = max(wind - o2_flow, 0.0)
    air_kg_day = air_only_nm3h * 24.0 * RHO_AIR_NTP

    o_in_air_t = air_kg_day * AIR_O2_MASS_FRAC / 1000.0
    n_in_air_t = air_kg_day * AIR_N2_MASS_FRAC / 1000.0

    # O2 enrichment: pure O2 stream — only oxygen, mass = vol × 32/22.414.
    o_enrich_t = o2_flow * 24.0 * (32.0 / MOLAR_VOLUME_NTP) / 1000.0

    return (
        {
            "blast_O_t": o_in_air_t,
            "blast_N_t": n_in_air_t,
            "enrich_O_t": o_enrich_t,
        },
        {
            "wind_nm3h": wind,
            "o2_flow_nm3h": o2_flow,
            "air_only_nm3h": air_only_nm3h,
        },
    )


def compute_steam_elements(online: Dict[str, float]) -> Dict[str, float]:
    """Compute H + O tonnes from steam injection (kg/h).

    Args:
        online (dict): Day-averaged ``process_params`` fields. Must
            contain ``steam_injection``.

    Returns:
        dict: ``{"steam_H_t": float, "steam_O_t": float}`` in tonnes.
    """
    steam_kgh = float(online.get("steam_injection", 0.0) or 0.0)
    if steam_kgh <= 0:
        return {"steam_H_t": 0.0, "steam_O_t": 0.0}

    steam_kg_day = steam_kgh * 24.0
    h_t = steam_kg_day * (2.0 * ATOMIC_WEIGHTS["H"] / 18.015) / 1000.0
    o_t = steam_kg_day * (ATOMIC_WEIGHTS["O"] / 18.015) / 1000.0
    return {"steam_H_t": h_t, "steam_O_t": o_t}


def compute_top_gas_elements(
    online: Dict[str, float],
    warnings: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute C, O, H, N tonnes leaving the furnace through the top gas.

    Volume method: **N₂ balance**.  N₂ is chemically inert through the
    furnace, so N₂ input (from blast air, Nm³/day) equals N₂ in top gas.
    Total top-gas volume follows from the measured top-gas N₂ fraction:

    .. code-block::

        n2_blast  = (wind_daily − o2_daily) × 0.792
        top_gas   = n2_blast / (N2_frac_in_top_gas)

    This gives a physically consistent volume (~1.2–1.5× blast) without
    relying on empirical bosh-vol formulas.

    Fallback to ``1.4 × wind`` if measured N₂ fraction is < 5 % (sensor
    gap / startup transient).

    Args:
        online (dict): Day-averaged ``process_params`` fields. Must
            contain ``hot_blast_vol_nm3h``, ``oxygen_enrichment_pct``,
            ``co_pct``, ``co2_pct``, ``h2_pct``.
        warnings (list): Mutable list — sanity-check notices are
            appended.

    Returns:
        tuple: ``({element: tonnes}, debug_dict)`` where the element
        dict has keys ``C``, ``O``, ``H``, ``N`` and *debug_dict*
        exposes ``top_gas_nm3_per_day`` and per-species kmol values.
    """
    wind = float(online.get("hot_blast_vol_nm3h", 0.0) or 0.0)
    co_pct = float(online.get("co_pct", 0.0) or 0.0)
    co2_pct = float(online.get("co2_pct", 0.0) or 0.0)
    h2_pct = float(online.get("h2_pct", 0.0) or 0.0)

    if wind <= 0 or (co_pct + co2_pct + h2_pct) <= 0:
        return {}, {"top_gas_nm3_per_day": 0.0}

    o2_enrich = float(online.get("oxygen_enrichment_pct", 0.0) or 0.0)
    effective = 20.8 + o2_enrich
    o2_flow_nm3h = max((wind * effective / 100.0 - 0.208 * wind) / 0.792, 0.0)

    # N2 in blast (Nm³/day) — N2 from air only; pure O2 injection adds no N2
    n2_blast_nm3_day = (wind * 24.0 - o2_flow_nm3h * 24.0) * 0.792

    # N2 fraction in measured top gas
    n2_pct = 100.0 - co_pct - co2_pct - h2_pct
    if n2_pct < 0:
        warnings.append(
            f"Top-gas components sum > 100 % "
            f"(CO+CO₂+H₂ = {co_pct + co2_pct + h2_pct:.1f} %). "
            "Clipping N₂ to 0 and falling back to 1.4 × wind."
        )
        n2_pct = 0.0

    daily_wind_nm3 = wind * 24.0
    if n2_pct >= 5.0:
        top_gas_nm3_per_day = n2_blast_nm3_day / (n2_pct / 100.0)
    else:
        warnings.append(
            f"Top-gas N₂% = {n2_pct:.1f} % (< 5 %) — N₂ balance unreliable. "
            "Using fallback top_gas ≈ 1.4 × wind."
        )
        top_gas_nm3_per_day = 1.4 * daily_wind_nm3

    # Final sanity bounds: expected ~1.0–1.8× daily wind
    if not (0.8 * daily_wind_nm3 <= top_gas_nm3_per_day <= 2.0 * daily_wind_nm3):
        warnings.append(
            f"Top-gas volume ({top_gas_nm3_per_day / daily_wind_nm3:.2f}× wind) "
            "outside expected 0.8–2.0× range — using fallback 1.4 × wind."
        )
        top_gas_nm3_per_day = 1.4 * daily_wind_nm3

    total_kmol = top_gas_nm3_per_day / MOLAR_VOLUME_NTP
    co_kmol = total_kmol * co_pct / 100.0
    co2_kmol = total_kmol * co2_pct / 100.0
    h2_kmol = total_kmol * h2_pct / 100.0
    n2_kmol = total_kmol * n2_pct / 100.0

    c_t = (co_kmol + co2_kmol) * ATOMIC_WEIGHTS["C"] / 1000.0
    o_t = (co_kmol * ATOMIC_WEIGHTS["O"] + co2_kmol * 2.0 * ATOMIC_WEIGHTS["O"]) / 1000.0
    h_t = h2_kmol * 2.0 * ATOMIC_WEIGHTS["H"] / 1000.0
    n_t = n2_kmol * 2.0 * ATOMIC_WEIGHTS["N"] / 1000.0

    return (
        {"C": c_t, "O": o_t, "H": h_t, "N": n_t},
        {
            "top_gas_nm3_per_day": top_gas_nm3_per_day,
            "co_kmol": co_kmol,
            "co2_kmol": co2_kmol,
            "h2_kmol": h2_kmol,
            "n2_kmol": n2_kmol,
        },
    )


# ---------------------------------------------------------------------------
# Outputs: HM and slag
# ---------------------------------------------------------------------------


def hm_to_elements(hm_mass_t: float, hm_row: pd.Series) -> Dict[str, float]:
    """Hot metal chemistry × HM tonnage → element tonnes.

    Args:
        hm_mass_t (float): Total hot metal produced for the day [t].
        hm_row (pd.Series): Row with ``chem_pct_*`` columns.

    Returns:
        dict: ``{element_symbol: tonnes}``.
    """
    out: Dict[str, float] = {}
    if hm_mass_t <= 0:
        return out
    for col, elem in HM_COMPOSITION.items():
        if elem not in ELEMENTS:
            continue
        pct = _get_pct(hm_row, col)
        if pct > 0:
            out[elem] = out.get(elem, 0.0) + hm_mass_t * pct / 100.0
    return out


def slag_to_elements(slag_mass_t: float, slag_row: pd.Series) -> Dict[str, float]:
    """Slag chemistry × slag tonnage → element tonnes (oxide + S).

    Args:
        slag_mass_t (float): Total slag produced for the day [t].
        slag_row (pd.Series): Row with ``slag_pct_*`` columns.

    Returns:
        dict: ``{element_symbol: tonnes}``.
    """
    out: Dict[str, float] = {}
    if slag_mass_t <= 0:
        return out
    for col, (token, kind) in SLAG_COMPOSITION.items():
        pct = _get_pct(slag_row, col)
        if pct <= 0:
            continue
        contrib = slag_mass_t * pct / 100.0
        if kind == "direct":
            if token in ELEMENTS:
                out[token] = out.get(token, 0.0) + contrib
        else:  # oxide
            mfrac_entry = OXIDE_TO_ELEMENT_MASS_FRAC.get(token)
            if mfrac_entry is None:
                continue
            elem, mfrac = mfrac_entry
            if elem in ELEMENTS:
                out[elem] = out.get(elem, 0.0) + contrib * mfrac
            out["O"] = out.get("O", 0.0) + contrib * (1.0 - mfrac)
    return out


# ---------------------------------------------------------------------------
# Dust catcher (manually entered)
# ---------------------------------------------------------------------------


def dust_catcher_to_elements(
    dust_t: float,
    composition_pct: Dict[str, float],
) -> Dict[str, float]:
    """Distribute manually-entered dust catcher tonnage into element tonnes.

    The composition is specified as oxide wt% (e.g. ``{"Fe2O3": 40, ...}``)
    or element wt% for species like ``"C"``.  The ``"other"`` / ``"Other"``
    key is silently ignored.

    Args:
        dust_t (float): Total dry-dust tonnage for the day [t].
        composition_pct (dict): e.g. ``{"Fe2O3": 40, "C": 12, "SiO2": 20, ...}``.

    Returns:
        dict: ``{element_symbol: tonnes}`` ready to add to the outputs dict.
    """
    elements: Dict[str, float] = {}
    if dust_t <= 0 or not composition_pct:
        return elements

    for species, pct in composition_pct.items():
        if species.lower() == "other" or pct <= 0:
            continue
        contrib = dust_t * float(pct) / 100.0
        if species == "C":
            elements["C"] = elements.get("C", 0.0) + contrib
        else:
            mfrac_entry = OXIDE_TO_ELEMENT_MASS_FRAC.get(species)
            if mfrac_entry is None:
                continue
            elem, mfrac = mfrac_entry
            elements[elem] = elements.get(elem, 0.0) + contrib * mfrac
            elements["O"] = elements.get("O", 0.0) + contrib * (1.0 - mfrac)

    return elements


# ---------------------------------------------------------------------------
# Future-stream stub (sludge / granulation losses)
# ---------------------------------------------------------------------------


def compute_unaccounted_solids(online: Dict[str, float]) -> Dict[str, float]:
    """Placeholder for sludge / granulation losses (not dust catcher).

    Dust catcher is handled separately via :func:`dust_catcher_to_elements`.
    This function remains for future sludge / granulation integration.
    Returns ``{}`` in v1.
    """
    return {}


# ---------------------------------------------------------------------------
# Closure table
# ---------------------------------------------------------------------------


def build_closure_table(
    inputs: Dict[str, Dict[str, float]],
    outputs: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Build the per-element closure DataFrame consumed by the page.

    Closure% is ``Out_t / In_t × 100``, shown as ``NaN`` when ``In_t``
    is zero (no recovery possible).

    Args:
        inputs (dict): ``{element: {stream: tonnes}}`` from the input side.
        outputs (dict): ``{element: {stream: tonnes}}`` from the output side.

    Returns:
        pd.DataFrame: Columns ``Element, In_t, Out_t, Closure_pct, Delta_t``
        with one row per element in :data:`~utils.material_balance.constants.ELEMENTS`.
    """
    rows = []
    for el in ELEMENTS:
        in_t = sum(inputs.get(el, {}).values())
        out_t = sum(outputs.get(el, {}).values())
        closure = (out_t / in_t * 100.0) if in_t > 0 else float("nan")
        rows.append(
            {
                "Element": el,
                "In_t": round(in_t, 2),
                "Out_t": round(out_t, 2),
                "Closure_pct": round(closure, 1) if in_t > 0 else None,
                "Delta_t": round(out_t - in_t, 2),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_full_balance(
    day: date,
    rm_lag_hours: int = 0,
    blast_lag_hours: int = 0,
    dust_catcher_t: float = 0.0,
) -> BalanceResult:
    """Compute the full element balance for one IST calendar day.

    Primary data source is the static ML-dataset CSV (via
    :func:`~utils.material_balance.data_sources.fetch_static_rm_for_day` and
    siblings).  DPR from InfluxDB is used as an optional override for masses
    when the mapping is configured.

    Args:
        day (date): IST calendar date to compute the balance for.
        rm_lag_hours (int): Shift the RM / composition input window this
            many hours backward (0 = same-day, 96 = 4 days prior).
        blast_lag_hours (int): Shift the blast / process-param window
            this many hours backward.
        dust_catcher_t (float): Manually-entered daily dust catcher
            tonnage (dry basis) [t].  0 = not measured / not modelled.

    Returns:
        BalanceResult: Inputs, outputs, closure table, gas-phase debug
        quantities, material masses, and any warnings raised.
    """
    # Local imports keep this module Streamlit-free at import time.
    from utils.material_balance.data_sources import (
        fetch_dpr_for_day,
        fetch_static_hm_slag_for_day,
        fetch_static_online_for_day,
        fetch_static_rm_for_day,
    )

    cfg = load_full_config()
    ash_assumptions = {
        "coke":    cfg.get("coke_ash_analysis_pct") or cfg.get("coke_ash_assumption_pct", {}),
        "nutcoke": cfg.get("nutcoke_ash_analysis_pct", {}),
        "pci":     cfg.get("pci_ash_analysis_pct") or cfg.get("pci_ash_assumption_pct", {}),
    }
    dust_composition = cfg.get("dust_catcher_composition_pct", {}) or {}

    # ── Data fetch ──────────────────────────────────────────────────────
    rm_df = fetch_static_rm_for_day(day, lag_hours=rm_lag_hours)
    hm_slag_df = fetch_static_hm_slag_for_day(day)
    online = fetch_static_online_for_day(day, lag_hours=blast_lag_hours)
    dpr_df = fetch_dpr_for_day(day)  # InfluxDB DPR (optional; usually empty)

    warnings: List[str] = []
    if rm_df is None or rm_df.empty:
        warnings.append(
            f"No raw-material data found for {day.isoformat()} in the static "
            "CSV. Select a different date or check the CSV date range."
        )

    rm_row = (
        rm_df.iloc[0]
        if (rm_df is not None and not rm_df.empty)
        else pd.Series(dtype=float)
    )
    hm_slag_row = (
        hm_slag_df.iloc[0]
        if (hm_slag_df is not None and not hm_slag_df.empty)
        else pd.Series(dtype=float)
    )
    n_rm_rows = int(rm_df.attrs.get("n_rows", 0)) if rm_df is not None else 0
    if 0 < n_rm_rows < 20:
        warnings.append(
            f"Only {n_rm_rows} hourly rows available for this day "
            f"(expected ≈24) — averages may be less representative."
        )
    if rm_lag_hours:
        warnings.append(
            f"RM lag applied: input composition taken from "
            f"{rm_lag_hours}h earlier (≈{rm_lag_hours // 24} day(s) prior)."
        )
    if blast_lag_hours:
        warnings.append(
            f"Blast lag applied: process params taken from "
            f"{blast_lag_hours}h earlier."
        )

    dpr_mapping = load_dpr_mapping()
    dpr_masses = apply_dpr_mapping(dpr_df, dpr_mapping)

    # Override HM / slag mass with the known DPR fields that require no
    # user-configured mapping.  Field names are fixed in dpr_data:
    #   total_hot_metal_mt  — total hot metal produced for the day (t)
    #   slag_generation_mt  — total slag generated for the day (t)
    # DPR is daily cadence so .iloc[-1] picks the most recent row if
    # multiple rows fall in the UTC window.
    if dpr_df is not None and not dpr_df.empty:
        for dpr_field, mass_key in (
            ("total_hot_metal_mt", "hm_mass_t"),
            ("slag_generation_mt", "slag_mass_t"),
        ):
            if dpr_field in dpr_df.columns:
                series = dpr_df[dpr_field].dropna()
                if not series.empty:
                    dpr_masses[mass_key] = float(series.iloc[-1])

    # ── Inputs ──────────────────────────────────────────────────────────
    material_masses, mass_warnings, used_dpr = resolve_material_masses(
        rm_row, dpr_masses, online
    )
    warnings.extend(mass_warnings)

    inputs = _ensure_element_dict()
    for spec in MATERIAL_REGISTRY:
        elements = material_to_elements(
            material_masses.get(spec.name, 0.0), rm_row, spec, ash_assumptions
        )
        for el, t in elements.items():
            _add_element(inputs, el, spec.name, t)

    blast_els, blast_dbg = compute_blast_elements(online)
    if blast_els:
        _add_element(inputs, "O", GAS_INPUT_BLAST, blast_els.get("blast_O_t", 0.0))
        _add_element(inputs, "N", GAS_INPUT_BLAST, blast_els.get("blast_N_t", 0.0))
        _add_element(inputs, "O", GAS_INPUT_O2, blast_els.get("enrich_O_t", 0.0))

    steam_els = compute_steam_elements(online)
    _add_element(inputs, "H", GAS_INPUT_STEAM, steam_els.get("steam_H_t", 0.0))
    _add_element(inputs, "O", GAS_INPUT_STEAM, steam_els.get("steam_O_t", 0.0))

    # ── Outputs ─────────────────────────────────────────────────────────
    hm_mass_t, slag_mass_t = resolve_hm_slag_masses(
        dpr_masses, online, rm_row, warnings
    )

    outputs = _ensure_element_dict()
    for el, t in hm_to_elements(hm_mass_t, hm_slag_row).items():
        _add_element(outputs, el, OUT_HM, t)
    for el, t in slag_to_elements(slag_mass_t, hm_slag_row).items():
        _add_element(outputs, el, OUT_SLAG, t)

    top_gas_els, top_gas_dbg = compute_top_gas_elements(online, warnings)
    for el, t in top_gas_els.items():
        _add_element(outputs, el, OUT_TOPGAS, t)

    # Dust catcher (manually entered)
    if dust_catcher_t > 0:
        if dust_composition:
            dust_els = dust_catcher_to_elements(dust_catcher_t, dust_composition)
            for el, t in dust_els.items():
                _add_element(outputs, el, OUT_DUST, t)
        else:
            warnings.append(
                "Dust catcher tonnes entered but no composition configured in "
                "material_balance.yml (dust_catcher_composition_pct). "
                "Dust mass added as unresolved loss."
            )
            # Still record total dust for Sankey display
            _add_element(outputs, "Fe", OUT_DUST, dust_catcher_t * 0.40)

    # Sludge / other future-stream placeholder — always empty in v1.
    for el, t in compute_unaccounted_solids(online).items():
        _add_element(outputs, el, OUT_UNACCOUNTED, t)

    closure = build_closure_table(inputs, outputs)

    gas_phase: Dict[str, float] = {}
    gas_phase.update(blast_dbg)
    gas_phase.update(top_gas_dbg)
    gas_phase["steam_kgh"] = float(online.get("steam_injection", 0.0) or 0.0)
    gas_phase["hm_mass_t"] = hm_mass_t
    gas_phase["slag_mass_t"] = slag_mass_t
    gas_phase["dust_catcher_t"] = dust_catcher_t

    return BalanceResult(
        day=day,
        inputs=inputs,
        outputs=outputs,
        closure_table=closure,
        material_masses=material_masses,
        gas_phase=gas_phase,
        warnings=warnings,
        used_dpr=used_dpr,
        n_rm_rows=n_rm_rows,
        rm_lag_hours=rm_lag_hours,
        blast_lag_hours=blast_lag_hours,
        dust_catcher_t=dust_catcher_t,
    )
