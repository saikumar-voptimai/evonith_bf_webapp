"""Orchestrator for the Material Balance page.

Public API:
    - :func:`run_full_balance` — single entry point, returns :class:`BalanceResult`
    - :class:`BalanceResult` — dataclass bundle consumed by plotters and the page

All heavy-lifting is delegated to sibling modules:
    - :mod:`elements` — material → element conversion (kind-switch engine)
    - :mod:`mass_resolution` — DPR-first mass picking with RM fallback
    - :mod:`gas_phase` — blast / steam / top-gas element math
    - :mod:`outputs` — HM, slag, dust-catcher, unaccounted-solids converters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List

import pandas as pd

from furnace_data.material_balance.constants import ELEMENTS, MATERIAL_REGISTRY
from furnace_data.material_balance.dpr_mapping import (
    apply_dpr_mapping,
    load_dpr_mapping,
    load_full_config,
)
from furnace_data.material_balance.elements import material_to_elements
from furnace_data.material_balance.gas_phase import (
    compute_blast_elements,
    compute_steam_elements,
    compute_top_gas_elements,
)
from furnace_data.material_balance.mass_resolution import (
    resolve_hm_slag_masses,
    resolve_material_masses,
)
from furnace_data.material_balance.outputs import (
    compute_unaccounted_solids,
    dust_catcher_to_elements,
    hm_to_elements,
    slag_to_elements,
)

log = logging.getLogger("root")

# Stream labels used in the inputs/outputs nested dicts.
#TODO: Standardise the name. Maybe all caps?
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
        n_rm_rows: How many hourly static-dataset rows were available for the day.
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
# Orchestrator-only helpers
# ---------------------------------------------------------------------------


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

    Primary data source is the static ML dataset (via
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
    from furnace_data.material_balance.data_sources import (
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
        # Species whose wt% is reported on net-fuel basis (after moisture), not ash basis.
        "coke_net_fuel_basis_species":    cfg.get("coke_net_fuel_basis_species", []),
        "nutcoke_net_fuel_basis_species": cfg.get("nutcoke_net_fuel_basis_species", []),
        "pci_net_fuel_basis_species":     cfg.get("pci_net_fuel_basis_species", []),
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
            "static dataset. Select a different date or check the available date range."
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
    gas_phase["steam_kgh"] = float(online.get("steam_kgs_hr", 0.0) or 0.0)
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
