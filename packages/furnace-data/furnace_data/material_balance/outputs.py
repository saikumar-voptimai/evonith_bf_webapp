"""Output-stream element converters: hot metal, slag, dust catcher.

Each function takes a tonnage + chemistry and returns element tonnes.
All functions are pure — no Streamlit, no I/O.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from furnace_data.material_balance._helpers import _get_pct
from furnace_data.material_balance.constants import (
    ELEMENTS,
    HM_COMPOSITION,
    OXIDE_TO_ELEMENT_MASS_FRAC,
    SLAG_COMPOSITION,
)


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


def compute_unaccounted_solids(online: Dict[str, float]) -> Dict[str, float]:
    """Placeholder for sludge / granulation losses (not dust catcher).

    Dust catcher is handled separately via :func:`dust_catcher_to_elements`.
    This function remains for future sludge / granulation integration.
    Returns ``{}`` in v1.
    """
    return {}
