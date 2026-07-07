"""Material → element conversion engine.

Converts one raw material's composition row + tonnage into element tonnes,
handling all ``kind`` switches defined in :mod:`constants`:

    direct, oxide, fe_as_fe2o3, fe_as_fe2o3_minus_feo, H2O, ASH, LOI
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from furnace_data.material_balance._helpers import _get_pct
from furnace_data.material_balance.constants import (
    ATOMIC_WEIGHTS,
    OXIDE_TO_ELEMENT_MASS_FRAC,
    MaterialSpec,
)


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

            # Some species (e.g. S) in the ash analysis report are given as
            # wt% of net fuel (gross mass minus moisture), NOT wt% of ash.
            # Identify those species and compute their correct base mass.
            net_fuel_species = set(
                ash_assumptions.get(
                    f"{spec.ash_assumption_key}_net_fuel_basis_species", []
                ) or []
            )
            if net_fuel_species:
                moist_pct = _get_pct(row, spec.moisture_col) if spec.moisture_col else 0.0
                net_fuel_t = mass_t * (1.0 - moist_pct / 100.0)
            else:
                net_fuel_t = mass_t  # unused if net_fuel_species is empty

            # Distribute ash mass (or net-fuel mass) into species.
            # Species can be oxides (split into element + O) OR direct elements
            # (e.g. S, P).  Base mass depends on which reporting convention applies.
            for species_key, species_pct in assumption.items():
                if species_key.lower() == "other" or (species_pct or 0.0) <= 0:
                    continue
                if species_key in net_fuel_species:
                    # Net-fuel basis: S% = wt% of (gross − moisture)
                    species_mass = net_fuel_t * float(species_pct) / 100.0
                else:
                    # Ash basis: species% = wt% of ash fraction only
                    species_mass = contrib * float(species_pct) / 100.0
                mfrac_entry = OXIDE_TO_ELEMENT_MASS_FRAC.get(species_key)
                if mfrac_entry is not None:
                    # Oxide: split into metal element + oxygen
                    elem, mfrac = mfrac_entry
                    elements[elem] = elements.get(elem, 0.0) + species_mass * mfrac
                    elements["O"] = elements.get("O", 0.0) + species_mass * (1.0 - mfrac)
                elif species_key in ATOMIC_WEIGHTS:
                    # Direct element (e.g. S, P)
                    elements[species_key] = elements.get(species_key, 0.0) + species_mass
                # else: unknown key → skip silently

        elif kind == "LOI":
            # TODO(material-balance): split LOI into CO2/H2O via per-material
            # loi_split_pct from the yml. Dropped in v1.
            continue

    return elements
