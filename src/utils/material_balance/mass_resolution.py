"""DPR-first mass resolution with RM-sum fallback.

Determines how many tonnes of each material, hot metal, and slag were
produced on a given day. Prefers DPR (Daily Production Report) masses
when available; otherwise falls back to RM ``*_mt`` columns or
``production_per_hour × 24``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from utils.material_balance._helpers import _get_pct
from utils.material_balance.constants import MATERIAL_REGISTRY

# Production fallback assumes ~30 % slag/HM ratio when DPR is missing.
SLAG_TO_HM_FALLBACK_RATIO = 0.30


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
        3. ``production_per_hour * 24`` from the static dataset (HM fallback)
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
