"""Gas-phase element math: blast air, O₂ enrichment, steam, and top gas.

Computes element tonnes (O, N, H, C) entering and leaving the furnace
via gas-phase streams. All functions are pure — no Streamlit, no I/O.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from utils.material_balance.constants import (
    AIR_N2_MASS_FRAC,
    AIR_O2_MASS_FRAC,
    ATOMIC_WEIGHTS,
    MOLAR_VOLUME_NTP,
    RHO_AIR_NTP,
)


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
