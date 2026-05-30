"""Blend chemistry and cost calculations for BMO.

This module evaluates solved wet ore quantities by first converting each ore
to dry weight, then calculating dry-weight Fe contribution, slag-forming oxide
contribution, final Fe percent, slag rate, and Rs/THM economics for BMO results.
"""

from __future__ import annotations

from math import isfinite

from utils.bmo.types import BlendEvaluation, OreInput

FE_FROM_FEO_FACTOR = 55.845 / 71.844


def _safe_pct(value: float) -> float:
    """
    Normalize a percentage value into a safe numeric range.

    BMO chemistry values come from configuration, live chemistry rows, and user
    editable inputs, so this guard prevents negative or impossible percentages
    from leaking into dry-weight and Fe contribution math.

    Args:
         - value: float - Raw percentage value.

    Returns:
         - return float - Percentage clamped between 0 and 100.
    """

    try:
        pct = float(value or 0.0)
    except (TypeError, ValueError):
        pct = 0.0
    return min(max(pct, 0.0), 100.0)


def compute_dry_fraction(moisture_pct: float) -> float:
    """
    Calculate the dry fraction remaining after removing moisture.

    This implements the first part of the plant formula:
    ``DryWeight = WetWeight * (100 - Moisture%) / 100``. The returned fraction
    can be multiplied by any wet weight to get dry weight.

    Args:
         - moisture_pct: float - Moisture percentage in wet material.

    Returns:
         - return float - Dry fraction between 0 and 1.
    """

    return (100.0 - _safe_pct(moisture_pct)) / 100.0


def compute_dry_weight_mt(wet_weight_mt: float, moisture_pct: float) -> float:
    """
    Convert wet material quantity into dry material quantity.

    The optimizer still plans wet burden quantities because stock, share, and
    purchase values are tracked on that basis. Chemistry calculations use this
    helper to remove moisture before calculating Fe contribution.

    Args:
         - wet_weight_mt: float - Wet material quantity in MT.
         - moisture_pct: float - Moisture percentage in wet material.

    Returns:
         - return float - Dry material quantity in MT.
    """

    try:
        wet_weight = float(wet_weight_mt or 0.0)
    except (TypeError, ValueError):
        wet_weight = 0.0
    return max(0.0, wet_weight) * compute_dry_fraction(moisture_pct)


def compute_fe_contribution_mt(dry_weight_mt: float, fe_t_pct: float) -> float:
    """
    Calculate Fe contribution from dry material weight and Fe percentage.

    This is the second step of the final Fe formula. Each material contributes
    ``DryWeight * Fe% / 100`` MT of Fe, and the blend-level Fe production is the
    sum of these contributions.

    Args:
         - dry_weight_mt: float - Dry material quantity in MT.
         - fe_t_pct: float - Total Fe percentage on dry basis.

    Returns:
         - return float - Fe contribution in MT.
    """

    try:
        dry_weight = float(dry_weight_mt or 0.0)
    except (TypeError, ValueError):
        dry_weight = 0.0
    return max(0.0, dry_weight) * (_safe_pct(fe_t_pct) / 100.0)


def compute_slag_forming_oxides_pct(
    *,
    sio2_pct: float,
    al2o3_pct: float,
    cao_pct: float,
    mgo_pct: float,
    tio2_pct: float,
    mno_pct: float,
    na2o_pct: float = 0.0,
    k2o_pct: float = 0.0,
) -> float:
    """
    Sum the dry-basis oxides that report to slag.

    This follows the simplified plant formula and intentionally excludes Fe,
    moisture, LOI, and fuel volatiles. Each input is treated as an oxide
    percentage on dry material weight.

    Args:
         - sio2_pct: float - Dry-basis SiO2 percentage.
         - al2o3_pct: float - Dry-basis Al2O3 percentage.
         - cao_pct: float - Dry-basis CaO percentage.
         - mgo_pct: float - Dry-basis MgO percentage.
         - tio2_pct: float - Dry-basis TiO2 percentage.
         - mno_pct: float - Dry-basis MnO percentage.
         - na2o_pct: float - Dry-basis Na2O percentage.
         - k2o_pct: float - Dry-basis K2O percentage.

    Returns:
         - return float - Total slag-forming oxide percentage.
    """

    return float(
        _safe_pct(sio2_pct)
        + _safe_pct(al2o3_pct)
        + _safe_pct(cao_pct)
        + _safe_pct(mgo_pct)
        + _safe_pct(tio2_pct)
        + _safe_pct(mno_pct)
        + _safe_pct(na2o_pct)
        + _safe_pct(k2o_pct)
    )


def compute_slag_contribution_mt(
    dry_weight_mt: float,
    *,
    sio2_pct: float,
    al2o3_pct: float,
    cao_pct: float,
    mgo_pct: float,
    tio2_pct: float,
    mno_pct: float,
    na2o_pct: float = 0.0,
    k2o_pct: float = 0.0,
) -> float:
    """
    Calculate slag contribution from one dry material quantity.

    This implements ``DryWeight * slag-forming oxides / 100`` for one ore or
    burden material. Blend-level slag MT is the sum of this value across all
    selected materials.

    Args:
         - dry_weight_mt: float - Dry material quantity in MT.
         - sio2_pct: float - Dry-basis SiO2 percentage.
         - al2o3_pct: float - Dry-basis Al2O3 percentage.
         - cao_pct: float - Dry-basis CaO percentage.
         - mgo_pct: float - Dry-basis MgO percentage.
         - tio2_pct: float - Dry-basis TiO2 percentage.
         - mno_pct: float - Dry-basis MnO percentage.
         - na2o_pct: float - Dry-basis Na2O percentage.
         - k2o_pct: float - Dry-basis K2O percentage.

    Returns:
         - return float - Slag contribution in MT.
    """

    try:
        dry_weight = float(dry_weight_mt or 0.0)
    except (TypeError, ValueError):
        dry_weight = 0.0
    slag_pct = compute_slag_forming_oxides_pct(
        sio2_pct=sio2_pct,
        al2o3_pct=al2o3_pct,
        cao_pct=cao_pct,
        mgo_pct=mgo_pct,
        tio2_pct=tio2_pct,
        mno_pct=mno_pct,
        na2o_pct=na2o_pct,
        k2o_pct=k2o_pct,
    )
    return max(0.0, dry_weight) * (slag_pct / 100.0)


def _dry_quantities_by_ore(
    ores: list[OreInput], quantities: dict[str, float]
) -> dict[str, float]:
    """
    Convert wet ore quantities into dry quantities by ore id.

    The returned dictionary keeps the original ore ids so downstream diagnostics
    and UI tables can show how each wet quantity changes after moisture removal.

    Args:
         - ores: list[OreInput] - Ores included in the evaluated blend.
         - quantities: dict[str, float] - Wet quantities keyed by ore id.

    Returns:
         - return dict[str, float] - Dry quantities keyed by ore id.
    """

    return {
        ore.ore_id: compute_dry_weight_mt(
            quantities.get(ore.ore_id, 0.0), ore.chemistry.moisture_pct
        )
        for ore in ores
    }


def _weighted_avg(ores: list[OreInput], weights: dict[str, float], attr: str) -> float:
    """
    Calculate a weighted chemistry average using supplied ore weights.

    For BMO chemistry metrics this should usually receive dry weights, because
    oxide and Fe percentages are interpreted after moisture is removed.

    Args:
         - ores: list[OreInput] - Ores included in the evaluated blend.
         - weights: dict[str, float] - Weights keyed by ore id.
         - attr: str - OreChemistry attribute to average.

    Returns:
         - return float - Weighted average chemistry value.
    """

    total_qty = sum(weights.values())
    if total_qty <= 0:
        return 0.0
    acc = 0.0
    for ore in ores:
        qty = float(weights.get(ore.ore_id, 0.0))
        if qty <= 0:
            continue
        value = float(getattr(ore.chemistry, attr, 0.0) or 0.0)
        acc += (qty / total_qty) * value
    return acc


def compute_effective_fe_pct(
    fe_t_pct: float, feo_pct: float, feo_in_slag_pct: float
) -> float:
    """
    Calculate FeO-adjusted Fe percentage retained for diagnostics.

    Final Fe production now follows the dry-weight Fe contribution formula.
    This helper is kept as a separate reported metric so existing users can
    still inspect the FeO-adjusted value without changing the final Fe formula.

    Args:
         - fe_t_pct: float - Final dry-weight Fe percentage.
         - feo_pct: float - Dry-weight FeO percentage.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.

    Returns:
         - return float - FeO-adjusted Fe percentage.
    """

    return float(fe_t_pct + (feo_pct - feo_in_slag_pct) * FE_FROM_FEO_FACTOR)


def evaluate_blend(
    ores: list[OreInput],
    quantities_mt: dict[str, float],
    feo_in_slag_pct: float,
    fuel_cost_per_thm_rs: float = 0.0,
) -> BlendEvaluation:
    """
    Evaluate one BMO blend using dry-weight Fe calculation.

    This is the central BMO calculation path. It starts with optimized wet
    quantities, converts each material to dry weight, calculates Fe MT per
    material, and then derives final Fe% as ``total Fe MT / total dry MT * 100``.
    Slag MT is calculated separately as the sum of dry-weight oxide
    contributions from SiO2, Al2O3, CaO, MgO, TiO2, MnO, Na2O, and K2O.

    Args:
         - ores: list[OreInput] - Ores included in the blend.
         - quantities_mt: dict[str, float] - Wet ore quantities keyed by ore id.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - fuel_cost_per_thm_rs: float - Fuel cost to include in total objective.

    Returns:
         - return BlendEvaluation - Blend cost, chemistry, production, and diagnostics.
    """

    total_qty_mt = float(sum(quantities_mt.values()))
    shares_pct: dict[str, float] = {}

    if total_qty_mt > 0:
        shares_pct = {
            ore_id: (qty / total_qty_mt) * 100.0
            for ore_id, qty in quantities_mt.items()
        }
    else:
        shares_pct = {ore_id: 0.0 for ore_id in quantities_mt}

    dry_quantities_mt = _dry_quantities_by_ore(ores, quantities_mt)
    total_dry_qty_mt = float(sum(dry_quantities_mt.values()))
    fe_contribution_mt_by_ore = {
        ore.ore_id: compute_fe_contribution_mt(
            dry_quantities_mt.get(ore.ore_id, 0.0), ore.chemistry.fe_t_pct
        )
        for ore in ores
    }
    fe_production_mt = float(sum(fe_contribution_mt_by_ore.values()))

    fe_t_pct = (
        (fe_production_mt / total_dry_qty_mt) * 100.0 if total_dry_qty_mt > 0 else 0.0
    )
    feo_pct = _weighted_avg(ores, dry_quantities_mt, "feo_pct")
    sio2_pct = _weighted_avg(ores, dry_quantities_mt, "sio2_pct")
    al2o3_pct = _weighted_avg(ores, dry_quantities_mt, "al2o3_pct")
    cao_pct = _weighted_avg(ores, dry_quantities_mt, "cao_pct")
    mgo_pct = _weighted_avg(ores, dry_quantities_mt, "mgo_pct")
    mno_pct = _weighted_avg(ores, dry_quantities_mt, "mno_pct")
    tio2_pct = _weighted_avg(ores, dry_quantities_mt, "tio2_pct")
    na2o_pct = _weighted_avg(ores, dry_quantities_mt, "na2o_pct")
    k2o_pct = _weighted_avg(ores, dry_quantities_mt, "k2o_pct")
    slag_contribution_mt_by_ore = {
        ore.ore_id: compute_slag_contribution_mt(
            dry_quantities_mt.get(ore.ore_id, 0.0),
            sio2_pct=ore.chemistry.sio2_pct,
            al2o3_pct=ore.chemistry.al2o3_pct,
            cao_pct=ore.chemistry.cao_pct,
            mgo_pct=ore.chemistry.mgo_pct,
            tio2_pct=ore.chemistry.tio2_pct,
            mno_pct=ore.chemistry.mno_pct,
            na2o_pct=ore.chemistry.na2o_pct,
            k2o_pct=ore.chemistry.k2o_pct,
        )
        for ore in ores
    }

    effective_fe_pct = compute_effective_fe_pct(fe_t_pct, feo_pct, feo_in_slag_pct)
    slag_pct = compute_slag_forming_oxides_pct(
        sio2_pct=sio2_pct,
        al2o3_pct=al2o3_pct,
        cao_pct=cao_pct,
        mgo_pct=mgo_pct,
        tio2_pct=tio2_pct,
        mno_pct=mno_pct,
        na2o_pct=na2o_pct,
        k2o_pct=k2o_pct,
    )

    slag_mt = float(sum(slag_contribution_mt_by_ore.values()))
    slag_rate_kg_per_thm = (
        (slag_mt / fe_production_mt) * 1000.0 if fe_production_mt > 0 else 0.0
    )

    ore_cost_total_rs = 0.0
    for ore in ores:
        qty = float(quantities_mt.get(ore.ore_id, 0.0))
        ore_cost_total_rs += qty * float(ore.price_rs_per_mt)

    if fe_production_mt > 0 and isfinite(fe_production_mt):
        ore_cost_per_thm_rs = ore_cost_total_rs / fe_production_mt
    else:
        ore_cost_per_thm_rs = float("inf")

    objective_rs_per_thm = ore_cost_per_thm_rs + float(fuel_cost_per_thm_rs)

    return BlendEvaluation(
        quantities_mt=quantities_mt,
        shares_pct=shares_pct,
        total_qty_mt=total_qty_mt,
        ore_cost_total_rs=float(ore_cost_total_rs),
        ore_cost_per_thm_rs=float(ore_cost_per_thm_rs),
        fuel_cost_per_thm_rs=float(fuel_cost_per_thm_rs),
        objective_rs_per_thm=float(objective_rs_per_thm),
        fe_t_pct=float(fe_t_pct),
        effective_fe_pct=float(effective_fe_pct),
        fe_production_mt=float(fe_production_mt),
        slag_pct=float(slag_pct),
        slag_mt=float(slag_mt),
        feasible=True,
        violations=[],
        slag_rate_kg_per_thm=float(slag_rate_kg_per_thm),
        diagnostics={
            "formula": "dry_weight_fe_and_oxide_sum_slag",
            "total_dry_qty_mt": float(total_dry_qty_mt),
            "dry_weight_mt_by_ore": {
                ore_id: float(value) for ore_id, value in dry_quantities_mt.items()
            },
            "fe_contribution_mt_by_ore": {
                ore_id: float(value)
                for ore_id, value in fe_contribution_mt_by_ore.items()
            },
            "moisture_pct_by_ore": {
                ore.ore_id: float(ore.chemistry.moisture_pct) for ore in ores
            },
            "slag_contribution_mt_by_ore": {
                ore_id: float(value)
                for ore_id, value in slag_contribution_mt_by_ore.items()
            },
            "slag_rate_kg_per_thm": float(slag_rate_kg_per_thm),
            "slag_rate_denominator_mt": float(fe_production_mt),
            "slag_rate_denominator": "fe_production_mt",
        },
    )
