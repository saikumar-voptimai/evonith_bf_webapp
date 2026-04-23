from __future__ import annotations

from math import isfinite

from domain.bmo.types import BlendEvaluation, OreInput

FE_FROM_FEO_FACTOR = 55.845 / 71.844
SIO2_FROM_SI_FACTOR = 60.0 / 28.0


def _weighted_avg(ores: list[OreInput], quantities: dict[str, float], attr: str) -> float:
    total_qty = sum(quantities.values())
    if total_qty <= 0:
        return 0.0
    acc = 0.0
    for ore in ores:
        qty = float(quantities.get(ore.ore_id, 0.0))
        if qty <= 0:
            continue
        value = float(getattr(ore.chemistry, attr, 0.0) or 0.0)
        acc += (qty / total_qty) * value
    return acc


def compute_effective_fe_pct(fe_t_pct: float, feo_pct: float, feo_in_slag_pct: float) -> float:
    return float(fe_t_pct + (feo_pct - feo_in_slag_pct) * FE_FROM_FEO_FACTOR)


def compute_slag_pct(
    sio2_pct: float,
    al2o3_pct: float,
    cao_pct: float,
    mgo_pct: float,
    mno_pct: float,
    si_in_slag_pct: float,
) -> float:
    return float(
        (sio2_pct - (si_in_slag_pct * SIO2_FROM_SI_FACTOR))
        + al2o3_pct
        + cao_pct
        + mgo_pct
        + mno_pct
    )


def evaluate_blend(
    ores: list[OreInput],
    quantities_mt: dict[str, float],
    feo_in_slag_pct: float,
    si_in_slag_pct: float,
    fuel_cost_per_thm_rs: float = 0.0,
) -> BlendEvaluation:
    total_qty_mt = float(sum(quantities_mt.values()))
    shares_pct: dict[str, float] = {}

    if total_qty_mt > 0:
        shares_pct = {
            ore_id: (qty / total_qty_mt) * 100.0 for ore_id, qty in quantities_mt.items()
        }
    else:
        shares_pct = {ore_id: 0.0 for ore_id in quantities_mt}

    fe_t_pct = _weighted_avg(ores, quantities_mt, "fe_t_pct")
    feo_pct = _weighted_avg(ores, quantities_mt, "feo_pct")
    sio2_pct = _weighted_avg(ores, quantities_mt, "sio2_pct")
    al2o3_pct = _weighted_avg(ores, quantities_mt, "al2o3_pct")
    cao_pct = _weighted_avg(ores, quantities_mt, "cao_pct")
    mgo_pct = _weighted_avg(ores, quantities_mt, "mgo_pct")
    mno_pct = _weighted_avg(ores, quantities_mt, "mno_pct")

    effective_fe_pct = compute_effective_fe_pct(fe_t_pct, feo_pct, feo_in_slag_pct)
    slag_pct = compute_slag_pct(
        sio2_pct=sio2_pct,
        al2o3_pct=al2o3_pct,
        cao_pct=cao_pct,
        mgo_pct=mgo_pct,
        mno_pct=mno_pct,
        si_in_slag_pct=si_in_slag_pct,
    )

    fe_production_mt = (effective_fe_pct / 100.0) * total_qty_mt if total_qty_mt > 0 else 0.0
    slag_mt = (slag_pct / 100.0) * total_qty_mt if total_qty_mt > 0 else 0.0

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
        diagnostics={},
    )

