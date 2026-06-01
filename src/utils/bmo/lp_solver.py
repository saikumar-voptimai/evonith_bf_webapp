"""LP baseline solver for BMO wet burden planning.

This module builds the deterministic ore-cost LP baseline using wet quantity
bounds while applying dry-weight Fe and slag-forming oxide coefficients for
production and quality constraints.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from utils.bmo.calculations import (
    compute_dry_fraction,
    compute_slag_forming_oxides_pct,
    evaluate_blend,
)
from utils.bmo.constraints import check_blend_constraints, validate_ore_bounds
from utils.bmo.types import BlendEvaluation, OreInput


def run_lp_baseline(
    ores: list[OreInput],
    *,
    target_production_mt: float,
    target_slag_qty_mt: float,
    feo_in_slag_pct: float,
) -> tuple[BlendEvaluation | None, list[str]]:
    """
    Run the deterministic LP baseline for selected BMO ores.

    The LP variables are wet ore quantities, and total blend quantity is a
    solver output. Share limits are represented as linear relationships against
    ``sum(qty)`` instead of using a fixed target burden quantity.

    Args:
         - ores: list[OreInput] - Ores selected for optimization.
         - target_production_mt: float - Target hot-metal production in MT.
         - target_slag_qty_mt: float - Maximum dry-weight slag quantity in MT.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.

    Returns:
         - return tuple[BlendEvaluation | None, list[str]] - LP blend and errors.
    """

    pre_errors = validate_ore_bounds(ores)
    if pre_errors:
        return None, pre_errors

    n = len(ores)
    c = np.array([float(ore.price_rs_per_mt) for ore in ores], dtype=float)

    fe_coeff = np.array(
        [
            compute_dry_fraction(ore.chemistry.moisture_pct)
            * (float(ore.chemistry.fe_t_pct) / 100.0)
            for ore in ores
        ],
        dtype=float,
    )
    slag_coeff = np.array(
        [
            compute_dry_fraction(ore.chemistry.moisture_pct)
            * compute_slag_forming_oxides_pct(
                sio2_pct=ore.chemistry.sio2_pct,
                al2o3_pct=ore.chemistry.al2o3_pct,
                cao_pct=ore.chemistry.cao_pct,
                mgo_pct=ore.chemistry.mgo_pct,
                tio2_pct=ore.chemistry.tio2_pct,
                mno_pct=ore.chemistry.mno_pct,
                na2o_pct=ore.chemistry.na2o_pct,
                k2o_pct=ore.chemistry.k2o_pct,
            )
            / 100.0
            for ore in ores
        ],
        dtype=float,
    )

    a_ub_rows = [
        -fe_coeff,  # Fe >= target
        slag_coeff,  # Slag <= target
    ]
    b_ub_values = [
        -float(target_production_mt),
        float(target_slag_qty_mt),
    ]

    for idx, ore in enumerate(ores):
        min_share = float(ore.min_share_pct) / 100.0
        max_share = float(ore.max_share_pct) / 100.0

        min_share_row = np.full(n, min_share, dtype=float)
        min_share_row[idx] -= 1.0
        a_ub_rows.append(min_share_row)
        b_ub_values.append(0.0)

        max_share_row = np.full(n, -max_share, dtype=float)
        max_share_row[idx] += 1.0
        a_ub_rows.append(max_share_row)
        b_ub_values.append(0.0)

    A_ub = np.vstack(a_ub_rows)
    b_ub = np.array(b_ub_values, dtype=float)

    bounds: list[tuple[float, float]] = []
    for ore in ores:
        bounds.append((0.0, max(0.0, float(ore.stock_mt))))

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not result.success or result.x is None:
        err = result.message or "LP solver failed."
        return None, [f"LP infeasible or failed: {err}"]

    quantities = {ore.ore_id: float(result.x[idx]) for idx, ore in enumerate(ores)}
    blend = evaluate_blend(
        ores=ores,
        quantities_mt=quantities,
        feo_in_slag_pct=feo_in_slag_pct,
        fuel_cost_per_thm_rs=0.0,
    )
    violations = check_blend_constraints(
        blend,
        ores,
        target_production_mt=target_production_mt,
        target_slag_qty_mt=target_slag_qty_mt,
    )
    blend.feasible = len(violations) == 0
    blend.violations = violations
    return blend, []
