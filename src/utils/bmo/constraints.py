"""Constraint validation helpers for BMO blend candidates.

This module checks optimized blend outputs against production, slag, stock,
and ore-share requirements. The same checks are used by LP and DE paths so
the UI reports constraint warnings consistently across optimization methods.
"""

from __future__ import annotations

from utils.bmo.types import BlendEvaluation, OreInput


def check_blend_constraints(
    blend: BlendEvaluation,
    ores: list[OreInput],
    *,
    target_production_mt: float,
    target_slag_qty_mt: float,
    fe_tolerance_mt: float = 0.5,
    slag_tolerance_mt: float = 0.5,
) -> list[str]:
    """
    Check a completed blend against BMO physical and planning constraints.

    LP and DE both call this helper after evaluating a candidate. Keeping the
    final validation shared ensures the UI reports stock, share, Fe, and slag
    violations the same way for every optimization path.

    Args:
         - blend: BlendEvaluation - Evaluated blend to validate.
         - ores: list[OreInput] - Ore inputs used to produce the blend.
         - target_production_mt: float - Target hot-metal production in MT.
         - target_slag_qty_mt: float - Maximum allowed slag quantity in MT.
         - fe_tolerance_mt: float - Allowed Fe production tolerance in MT.
         - slag_tolerance_mt: float - Allowed slag tolerance in MT.

    Returns:
         - return list[str] - Human-readable constraint violation messages.
    """

    violations: list[str] = []

    if blend.fe_production_mt < target_production_mt - fe_tolerance_mt:
        violations.append(
            f"Hot metal below target: {blend.fe_production_mt:.2f} < {target_production_mt:.2f} MT."
        )

    if blend.slag_mt > target_slag_qty_mt + slag_tolerance_mt:
        violations.append(
            f"Slag exceeds bound: {blend.slag_mt:.2f} > {target_slag_qty_mt:.2f} MT."
        )

    for ore in ores:
        qty = float(blend.quantities_mt.get(ore.ore_id, 0.0))
        share_pct = float(blend.shares_pct.get(ore.ore_id, 0.0))

        if qty - float(ore.stock_mt) > 1e-6:
            violations.append(
                f"{ore.display_name}: quantity {qty:.2f} MT exceeds stock {ore.stock_mt:.2f} MT."
            )

        if share_pct + 1e-6 < float(ore.min_share_pct):
            violations.append(
                f"{ore.display_name}: share {share_pct:.2f}% below min {ore.min_share_pct:.2f}%."
            )

        if share_pct - 1e-6 > float(ore.max_share_pct):
            violations.append(
                f"{ore.display_name}: share {share_pct:.2f}% above max {ore.max_share_pct:.2f}%."
            )

    return violations


def validate_ore_bounds(ores: list[OreInput]) -> list[str]:
    """
    Validate ore share and stock bounds before running an optimizer.

    These checks catch impossible input limits before SciPy or HiGHS is called.
    They are intentionally limited to static ore properties that can be verified
    without solving the full blend problem.

    Args:
         - ores: list[OreInput] - Candidate ores with stock and share limits.

    Returns:
         - return list[str] - Pre-run infeasibility or input validation errors.
    """

    errors: list[str] = []
    min_sum = 0.0
    max_sum = 0.0

    for ore in ores:
        if ore.min_share_pct > ore.max_share_pct:
            errors.append(
                f"{ore.display_name}: min_share_pct ({ore.min_share_pct}) > max_share_pct ({ore.max_share_pct})."
            )
        if ore.stock_mt < 0:
            errors.append(f"{ore.display_name}: stock_mt cannot be negative.")
        if ore.price_rs_per_mt < 0:
            errors.append(f"{ore.display_name}: price_rs_per_mt cannot be negative.")

        min_sum += ore.min_share_pct / 100.0
        max_sum += ore.max_share_pct / 100.0

    if min_sum > 1.0 + 1e-6:
        errors.append(
            f"Sum of minimum shares is {min_sum * 100.0:.2f}% (must be <= 100%)."
        )
    if max_sum < 1.0 - 1e-6:
        errors.append(
            f"Sum of maximum shares is {max_sum * 100.0:.2f}% (must be >= 100%)."
        )
    if all(float(ore.stock_mt) <= 0.0 for ore in ores):
        errors.append("At least one selected ore must have positive stock.")

    return errors
