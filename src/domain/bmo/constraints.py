from __future__ import annotations

from domain.bmo.types import BlendEvaluation, OreInput


def check_blend_constraints(
    blend: BlendEvaluation,
    ores: list[OreInput],
    *,
    target_total_qty_mt: float,
    min_fe_production_mt: float,
    max_fe_production_mt: float,
    target_slag_qty_mt: float,
    qty_tolerance_mt: float = 0.5,
) -> list[str]:
    violations: list[str] = []

    if abs(blend.total_qty_mt - target_total_qty_mt) > qty_tolerance_mt:
        violations.append(
            f"Total quantity mismatch: {blend.total_qty_mt:.2f} vs target {target_total_qty_mt:.2f} MT."
        )

    if blend.fe_production_mt < min_fe_production_mt:
        violations.append(
            f"Fe production below minimum: {blend.fe_production_mt:.2f} < {min_fe_production_mt:.2f} MT."
        )

    if blend.fe_production_mt > max_fe_production_mt:
        violations.append(
            f"Fe production above maximum: {blend.fe_production_mt:.2f} > {max_fe_production_mt:.2f} MT."
        )

    if blend.slag_mt > target_slag_qty_mt:
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


def validate_ore_bounds(ores: list[OreInput], target_total_qty_mt: float) -> list[str]:
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

        min_qty = (ore.min_share_pct / 100.0) * target_total_qty_mt
        max_qty = min((ore.max_share_pct / 100.0) * target_total_qty_mt, ore.stock_mt)
        if min_qty > max_qty + 1e-6:
            errors.append(
                (
                    f"{ore.display_name}: infeasible quantity bounds after stock cap "
                    f"(min_qty={min_qty:.2f}, max_qty={max_qty:.2f})."
                )
            )

    if min_sum > 1.0 + 1e-6:
        errors.append(
            f"Sum of minimum shares is {min_sum * 100.0:.2f}% (must be <= 100%)."
        )
    if max_sum < 1.0 - 1e-6:
        errors.append(
            f"Sum of maximum shares is {max_sum * 100.0:.2f}% (must be >= 100%)."
        )

    return errors

