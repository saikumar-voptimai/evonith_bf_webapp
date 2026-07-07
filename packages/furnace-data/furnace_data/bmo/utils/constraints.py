"""Constraint validation helpers for BMO blend candidates.

This module checks optimized blend outputs against target production, slag,
stock, and ore-share requirements. The same checks are used by LP and DE paths
so the UI reports constraint warnings consistently across optimization methods.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite

import pandas as pd

from furnace_data.bmo.utils.types import BlendEvaluation, OreInput


def check_blend_constraints(
    blend: BlendEvaluation,
    ores: list[OreInput],
    *,
    target_production_mt: float,
    target_slag_qty_mt: float,
    target_slag_basicity_min: float | None = None,
    target_slag_basicity_max: float | None = None,
    target_slag_t_basicity_min: float | None = None,
    target_slag_t_basicity_max: float | None = None,
    fe_tolerance_mt: float = 0.5,
    slag_tolerance_mt: float = 0.0,
    basicity_tolerance: float = 1e-6,
) -> list[str]:
    """
    Check a completed blend against BMO physical and planning constraints.

    LP and DE both call this helper after evaluating a candidate. Keeping the
    final validation shared ensures the UI reports stock, share, target hot
    metal, and slag violations the same way for every optimization path. Target
    hot metal is checked on both sides so DE cannot treat over-production as a
    feasible result.

    Args:
         - blend: BlendEvaluation - Evaluated blend to validate.
         - ores: list[OreInput] - Ore inputs used to produce the blend.
         - target_production_mt: float - Target hot-metal production in MT.
         - target_slag_qty_mt: float - Maximum allowed slag quantity in MT.
         - target_slag_basicity_min: float | None - Minimum CaO / SiO2 basicity.
         - target_slag_basicity_max: float | None - Maximum CaO / SiO2 basicity.
         - target_slag_t_basicity_min: float | None - Minimum (CaO + MgO) / SiO2 basicity.
         - target_slag_t_basicity_max: float | None - Maximum (CaO + MgO) / SiO2 basicity.
         - fe_tolerance_mt: float - Allowed Fe production tolerance in MT.
         - slag_tolerance_mt: float - Allowed slag tolerance in MT. The
           default is zero because BMO treats max slag as a strict cap.
         - basicity_tolerance: float - Numeric tolerance for basicity bounds.

    Returns:
         - return list[str] - Human-readable constraint violation messages.
    """

    violations: list[str] = []

    if blend.fe_production_mt < target_production_mt - fe_tolerance_mt:
        violations.append(
            f"Fe production below required target: {blend.fe_production_mt:.2f} < {target_production_mt:.2f} MT."
        )
    if blend.fe_production_mt > target_production_mt + fe_tolerance_mt:
        violations.append(
            f"Fe production above required target: {blend.fe_production_mt:.2f} > {target_production_mt:.2f} MT."
        )

    if blend.slag_mt > target_slag_qty_mt + slag_tolerance_mt:
        violations.append(
            f"Slag exceeds bound: {blend.slag_mt:.2f} > {target_slag_qty_mt:.2f} MT."
        )

    def _check_basicity(
        *,
        label: str,
        value: float,
        denominator_key: str,
        min_value: float | None,
        max_value: float | None,
    ) -> None:
        if min_value is None and max_value is None:
            return
        denominator = float(
            blend.diagnostics.get(denominator_key, 0.0) or 0.0
        )
        basicity = float(value or 0.0)
        if denominator <= 0.0 or not isfinite(basicity):
            violations.append(
                f"{label} unavailable: SiO2 denominator is zero."
            )
        else:
            if min_value is not None and basicity < min_value - basicity_tolerance:
                violations.append(
                    f"{label} below bound: {basicity:.3f} < {min_value:.3f}."
                )
            if max_value is not None and basicity > max_value + basicity_tolerance:
                violations.append(
                    f"{label} above bound: {basicity:.3f} > {max_value:.3f}."
                )

    _check_basicity(
        label="Slag basicity",
        value=float(getattr(blend, "slag_basicity", 0.0) or 0.0),
        denominator_key="slag_basicity_denominator_mt",
        min_value=(
            float(target_slag_basicity_min)
            if target_slag_basicity_min is not None
            else None
        ),
        max_value=(
            float(target_slag_basicity_max)
            if target_slag_basicity_max is not None
            else None
        ),
    )
    _check_basicity(
        label="Slag T Basicity",
        value=float(getattr(blend, "slag_t_basicity", 0.0) or 0.0),
        denominator_key="slag_t_basicity_denominator_mt",
        min_value=(
            float(target_slag_t_basicity_min)
            if target_slag_t_basicity_min is not None
            else None
        ),
        max_value=(
            float(target_slag_t_basicity_max)
            if target_slag_t_basicity_max is not None
            else None
        ),
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


def validate_selected_pellet_inputs(
    ores: list[OreInput],
    *,
    max_chemistry_age_days: int = 30,
    now: datetime | None = None,
) -> list[str]:
    """
    Return blocking pre-run issues for selected pellet materials.

    Pellet can materially change Fe, slag, and fuel-model inputs. If stock or
    chemistry is not backed by fresh source data, the operator must explicitly
    review the editor values before running BMO.
    """

    issues: list[str] = []
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)

    for ore in ores:
        material_key = str(ore.metadata.get("material_key", "")).lower()
        if "pellet" not in material_key and "pellet" not in ore.display_name.lower():
            continue

        stock_source = str(ore.metadata.get("stock_source", ""))
        if float(ore.stock_mt or 0.0) <= 0.0:
            issues.append(f"{ore.display_name}: enter positive pellet stock before running.")
        elif stock_source != "offline_db":
            issues.append(
                f"{ore.display_name}: stock is not from raw_material_stock; confirm the editor stock value."
            )

        chemistry_source = str(ore.metadata.get("chemistry_source", ""))
        sample_timestamp = str(ore.metadata.get("chemistry_sample_timestamp", "") or "")
        if chemistry_source == "fallback":
            issues.append(
                f"{ore.display_name}: chemistry is using fallback values; confirm or edit chemistry before running."
            )
            continue
        if not sample_timestamp:
            issues.append(
                f"{ore.display_name}: chemistry timestamp is missing; confirm or edit chemistry before running."
            )
            continue

        sample_time = pd.to_datetime(sample_timestamp, utc=True, errors="coerce")
        if pd.isna(sample_time):
            issues.append(
                f"{ore.display_name}: chemistry timestamp is invalid; confirm or edit chemistry before running."
            )
            continue
        age_days = (pd.Timestamp(current) - sample_time).total_seconds() / 86400.0
        if age_days > float(max_chemistry_age_days):
            issues.append(
                f"{ore.display_name}: chemistry sample is {age_days:.0f} days old; confirm or edit chemistry before running."
            )

    return issues
