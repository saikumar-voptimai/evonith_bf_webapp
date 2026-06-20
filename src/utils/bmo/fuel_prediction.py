"""Fuel-cost enrichment for evaluated BMO blends.

This module connects deterministic blend chemistry calculations with the BMO
fuel-cost model service. It builds the model feature payload for a solved blend,
predicts fuel Rs/THM, and re-evaluates the blend so LP and DE results expose the
same total-cost fields.
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from utils.bmo.calculations import evaluate_blend
from utils.bmo.feature_builder import PreBuiltFeatureContext, build_feature_payload
from utils.bmo.fuel_rates import estimate_fuel_rates_from_cost
from utils.bmo.types import (
    BlendEvaluation,
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceSettings,
)

if TYPE_CHECKING:
    from utils.bmo.model_service import FuelUnitCostModelService


def evaluate_blend_with_fuel_prediction(
    *,
    ores: list[OreInput],
    quantities_mt: Mapping[str, float],
    feo_in_slag_pct: float,
    model_service: FuelUnitCostModelService,
    process_context: Mapping[str, Any] | None,
    history_df: pd.DataFrame | None,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
    slag_balance_settings: SlagBalanceSettings | None = None,
    prebuilt_context: PreBuiltFeatureContext | None = None,
    hot_metal_target_mt: float | None = None,
) -> BlendEvaluation:
    """
    Evaluate a blend and attach the model-predicted fuel unit cost.

    LP optimization still decides quantities from ore economics and physical
    constraints. This helper is used after a quantity vector exists so LP and
    DE results can both show a comparable fuel-cost prediction.

    Args:
         - ores: list[OreInput] - Ores included in the solved blend.
         - quantities_mt: Mapping[str, float] - Solved ore quantities keyed by ore id.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - model_service: FuelUnitCostModelService - Fuel-cost prediction service.
         - process_context: Mapping[str, Any] | None - Latest process variables.
         - history_df: pd.DataFrame | None - Historical process data for lagged features.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
         - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
         - dust_inputs: list[DustInput] | None - Dust rows deducted in final balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.
         - hot_metal_target_mt: float | None - Operator HM basis for cost/slag/model THM fields.

    Returns:
         - return BlendEvaluation - Blend metrics with fuel prediction diagnostics.
    """

    quantities = {str(ore_id): float(qty) for ore_id, qty in quantities_mt.items()}
    ore_name_by_id = {ore.ore_id: ore.display_name for ore in ores}
    if prebuilt_context is not None:
        prediction = model_service.predict_with_prebuilt(
            prebuilt_context, quantities, history_df
        )
    else:
        feature_payload = build_feature_payload(
            quantities_mt=quantities,
            ore_display_name_by_id=ore_name_by_id,
            process_context=process_context,
            ores=ores,
            hot_metal_target_mt=hot_metal_target_mt,
        )
        prediction = model_service.predict(feature_payload, history_df)
    blend = evaluate_blend(
        ores=ores,
        quantities_mt=quantities,
        feo_in_slag_pct=feo_in_slag_pct,
        fuel_cost_per_thm_rs=float(prediction.value),
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=flux_inputs,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
        hot_metal_target_mt=hot_metal_target_mt,
    )
    blend.diagnostics["model_prediction"] = prediction
    blend.diagnostics["feature_details"] = prediction.details
    fuel_rates = estimate_fuel_rates_from_cost(
        fuel_cost_per_thm_rs=float(prediction.value),
        process_context=process_context,
        history_df=history_df,
    )
    if fuel_rates is not None:
        blend.diagnostics["fuel_rate_estimate"] = fuel_rates.to_dict()
    return blend
