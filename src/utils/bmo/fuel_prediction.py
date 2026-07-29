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
from utils.bmo.fuel_rates import (
    ASSUMED_FUEL_PRICES_RS_PER_KG,
    estimate_fuel_rates_from_cost,
    estimate_fuel_rates_from_inputs,
)
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

# Fuel-ash ``fuel_id`` -> re-price key used by ASSUMED_FUEL_PRICES_RS_PER_KG.
_FUEL_ID_TO_PRICE_KEY = {"coke": "coke", "nut_coke": "nut_coke", "pci": "pci"}


def _current_fuel_prices_rs_per_kg(
    fuel_ash_inputs: list[FuelAshInput] | None,
) -> dict[str, float]:
    """Operator's current fuel prices (Rs/kg), falling back to assumed prices.

    Reads ``price_rs_per_mt`` from the coke / nut coke / PCI rows of the fuel-ash
    editor and converts to Rs/kg. Any missing or non-positive price keeps the
    assumed (model-baseline) price so that component is unchanged.
    """

    prices = dict(ASSUMED_FUEL_PRICES_RS_PER_KG)
    for fuel in fuel_ash_inputs or []:
        key = _FUEL_ID_TO_PRICE_KEY.get(str(fuel.fuel_id).strip().lower())
        if key is None:
            continue
        price_per_mt = float(getattr(fuel, "price_rs_per_mt", 0.0) or 0.0)
        if price_per_mt > 0.0:
            prices[key] = price_per_mt / 1000.0
    return prices


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
    fuel_rate_basis: str = "model_cost",
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
         - fuel_rate_basis: str - Where the displayed (re-priced) fuel rates come
           from. ``"model_cost"`` (optimized blends): decompose the ML-predicted
           fuel cost back into physical rates at the model's baseline prices,
           then re-price at the operator's current prices — blend-sensitive.
           ``"inputs"`` (manual blend): use the actual current coke / nut coke /
           PCI rates from the Fuel Ash table (seeded from the latest static
           dataset row), re-priced at current prices — the realised fuel cost.
           Either basis falls back to the other when its data is unavailable.

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
    # Rate basis differs by blend kind (see docstring): manual blends show the
    # realised cost from actual current rates; optimized blends convert the
    # ML-predicted cost so the display stays blend-sensitive.
    if fuel_rate_basis == "inputs":
        fuel_rates = estimate_fuel_rates_from_inputs(fuel_ash_inputs)
        fuel_rate_source = "fuel_ash_inputs"
        if fuel_rates is None:
            fuel_rates = estimate_fuel_rates_from_cost(
                fuel_cost_per_thm_rs=float(prediction.value),
                process_context=process_context,
                history_df=history_df,
                fuel_ash_inputs=fuel_ash_inputs,
            )
            fuel_rate_source = "model_cost_residual"
    else:
        # ``fuel_ash_inputs`` pins nut coke + PCI to the operator's visible run
        # inputs so only coke is back-solved from the predicted cost. Without
        # them the decomposition re-derived nut coke from history and reported
        # ~30 kg/THM against the plant's 70 kg/THM, which also threw the coke
        # residual (and so the re-priced fuel cost) off by the difference.
        fuel_rates = estimate_fuel_rates_from_cost(
            fuel_cost_per_thm_rs=float(prediction.value),
            process_context=process_context,
            history_df=history_df,
            fuel_ash_inputs=fuel_ash_inputs,
        )
        fuel_rate_source = "model_cost_residual"
        if fuel_rates is None:
            fuel_rates = estimate_fuel_rates_from_inputs(fuel_ash_inputs)
            fuel_rate_source = "fuel_ash_inputs"
    if fuel_rates is not None:
        blend.diagnostics["fuel_rate_estimate"] = fuel_rates.to_dict()
        blend.diagnostics["fuel_rate_estimate_source"] = fuel_rate_source
        # Re-price physical fuel rates at the operator's current prices. Prefer
        # the Fuel Ash editor rates; only reverse-solve from model cost when
        # those run inputs are unavailable. Stored in diagnostics only (display
        # use); the optimizer keeps minimising on the model's baseline-price
        # ``objective_rs_per_thm``.
        current_prices = _current_fuel_prices_rs_per_kg(fuel_ash_inputs)
        adjusted_fuel_cost = (
            fuel_rates.coke_rate_kg_thm * current_prices["coke"]
            + fuel_rates.nut_coke_rate_kg_thm * current_prices["nut_coke"]
            + fuel_rates.pci_rate_kg_thm * current_prices["pci"]
        )
        ore_cost = float(blend.objective_rs_per_thm) - float(blend.fuel_cost_per_thm_rs)
        blend.diagnostics["adjusted_fuel_cost_per_thm_rs"] = float(adjusted_fuel_cost)
        blend.diagnostics["adjusted_objective_rs_per_thm"] = float(
            ore_cost + adjusted_fuel_cost
        )
        blend.diagnostics["current_fuel_prices_rs_per_kg"] = current_prices
    return blend
