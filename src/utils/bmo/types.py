"""Typed data contracts for the BMO optimization workflow.

This module defines the ore chemistry, ore input, blend evaluation, and model
prediction structures shared by the BMO page, solvers, model service, and UI
components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Element mass fractions in the corresponding oxides.  These constants match
# the plant workbook and are also used by the editor ingestion layer so an
# elemental Mn/Ti analysis is converted exactly once before entering the
# calculation ledger.
MN_IN_MNO_FRACTION = 0.774461846
TI_IN_TIO2_FRACTION = 0.599341397


def oxide_pct_from_basis(value_pct: float, basis: str, *, element: str) -> float:
    """Return a canonical oxide percentage from an elemental-or-oxide input."""

    try:
        value = max(0.0, float(value_pct or 0.0))
    except (TypeError, ValueError):
        value = 0.0
    normalized = str(basis or "").strip().lower().replace("_", "")
    if element == "mn":
        return (
            value / MN_IN_MNO_FRACTION
            if normalized in {"mn", "elemental", "elementalmn"}
            else value
        )
    if element == "ti":
        return (
            value / TI_IN_TIO2_FRACTION
            if normalized in {"ti", "elemental", "elementalti"}
            else value
        )
    raise ValueError(f"Unsupported chemistry element: {element}")


@dataclass
class OreChemistry:
    """
    Store chemistry values for one ore or burden material.

    Fe, oxide, alkali, and moisture fields are kept together because blend
    evaluation must convert wet quantity to dry quantity before applying
    chemistry percentages. Defaults are intentionally zero so partial fallback
    mappings remain valid.

    Args:
         - fe_t_pct: float - Total iron percentage on dry basis.
         - moisture_pct: float - Moisture or total moisture percentage.
         - feo_pct: float - FeO percentage on dry basis.
         - sio2_pct: float - SiO2 percentage on dry basis.
         - al2o3_pct: float - Al2O3 percentage on dry basis.
         - cao_pct: float - CaO percentage on dry basis.
         - mgo_pct: float - MgO percentage on dry basis.
         - mno_pct: float - MnO percentage on dry basis.
         - tio2_pct: float - TiO2 percentage on dry basis.
         - p_pct: float - Phosphorus percentage on dry basis.
         - s_pct: float - Sulphur percentage on dry basis.
         - zn_pct: float - Zinc percentage on dry basis.
         - na2o_pct: float - Na2O percentage on dry basis.
         - k2o_pct: float - K2O percentage on dry basis.

    Returns:
         - return OreChemistry - Chemistry record used by blend calculations.
    """

    fe_t_pct: float
    moisture_pct: float = 0.0
    feo_pct: float = 0.0
    sio2_pct: float = 0.0
    al2o3_pct: float = 0.0
    cao_pct: float = 0.0
    mgo_pct: float = 0.0
    mno_pct: float = 0.0
    tio2_pct: float = 0.0
    p_pct: float = 0.0
    s_pct: float = 0.0
    zn_pct: float = 0.0
    na2o_pct: float = 0.0
    k2o_pct: float = 0.0


@dataclass
class OreInput:
    """
    Store optimizer input data for one selectable ore.

    Quantities, stock, and share limits are treated as wet-basis planning values.
    The nested chemistry record supplies the moisture needed to convert those
    wet quantities into dry weights for Fe and slag calculations.

    Args:
         - ore_id: str - Stable ore identifier used in quantity maps.
         - display_name: str - Human-readable ore name shown in the UI.
         - stock_mt: float - Available wet stock quantity in MT.
         - price_rs_per_mt: float - Ore price in Rs/MT.
         - min_share_pct: float - Minimum wet burden share percentage.
         - max_share_pct: float - Maximum wet burden share percentage.
         - chemistry: OreChemistry - Chemistry and moisture values for the ore.
         - metadata: dict[str, Any] - Source mapping metadata for diagnostics.

    Returns:
         - return OreInput - Typed ore input consumed by LP and DE solvers.
    """

    ore_id: str
    display_name: str
    stock_mt: float
    price_rs_per_mt: float
    min_share_pct: float
    max_share_pct: float
    chemistry: OreChemistry
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FuelAshInput:
    """
    Store fuel rate and ash chemistry values for slag contribution.

    Fuel quantity is represented as kg/THM because coke, nut coke, and PCI are
    usually tracked as furnace fuel rates. The ash chemistry fields describe
    the oxide composition inside the ash fraction, not the whole wet fuel. The
    workbook reports S and P as percentages in the dry fuel, so those two fields
    are handled on dry-fuel basis instead of ash basis.

    Args:
         - fuel_id: str - Stable fuel identifier used in diagnostics.
         - display_name: str - Human-readable fuel name shown in the UI.
         - enabled: bool - Whether this fuel should contribute ash to slag.
         - rate_kg_per_thm: float - Fuel rate in kg per THM on ``rate_basis``.
         - rate_basis: str - ``wet`` or ``dry``; controls the one moisture conversion.
         - price_rs_per_mt: float - Operator's current fuel price in Rs per MT (Rs/tonne).
         - moisture_pct: float - Fuel TM for coke/nut coke, or IM for PCI.
         - vm_pct: float - Volatile-matter (VM) percentage used by ash analysis.
         - ash_pct: float - Ash percentage in the fuel.
         - sio2_pct: float - SiO2 percentage inside the ash.
         - al2o3_pct: float - Al2O3 percentage inside the ash.
         - cao_pct: float - CaO percentage inside the ash.
         - mgo_pct: float - MgO percentage inside the ash.
         - fe2o3_pct: float - Fe2O3 percentage inside the ash for diagnostics.
         - mno_pct: float - Canonical MnO percentage inside the ash.
         - tio2_pct: float - TiO2 percentage inside the ash.
         - alkali_pct: float - Total alkali percentage inside ash, when reported.
         - na2o_pct: float - Na2O percentage inside the ash.
         - k2o_pct: float - K2O percentage inside the ash.
         - s_pct: float - Sulphur percentage on dry fuel basis.
         - p_pct: float - Phosphorus percentage on dry fuel basis.

    Returns:
         - return FuelAshInput - Fuel ash record used by blend calculations.
    """

    fuel_id: str
    display_name: str
    enabled: bool = True
    rate_kg_per_thm: float = 0.0
    rate_basis: str = "wet"
    price_rs_per_mt: float = 0.0
    moisture_pct: float = 0.0
    vm_pct: float = 0.0
    ash_pct: float = 0.0
    sio2_pct: float = 0.0
    al2o3_pct: float = 0.0
    cao_pct: float = 0.0
    mgo_pct: float = 0.0
    fe2o3_pct: float = 0.0
    mno_pct: float = 0.0
    tio2_pct: float = 0.0
    alkali_pct: float = 0.0
    na2o_pct: float = 0.0
    k2o_pct: float = 0.0
    s_pct: float = 0.0
    p_pct: float = 0.0
    chemistry_source: str = "default"


@dataclass
class FluxInput:
    """
    Store fixed flux quantity and chemistry values for slag contribution.

    Fluxes are not optimizer decision variables in the current BMO workflow.
    Operators provide wet quantity and chemistry values, then the calculation
    converts that quantity to dry weight and adds dry-basis flux oxides to the
    total slag estimate.

    Args:
         - flux_id: str - Stable flux identifier used in diagnostics.
         - display_name: str - Human-readable flux name shown in the UI.
         - enabled: bool - Whether this flux should contribute to slag.
         - wet_qty_mt: float - Wet flux quantity in MT.
         - moisture_pct: float - Flux moisture or total moisture percentage.
         - sio2_pct: float - SiO2 percentage on dry flux basis.
         - al2o3_pct: float - Al2O3 percentage on dry flux basis.
         - cao_pct: float - CaO percentage on dry flux basis.
         - mgo_pct: float - MgO percentage on dry flux basis.
         - fe2o3_pct: float - Fe2O3 percentage retained for diagnostics.
         - mno_pct: float - MnO percentage on dry flux basis.
         - tio2_pct: float - TiO2 percentage on dry flux basis.
         - na2o_pct: float - Na2O percentage on dry flux basis.
         - k2o_pct: float - K2O percentage on dry flux basis.
         - caf2_pct: float - CaF2 percentage on dry flux basis.
         - p_pct: float - Phosphorus percentage on dry flux basis.
         - s_pct: float - Sulphur percentage on dry flux basis.
         - zn_pct: float - Zinc percentage on dry flux basis.
         - loi_pct: float - LOI percentage retained for diagnostics.
         - price_rs_per_mt: float - Flux price, used when the LP optimises flux quantity.
         - stock_mt: float - Available flux stock; upper bound when the LP optimises flux.
         - optimizable: bool - If True, the LP decides this flux's quantity (0..stock)
           to satisfy the slag basicity bounds; otherwise it is a fixed addition.

    Returns:
         - return FluxInput - Flux record used by blend calculations.
    """

    flux_id: str
    display_name: str
    enabled: bool = True
    wet_qty_mt: float = 0.0
    moisture_pct: float = 0.0
    sio2_pct: float = 0.0
    al2o3_pct: float = 0.0
    cao_pct: float = 0.0
    mgo_pct: float = 0.0
    fe2o3_pct: float = 0.0
    mno_pct: float = 0.0
    tio2_pct: float = 0.0
    na2o_pct: float = 0.0
    k2o_pct: float = 0.0
    caf2_pct: float = 0.0
    p_pct: float = 0.0
    s_pct: float = 0.0
    zn_pct: float = 0.0
    loi_pct: float = 0.0
    price_rs_per_mt: float = 0.0
    stock_mt: float = 0.0
    optimizable: bool = False


@dataclass
class DustInput:
    """
    Store BF gas dust loss quantity and chemistry values.

    Dust is deducted from the total BF component balance after ore, flux, and
    fuel ash contributions are added. Values are stored as wet quantity plus
    dry-basis component percentages so the same dry-weight conversion can be
    applied before subtraction.

    Args:
         - dust_id: str - Stable dust identifier used in diagnostics.
         - display_name: str - Human-readable dust name shown in the UI.
         - enabled: bool - Whether this dust row should be deducted.
         - wet_qty_mt: float - Wet dust quantity in MT/day when basis is ``mt_per_day``.
         - quantity_kg_per_charge: float - Wet dust kg/charge when basis is
           ``kg_per_charge``.
         - rate_basis: str - ``mt_per_day`` or ``kg_per_charge``.
         - source: str - Provenance shown to the operator.
         - moisture_pct: float - Dust moisture percentage.
         - sio2_pct: float - SiO2 percentage on dry dust basis.
         - al2o3_pct: float - Al2O3 percentage on dry dust basis.
         - cao_pct: float - CaO percentage on dry dust basis.
         - mgo_pct: float - MgO percentage on dry dust basis.
         - fe_pct: float - Fe percentage on dry dust basis.
         - mn_pct: float - Mn percentage on dry dust basis.
         - p_pct: float - Phosphorus percentage on dry dust basis.
         - s_pct: float - Sulphur percentage on dry dust basis.
         - ti_pct: float - Titanium percentage on dry dust basis.
         - zn_pct: float - Zinc percentage on dry dust basis.
         - na2o_pct: float - Na2O percentage on dry dust basis.
         - k2o_pct: float - K2O percentage on dry dust basis.
         - caf2_pct: float - CaF2 percentage on dry dust basis.

    Returns:
         - return DustInput - Dust loss record used by full slag balance.
    """

    dust_id: str
    display_name: str
    enabled: bool = True
    wet_qty_mt: float = 0.0
    quantity_kg_per_charge: float = 0.0
    rate_basis: str = "mt_per_day"
    source: str = "default"
    moisture_pct: float = 0.0
    sio2_pct: float = 0.0
    al2o3_pct: float = 0.0
    cao_pct: float = 0.0
    mgo_pct: float = 0.0
    fe_pct: float = 0.0
    mn_pct: float = 0.0
    p_pct: float = 0.0
    s_pct: float = 0.0
    ti_pct: float = 0.0
    zn_pct: float = 0.0
    na2o_pct: float = 0.0
    k2o_pct: float = 0.0
    caf2_pct: float = 0.0


@dataclass
class SlagBalanceSettings:
    """
    Store plant assumptions used by the full BF slag balance.

    These values describe the split between pig iron, gas, and slag after ore,
    flux, fuel ash, and dust component masses are calculated. The correction
    factor lets BMO calibrate calculated slag to observed plant behavior
    without exposing PI C/Si/S/Others in the operator UI.

    Args:
         - enabled: bool - Whether full slag balance should replace simplified slag.
         - carbon_pct: float - Legacy carbon percentage assumed in pig iron.
         - silicon_pct: float - Legacy silicon percentage assumed in pig iron.
         - sulphur_pct: float - Legacy sulphur percentage assumed in pig iron.
         - other_pct: float - Legacy other non-Fe pig iron percentage.
         - pi_loss_pct: float - Pig iron loss percentage deducted from theoretical PI.
         - fe_to_pig_iron_fraction: float - Fraction of net Fe reporting to pig iron.
         - mn_recovery_pct: float - Mn and Ti recovery percentage to pig iron.
         - sulphur_gas_loss_pct: float - Sulphur percentage reporting to gas.
         - alkali_to_slag_fraction: float - Fraction of net alkali reporting to slag.
         - si_to_sio2_factor: float - Conversion factor from Si to SiO2.
         - fe_to_feo_factor: float - Conversion factor from remaining Fe to FeO.
         - mn_to_mno_factor: float - Conversion factor from remaining Mn to MnO.
         - slag_correction_factor: float - Empirical factor applied to final slag.

    Returns:
         - return SlagBalanceSettings - Full slag balance configuration.
    """

    enabled: bool = False
    carbon_pct: float = 0.0
    silicon_pct: float = 0.0
    sulphur_pct: float = 0.0
    other_pct: float = 0.0
    mn_pct: float = 0.0
    ti_pct: float = 0.0
    pi_loss_pct: float = 0.2
    fe_to_pig_iron_fraction: float = 0.999
    mn_recovery_pct: float = 60.0
    sulphur_gas_loss_pct: float = 10.0
    alkali_to_slag_fraction: float = 0.8
    si_to_sio2_factor: float = 2.14
    fe_to_feo_factor: float = 72.0 / 56.0
    mn_to_mno_factor: float = 1.291
    slag_correction_factor: float = 1.0


@dataclass
class SlagBalanceResult:
    """
    Store full BF slag balance output and diagnostics.

    The result keeps both the final slag MT and the intermediate component
    maps so calculations can be audited against the workbook sequence.

    Args:
         - total_slag_mt: float - Final slag quantity from full BF balance.
         - actual_pig_iron_mt: float - Actual pig iron quantity calculated by settings.
         - theoretical_pig_iron_mt: float - Theoretical pig iron before PI loss.
         - ore_components_mt: dict[str, float] - Component totals from ores.
         - flux_components_mt: dict[str, float] - Component totals from fluxes.
         - fuel_ash_components_mt: dict[str, float] - Component totals from fuel ash.
         - dust_components_mt: dict[str, float] - Component totals deducted as dust.
         - total_into_bf_mt: dict[str, float] - Ore + flux + fuel ash component totals.
         - net_into_bf_mt: dict[str, float] - Component totals after dust deduction.
         - slag_components_mt: dict[str, float] - Final slag component masses.
         - diagnostics: dict[str, Any] - Additional trace values for review.

    Returns:
         - return SlagBalanceResult - Full slag balance result and trace data.
    """

    total_slag_mt: float
    actual_pig_iron_mt: float
    theoretical_pig_iron_mt: float
    ore_components_mt: dict[str, float]
    flux_components_mt: dict[str, float]
    fuel_ash_components_mt: dict[str, float]
    dust_components_mt: dict[str, float]
    total_into_bf_mt: dict[str, float]
    net_into_bf_mt: dict[str, float]
    slag_components_mt: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlendEvaluation:
    """
    Store calculated metrics for one evaluated BMO blend.

    This object carries both planning values, such as wet quantities and ore
    cost, and chemistry values, such as dry-weight Fe production. Diagnostics
    include per-ore dry weights and Fe contributions for review.

    Args:
         - quantities_mt: dict[str, float] - Wet ore quantities keyed by ore id.
         - shares_pct: dict[str, float] - Wet burden shares keyed by ore id.
         - total_qty_mt: float - Total wet burden quantity in MT.
         - ore_cost_total_rs: float - Total wet ore purchase cost in Rs.
         - ore_cost_per_thm_rs: float - Ore cost divided by Fe production.
         - fuel_cost_per_thm_rs: float - Predicted or fallback fuel cost in Rs/THM.
         - objective_rs_per_thm: float - Ore plus fuel objective in Rs/THM.
         - fe_t_pct: float - Final dry-weight Fe percentage.
         - effective_fe_pct: float - FeO-adjusted Fe percentage retained for diagnostics.
         - fe_production_mt: float - Final Fe contribution from dry weights in MT.
         - slag_pct: float - Dry-weight slag percentage estimate.
         - slag_mt: float - Slag quantity estimate in MT.
         - feasible: bool - Whether the blend satisfies hard constraints.
         - violations: list[str] - Human-readable constraint violations.
         - slag_rate_kg_per_thm: float - Slag rate against the app's THM denominator.
         - slag_basicity: float - Slag basicity, CaO / SiO2.
         - slag_t_basicity: float - Total slag basicity, (CaO + MgO) / SiO2.
         - slag_ib4: float - IB4, (CaO + MgO) / (SiO2 + Al2O3).
         - slag_al2o3_pct: float - Al2O3 percentage of final slag.
         - slag_mgo_pct: float - MgO percentage of final slag.
         - slag_mgo_al2o3_ratio: float - MgO / Al2O3 mass ratio in final slag. Unlike
           the two percentages this is scale-free: it does not move when total slag
           mass changes, so it is purely a burden-selection quantity.
         - diagnostics: dict[str, Any] - Additional calculation and solver details.

    Returns:
         - return BlendEvaluation - Evaluated blend metrics and diagnostics.
    """

    quantities_mt: dict[str, float]
    shares_pct: dict[str, float]
    total_qty_mt: float
    ore_cost_total_rs: float
    ore_cost_per_thm_rs: float
    fuel_cost_per_thm_rs: float
    objective_rs_per_thm: float
    fe_t_pct: float
    effective_fe_pct: float
    fe_production_mt: float
    slag_pct: float
    slag_mt: float
    feasible: bool
    violations: list[str]
    slag_rate_kg_per_thm: float = 0.0
    slag_basicity: float = 0.0
    slag_t_basicity: float = 0.0
    slag_ib4: float = 0.0
    slag_al2o3_pct: float = 0.0
    slag_mgo_pct: float = 0.0
    slag_mgo_al2o3_ratio: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPrediction:
    """
    Store BMO fuel-cost model prediction output.

    Prediction records are attached to blend diagnostics so the page can show
    whether the ML model or deterministic fallback produced the fuel-cost term.

    Args:
         - value: float - Predicted or fallback fuel cost in Rs/THM.
         - model_loaded: bool - Whether the model artifact was loaded.
         - scaler_loaded: bool - Whether the scaler artifact was loaded.
         - used_fallback: bool - Whether fallback logic was used for the value.
         - missing_features: list[str] - Feature names not resolved before defaults.
         - imputed_features: list[str] - Feature names populated from defaults.
         - details: dict[str, Any] - Additional inference diagnostics.

    Returns:
         - return ModelPrediction - Fuel-cost prediction record.
    """

    value: float
    model_loaded: bool
    scaler_loaded: bool
    used_fallback: bool
    missing_features: list[str] = field(default_factory=list)
    imputed_features: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
