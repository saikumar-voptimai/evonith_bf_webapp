"""Objective evaluation for BMO nonlinear total-cost optimization.

This module converts DE candidate wet-quantity vectors into blend evaluations,
predicts fuel cost for each candidate, applies constraint penalties, and returns
rich objective diagnostics to the shared optimization runtime.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from domain.optimization_runtime import ObjectiveResult
from utils.bmo.coke_correction import (
    CokeCorrectionReference,
    CokeCorrectionSettings,
)
from utils.bmo.constraints import check_blend_constraints
from utils.bmo.feature_builder import PreBuiltFeatureContext
from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.types import (
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceSettings,
)


class BmoObjectiveEvaluator:
    """
    Evaluate BMO candidate quantities as total-cost objective values.

    DE proposes wet ore quantity vectors. This evaluator converts each vector
    into a full blend evaluation, asks the fuel model for Rs/THM, and applies
    soft penalties so infeasible candidates can still guide the search.

    Args:
         - ores: list[OreInput] - Ores available to the optimizer.
         - target_production_mt: float - Target hot-metal production in MT.
         - target_slag_qty_mt: float - Maximum allowed slag quantity in MT.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - max_burden_qty_mt: float | None - Charging-throughput ceiling on total wet
           IBRM + flux in MT. ``None`` leaves the burden quantity unbounded.
         - model_service: FuelUnitCostModelService - Fuel-cost prediction service.
         - process_context: dict[str, float] | None - Latest process variables.
         - history_df: pd.DataFrame | None - Historical process data for lagged features.
         - penalty_cfg: dict[str, Any] - Penalty weights used for soft constraints.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
         - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
         - dust_inputs: list[DustInput] | None - Dust rows deducted in final balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.

    Returns:
         - return BmoObjectiveEvaluator - Configured evaluator for quantity vectors.
    """

    def __init__(
        self,
        *,
        ores: list[OreInput],
        target_production_mt: float,
        target_slag_qty_mt: float,
        feo_in_slag_pct: float,
        model_service: FuelUnitCostModelService,
        process_context: dict[str, float] | None,
        history_df: pd.DataFrame | None,
        penalty_cfg: dict[str, Any],
        target_slag_basicity_min: float | None = None,
        target_slag_basicity_max: float | None = None,
        target_slag_t_basicity_min: float | None = None,
        target_slag_t_basicity_max: float | None = None,
        target_slag_al2o3_max_pct: float | None = None,
        target_slag_mgo_min_pct: float | None = None,
        target_slag_mgo_al2o3_ratio_min: float | None = None,
        max_burden_qty_mt: float | None = None,
        fuel_ash_inputs: list[FuelAshInput] | None = None,
        flux_inputs: list[FluxInput] | None = None,
        dust_inputs: list[DustInput] | None = None,
        slag_balance_settings: SlagBalanceSettings | None = None,
        prebuilt_context: PreBuiltFeatureContext | None = None,
        hot_metal_target_mt: float | None = None,
        fe_tolerance_mt: float = 0.5,
        coke_correction_settings: CokeCorrectionSettings | None = None,
        coke_correction_reference: CokeCorrectionReference | None = None,
        hot_metal_si_pct: float | None = None,
        fuel_rate_anchor_basis: str = "model_cost",
        charge_mass_mt: float = 26.4,
    ) -> None:
        """
        Store optimizer inputs and precompute array forms of bounds.

        These cached arrays keep each DE objective call lightweight. Constraint
        targets and penalty settings are also normalized to floats so candidate
        evaluation is deterministic across Streamlit reruns and tests.

        Args:
             - ores: list[OreInput] - Ores available to the optimizer.
             - target_production_mt: float - Target hot-metal production in MT.
             - target_slag_qty_mt: float - Maximum allowed slag quantity in MT.
             - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
             - model_service: FuelUnitCostModelService - Fuel-cost prediction service.
             - process_context: dict[str, float] | None - Latest process variables.
             - history_df: pd.DataFrame | None - Historical process data for lagged features.
             - penalty_cfg: dict[str, Any] - Penalty weights used for soft constraints.
             - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
             - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
             - dust_inputs: list[DustInput] | None - Dust rows deducted in final balance.
             - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.

        Returns:
             - return None - Initializes evaluator state.
        """

        self.ores = ores
        self.target_production_mt = float(target_production_mt)
        self.target_slag_qty_mt = float(target_slag_qty_mt)
        self.feo_in_slag_pct = float(feo_in_slag_pct)
        self.target_slag_basicity_min = (
            float(target_slag_basicity_min)
            if target_slag_basicity_min is not None
            else None
        )
        self.target_slag_basicity_max = (
            float(target_slag_basicity_max)
            if target_slag_basicity_max is not None
            else None
        )
        self.target_slag_t_basicity_min = (
            float(target_slag_t_basicity_min)
            if target_slag_t_basicity_min is not None
            else None
        )
        self.target_slag_t_basicity_max = (
            float(target_slag_t_basicity_max)
            if target_slag_t_basicity_max is not None
            else None
        )
        # Slag-quality limits. DE penalises these rather than rejecting outright,
        # so an over-constrained problem still returns a best-effort blend with
        # its violations named instead of an empty result pane.
        self.target_slag_al2o3_max_pct = (
            float(target_slag_al2o3_max_pct)
            if target_slag_al2o3_max_pct is not None
            else None
        )
        self.target_slag_mgo_min_pct = (
            float(target_slag_mgo_min_pct)
            if target_slag_mgo_min_pct is not None
            else None
        )
        self.target_slag_mgo_al2o3_ratio_min = (
            float(target_slag_mgo_al2o3_ratio_min)
            if target_slag_mgo_al2o3_ratio_min is not None
            else None
        )
        self.max_burden_qty_mt = (
            float(max_burden_qty_mt)
            if max_burden_qty_mt is not None and float(max_burden_qty_mt) > 0.0
            else None
        )
        self.model_service = model_service
        self.process_context = process_context or {}
        self.history_df = history_df
        self.fuel_ash_inputs = fuel_ash_inputs
        self.flux_inputs = flux_inputs
        # Optimisable fluxes (dolomite/quartz) become DE decision variables so the
        # solver can add flux to satisfy basicity; the rest are fixed additions.
        _all_fluxes = list(flux_inputs or [])
        self.variable_fluxes = [
            flux
            for flux in _all_fluxes
            if flux.optimizable and flux.enabled and float(flux.stock_mt) > 0.0
        ]
        _variable_flux_ids = {flux.flux_id for flux in self.variable_fluxes}
        self.fixed_fluxes = [
            flux for flux in _all_fluxes if flux.flux_id not in _variable_flux_ids
        ]
        self.n_flux = len(self.variable_fluxes)
        self.dust_inputs = dust_inputs
        self.slag_balance_settings = slag_balance_settings
        # Both are resolved once, before the search starts, and held frozen for
        # every candidate. That is what makes the corrected objective a single
        # consistent function: a term whose reference could not be resolved is
        # off for the whole run rather than flickering between iterations, so
        # the search never chases a moving cost surface.
        self.coke_correction_settings = coke_correction_settings
        self.coke_correction_reference = coke_correction_reference
        # Constant across candidates by design. The Si model is blend-flat, and
        # calling it per candidate would cost thousands of XGBoost inferences to
        # move the objective by a fraction of a kg/THM; a constant offset does
        # not distort the search at all.
        self.hot_metal_si_pct = hot_metal_si_pct
        # The chosen anchor is propagated into the final fuel-ash ledger, so the
        # objective and the displayed slag use the same physical fuel rates.
        self.fuel_rate_anchor_basis = fuel_rate_anchor_basis
        self.charge_mass_mt = float(charge_mass_mt)
        self.penalty_cfg = penalty_cfg
        self.prebuilt_context = prebuilt_context
        self.hot_metal_target_mt = hot_metal_target_mt
        self.fe_tolerance_mt = float(fe_tolerance_mt)
        self.stocks = np.array([float(ore.stock_mt) for ore in ores], dtype=float)
        self.min_shares = np.array(
            [float(ore.min_share_pct) / 100.0 for ore in ores], dtype=float
        )
        self.max_shares = np.array(
            [float(ore.max_share_pct) / 100.0 for ore in ores], dtype=float
        )

    def evaluate_quantities(
        self,
        quantities_vector: np.ndarray,
        flux_quantities: np.ndarray | None = None,
    ) -> ObjectiveResult:
        """
        Evaluate one candidate wet-quantity vector for DE optimization.

        The returned objective contains both the scalar value required by SciPy
        and rich diagnostics for the Streamlit comparison view. Production
        shortfall, production excess, slag, stock, and share violations are
        penalized but also reported explicitly. Penalizing production excess
        keeps DE anchored to the operator's target hot metal without requiring
        a separate maximum-production UI input.

        Args:
             - quantities_vector: np.ndarray - Candidate wet ore quantities in MT.
             - flux_quantities: np.ndarray | None - Candidate quantities (MT) for the
               optimisable fluxes (dolomite/quartz), in ``self.variable_fluxes`` order.

        Returns:
             - return ObjectiveResult - Penalized objective, components, and diagnostics.
        """

        qty = np.asarray(quantities_vector, dtype=float)
        quantities = {ore.ore_id: float(qty[idx]) for idx, ore in enumerate(self.ores)}

        flux_qty = (
            np.asarray(flux_quantities, dtype=float)
            if flux_quantities is not None
            else np.zeros(self.n_flux, dtype=float)
        )
        # Rebuild the flux list with candidate quantities for optimisable fluxes so
        # the blend (and its basicity) reflects the flux the DE candidate adds.
        candidate_flux_inputs = list(self.fixed_fluxes) + [
            replace(flux, wet_qty_mt=float(flux_qty[j]))
            for j, flux in enumerate(self.variable_fluxes)
        ]

        # DE prices the CORRECTED fuel in its objective, so its slag has to be on
        # the corrected-fuel basis too. The page's LP path deliberately does not
        # do this - see ``evaluate_blend_with_fuel_prediction``.
        blend = evaluate_blend_with_fuel_prediction(
            recompute_slag_with_corrected_fuel=True,
            ores=self.ores,
            quantities_mt=quantities,
            feo_in_slag_pct=self.feo_in_slag_pct,
            model_service=self.model_service,
            process_context=self.process_context,
            history_df=self.history_df,
            fuel_ash_inputs=self.fuel_ash_inputs,
            flux_inputs=candidate_flux_inputs,
            dust_inputs=self.dust_inputs,
            slag_balance_settings=self.slag_balance_settings,
            prebuilt_context=self.prebuilt_context,
            hot_metal_target_mt=self.hot_metal_target_mt,
            coke_correction_settings=self.coke_correction_settings,
            coke_correction_reference=self.coke_correction_reference,
            hot_metal_si_pct=self.hot_metal_si_pct,
            fuel_rate_anchor_basis=self.fuel_rate_anchor_basis,
            charge_mass_mt=self.charge_mass_mt,
        )
        # Flux cost per THM keeps DE from over-dosing flux (it costs money, like ore).
        thm_basis = float(self.hot_metal_target_mt or blend.fe_production_mt or 0.0)
        flux_cost_total_rs = float(
            sum(
                float(flux_qty[j]) * float(self.variable_fluxes[j].price_rs_per_mt)
                for j in range(self.n_flux)
            )
        )
        flux_cost_per_thm = flux_cost_total_rs / thm_basis if thm_basis > 0.0 else 0.0
        blend.diagnostics["lp_flux_quantities_mt"] = {
            flux.flux_id: float(flux_qty[j])
            for j, flux in enumerate(self.variable_fluxes)
        }
        # Persist flux cost on the blend so the displayed total cost includes the
        # flux the optimizer bought (it is a real spend, like ore).
        blend.diagnostics["flux_cost_per_thm_rs"] = float(flux_cost_per_thm)

        penalty_stock = float(self.penalty_cfg.get("penalty_stock", 2500.0))
        penalty_share = float(self.penalty_cfg.get("penalty_share", 2500.0))
        penalty_fe = float(self.penalty_cfg.get("penalty_fe", 3000.0))
        penalty_production_excess = float(
            self.penalty_cfg.get("penalty_production_excess", penalty_fe)
        )
        penalty_slag = float(self.penalty_cfg.get("penalty_slag", 3000.0))
        penalty_burden = float(self.penalty_cfg.get("penalty_burden", penalty_slag))
        penalty_basicity = float(
            self.penalty_cfg.get("penalty_basicity", penalty_slag * 100.0)
        )
        # Per percentage POINT of Al2O3 / MgO deviation, so it defaults an order of
        # magnitude below penalty_basicity while still dominating the ~13,000
        # Rs/THM objective for any violation worth caring about.
        penalty_slag_chemistry = float(
            self.penalty_cfg.get("penalty_slag_chemistry", penalty_basicity / 10.0)
        )
        penalty_large = float(self.penalty_cfg.get("penalty_large", 1_000_000.0))

        stock_violation_mt = float(np.sum(np.clip(qty - self.stocks, 0.0, None)))
        stock_penalty = stock_violation_mt * penalty_stock

        shares = (
            qty / float(blend.total_qty_mt)
            if blend.total_qty_mt > 0
            else np.zeros_like(qty)
        )
        share_violation = float(
            np.sum(np.clip(self.min_shares - shares, 0.0, None))
            + np.sum(np.clip(shares - self.max_shares, 0.0, None))
        )
        share_penalty = share_violation * penalty_share

        production_value_mt = float(blend.fe_production_mt)
        production_target_mt = self.target_production_mt
        if blend.diagnostics.get("iron_closure_basis") == "actual_pig_iron":
            production_value_mt = float(
                blend.diagnostics.get("iron_closure_production_mt", 0.0) or 0.0
            )
            production_target_mt = float(
                blend.diagnostics.get("iron_closure_target_mt", 0.0)
                or self.hot_metal_target_mt
                or self.target_production_mt
            )

        fe_penalty = 0.0
        fe_shortfall_mt = max(
            0.0,
            production_target_mt - self.fe_tolerance_mt - production_value_mt,
        )
        if fe_shortfall_mt > 0.0:
            fe_penalty += (fe_shortfall_mt) * penalty_fe
        production_excess_penalty = 0.0
        fe_excess_mt = max(
            0.0,
            production_value_mt - production_target_mt - self.fe_tolerance_mt,
        )
        if fe_excess_mt > 0.0:
            production_excess_penalty += (fe_excess_mt) * penalty_production_excess

        slag_penalty = 0.0
        if blend.slag_mt > self.target_slag_qty_mt:
            slag_penalty += (blend.slag_mt - self.target_slag_qty_mt) * penalty_slag

        # Charging-throughput cap on IBRM + flux. DE reaches the Fe target by
        # scaling total burden, so without this penalty a lean, cheap blend just
        # buys more tonnes -- tonnes the charging system cannot deliver.
        burden_penalty = 0.0
        burden_qty_mt = float(
            blend.diagnostics.get("total_burden_qty_mt", blend.total_qty_mt) or 0.0
        )
        if self.max_burden_qty_mt is not None:
            burden_excess_mt = burden_qty_mt - self.max_burden_qty_mt
            if burden_excess_mt > 0.0:
                burden_penalty = burden_excess_mt * penalty_burden

        def _bound_penalty(
            *,
            value: float,
            denominator_key: str,
            min_value: float | None,
            max_value: float | None,
            weight: float,
        ) -> float:
            penalty = 0.0
            denominator = float(blend.diagnostics.get(denominator_key, 0.0) or 0.0)
            if min_value is not None:
                if denominator <= 0.0 or not math.isfinite(value):
                    penalty += penalty_large
                elif value < min_value:
                    penalty += (min_value - value) * weight
            if max_value is not None:
                if denominator <= 0.0 or not math.isfinite(value):
                    penalty += penalty_large
                elif value > max_value:
                    penalty += (value - max_value) * weight
            return float(penalty)

        basicity_penalty = _bound_penalty(
            value=float(getattr(blend, "slag_basicity", 0.0) or 0.0),
            denominator_key="slag_basicity_denominator_mt",
            min_value=self.target_slag_basicity_min,
            max_value=self.target_slag_basicity_max,
            weight=penalty_basicity,
        )
        t_basicity_penalty = _bound_penalty(
            value=float(getattr(blend, "slag_t_basicity", 0.0) or 0.0),
            denominator_key="slag_t_basicity_denominator_mt",
            min_value=self.target_slag_t_basicity_min,
            max_value=self.target_slag_t_basicity_max,
            weight=penalty_basicity,
        )
        # Al2O3 and MgO violations are measured in percentage POINTS, an order of
        # magnitude larger than a basicity deviation, so they get their own weight
        # rather than sharing penalty_basicity and swamping every other term.
        al2o3_penalty = _bound_penalty(
            value=float(getattr(blend, "slag_al2o3_pct", 0.0) or 0.0),
            denominator_key="slag_chemistry_denominator_mt",
            min_value=None,
            max_value=self.target_slag_al2o3_max_pct,
            weight=penalty_slag_chemistry,
        )
        mgo_penalty = _bound_penalty(
            value=float(getattr(blend, "slag_mgo_pct", 0.0) or 0.0),
            denominator_key="slag_chemistry_denominator_mt",
            min_value=self.target_slag_mgo_min_pct,
            max_value=None,
            weight=penalty_slag_chemistry,
        )
        mgo_al2o3_penalty = _bound_penalty(
            value=float(getattr(blend, "slag_mgo_al2o3_ratio", 0.0) or 0.0),
            denominator_key="slag_mgo_al2o3_denominator_mt",
            min_value=self.target_slag_mgo_al2o3_ratio_min,
            max_value=None,
            weight=penalty_basicity,
        )

        finite_penalty = 0.0
        if not math.isfinite(blend.objective_rs_per_thm):
            finite_penalty = penalty_large
        if blend.total_qty_mt <= 0.0:
            finite_penalty += penalty_large

        total_penalty = (
            stock_penalty
            + share_penalty
            + fe_penalty
            + production_excess_penalty
            + slag_penalty
            + burden_penalty
            + basicity_penalty
            + t_basicity_penalty
            + al2o3_penalty
            + mgo_penalty
            + mgo_al2o3_penalty
            + finite_penalty
        )
        objective_value = float(
            blend.objective_rs_per_thm + flux_cost_per_thm + total_penalty
        )

        violations = check_blend_constraints(
            blend,
            self.ores,
            target_production_mt=self.target_production_mt,
            target_slag_qty_mt=self.target_slag_qty_mt,
            target_slag_basicity_min=self.target_slag_basicity_min,
            target_slag_basicity_max=self.target_slag_basicity_max,
            target_slag_t_basicity_min=self.target_slag_t_basicity_min,
            target_slag_t_basicity_max=self.target_slag_t_basicity_max,
            target_slag_al2o3_max_pct=self.target_slag_al2o3_max_pct,
            target_slag_mgo_min_pct=self.target_slag_mgo_min_pct,
            target_slag_mgo_al2o3_ratio_min=self.target_slag_mgo_al2o3_ratio_min,
            max_burden_qty_mt=self.max_burden_qty_mt,
        )
        feasible = len(violations) == 0

        return ObjectiveResult(
            objective_value=objective_value,
            components={
                "ore_cost_per_thm_rs": float(blend.ore_cost_per_thm_rs),
                "fuel_cost_per_thm_rs": float(blend.fuel_cost_per_thm_rs),
                "flux_cost_per_thm_rs": float(flux_cost_per_thm),
                "base_objective_rs_per_thm": float(blend.objective_rs_per_thm),
                "penalty_total": float(total_penalty),
                "penalty_stock": float(stock_penalty),
                "penalty_share_bounds": float(share_penalty),
                "penalty_fe": float(fe_penalty),
                "penalty_production_excess": float(production_excess_penalty),
                "penalty_slag": float(slag_penalty),
                "penalty_burden_capacity": float(burden_penalty),
                "total_burden_qty_mt": float(burden_qty_mt),
                "penalty_slag_basicity": float(basicity_penalty),
                "penalty_slag_t_basicity": float(t_basicity_penalty),
                "penalty_slag_al2o3": float(al2o3_penalty),
                "penalty_slag_mgo": float(mgo_penalty),
                "penalty_slag_mgo_al2o3": float(mgo_al2o3_penalty),
                "penalty_non_finite": float(finite_penalty),
                "coke_correction_delta_kg_thm": float(
                    blend.diagnostics.get("coke_correction_delta_kg_thm", 0.0) or 0.0
                ),
                "fuel_cost_per_thm_rs_uncorrected": float(
                    blend.diagnostics.get(
                        "fuel_cost_per_thm_rs_uncorrected",
                        blend.fuel_cost_per_thm_rs,
                    )
                ),
            },
            feasible=feasible,
            violations=violations,
            diagnostics={
                "blend": blend,
                "model_prediction": blend.diagnostics.get("model_prediction"),
                "feature_details": blend.diagnostics.get("feature_details", {}),
            },
        )
