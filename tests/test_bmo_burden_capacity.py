'''Regression tests for the BMO IBRM + flux charging-capacity safeguard.'''

from __future__ import annotations

import numpy as np
import pytest

from utils.bmo.calculations import compute_charging_requirements, evaluate_blend
from utils.bmo.constraints import (
    burden_capacity_ratio_mt_per_thm,
    check_blend_constraints,
)
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.objective import BmoObjectiveEvaluator
from utils.bmo.types import FluxInput, OreChemistry, OreInput


def _ore() -> OreInput:
    return OreInput(
        ore_id='ore_a',
        display_name='ORE A',
        stock_mt=1000.0,
        price_rs_per_mt=1000.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(fe_t_pct=50.0, moisture_pct=0.0),
    )


def _fixed_flux(quantity_mt: float = 10.0) -> FluxInput:
    return FluxInput(
        flux_id='limestone',
        display_name='Limestone',
        enabled=True,
        wet_qty_mt=quantity_mt,
        optimizable=False,
    )


class _Prediction:
    value = 12_900.0
    details: dict = {}
    used_fallback = False


class _ModelService:
    def predict(self, feature_payload, history_df):  # noqa: ANN001
        return _Prediction()


def test_default_capacity_ratio_reconstructs_4592_mt_reference_limit() -> None:
    ratio = burden_capacity_ratio_mt_per_thm()

    assert ratio == pytest.approx((26.4 * 7.5 * 24.0 - 160.0) / 2350.0)
    assert ratio * 2350.0 == pytest.approx(4592.0)


def test_blend_diagnostics_and_constraint_include_wet_fixed_flux() -> None:
    ore = _ore()
    blend = evaluate_blend(
        ores=[ore],
        quantities_mt={ore.ore_id: 100.0},
        feo_in_slag_pct=0.0,
        flux_inputs=[_fixed_flux()],
        hot_metal_target_mt=50.0,
    )

    assert blend.diagnostics['total_flux_wet_qty_mt'] == pytest.approx(10.0)
    assert blend.diagnostics['total_burden_qty_mt'] == pytest.approx(110.0)

    violations = check_blend_constraints(
        blend,
        [ore],
        target_production_mt=50.0,
        target_slag_qty_mt=1000.0,
        max_burden_qty_mt=109.0,
    )
    assert any('Charging capacity exceeded' in item for item in violations)


def test_flux_rate_and_required_charges_use_charge_mass_and_nut_coke() -> None:
    ore = _ore()
    blend = evaluate_blend(
        ores=[ore],
        quantities_mt={ore.ore_id: 4183.0},
        feo_in_slag_pct=0.0,
        flux_inputs=[_fixed_flux()],
        hot_metal_target_mt=100.0,
    )
    blend.diagnostics['fuel_rate_estimate'] = {
        'coke_rate_kg_thm': 400.0,
        'nut_coke_rate_kg_thm': 70.0,
        'pci_rate_kg_thm': 150.0,
    }

    charging = compute_charging_requirements(
        blend,
        charge_mass_mt=26.4,
        hours_per_day=24.0,
    )

    # IBRM 4,183 + flux 10 + nut coke 7 = 4,200 MT/day.
    assert charging['flux_rate_kg_per_thm'] == pytest.approx(100.0)
    assert charging['coke_total_mt'] == pytest.approx(40.0)
    assert charging['nut_coke_total_mt'] == pytest.approx(7.0)
    assert charging['pci_total_mt'] == pytest.approx(15.0)
    assert charging['total_charge_mix_mt'] == pytest.approx(4200.0)
    assert charging['charge_mix_mt_per_hour'] == pytest.approx(175.0)
    assert charging['required_charges_per_hour'] == pytest.approx(175.0 / 26.4)
    assert charging['hot_metal_per_charge_mt'] == pytest.approx(
        100.0 / ((175.0 / 26.4) * 24.0)
    )


def test_lp_reserves_fixed_flux_from_shared_charging_capacity() -> None:
    ore = _ore()
    fluxes = [_fixed_flux()]

    feasible, errors = run_lp_baseline(
        [ore],
        target_production_mt=50.0,
        target_slag_qty_mt=1000.0,
        feo_in_slag_pct=0.0,
        max_burden_qty_mt=110.0,
        flux_inputs=fluxes,
        hot_metal_target_mt=50.0,
    )
    assert errors == []
    assert feasible is not None
    assert feasible.diagnostics['total_burden_qty_mt'] == pytest.approx(110.0)

    infeasible, errors = run_lp_baseline(
        [ore],
        target_production_mt=50.0,
        target_slag_qty_mt=1000.0,
        feo_in_slag_pct=0.0,
        max_burden_qty_mt=109.0,
        flux_inputs=fluxes,
        hot_metal_target_mt=50.0,
    )
    assert infeasible is None
    assert any('IBRM + flux' in item or 'charging' in item.lower() for item in errors)


def test_de_objective_penalizes_and_rejects_over_capacity_candidate() -> None:
    ore = _ore()
    evaluator = BmoObjectiveEvaluator(
        ores=[ore],
        target_production_mt=50.0,
        target_slag_qty_mt=1000.0,
        feo_in_slag_pct=0.0,
        max_burden_qty_mt=105.0,
        model_service=_ModelService(),
        process_context={},
        history_df=None,
        penalty_cfg={'penalty_burden': 100.0},
        flux_inputs=[_fixed_flux()],
        hot_metal_target_mt=50.0,
    )

    result = evaluator.evaluate_quantities(np.array([100.0]))

    assert result.components['total_burden_qty_mt'] == pytest.approx(110.0)
    assert result.components['penalty_burden_capacity'] == pytest.approx(500.0)
    assert result.feasible is False
    assert any('Charging capacity exceeded' in item for item in result.violations)
