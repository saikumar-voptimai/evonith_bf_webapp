'''Regression tests for the BMO IBRM + flux charging-capacity safeguard.'''

from __future__ import annotations

import numpy as np
import pytest

from utils.bmo.calculations import compute_charging_requirements, evaluate_blend
from utils.bmo.constraints import (
    CHARGING_HOURS_PER_DAY,
    check_blend_constraints,
    max_ibrm_flux_capacity_mt,
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


def test_default_capacity_reproduces_the_plant_reference_limit() -> None:
    capacity = max_ibrm_flux_capacity_mt(target_hot_metal_mt=2350.0)

    assert capacity == pytest.approx(26.4 * 7.5 * 24.0 - 70.0 * 2350.0 / 1000.0)
    assert capacity == pytest.approx(4587.5)


def test_capacity_tracks_operator_charge_rate_and_charge_mass() -> None:
    """Charges/hr and MT/charge are operator inputs, not yml constants.

    The page rebuilds the ceiling from whatever the operator typed, so both
    numbers have to move it proportionally. The plant's charge reports show an
    actual mean charge mass near 30.1 MT against the 26.4 default (findings
    section 6), which is a ~14% ceiling difference - large enough that it must
    not be stuck in config.
    """

    base = max_ibrm_flux_capacity_mt(
        {"max_charges_per_hour": 7.5, "charge_mass_mt": 26.4},
        target_hot_metal_mt=2350.0,
    )

    heavier_charge = max_ibrm_flux_capacity_mt(
        {"max_charges_per_hour": 7.5, "charge_mass_mt": 30.1},
        target_hot_metal_mt=2350.0,
    )
    assert heavier_charge > base
    assert heavier_charge == pytest.approx(30.1 * 7.5 * 24.0 - 164.5)

    faster_charging = max_ibrm_flux_capacity_mt(
        {"max_charges_per_hour": 8.0, "charge_mass_mt": 26.4},
        target_hot_metal_mt=2350.0,
    )
    assert faster_charging > base
    assert faster_charging == pytest.approx(26.4 * 8.0 * 24.0 - 164.5)


def test_capacity_is_absolute_and_never_exceeds_what_the_skips_can_deliver() -> None:
    """The ceiling must not scale with the HM target.

    The charging system runs the same 24 hours whatever iron is asked for, so
    only the nut-coke deduction moves with the target. The superseded per-THM
    ratio (cap = 1.954 x HM) got this wrong in both directions: it understated
    the room by ~15% at a 2000 t target, and at 2600 t it allowed 5,076 MT -
    more than the 4,752 MT the skips can physically carry in a day.
    """

    cfg = {"max_charges_per_hour": 7.5, "charge_mass_mt": 26.4}
    physical_capacity_mt = 26.4 * 7.5 * CHARGING_HOURS_PER_DAY

    caps = {
        hm: max_ibrm_flux_capacity_mt(cfg, target_hot_metal_mt=hm)
        for hm in (2000.0, 2350.0, 2600.0)
    }

    for hm, cap in caps.items():
        assert cap < physical_capacity_mt, hm
        assert cap == pytest.approx(physical_capacity_mt - 70.0 * hm / 1000.0)

    # Higher HM consumes more nut coke, so it leaves LESS room, never more.
    assert caps[2000.0] > caps[2350.0] > caps[2600.0]

    # The old ratio form would have allowed more than the skips can deliver.
    superseded_ratio_cap = (physical_capacity_mt - 164.5) / 2350.0 * 2600.0
    assert superseded_ratio_cap > physical_capacity_mt
    assert caps[2600.0] < superseded_ratio_cap


def test_capacity_responds_to_the_nut_coke_rate() -> None:
    """Nut coke is derived from its rate, not entered as a tonnage."""

    cfg = {"max_charges_per_hour": 7.5, "charge_mass_mt": 26.4}
    at_70 = max_ibrm_flux_capacity_mt(
        cfg, target_hot_metal_mt=2350.0, nut_coke_rate_kg_per_thm=70.0
    )
    at_90 = max_ibrm_flux_capacity_mt(
        cfg, target_hot_metal_mt=2350.0, nut_coke_rate_kg_per_thm=90.0
    )

    assert at_70 - at_90 == pytest.approx(20.0 * 2350.0 / 1000.0)


def test_capacity_is_zero_when_the_operator_zeroes_the_charging_plant() -> None:
    """A zero charge rate or charge mass must disable the cap, not go negative."""

    for zeroed in ("max_charges_per_hour", "charge_mass_mt"):
        cfg = {"max_charges_per_hour": 7.5, "charge_mass_mt": 26.4, zeroed: 0.0}
        assert (
            max_ibrm_flux_capacity_mt(cfg, target_hot_metal_mt=2350.0) == 0.0
        )


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
    assert charging['chemical_hot_metal_per_charge_mt'] is None
    assert 'hot_metal_per_charge_mt' not in charging
    assert 'planning_hot_metal_per_charge_mt' not in charging


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
