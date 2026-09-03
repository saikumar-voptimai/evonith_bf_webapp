"""Slag-quality window: Al2O3 cap, MgO floor, and MgO/Al2O3 floor.

The physics these tests pin down, and why the three limits behave differently:

Al2O3 and MgO are inert - neither is reduced nor volatilised, so all of each
reports to slag and their MASSES are set entirely by what is charged. Their
PERCENTAGES are therefore inversely proportional to total slag:

    at 330 kg/THM the plant runs 18.71% Al2O3 (mean of 6,402 hours)
    the same Al2O3 mass at 290 kg/THM reads 18.71 * 330/290 = 21.3%

So cutting the slag rate pushes Al2O3 UP (toward its cap) and MgO UP as well
(away from its floor). The Al2O3 cap is what actually binds a slag-rate
reduction; the MgO floor gets easier. MgO/Al2O3 is a ratio of two masses, so
total slag cancels and it does not move with slag rate at all - it constrains
the burden alone.

See docs/bmo_fuel_slag_si_findings.md for the plant reference distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.bmo.calculations import evaluate_blend
from utils.bmo.constraints import check_blend_constraints
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.objective import BmoObjectiveEvaluator
from utils.bmo.types import FluxInput, OreChemistry, OreInput


def _ore(
    ore_id: str,
    *,
    price: float = 1000.0,
    al2o3: float = 2.0,
    mgo: float = 1.0,
    sio2: float = 4.0,
    cao: float = 4.0,
    fe: float = 50.0,
    stock: float = 5000.0,
    max_share: float = 100.0,
) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=ore_id.upper(),
        stock_mt=stock,
        price_rs_per_mt=price,
        min_share_pct=0.0,
        max_share_pct=max_share,
        chemistry=OreChemistry(
            fe_t_pct=fe,
            sio2_pct=sio2,
            al2o3_pct=al2o3,
            cao_pct=cao,
            mgo_pct=mgo,
        ),
    )


def _blend(ores: list[OreInput], quantities: dict[str, float]):
    return evaluate_blend(
        ores=ores,
        quantities_mt=quantities,
        feo_in_slag_pct=0.0,
        fuel_cost_per_thm_rs=0.0,
    )


# --- The percentages themselves ---------------------------------------------


def test_slag_chemistry_percentages_are_computed_against_total_slag():
    ores = [_ore("ore", al2o3=2.0, mgo=1.0, sio2=4.0, cao=4.0)]
    blend = _blend(ores, {"ore": 100.0})

    slag_mt = blend.slag_mt
    assert slag_mt > 0.0
    assert blend.slag_al2o3_pct == pytest.approx(
        blend.diagnostics["slag_al2o3_mt"] / slag_mt * 100.0
    )
    assert blend.slag_mgo_pct == pytest.approx(
        blend.diagnostics["slag_mgo_mt"] / slag_mt * 100.0
    )
    assert blend.slag_mgo_al2o3_ratio == pytest.approx(
        blend.diagnostics["slag_mgo_mt"] / blend.diagnostics["slag_al2o3_mt"]
    )


def test_mgo_al2o3_ratio_is_invariant_to_burden_scale():
    """The ratio is scale-free: doubling the burden must not move it.

    This is what makes MgO/Al2O3 a burden-selection constraint rather than a
    slag-rate one, and why cutting slag cannot be used to satisfy it.
    """

    ores = [_ore("ore", al2o3=2.5, mgo=1.0)]
    small = _blend(ores, {"ore": 100.0})
    large = _blend(ores, {"ore": 400.0})

    assert large.slag_mt > small.slag_mt * 3.0
    assert large.slag_mgo_al2o3_ratio == pytest.approx(small.slag_mgo_al2o3_ratio)
    # ...whereas the percentages are unchanged only because composition is fixed;
    # what moves them is the RATIO of inert mass to slag, tested below.
    assert large.slag_al2o3_pct == pytest.approx(small.slag_al2o3_pct)


def test_cutting_slag_raises_al2o3_pct_and_mgo_pct_together():
    """The concentration effect behind the whole 330 -> 290 kg/THM problem.

    Holding the inert masses fixed and shrinking everything else raises both
    percentages. Al2O3 rises into its cap; MgO rises away from its floor.
    """

    inert_heavy = [_ore("ore", al2o3=2.0, mgo=1.0, sio2=8.0, cao=8.0)]
    inert_light = [_ore("ore", al2o3=2.0, mgo=1.0, sio2=3.0, cao=3.0)]

    high_slag = _blend(inert_heavy, {"ore": 100.0})
    low_slag = _blend(inert_light, {"ore": 100.0})

    assert low_slag.slag_mt < high_slag.slag_mt
    assert low_slag.slag_al2o3_pct > high_slag.slag_al2o3_pct
    assert low_slag.slag_mgo_pct > high_slag.slag_mgo_pct
    # The ratio does not care.
    assert low_slag.slag_mgo_al2o3_ratio == pytest.approx(
        high_slag.slag_mgo_al2o3_ratio
    )


# --- Constraint validation ---------------------------------------------------


def test_check_blend_constraints_flags_each_slag_quality_limit():
    ores = [_ore("ore", al2o3=6.0, mgo=0.1, sio2=4.0, cao=4.0)]
    blend = _blend(ores, {"ore": 100.0})

    violations = check_blend_constraints(
        blend,
        ores,
        target_production_mt=blend.fe_production_mt,
        target_slag_qty_mt=1.0e9,
        target_slag_al2o3_max_pct=5.0,
        target_slag_mgo_min_pct=7.0,
        target_slag_mgo_al2o3_ratio_min=0.36,
    )

    assert any("Slag Al2O3 above bound" in item for item in violations)
    assert any("Slag MgO below bound" in item for item in violations)
    assert any("Slag MgO/Al2O3 below bound" in item for item in violations)


def test_slag_quality_limits_are_optional():
    """None means off, so existing callers keep their current behaviour."""

    ores = [_ore("ore", al2o3=6.0, mgo=0.1)]
    blend = _blend(ores, {"ore": 100.0})

    violations = check_blend_constraints(
        blend,
        ores,
        target_production_mt=blend.fe_production_mt,
        target_slag_qty_mt=1.0e9,
    )

    assert not any("Al2O3" in item or "MgO" in item for item in violations)


# --- LP enforcement ----------------------------------------------------------


def _lp(**overrides):
    ores = overrides.pop(
        "ores",
        [
            # Cheap but alumina-rich, and MgO-poor: the blend the optimizer wants
            # on price alone and which the quality window must reject.
            _ore("high_alumina", price=800.0, al2o3=6.0, mgo=0.2),
            # Dearer, low-alumina, MgO-bearing.
            _ore("clean", price=2000.0, al2o3=1.0, mgo=2.5),
        ],
    )
    kwargs = dict(
        target_production_mt=50.0,
        target_slag_qty_mt=1.0e6,
        feo_in_slag_pct=0.0,
    )
    kwargs.update(overrides)
    return run_lp_baseline(ores, **kwargs), ores


def test_lp_without_the_window_buys_the_cheap_high_alumina_ore():
    (blend, errors), _ = _lp()

    assert errors == []
    assert blend is not None
    assert blend.quantities_mt["high_alumina"] > blend.quantities_mt["clean"]


def test_lp_al2o3_cap_shifts_the_blend_to_low_alumina_material():
    (unbounded, _), _ = _lp()
    (bounded, errors), _ = _lp(target_slag_al2o3_max_pct=20.0)

    assert errors == []
    assert bounded is not None
    assert bounded.feasible is True
    assert bounded.slag_al2o3_pct <= 20.0 + 1e-2
    # It had to buy the dearer clean ore to get there.
    assert bounded.quantities_mt["clean"] > unbounded.quantities_mt["clean"]
    assert bounded.ore_cost_per_thm_rs > unbounded.ore_cost_per_thm_rs


def test_lp_mgo_floor_and_ratio_floor_are_enforced_exactly():
    (blend, errors), _ = _lp(
        target_slag_mgo_min_pct=8.0,
        target_slag_mgo_al2o3_ratio_min=0.36,
    )

    assert errors == []
    assert blend is not None
    assert blend.feasible is True
    assert blend.slag_mgo_pct >= 8.0 - 1e-2
    assert blend.slag_mgo_al2o3_ratio >= 0.36 - 1e-3


def test_lp_reports_which_slag_quality_limit_blocks_an_infeasible_blend():
    """With six limits live, a bare 'infeasible' is not actionable."""

    ores = [_ore("only_ore", al2o3=8.0, mgo=0.0)]
    (blend, errors), _ = _lp(ores=ores, target_slag_mgo_min_pct=7.0)

    assert blend is None
    assert any("MgO floor" in error for error in errors)


def test_lp_al2o3_cap_and_slag_cap_interact_the_way_the_physics_says():
    """Tightening the slag cap makes the Al2O3 cap harder, not easier.

    Same burden, same Al2O3 limit: the run with the tighter slag cap must end up
    at a higher (or equal) Al2O3 percentage, because the inert mass is being
    divided by less slag.
    """

    loose, _ = _lp(target_slag_al2o3_max_pct=25.0, target_slag_qty_mt=1.0e6)
    loose_blend, loose_errors = loose
    assert loose_errors == []
    assert loose_blend is not None

    tight, _ = _lp(
        target_slag_al2o3_max_pct=25.0,
        target_slag_qty_mt=loose_blend.slag_mt * 0.9,
    )
    tight_blend, tight_errors = tight

    if tight_blend is not None:
        assert tight_blend.slag_mt <= loose_blend.slag_mt + 1e-6
        assert tight_blend.slag_al2o3_pct >= loose_blend.slag_al2o3_pct - 1e-6
    else:
        # Refusing is also a correct answer; it must say why.
        assert tight_errors


def test_lp_can_buy_dolomite_to_satisfy_the_mgo_floor():
    """The optimisable-flux path reaches the MgO limits, not just basicity."""

    ores = [_ore("ore", al2o3=3.0, mgo=0.0)]
    dolomite = FluxInput(
        flux_id="dolomite",
        display_name="DOLOMITE",
        enabled=True,
        wet_qty_mt=0.0,
        mgo_pct=22.3,
        cao_pct=30.0,
        sio2_pct=1.2,
        al2o3_pct=0.24,
        price_rs_per_mt=6000.0,
        stock_mt=2000.0,
        optimizable=True,
    )

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=50.0,
        target_slag_qty_mt=1.0e6,
        feo_in_slag_pct=0.0,
        flux_inputs=[dolomite],
        target_slag_mgo_min_pct=6.0,
    )

    assert errors == []
    assert blend is not None
    assert blend.feasible is True
    assert blend.slag_mgo_pct >= 6.0 - 1e-2
    assert blend.diagnostics["lp_flux_quantities_mt"]["dolomite"] > 0.0


# --- DE enforcement ----------------------------------------------------------


def _evaluator(**overrides) -> BmoObjectiveEvaluator:
    ores = [_ore("ore", al2o3=6.0, mgo=0.1)]
    kwargs = dict(
        ores=ores,
        target_production_mt=50.0,
        target_slag_qty_mt=1.0e6,
        feo_in_slag_pct=0.0,
        model_service=FuelUnitCostModelService(
            bundle_cfg={
                "model_path": "src/assets/models/bmo_fuel/definitely_missing.joblib",
                "scaler_path": "src/assets/models/bmo_fuel/definitely_missing.joblib",
            },
            fallback_cfg={},
        ),
        process_context={},
        history_df=pd.DataFrame(),
        penalty_cfg={"penalty_basicity": 1000.0, "penalty_slag_chemistry": 100.0},
    )
    kwargs.update(overrides)
    return BmoObjectiveEvaluator(**kwargs)


def test_de_penalizes_each_slag_quality_violation():
    result = _evaluator(
        target_slag_al2o3_max_pct=5.0,
        target_slag_mgo_min_pct=7.0,
        target_slag_mgo_al2o3_ratio_min=0.36,
    ).evaluate_quantities(np.array([100.0], dtype=float))

    assert result.components["penalty_slag_al2o3"] > 0.0
    assert result.components["penalty_slag_mgo"] > 0.0
    assert result.components["penalty_slag_mgo_al2o3"] > 0.0
    assert result.feasible is False


def test_de_applies_no_slag_quality_penalty_when_limits_are_met():
    result = _evaluator(
        target_slag_al2o3_max_pct=99.0,
        target_slag_mgo_min_pct=0.0,
        target_slag_mgo_al2o3_ratio_min=0.0,
    ).evaluate_quantities(np.array([100.0], dtype=float))

    assert result.components["penalty_slag_al2o3"] == pytest.approx(0.0)
    assert result.components["penalty_slag_mgo"] == pytest.approx(0.0)
    assert result.components["penalty_slag_mgo_al2o3"] == pytest.approx(0.0)


def test_de_slag_chemistry_penalty_weight_is_separately_configurable():
    """Percentage-point deviations must not inherit the basicity weight.

    A 1 pp Al2O3 breach and a 1.0 basicity breach are wildly different physical
    events; sharing one weight would let Al2O3 swamp every other term.
    """

    quantities = np.array([100.0], dtype=float)
    cheap = _evaluator(
        target_slag_al2o3_max_pct=5.0,
        penalty_cfg={"penalty_basicity": 1000.0, "penalty_slag_chemistry": 10.0},
    ).evaluate_quantities(quantities)
    dear = _evaluator(
        target_slag_al2o3_max_pct=5.0,
        penalty_cfg={"penalty_basicity": 1000.0, "penalty_slag_chemistry": 100.0},
    ).evaluate_quantities(quantities)

    assert dear.components["penalty_slag_al2o3"] == pytest.approx(
        cheap.components["penalty_slag_al2o3"] * 10.0
    )


# --- The three basicity indices are distinct and must not be confused --------


def test_b2_t_basicity_and_ib4_are_three_different_indices():
    """Pins the scale of each, because IB4 already got mistaken for T Basicity.

    IB4 puts Al2O3 (~19% of slag) into the denominator, so it reads far LOWER
    than T Basicity off the same slag. Plant reference over 6,515 hours of
    HM_SLAG analysis:

        B2  = CaO/SiO2               mean 1.079
        T.B = (CaO+MgO)/SiO2         mean 1.309
        IB4 = (CaO+MgO)/(SiO2+Al2O3) mean 0.844

    A ~0.85 IB4 is correct. A ~0.85 T Basicity would not be.
    """

    ores = [_ore("ore", al2o3=2.5, mgo=1.2, sio2=5.0, cao=5.5)]
    blend = _blend(ores, {"ore": 1000.0})

    cao = blend.diagnostics["slag_basicity_cao_mt"]
    mgo = blend.diagnostics["slag_mgo_mt"]
    sio2 = blend.diagnostics["slag_basicity_sio2_mt"]
    al2o3 = blend.diagnostics["slag_al2o3_mt"]

    assert blend.slag_basicity == pytest.approx(cao / sio2)
    assert blend.slag_t_basicity == pytest.approx((cao + mgo) / sio2)
    assert blend.slag_ib4 == pytest.approx((cao + mgo) / (sio2 + al2o3))

    # Ordering is structural: same numerator, bigger denominator.
    assert blend.slag_ib4 < blend.slag_t_basicity
    assert blend.slag_basicity < blend.slag_t_basicity


def test_ib4_and_t_basicity_reproduce_the_plant_slag_analysis():
    """Both indices, computed from a slag matching the plant's mean analysis."""

    # Plant mean slag: CaO 36.66, MgO 7.82, SiO2 34.02, Al2O3 18.71 (%).
    cao, mgo, sio2, al2o3 = 36.66, 7.82, 34.02, 18.71

    assert (cao + mgo) / sio2 == pytest.approx(1.309, abs=0.01)
    assert (cao + mgo) / (sio2 + al2o3) == pytest.approx(0.844, abs=0.01)
    # The gap between them is the whole reason they get confused.
    assert (cao + mgo) / sio2 - (cao + mgo) / (sio2 + al2o3) > 0.4
