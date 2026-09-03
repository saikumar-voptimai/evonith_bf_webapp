"""The transition ladder: getting from today's blend to the optimum, in steps.

The LP says where to go. It does not say how to get there from what the plant is
charging today, and a 20-point share change is not an instruction anyone can act
on - burden descent takes 6-7 hours and ore contracts do not turn overnight.

Every rung is a genuine LP solve that independently satisfies all six slag
limits, so no step on the path is one the furnace could not actually run.
"""

from __future__ import annotations

import pytest

from utils.bmo.transition import build_transition_ladder
from utils.bmo.types import OreChemistry, OreInput


def _ore(ore_id, name, *, fe, sio2, cao, mgo, al2o3, price, max_share=100.0):
    return OreInput(
        ore_id=ore_id, display_name=name, stock_mt=60_000.0,
        price_rs_per_mt=price, min_share_pct=0.0, max_share_pct=max_share,
        chemistry=OreChemistry(
            fe_t_pct=fe, sio2_pct=sio2, cao_pct=cao, mgo_pct=mgo,
            al2o3_pct=al2o3, moisture_pct=3.0,
        ),
    )


def _ores():
    return [
        _ore("sinter", "SINTER", fe=56.5, sio2=5.4, cao=10.9, mgo=2.4,
             al2o3=2.1, price=7789.0, max_share=70.0),
        _ore("clo", "CLO", fe=62.0, sio2=3.2, cao=0.2, mgo=0.1,
             al2o3=2.6, price=5349.0, max_share=40.0),
        _ore("pellet", "PELLET", fe=64.5, sio2=2.6, cao=1.1, mgo=0.5,
             al2o3=0.7, price=9650.0, max_share=20.0),
    ]


MANUAL = {"sinter": 70.0, "clo": 20.0, "pellet": 10.0}
LP = dict(
    target_production_mt=2220.0, target_slag_qty_mt=800.0, feo_in_slag_pct=0.4,
    hot_metal_target_mt=2350.0,
)


def _ladder(**overrides):
    kwargs = {**LP, **overrides}
    move = kwargs.pop("max_share_move_pct", 5.0)
    return build_transition_ladder(
        _ores(), MANUAL, max_share_move_pct=move, **kwargs
    )


# --- the path itself -----------------------------------------------------------


def test_ladder_reaches_the_unconstrained_optimum():
    """Greedy steps must not stall short of where the LP would go in one jump."""

    from utils.bmo.lp_solver import run_lp_baseline

    direct, _ = run_lp_baseline(_ores(), target_slag_qty_mt=1.0e6, **{
        k: v for k, v in LP.items() if k != "target_slag_qty_mt"
    })
    ladder = _ladder(target_slag_basicity_max=1.70)

    assert ladder.converged
    assert ladder.destination is not None
    assert ladder.destination.blend.ore_cost_per_thm_rs == pytest.approx(
        direct.ore_cost_per_thm_rs, rel=1e-3
    )


def test_no_single_step_exceeds_the_move_limit():
    """The whole point: each rung must be something the operator can do next."""

    move = 5.0
    ladder = _ladder(max_share_move_pct=move, target_slag_basicity_max=1.70)

    previous = ladder.start_shares_pct
    for rung in ladder.rungs:
        assert rung.feasible
        for ore_id, share in rung.shares_pct.items():
            assert abs(share - previous.get(ore_id, 0.0)) <= move + 1e-6
        previous = rung.shares_pct


def test_a_bigger_step_reaches_the_optimum_in_fewer_rungs():
    small = _ladder(max_share_move_pct=2.0, target_slag_basicity_max=1.70)
    large = _ladder(max_share_move_pct=20.0, target_slag_basicity_max=1.70)

    assert len(large.rungs) < len(small.rungs)
    # Both should still land in the same place.
    assert large.destination.blend.ore_cost_per_thm_rs == pytest.approx(
        small.destination.blend.ore_cost_per_thm_rs, rel=1e-3
    )


def test_every_rung_is_a_real_feasible_blend():
    """No rung may violate a slag limit, even mid-path."""

    ladder = _ladder(target_slag_basicity_max=1.70)

    for rung in ladder.rungs:
        assert rung.feasible
        assert rung.blend.feasible
        assert rung.blend.violations == []


def test_the_path_saves_money_monotonically():
    ladder = _ladder(target_slag_basicity_max=1.70)

    costs = [r.blend.ore_cost_per_thm_rs for r in ladder.rungs]
    assert costs == sorted(costs, reverse=True)
    assert ladder.ore_cost_saving_rs_per_thm() > 0.0


# --- honesty about where the path stops -----------------------------------------


def test_an_out_of_bounds_start_still_gets_a_path():
    """The operator needs an instruction, not just a complaint.

    Reporting "your current blend breaches the limits" and stopping leaves them
    with a violation and nothing to do about it - which is exactly what happened
    in the field on a blend that was out by 0.009 on T-basicity and 0.12% on
    Al2O3. The first rung must recover feasibility, and every later rung carries
    on toward the optimum as normal.
    """

    ladder = _ladder(target_slag_basicity_max=1.55)

    assert not ladder.start_is_admissible
    assert ladder.rungs, "an inadmissible start must still produce steps"
    first = ladder.rungs[0]
    assert first.is_recovery
    assert first.feasible
    assert first.blend.violations == []


def test_recovery_widens_the_step_only_when_it_has_to():
    """A routine step policy may be too small to climb back inside the limits.

    Widening is confined to the first rung and only when the start is already
    out of bounds - the operator's step policy must still govern every ordinary
    move. Being told "this one needs 12%, not your usual 5%" is itself the
    useful finding.
    """

    tight = _ladder(max_share_move_pct=0.5, target_slag_basicity_max=1.55)

    assert not tight.start_is_admissible
    assert tight.rungs and tight.rungs[0].feasible
    widened = tight.diagnostics.get("recovery_move_pct")
    if widened is not None:
        assert widened > 0.5
        # Only the recovery rung may exceed the policy.
        for rung in tight.rungs[1:]:
            assert (rung.move_used_pct or 0.0) <= 0.5 + 1e-9


def test_a_feasible_start_never_widens_or_flags_recovery():
    """Widening must not leak into the normal path."""

    ladder = _ladder(target_slag_basicity_max=1.70)

    assert ladder.start_is_admissible
    assert ladder.diagnostics.get("recovery_move_pct") is None
    assert not any(rung.is_recovery for rung in ladder.rungs)


def test_a_blend_already_out_of_bounds_is_reported_as_such():
    """A different problem from 'the optimum is far away', and needs saying.

    Every rung is anchored to the previous one, and the first to the current
    blend. If the current blend already breaches a limit there is no ladder -
    and a bare 'infeasible' would send the operator hunting the wrong thing.
    """

    ladder = _ladder(target_slag_basicity_max=1.55)

    assert not ladder.start_is_admissible
    assert any("basicity" in v.lower() for v in ladder.start_violations)


def test_binding_limits_name_share_caps_not_only_slag_limits():
    """Often it is an ore's own maximum share that stops the path."""

    ladder = _ladder(target_slag_basicity_max=1.70)
    all_binding = [b for rung in ladder.rungs for b in rung.binding_limits]

    assert any("max share" in b for b in all_binding)


def test_convergence_does_not_leave_a_duplicate_final_step():
    """The repeat rung proves convergence; it is not a step to carry out."""

    ladder = _ladder(target_slag_basicity_max=1.70)

    assert ladder.converged
    assert len(ladder.rungs) >= 2
    last, penultimate = ladder.rungs[-1], ladder.rungs[-2]
    assert last.shares_pct != pytest.approx(penultimate.shares_pct)


# --- inputs ---------------------------------------------------------------------


def test_manual_shares_outside_the_bounds_are_clamped_and_renormalised():
    """The plant may be charging something the current share limits disallow."""

    ladder = build_transition_ladder(
        _ores(), {"sinter": 95.0, "clo": 5.0, "pellet": 0.0},
        max_share_move_pct=5.0, target_slag_basicity_max=1.70, **LP,
    )

    assert sum(ladder.start_shares_pct.values()) == pytest.approx(100.0)
    assert ladder.start_shares_pct["sinter"] <= 70.0 + 1e-6
