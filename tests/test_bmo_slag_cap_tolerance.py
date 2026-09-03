"""The slag cap must not report a violation the operator cannot see.

THE BUG THIS PINS DOWN.

The LP drives slag exactly onto its cap - that is what a cost minimiser does
with a binding constraint. The page then RE-EVALUATES the displayed blend
through a different path (fuel re-priced, slag recomputed on corrected fuel
rates) and re-checks it. A strict ``>`` on a value that has drifted a few parts
in 1e12 then produced:

    Slag exceeds bound: 775.50 > 775.50 MT

which is unactionable, contradicts the solver that had just declared the blend
feasible, and trains operators to ignore the violations panel - which is the
real cost, because the panel also carries the violations that matter.

THE INVARIANT. A violation message whose two numbers print identically must
never be emitted. The tolerance is therefore tied to the message's own display
resolution rather than to a guess about float noise.

WHAT WOULD INVALIDATE THIS. If the message's format string ever changes from
two decimals, ``SLAG_DISPLAY_TOLERANCE_MT`` has to move with it, and
``test_the_tolerance_matches_the_message_precision`` fails until it does.
"""

from __future__ import annotations

import pytest

from utils.bmo.constraints import SLAG_DISPLAY_TOLERANCE_MT, check_blend_constraints


class _Blend:
    """Only the fields the slag-cap check reads."""

    def __init__(self, slag_mt: float) -> None:
        self.slag_mt = slag_mt
        self.quantities_mt: dict[str, float] = {}
        self.total_qty_mt = 0.0
        self.fe_production_mt = 100.0
        self.diagnostics: dict[str, object] = {}


def _slag_violations(slag_mt: float, cap_mt: float, **kwargs) -> list[str]:
    return [
        v
        for v in check_blend_constraints(
            _Blend(slag_mt),
            [],
            target_production_mt=100.0,
            target_slag_qty_mt=cap_mt,
            **kwargs,
        )
        if "Slag exceeds bound" in v
    ]


# --- the reported bug ---------------------------------------------------------


def test_slag_a_hair_over_the_cap_is_not_a_violation():
    """The exact failure from the field: 775.50 reported as exceeding 775.50."""

    assert _slag_violations(775.5000000001, 775.5) == []


def test_slag_exactly_on_the_cap_is_not_a_violation():
    """A binding constraint is satisfied, not breached."""

    assert _slag_violations(775.5, 775.5) == []


@pytest.mark.parametrize("over", [1e-12, 1e-9, 1e-6, 0.001, 0.004])
def test_nothing_that_prints_the_same_is_flagged(over):
    """Anything inside half a display step rounds to the cap as printed."""

    assert _slag_violations(775.5 + over, 775.5) == []


# --- and it must still catch a real breach -------------------------------------


def test_a_real_overshoot_is_still_reported():
    """The tolerance must not become a licence to exceed the cap.

    5 MT over a 775.5 MT cap is a genuine planning error and has to surface.
    """

    violations = _slag_violations(780.5, 775.5)

    assert len(violations) == 1
    assert "780.50" in violations[0]


def test_the_smallest_visible_overshoot_is_reported():
    """One display step over must be caught, or the guard is too generous.

    This is the boundary that stops SLAG_DISPLAY_TOLERANCE_MT being quietly
    inflated later: at 0.01 MT over, the message reads 775.51 > 775.50, the two
    numbers differ on screen, and it is therefore actionable.
    """

    violations = _slag_violations(775.51, 775.5)

    assert len(violations) == 1
    assert "775.51" in violations[0] and "775.50" in violations[0]


def test_the_tolerance_matches_the_message_precision():
    """Half of the 0.01 MT step the message prints at.

    Derived from the format, not chosen freely: if the message ever moves to a
    different precision this has to move with it.
    """

    display_step_mt = 0.01
    assert SLAG_DISPLAY_TOLERANCE_MT == pytest.approx(display_step_mt / 2.0)


def test_no_emitted_violation_ever_prints_two_equal_numbers():
    """The invariant itself, swept across the boundary.

    Asserting on the MESSAGE rather than on the comparison is what makes this
    survive a future refactor of how the check is written.
    """

    cap = 775.5
    for step in range(0, 400):
        slag = cap + step * 0.0001
        for message in _slag_violations(slag, cap):
            left, right = message.split(":")[1].split(">")
            assert left.strip() != right.strip().replace(" MT.", ""), (
                f"unactionable message at slag={slag!r}: {message}"
            )


# --- callers may still tighten it ----------------------------------------------


def test_a_caller_can_still_demand_an_exact_cap():
    """The LP's own post-solve check uses 1e-6, before anything is re-evaluated.

    The relaxed default is for the DISPLAY re-check. A caller that knows its
    value has not been through a second evaluation can still ask for exactness.
    """

    assert _slag_violations(775.5 + 1e-4, 775.5, slag_tolerance_mt=1e-6) != []
