"""Step 1 search order (SV-1.8 follow-up, #288): parallel oxygen bracket rounds in place of the golden section, and
the oversize options searched side by side as independent items of ``Engine.run_items``.

Fast tests: ``bracket_rounds`` on synthetic single-dipped functions against the golden section it replaces, and
``Engine.min_total_cost`` with the level search and the frontier replaced by synthetic ones (the calls to
``run_items`` and their items). That the items give the same plans, anchors and solve counts in solver processes
as one after the other is ``run_items``' own property (``tests/test_strata_parallel.py``, real network in the
nightly suite); a STRATA-ω Step 1 serial against parallel is ``scripts/optimiser_bench.py --mode step1 --compare``."""

from __future__ import annotations

import numpy as np
import pytest

from utils.burden_engine.engine import Engine, Limits, _item_frontier, _item_level, bracket_rounds

TOL_R = 1e-4
STEP = 0.0022467  # oxygen grid step of Step 1 (9 levels from today's O2/air to the 5.5 % enrichment limit)


def golden(f, a: float, b: float, tol: float) -> tuple[float, int]:
    """The oxygen search of Step 1 before #288: golden section, one level after the other. Returns (the cheapest
    evaluated x, the number of sequential evaluations)."""
    gr = (5**0.5 - 1) / 2
    seen: dict[float, float] = {}

    def at(x: float) -> float:
        seen[round(x, 7)] = f(x)
        return seen[round(x, 7)]

    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = at(c), at(d)
    while b - a > tol:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = at(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = at(d)
    return min(seen, key=seen.get), len(seen)


def dip(x0: float, p: float = 2.0, edge: float = np.inf):
    """Single-dipped cost with its minimum at x0 (Rs/tHM scale); infeasible (inf) beyond ``edge``."""
    return lambda x: np.inf if x > edge else 29_600.0 + 4e7 * abs(x - x0) ** p


def search(fns: dict, brackets: dict, k: int = 4):
    """``bracket_rounds`` on the functions ``fns`` (one per key), the grid points (bracket ends and middle) known."""
    seen = {key: {round(x, 7): fns[key](x) for x in (a, 0.5 * (a + b), b)} for key, (a, b) in brackets.items()}
    calls: list[list] = []

    def evaluate(points):
        calls.append(list(points))
        return [fns[key](x) for key, x in points]

    rounds = bracket_rounds(brackets, seen, evaluate, k=k, tol=TOL_R)
    best = {key: min(seen[key], key=seen[key].get) for key in brackets}
    return best, rounds, calls


A = 0.05209  # today's O2/air (calibration)


@pytest.mark.parametrize("x0", [A + 0.55 * STEP, A + 0.93 * STEP, A + STEP, A + 1.31 * STEP, A + 1.48 * STEP])
@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_bracket_rounds_find_the_golden_section_optimum_in_fewer_steps(x0, p) -> None:
    f = dip(x0, p)
    a, b = A, A + 2 * STEP  # the best grid level's neighbours
    x_gold, n_gold = golden(f, a, b, TOL_R)
    best, rounds, calls = search({0: f}, {0: (a, b)})
    assert abs(best[0] - x_gold) <= TOL_R
    assert abs(best[0] - x0) <= TOL_R
    assert rounds == len(calls) <= 5 < n_gold == 10  # rounds run one after the other; the golden section, levels
    assert all(len(c) <= 4 for c in calls) and len(calls[0]) == 4


def test_a_bracket_at_the_end_of_the_grid_and_an_infeasible_side() -> None:
    """Best grid level at today's oxygen (the bracket is one grid step), and a cost that falls until a limit makes the
    plan infeasible (more oxygen: more production until the blast or oxygen limit)."""
    best, rounds, _ = search({0: dip(A + 0.2 * STEP)}, {0: (A, A + STEP)})
    assert abs(best[0] - (A + 0.2 * STEP)) <= TOL_R and rounds <= 4
    edge = A + 1.21 * STEP
    f = dip(A + 3 * STEP, edge=edge)
    x_gold, _ = golden(f, A, A + 2 * STEP, TOL_R)
    best, rounds, _ = search({0: f}, {0: (A, A + 2 * STEP)})
    assert edge - TOL_R <= best[0] <= edge and abs(best[0] - x_gold) <= TOL_R and rounds <= 5


def test_the_points_of_all_keys_are_evaluated_in_one_call_per_round() -> None:
    fns = {False: dip(A + 0.7 * STEP), True: dip(A + 1.2 * STEP)}
    brackets = {False: (A, A + 2 * STEP), True: (A, A + 2 * STEP)}
    best, rounds, calls = search(fns, brackets)
    assert abs(best[False] - (A + 0.7 * STEP)) <= TOL_R and abs(best[True] - (A + 1.2 * STEP)) <= TOL_R
    assert len(calls) == rounds
    assert [key for key, _ in calls[0]] == [False] * 4 + [True] * 4  # in key order, both keys in the first round
    for c in calls:
        keys = [key for key, _ in c]
        assert keys == sorted(keys)


def test_the_number_of_points_per_round_is_a_parameter() -> None:
    f = dip(A + 1.1 * STEP)
    rounds = {k: search({0: f}, {0: (A, A + 2 * STEP)}, k=k)[1] for k in (2, 4, 8)}
    assert rounds[8] <= rounds[4] <= rounds[2]
    assert search({0: f}, {0: (A, A + 2 * STEP)}, k=1)[1] == rounds[2]  # one point would not narrow: at least 2


# --------------------------------------------------------------------------------------- Step 1 on the engine
R_OPT = {False: 0.37, True: 0.81}  # cheapest oxygen of each option, as a fraction of the grid span
EXTRA = {False: 0.0, True: 100.0}  # the oversize option costs 100 Rs/tHM more at its optimum


@pytest.fixture(scope="module")
def eng():
    return Engine()


def step1_with_synthetic_levels(eng: Engine, monkeypatch, o2_points: int = 4):
    """Step 1 (Fixed-ω) with a synthetic level search (cost single-dipped in oxygen) and frontier; returns (the plan,
    the run_items calls as (function, items), the oxygen grid)."""
    s = eng.cal_settings()
    L = Limits(process=True)
    r0, r1 = s.O2_per_air, eng.hy.o2_per_air_from_tag(L.enrich_max)
    grid = list(np.linspace(r0, r1, 9))

    def level(r, L, base, P_lo, on6, tol_P):
        x0 = r0 + R_OPT[on6] * (r1 - r0)
        total = 29_600.0 + EXTRA[on6] + 4e7 * (r - x0) ** 2
        plan = dict(feasible=True, cost=dict(total=total), settings=dict(O2_per_air=r), P_th=95.0 + 100 * r,
                    P_tday=24 * (95.0 + 100 * r), ore6_on=on6)  # fmt: skip
        return plan, 96.0 + 100 * r

    def frontier(s, L, P_list, fuel=True, ore6_on=False):
        return [dict(P_th=P, feasible=True, cost=dict(total=30_000.0 - s.O2_per_air), ore6_on=ore6_on) for P in P_list]

    calls: list = []
    run_items = eng.run_items

    def spy(fn, items, *, network):
        calls.append((fn, list(items)))
        return run_items(fn, items, network=network)

    monkeypatch.setattr(eng, "_ore6_options", lambda: [False, True])
    monkeypatch.setattr(eng, "_cheapest_at_level", level)
    monkeypatch.setattr(eng, "frontier", frontier)
    monkeypatch.setattr(eng, "run_items", spy)
    best = eng.min_total_cost(L, P_lo=84.0, step=1.0, o2_levels=grid, o2_points=o2_points)
    return best, calls, grid


def test_step1_runs_both_oversize_options_as_items(eng, monkeypatch) -> None:
    best, calls, grid = step1_with_synthetic_levels(eng, monkeypatch)
    fns = [fn for fn, _ in calls]
    # one call for the grid of both options, one per bracket round (both options together), one for the frontier
    assert fns[0] is _item_level and fns[-1] is _item_frontier
    assert all(fn is _item_level for fn in fns[1:-1]) and 3 <= len(fns) - 2 <= 5
    first = [(item[4], item[0]) for item in calls[0][1]]
    assert first == [(False, r) for r in grid] + [(True, r) for r in grid]
    assert [item[4] for item in calls[1][1]] == [False] * 4 + [True] * 4
    for _, items in calls[1:]:
        options = [item[4] for item in items]  # the option of a level item and of a frontier item
        assert options == sorted(options)  # in option order
    span = grid[-1] - grid[0]
    assert best["ore6_on"] is False
    assert abs(best["settings"]["O2_per_air"] - (grid[0] + R_OPT[False] * span)) <= TOL_R
    assert list(best["curves"]) == [False, True]
    # the frontier levels of each option: its grid, its cheapest level and its top-production level
    # (the top level is the grid's last, held once although its key is rounded)
    levels = {on6: [item[0].O2_per_air for item in calls[-1][1] if item[4] is on6] for on6 in (False, True)}
    for on6 in (False, True):
        assert len(levels[on6]) == len(set(levels[on6])) == len(grid) + 1
        assert set(grid) <= set(levels[on6]) and best["curves"][on6]


def test_step1_takes_the_cheaper_option_and_its_oxygen(eng, monkeypatch) -> None:
    monkeypatch.setitem(EXTRA, True, -50.0)  # now the oversize option is cheaper
    best, calls, grid = step1_with_synthetic_levels(eng, monkeypatch, o2_points=6)
    span = grid[-1] - grid[0]
    assert best["ore6_on"] is True
    assert abs(best["settings"]["O2_per_air"] - (grid[0] + R_OPT[True] * span)) <= TOL_R
    assert [item[4] for item in calls[1][1]] == [False] * 6 + [True] * 6
