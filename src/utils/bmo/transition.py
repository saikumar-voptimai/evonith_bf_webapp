"""A step-by-step path from the current manual blend to the optimal one.

The LP answers "what is the cheapest blend that meets the slag limits". It does
not answer "how do I get there from what I am charging today", and that is the
question an operator actually has. A 20-percentage-point share change is not an
instruction anyone can act on: burden descent takes 6-7 hours, ore contracts do
not turn overnight, and a large step lands the furnace somewhere nobody has
seen before.

So this builds a LADDER. Each rung is a re-solve of the same LP with one extra
restriction - no ore may move more than ``max_share_move_pct`` from where the
previous rung put it:

    |share_i(rung k) - share_i(rung k-1)|  <=  delta

That needs no change to the solver. The move cap is applied by tightening each
ore's own min/max share bounds around the previous rung, so every rung is a
genuine LP solve that independently satisfies ALL six slag limits - rate,
CaO/SiO2, T-basicity, Al2O3, MgO and MgO/Al2O3. A rung that cannot satisfy them
is reported as infeasible with the binding limit named, rather than being
quietly skipped.

Two properties worth knowing:

* Because the LP objective is ore + flux purchase cost only, every rung sits on
  whichever limit binds. The ladder therefore shows the operator which
  constraint is holding them back at each stage, which is usually the more
  useful half of the answer.

* The ladder converges when a rung stops moving. That is the unconstrained
  optimum, and the number of rungs taken to reach it is the honest answer to
  "how long will this take" - in shifts or days, depending on the step the
  operator chose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.types import BlendEvaluation, OreInput

DEFAULT_MAX_SHARE_MOVE_PCT = 5.0
DEFAULT_MAX_RUNGS = 12
CONVERGENCE_TOL_PCT = 0.05


@dataclass
class TransitionRung:
    """One step on the path from the manual blend to the optimum."""

    index: int
    shares_pct: dict[str, float]
    blend: BlendEvaluation | None
    errors: list[str] = field(default_factory=list)
    binding_limits: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.blend is not None


@dataclass
class TransitionLadder:
    """The whole path, plus what it is worth."""

    rungs: list[TransitionRung]
    start_shares_pct: dict[str, float]
    converged: bool
    max_share_move_pct: float
    start_blend: BlendEvaluation | None = None
    start_violations: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def start_is_admissible(self) -> bool:
        """Does what the plant is charging today already meet the limits?

        If not, no ladder exists: every rung is anchored to the previous one and
        the first is anchored to the current blend. The operator has to be told
        that the starting point itself is out of bounds, which is a different
        problem from 'the optimum is unreachable in one step'.
        """

        return not self.start_violations

    @property
    def destination(self) -> TransitionRung | None:
        for rung in reversed(self.rungs):
            if rung.feasible:
                return rung
        return None

    def ore_cost_saving_rs_per_thm(self) -> float | None:
        """Saving between the first and last feasible rung, per tHM."""

        feasible = [r for r in self.rungs if r.feasible]
        if len(feasible) < 2:
            return None
        return float(
            feasible[0].blend.ore_cost_per_thm_rs
            - feasible[-1].blend.ore_cost_per_thm_rs
        )


def _clamped_start(
    ores: list[OreInput], manual_shares_pct: dict[str, float]
) -> dict[str, float]:
    """Start from the manual blend, clamped into each ore's allowed range.

    The manual blend is what the plant actually charged, which may sit outside
    the share limits the operator has since set. Starting from an inadmissible
    point would make the first rung meaningless, so it is clamped and then
    renormalised to 100%.
    """

    clamped = {
        ore.ore_id: float(manual_shares_pct.get(ore.ore_id, 0.0) or 0.0)
        for ore in ores
    }
    if sum(clamped.values()) <= 0.0:
        # No usable manual blend: fall back to each ore's minimum.
        return {ore.ore_id: float(ore.min_share_pct) for ore in ores}

    # Clamping and renormalising fight each other: scaling a clamped set back to
    # 100% pushes ores past their caps again (70 out of 75 becomes 93). So clamp
    # once, then close the gap ADDITIVELY, spreading it across the ores that
    # still have room. Because the amount handed to each ore is at most its own
    # remaining headroom, no cap can be broken and one pass is exact.
    for ore in ores:
        clamped[ore.ore_id] = min(
            max(clamped[ore.ore_id], float(ore.min_share_pct)),
            float(ore.max_share_pct),
        )

    deficit = 100.0 - sum(clamped.values())
    if abs(deficit) > 1e-9:
        room = {
            ore.ore_id: (
                float(ore.max_share_pct) - clamped[ore.ore_id]
                if deficit > 0
                else clamped[ore.ore_id] - float(ore.min_share_pct)
            )
            for ore in ores
        }
        available = sum(max(0.0, value) for value in room.values())
        # If the caps cannot span 100% at all, leave it short: the LP reports
        # that as a structural infeasibility with a far clearer message.
        if available > 1e-9:
            movable = min(abs(deficit), available)
            sign = 1.0 if deficit > 0 else -1.0
            for ore_id, headroom in room.items():
                if headroom > 0.0:
                    clamped[ore_id] += sign * movable * headroom / available
    return clamped


def _ores_bounded_around(
    ores: list[OreInput], shares_pct: dict[str, float], delta_pct: float
) -> list[OreInput]:
    """Tighten every ore's share bounds to within ``delta`` of the last rung.

    This is how the move limit reaches the LP: no solver change is needed,
    because the LP already enforces per-ore share bounds.
    """

    out = []
    for ore in ores:
        here = float(shares_pct.get(ore.ore_id, 0.0) or 0.0)
        out.append(
            replace(
                ore,
                min_share_pct=max(float(ore.min_share_pct), here - delta_pct),
                max_share_pct=min(float(ore.max_share_pct), here + delta_pct),
            )
        )
    return out


def _binding_limits(
    blend: BlendEvaluation,
    targets: dict[str, Any],
    ores: list[OreInput] | None = None,
) -> list[str]:
    """What this rung is sitting on - a slag limit, or an ore share cap.

    Both matter. Often the ladder stops because an ore has hit its own
    maximum share, not because any slag limit binds, and telling the
    operator "raise the CLO cap" is far more actionable than silence.
    """

    hits: list[str] = []
    checks = (
        ("slag rate", "slag_rate_kg_per_thm", targets.get("slag_rate_cap"), 1.0),
        ("CaO/SiO2 max", "slag_basicity", targets.get("basicity_max"), 5e-3),
        ("CaO/SiO2 min", "slag_basicity", targets.get("basicity_min"), 5e-3),
        ("T-basicity max", "slag_t_basicity", targets.get("t_basicity_max"), 5e-3),
        ("T-basicity min", "slag_t_basicity", targets.get("t_basicity_min"), 5e-3),
        ("Al2O3 max", "slag_al2o3_pct", targets.get("al2o3_max"), 0.05),
        ("MgO min", "slag_mgo_pct", targets.get("mgo_min"), 0.05),
        ("MgO/Al2O3 min", "slag_mgo_al2o3_ratio", targets.get("mgo_al2o3_min"), 5e-3),
    )
    for label, attribute, bound, tolerance in checks:
        if bound is None:
            continue
        value = float(getattr(blend, attribute, 0.0) or 0.0)
        if abs(value - float(bound)) <= tolerance:
            hits.append(label)

    for ore in ores or []:
        share = float(blend.shares_pct.get(ore.ore_id, 0.0) or 0.0)
        if abs(share - float(ore.max_share_pct)) <= 0.05:
            hits.append(f"{ore.display_name} at max share {ore.max_share_pct:.0f}%")
        elif abs(share - float(ore.min_share_pct)) <= 0.05 and ore.min_share_pct > 0:
            hits.append(f"{ore.display_name} at min share {ore.min_share_pct:.0f}%")
    return hits


def _evaluate_start(
    ores: list[OreInput], shares_pct: dict[str, float], lp_kwargs: dict[str, Any]
) -> tuple[BlendEvaluation | None, list[str]]:
    """Solve the current blend as-charged, then check it against the limits.

    Shares are pinned so the LP has no freedom - it only scales the burden to
    the Fe target. The slag limits are then applied by the shared validator, so
    the violations reported here read exactly the same as anywhere else.
    """

    from utils.bmo.constraints import check_blend_constraints

    pinned = [
        replace(ore, min_share_pct=shares_pct.get(ore.ore_id, 0.0),
                max_share_pct=shares_pct.get(ore.ore_id, 0.0))
        for ore in ores
    ]
    unconstrained = dict(lp_kwargs)
    limit_keys = (
        "target_slag_basicity_min", "target_slag_basicity_max",
        "target_slag_t_basicity_min", "target_slag_t_basicity_max",
        "target_slag_al2o3_max_pct", "target_slag_mgo_min_pct",
        "target_slag_mgo_al2o3_ratio_min",
    )
    limits = {key: unconstrained.pop(key, None) for key in limit_keys}
    slag_cap = unconstrained.pop("target_slag_qty_mt", None)
    unconstrained["target_slag_qty_mt"] = 1.0e9

    blend, _ = run_lp_baseline(pinned, **unconstrained)
    if blend is None:
        return None, ["Current blend could not be evaluated against the Fe target."]

    violations = check_blend_constraints(
        blend, ores,
        target_production_mt=unconstrained.get("target_production_mt", 0.0),
        target_slag_qty_mt=slag_cap if slag_cap is not None else 1.0e9,
        **limits,
    )
    return blend, violations


def build_transition_ladder(
    ores: list[OreInput],
    manual_shares_pct: dict[str, float],
    *,
    max_share_move_pct: float = DEFAULT_MAX_SHARE_MOVE_PCT,
    max_rungs: int = DEFAULT_MAX_RUNGS,
    **lp_kwargs: Any,
) -> TransitionLadder:
    """
    Build the step-by-step path from the manual blend to the optimum.

    Args:
         - ores: list[OreInput] - Ores available, with their real share limits.
         - manual_shares_pct: dict[str, float] - What the plant is charging now,
           keyed by ore id. Clamped into the allowed range before starting.
         - max_share_move_pct: float - Largest share change any single ore may
           make in one rung. This is the operator's step-change policy.
         - max_rungs: int - Safety stop.
         - **lp_kwargs: Any - Passed straight to ``run_lp_baseline``: targets,
           slag limits, flux, dust, slag balance settings and so on.

    Returns:
         - return TransitionLadder - Rungs in order, whether it converged, and
           the ore-cost saving between first and last.
    """

    delta = max(0.1, float(max_share_move_pct))
    start = _clamped_start(ores, manual_shares_pct)
    targets = {
        "slag_rate_cap": lp_kwargs.get("_slag_rate_cap"),
        "basicity_min": lp_kwargs.get("target_slag_basicity_min"),
        "basicity_max": lp_kwargs.get("target_slag_basicity_max"),
        "t_basicity_min": lp_kwargs.get("target_slag_t_basicity_min"),
        "t_basicity_max": lp_kwargs.get("target_slag_t_basicity_max"),
        "al2o3_max": lp_kwargs.get("target_slag_al2o3_max_pct"),
        "mgo_min": lp_kwargs.get("target_slag_mgo_min_pct"),
        "mgo_al2o3_min": lp_kwargs.get("target_slag_mgo_al2o3_ratio_min"),
    }
    lp_kwargs.pop("_slag_rate_cap", None)

    # Evaluate the current blend first. If it already breaches the limits the
    # operator has set, that is the finding - not a bare "infeasible" from the
    # first rung.
    start_blend, start_violations = _evaluate_start(ores, start, dict(lp_kwargs))

    rungs: list[TransitionRung] = []
    current = dict(start)
    converged = False

    for index in range(1, int(max_rungs) + 1):
        stepped_ores = _ores_bounded_around(ores, current, delta)
        blend, errors = run_lp_baseline(stepped_ores, **lp_kwargs)
        if blend is None:
            rungs.append(TransitionRung(index=index, shares_pct=dict(current),
                                        blend=None, errors=list(errors)))
            break

        next_shares = {
            ore.ore_id: float(blend.shares_pct.get(ore.ore_id, 0.0) or 0.0)
            for ore in ores
        }
        rungs.append(
            TransitionRung(
                index=index,
                shares_pct=next_shares,
                blend=blend,
                binding_limits=_binding_limits(blend, targets, ores),
            )
        )

        moved = max(
            abs(next_shares[ore_id] - current.get(ore_id, 0.0)) for ore_id in next_shares
        )
        current = next_shares
        if moved <= CONVERGENCE_TOL_PCT:
            converged = True
            # This rung repeats the previous one - it proves convergence rather
            # than asking the operator to do anything, so it is not a step.
            if len(rungs) > 1:
                rungs.pop()
            break

    return TransitionLadder(
        rungs=rungs,
        start_shares_pct=start,
        converged=converged,
        max_share_move_pct=delta,
        start_blend=start_blend,
        start_violations=start_violations,
        diagnostics={
            "rungs_taken": len(rungs),
            "hit_max_rungs": len(rungs) >= int(max_rungs) and not converged,
            "targets": targets,
        },
    )
