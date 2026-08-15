"""Nonlinear BMO optimizer orchestration.

This module prepares share bounds, seeds DE around the LP baseline, evaluates
candidate wet-share vectors, and returns the best total-cost blend found by
the differential-evolution runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from domain.optimization_runtime import ObjectiveResult, OptimizerRunner
from utils.bmo.calculations import scale_ore_quantities_to_hot_metal
from utils.bmo.coke_correction import (
    CokeCorrectionReference,
    CokeCorrectionSettings,
)
from utils.bmo.constraints import check_blend_constraints, validate_ore_bounds
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.objective import BmoObjectiveEvaluator
from utils.bmo.types import (
    BlendEvaluation,
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceSettings,
)


def _project_shares(
    raw: np.ndarray, min_shares: np.ndarray, max_shares: np.ndarray
) -> np.ndarray:
    shares = np.clip(np.asarray(raw, dtype=float), min_shares, max_shares)
    for _ in range(20):
        diff = 1.0 - float(np.sum(shares))
        if abs(diff) <= 1e-10:
            break
        if diff > 0:
            capacity = np.clip(max_shares - shares, 0.0, None)
        else:
            capacity = np.clip(shares - min_shares, 0.0, None)
        cap_sum = float(np.sum(capacity))
        if cap_sum <= 1e-12:
            break
        shares += capacity / cap_sum * diff
        shares = np.clip(shares, min_shares, max_shares)
    return shares


def _quantities_from_shares(
    *,
    shares: np.ndarray,
    ores: list[OreInput],
    target_fe_mt: float,
    hot_metal_target_mt: float | None = None,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
    slag_balance_settings: SlagBalanceSettings | None = None,
    charge_mass_mt: float = 26.4,
) -> np.ndarray:
    fe_per_wet_mt = np.array(
        [
            max(0.0, 1.0 - float(ore.chemistry.moisture_pct) / 100.0)
            * max(0.0, float(ore.chemistry.fe_t_pct))
            / 100.0
            for ore in ores
        ],
        dtype=float,
    )
    fe_per_blend_mt = float(np.dot(shares, fe_per_wet_mt))
    if fe_per_blend_mt <= 0.0:
        return np.zeros(len(ores), dtype=float)
    total_wet_mt = float(target_fe_mt) / fe_per_blend_mt
    quantities = shares * total_wet_mt
    if (
        hot_metal_target_mt is not None
        and float(hot_metal_target_mt) > 0.0
        and slag_balance_settings is not None
        and slag_balance_settings.enabled
    ):
        scaled = scale_ore_quantities_to_hot_metal(
            ores=ores,
            reference_quantities_mt={
                ore.ore_id: float(quantities[index]) for index, ore in enumerate(ores)
            },
            target_hot_metal_mt=float(hot_metal_target_mt),
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            charge_mass_mt=charge_mass_mt,
        )
        quantities = np.array(
            [float(scaled.get(ore.ore_id, 0.0)) for ore in ores], dtype=float
        )
    return quantities


def _build_initial_share_population(
    *,
    lp_shares: np.ndarray,
    min_shares: np.ndarray,
    max_shares: np.ndarray,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    population: list[np.ndarray] = [_project_shares(lp_shares, min_shares, max_shares)]
    scales = [0.01, 0.025, 0.05, 0.075, 0.10]
    while len(population) < sample_count:
        scale = scales[(len(population) - 1) % len(scales)]
        noise = rng.normal(0.0, scale, size=lp_shares.shape)
        population.append(_project_shares(lp_shares + noise, min_shares, max_shares))
    return np.vstack(population)


def _build_random_share_population(
    *,
    min_shares: np.ndarray,
    max_shares: np.ndarray,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """
    Build a DE start population that does not depend on an LP solution.

    Used when the LP is infeasible, or when the operator asks for an independent
    search. Unlike the LP-seeded population, which explores a few percent around
    one point, this spreads across the whole share simplex: the midpoint of the
    bounds, one corner per ore with that ore pushed to its maximum, then Dirichlet
    draws. Dirichlet rather than independent uniforms because independent draws
    concentrate near the centre once projected back onto the simplex, which would
    leave the corners — where a lean or a rich burden lives — unexplored.

    Args:
         - min_shares: np.ndarray - Per-ore minimum share as a fraction.
         - max_shares: np.ndarray - Per-ore maximum share as a fraction.
         - sample_count: int - Number of population members to return.
         - seed: int - Seed for reproducibility.

    Returns:
         - return np.ndarray - ``(sample_count, n_ore)`` share population.
    """

    rng = np.random.default_rng(seed)
    n = int(len(min_shares))
    population: list[np.ndarray] = [
        _project_shares((min_shares + max_shares) / 2.0, min_shares, max_shares)
    ]
    for index in range(n):
        corner = np.array(min_shares, dtype=float)
        corner[index] = float(max_shares[index])
        population.append(_project_shares(corner, min_shares, max_shares))
    while len(population) < sample_count:
        population.append(
            _project_shares(rng.dirichlet(np.ones(n)), min_shares, max_shares)
        )
    return np.vstack(population[: max(sample_count, 1)])


def _build_random_flux_population(
    *,
    flux_bounds: list[tuple[float, float]],
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Uniform draws inside each flux's ``(0, stock)`` bound, plus an all-zero row."""

    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in flux_bounds], dtype=float)
    hi = np.array([b[1] for b in flux_bounds], dtype=float)
    # Charging no flux at all is a legitimate and common answer, so make sure it
    # is in the population rather than leaving DE to stumble onto it.
    rows = [lo.copy()]
    while len(rows) < sample_count:
        rows.append(lo + rng.random(lo.shape) * (hi - lo))
    return np.vstack(rows[: max(sample_count, 1)])


def run_nonlinear_optimizer(
    ores: list[OreInput],
    *,
    target_production_mt: float,
    target_slag_qty_mt: float,
    feo_in_slag_pct: float,
    model_service: FuelUnitCostModelService,
    process_context: dict[str, float] | None,
    history_df: pd.DataFrame | None,
    de_cfg: dict[str, Any],
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
    hot_metal_target_mt: float | None = None,
    charge_mass_mt: float = 26.4,
    coke_correction_settings: CokeCorrectionSettings | None = None,
    coke_correction_reference: CokeCorrectionReference | None = None,
    hot_metal_si_pct: float | None = None,
    fuel_rate_anchor_basis: str = "model_cost",
    progress_callback: (
        Callable[[int, float, float | None, int, float], bool] | None
    ) = None,
) -> tuple[BlendEvaluation | None, list[str]]:
    """
    Run nonlinear total-cost BMO optimization with DE.

    The nonlinear path normally starts from the feasible LP baseline, then
    explores wet quantity vectors with a fuel-aware objective. ``de_cfg`` key
    ``initial_solution`` decides what happens when the LP cannot solve:
    ``"lp"`` skips DE entirely, ``"random"`` never consults the LP, and
    ``"lp_else_random"`` (the default) falls back to a population spread across
    the share simplex so an infeasible LP no longer takes DE down with it.

    Args:
         - ores: list[OreInput] - Ores selected for optimization.
         - target_production_mt: float - Target hot-metal production in MT.
         - target_slag_qty_mt: float - Maximum allowed slag quantity in MT.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - target_slag_basicity_min: float | None - Minimum CaO / SiO2 basicity.
         - target_slag_basicity_max: float | None - Maximum CaO / SiO2 basicity.
         - target_slag_t_basicity_min/max: float | None - (CaO + MgO) / SiO2 bounds.
         - target_slag_al2o3_max_pct: float | None - Maximum Al2O3 % of final slag.
         - target_slag_mgo_min_pct: float | None - Minimum MgO % of final slag.
         - target_slag_mgo_al2o3_ratio_min: float | None - Minimum MgO / Al2O3 ratio.
         - max_burden_qty_mt: float | None - Charging-throughput ceiling on total wet
           IBRM + flux in MT. ``None`` leaves the burden quantity unbounded.
         - model_service: FuelUnitCostModelService - Fuel-cost prediction service.
         - process_context: dict[str, float] | None - Latest process variables.
         - history_df: pd.DataFrame | None - Historical process data for lagged features.
         - de_cfg: dict[str, Any] - Differential-evolution and penalty settings.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
         - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
         - dust_inputs: list[DustInput] | None - Dust rows deducted in final balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.
         - coke_correction_settings: CokeCorrectionSettings | None - Physics
           coke-rate correction settings, forwarded to the seed LP and to every
           DE candidate so both optimise the same objective.
         - coke_correction_reference: CokeCorrectionReference | None - Recent
           observed operating point, resolved once and held frozen for the run.
         - hot_metal_si_pct: float | None - Si used by the correction's Si term,
           held constant across candidates (see ``BmoObjectiveEvaluator``).
         - fuel_rate_anchor_basis: str - ``"observed"`` or ``"model_cost"``; see
           ``evaluate_blend_with_fuel_prediction``.

    Returns:
         - return tuple[BlendEvaluation | None, list[str]] - Best blend and errors.
    """

    pre_errors = validate_ore_bounds(ores)
    if pre_errors:
        return None, pre_errors

    # How DE gets its starting population.
    #   lp              - seed from the LP baseline; abort if the LP is infeasible
    #   random          - ignore the LP entirely and search from a spread population
    #   lp_else_random  - prefer the LP, fall back to a random spread when it fails
    # The default rescues the case the LP cannot solve. An infeasible LP does not
    # always mean the problem is unsatisfiable: the LP works on a linearised slag
    # and basicity model, so it can reject a burden the exact evaluation accepts.
    # DE scores with soft penalties, so it can return a best-effort blend there,
    # flagged with its violations, instead of the page showing nothing at all.
    seed_strategy = (
        str(de_cfg.get("initial_solution", "lp_else_random")).strip().lower()
    )
    if seed_strategy not in {"lp", "random", "lp_else_random"}:
        seed_strategy = "lp_else_random"

    lp_blend: BlendEvaluation | None = None
    lp_errors: list[str] = []
    if seed_strategy != "random":
        lp_blend, lp_errors = run_lp_baseline(
            ores,
            target_production_mt=target_production_mt,
            target_slag_qty_mt=target_slag_qty_mt,
            feo_in_slag_pct=feo_in_slag_pct,
            target_slag_basicity_min=target_slag_basicity_min,
            target_slag_basicity_max=target_slag_basicity_max,
            target_slag_t_basicity_min=target_slag_t_basicity_min,
            target_slag_t_basicity_max=target_slag_t_basicity_max,
            target_slag_al2o3_max_pct=target_slag_al2o3_max_pct,
            target_slag_mgo_min_pct=target_slag_mgo_min_pct,
            target_slag_mgo_al2o3_ratio_min=target_slag_mgo_al2o3_ratio_min,
            max_burden_qty_mt=max_burden_qty_mt,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            hot_metal_target_mt=hot_metal_target_mt,
            # The seed must be optimised against the same objective DE will use.
            # Seeding from an uncorrected LP would drop DE into the low-Fe corner
            # the correction exists to avoid, and DE only explores a few percent
            # around its seed - it would never find its way back out.
            #
            # This is why ``price_coke_correction`` exists as a flag rather than
            # being deleted: the page's own LP is a pure ore-cost solve, but DE
            # optimises ore + predicted fuel and its seed has to match.
            coke_correction_settings=coke_correction_settings,
            coke_correction_reference=coke_correction_reference,
            price_coke_correction=True,
            charge_mass_mt=charge_mass_mt,
        )
    if lp_blend is None and seed_strategy == "lp":
        return None, [
            "Total-cost optimizer skipped because hard LP constraints are infeasible.",
            *lp_errors,
        ]

    min_shares = np.array(
        [float(ore.min_share_pct) / 100.0 for ore in ores], dtype=float
    )
    max_shares = np.array(
        [float(ore.max_share_pct) / 100.0 for ore in ores], dtype=float
    )
    # Optimisable fluxes (dolomite/quartz) are extra DE variables (absolute MT,
    # 0..stock) so DE can add flux to satisfy basicity, just like the LP.
    n_ore = len(ores)
    variable_fluxes = [
        flux
        for flux in (flux_inputs or [])
        if flux.optimizable and flux.enabled and float(flux.stock_mt) > 0.0
    ]
    n_flux = len(variable_fluxes)
    flux_bounds = [(0.0, max(0.0, float(flux.stock_mt))) for flux in variable_fluxes]
    bounds = [
        (float(lo), float(hi)) for lo, hi in zip(min_shares, max_shares)
    ] + flux_bounds

    prebuilt_context = model_service.build_prebuilt_context(
        ores=ores,
        process_context=process_context,
        history_df=history_df,
        hot_metal_target_mt=hot_metal_target_mt,
    )

    evaluator = BmoObjectiveEvaluator(
        ores=ores,
        target_production_mt=float(target_production_mt),
        target_slag_qty_mt=float(target_slag_qty_mt),
        feo_in_slag_pct=float(feo_in_slag_pct),
        target_slag_basicity_min=target_slag_basicity_min,
        target_slag_basicity_max=target_slag_basicity_max,
        target_slag_t_basicity_min=target_slag_t_basicity_min,
        target_slag_t_basicity_max=target_slag_t_basicity_max,
        target_slag_al2o3_max_pct=target_slag_al2o3_max_pct,
        target_slag_mgo_min_pct=target_slag_mgo_min_pct,
        target_slag_mgo_al2o3_ratio_min=target_slag_mgo_al2o3_ratio_min,
        max_burden_qty_mt=max_burden_qty_mt,
        model_service=model_service,
        process_context=process_context,
        history_df=history_df,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=flux_inputs,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
        penalty_cfg=de_cfg,
        prebuilt_context=prebuilt_context,
        hot_metal_target_mt=hot_metal_target_mt,
        coke_correction_settings=coke_correction_settings,
        coke_correction_reference=coke_correction_reference,
        hot_metal_si_pct=hot_metal_si_pct,
        fuel_rate_anchor_basis=fuel_rate_anchor_basis,
        charge_mass_mt=charge_mass_mt,
    )

    # Every DE function evaluation is recorded here as a compact
    # (basicity, slag, total cost, feasible) row so the page can render the
    # cost-vs-basicity-vs-slag exploration scatter and the top-alternatives table.
    de_candidates: list[dict[str, float]] = []

    def objective(raw_x: np.ndarray) -> ObjectiveResult:
        """
        Evaluate a DE candidate wet-share vector.

        This nested adapter keeps candidate diagnostics attached to each SciPy
        evaluation. The outer runner can then recover the best blend and expose
        the quantities that produced it.

        Args:
             - raw_x: np.ndarray - Candidate shares from differential evolution.

        Returns:
             - return ObjectiveResult - Penalized objective result for the quantities.
        """

        raw = np.asarray(raw_x, dtype=float)
        ore_x = raw[:n_ore]
        flux_x = raw[n_ore : n_ore + n_flux]
        shares = _project_shares(ore_x, min_shares, max_shares)
        candidate_flux_inputs = list(evaluator.fixed_fluxes) + [
            replace(flux, wet_qty_mt=float(flux_x[j]))
            for j, flux in enumerate(variable_fluxes)
        ]
        quantities = _quantities_from_shares(
            shares=shares,
            ores=ores,
            target_fe_mt=float(target_production_mt),
            hot_metal_target_mt=hot_metal_target_mt,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=candidate_flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            charge_mass_mt=charge_mass_mt,
        )
        result = evaluator.evaluate_quantities(quantities, flux_quantities=flux_x)
        result.diagnostics["candidate_shares_pct"] = (shares * 100.0).tolist()
        result.diagnostics["raw_candidate_shares_pct"] = (ore_x * 100.0).tolist()
        result.diagnostics["candidate_quantities_mt"] = np.asarray(
            quantities, dtype=float
        ).tolist()
        result.diagnostics["candidate_flux_quantities_mt"] = flux_x.tolist()

        candidate_blend = result.diagnostics.get("blend")
        if candidate_blend is not None:
            flux_cost = float(
                candidate_blend.diagnostics.get("flux_cost_per_thm_rs", 0.0) or 0.0
            )
            total_cost = float(candidate_blend.objective_rs_per_thm) + flux_cost
            if math.isfinite(total_cost):
                de_candidates.append(
                    {
                        "total_cost_rs_per_thm": total_cost,
                        "slag_basicity": float(
                            getattr(candidate_blend, "slag_basicity", 0.0) or 0.0
                        ),
                        "slag_mt": float(
                            getattr(candidate_blend, "slag_mt", 0.0) or 0.0
                        ),
                        "feasible": bool(result.feasible),
                        # The blend combination behind this candidate, so the UI
                        # can list full alternative solutions, not just metrics.
                        "shares_pct": {
                            ore.ore_id: float(shares[idx] * 100.0)
                            for idx, ore in enumerate(ores)
                        },
                        "flux_mt": {
                            flux.flux_id: float(flux_x[j])
                            for j, flux in enumerate(variable_fluxes)
                        },
                    }
                )
        return result

    baseline_solution: dict[str, Any] | None = None
    initial_population: np.ndarray | None = None
    n_vars = n_ore + n_flux
    min_samples = max(5, int(de_cfg.get("popsize", 10)) * n_vars)
    sample_count = max(
        min_samples, int(de_cfg.get("initial_population_samples", min_samples))
    )
    seed_used = (
        "lp" if (lp_blend is not None and lp_blend.total_qty_mt > 0) else "random"
    )
    if lp_blend is not None and lp_blend.total_qty_mt > 0:
        lp_quantities = np.array(
            [float(lp_blend.quantities_mt.get(ore.ore_id, 0.0)) for ore in ores],
            dtype=float,
        )
        lp_shares = lp_quantities / float(np.sum(lp_quantities))
        # Seed DE from the LP solution, including the flux quantities the LP chose.
        lp_flux_map = lp_blend.diagnostics.get("lp_flux_quantities_mt", {}) or {}
        lp_flux_vec = np.array(
            [float(lp_flux_map.get(flux.flux_id, 0.0)) for flux in variable_fluxes],
            dtype=float,
        )
        baseline_x = np.concatenate([lp_shares, lp_flux_vec])
        baseline_result = objective(baseline_x)
        baseline_solution = {
            "x": baseline_x.tolist(),
            "objective": float(baseline_result.objective_value),
            "feasible": bool(baseline_result.feasible),
            "components": dict(baseline_result.components),
            "violations": list(baseline_result.violations),
            "diagnostics": dict(baseline_result.diagnostics),
        }
        ore_population = _build_initial_share_population(
            lp_shares=lp_shares,
            min_shares=min_shares,
            max_shares=max_shares,
            sample_count=sample_count,
            seed=int(de_cfg.get("seed", 42)),
        )
        if n_flux > 0:
            rng = np.random.default_rng(int(de_cfg.get("seed", 42)) + 1)
            flux_lo = np.array([lo for lo, _ in flux_bounds], dtype=float)
            flux_hi = np.array([hi for _, hi in flux_bounds], dtype=float)
            flux_rows = [lp_flux_vec]
            spread = np.maximum(flux_hi * 0.15, 1.0)
            while len(flux_rows) < ore_population.shape[0]:
                noise = rng.normal(0.0, 1.0, size=lp_flux_vec.shape) * spread
                flux_rows.append(np.clip(lp_flux_vec + noise, flux_lo, flux_hi))
            initial_population = np.hstack([ore_population, np.vstack(flux_rows)])
        else:
            initial_population = ore_population
    else:
        # No usable LP seed. Search from a population spread across the whole
        # share simplex instead of returning nothing. There is no baseline
        # solution to hand the runner, so DE keeps whatever it finds.
        ore_population = _build_random_share_population(
            min_shares=min_shares,
            max_shares=max_shares,
            sample_count=sample_count,
            seed=int(de_cfg.get("seed", 42)),
        )
        if n_flux > 0:
            flux_population = _build_random_flux_population(
                flux_bounds=flux_bounds,
                sample_count=ore_population.shape[0],
                seed=int(de_cfg.get("seed", 42)) + 1,
            )
            initial_population = np.hstack([ore_population, flux_population])
        else:
            initial_population = ore_population

    runner = OptimizerRunner(de_cfg)
    optimization_result = runner.run_differential_evolution(
        bounds=bounds,
        objective_fn=objective,
        baseline_solution=baseline_solution,
        initial_population=initial_population,
        progress_callback=progress_callback,
    )

    best_diag = dict(optimization_result.best_solution.get("diagnostics", {}))
    blend = best_diag.get("blend")
    if blend is None:
        msg = (
            optimization_result.diagnostics.get("de_result", {}).get("message")
            or "DE failed."
        )
        return None, [f"Nonlinear optimizer failed: {msg}"]

    blend = blend if isinstance(blend, BlendEvaluation) else None
    if blend is None:
        return None, [
            "Nonlinear optimizer failed: could not recover blend diagnostics."
        ]

    violations = check_blend_constraints(
        blend,
        ores,
        target_production_mt=target_production_mt,
        target_slag_qty_mt=target_slag_qty_mt,
        target_slag_basicity_min=target_slag_basicity_min,
        target_slag_basicity_max=target_slag_basicity_max,
        target_slag_t_basicity_min=target_slag_t_basicity_min,
        target_slag_t_basicity_max=target_slag_t_basicity_max,
        target_slag_al2o3_max_pct=target_slag_al2o3_max_pct,
        target_slag_mgo_min_pct=target_slag_mgo_min_pct,
        target_slag_mgo_al2o3_ratio_min=target_slag_mgo_al2o3_ratio_min,
        max_burden_qty_mt=max_burden_qty_mt,
    )
    blend.feasible = len(violations) == 0
    blend.violations = violations
    blend.diagnostics["de_result"] = optimization_result.diagnostics.get(
        "de_result", {}
    )
    blend.diagnostics["runtime"] = {
        "best_solution": optimization_result.best_solution,
        "compare_metrics": optimization_result.compare_metrics,
    }
    blend.diagnostics["de_candidates"] = de_candidates
    # Provenance of the start population. When the LP could not seed DE the
    # operator needs to know the result came from an unguided search and that the
    # LP's own reasons are worth reading, so carry both rather than dropping them.
    blend.diagnostics["de_seed"] = {
        "strategy_requested": seed_strategy,
        "strategy_used": seed_used,
        "lp_seed_available": lp_blend is not None,
        "lp_seed_errors": list(lp_errors),
    }
    return blend, []
