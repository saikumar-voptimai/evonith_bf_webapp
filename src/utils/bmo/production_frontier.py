"""Production-frontier simulation for the BMO page.

The simulator repeatedly solves the deterministic BMO LP across production
targets and bounded daily raw-material scenarios.  It deliberately lives
outside the main optimizer state: the UI can explore a different planning
envelope without changing the currently applied BMO inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from utils.bmo.constraints import max_ibrm_flux_capacity_mt
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.types import (
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceSettings,
)


@dataclass(frozen=True)
class FrontierSettings:
    """Independent controls for one production-frontier simulation."""

    production_min_mt: float = 1700.0
    production_max_mt: float = 2500.0
    production_step_mt: float = 5.0
    max_slag_rate_kg_per_thm: float = 375.0
    basicity_min: float = 1.0
    basicity_max: float = 1.2
    t_basicity_min: float | None = None
    t_basicity_max: float | None = None
    al2o3_max_pct: float | None = None
    mgo_min_pct: float | None = None
    mgo_al2o3_min: float | None = None
    max_charges_per_hour: float = 6.35
    charge_mass_mt: float = 26.4
    enforce_charging_capacity: bool = True
    nut_coke_rate_kg_per_thm: float = 70.0
    nut_coke_moisture_pct: float = 0.0
    variation_scenarios: int = 20
    stock_variation_pct: float = 15.0
    price_variation_pct: float = 7.5
    fe_variation_pp: float = 0.75
    gangue_variation_pct: float = 7.5
    near_minimum_tolerance_rs_per_thm: float = 25.0
    random_seed: int = 234


ProgressCallback = Callable[[dict[str, Any]], None]

# Largest run the form may request. The default 1,700-2,500 MT range at a 5 MT
# step has 161 targets, so the 1,000-scenario maximum needs 161,161 solves; at a
# few ms per LP that is minutes, and the progress panel shows the ETA.
MAX_VARIATION_SCENARIOS = 1000
MAX_TOTAL_SOLVES = 200_000


def production_targets(settings: FrontierSettings) -> np.ndarray:
    """Return the inclusive production grid after validating its shape."""

    low = float(settings.production_min_mt)
    high = float(settings.production_max_mt)
    step = float(settings.production_step_mt)
    if low <= 0.0 or high <= 0.0:
        raise ValueError("Production targets must be greater than zero.")
    if high < low:
        raise ValueError("Maximum production must be at least minimum production.")
    if step <= 0.0:
        raise ValueError("Production step must be greater than zero.")
    count = int(np.floor((high - low) / step + 1.0e-9)) + 1
    targets = low + np.arange(count, dtype=float) * step
    if targets[-1] < high - 1.0e-7:
        targets = np.append(targets, high)
    return targets


def validate_frontier_inputs(
    ores: list[OreInput], settings: FrontierSettings
) -> list[str]:
    """Return operator-facing errors before any scenario is solved."""

    errors: list[str] = []
    try:
        targets = production_targets(settings)
    except ValueError as exc:
        return [str(exc)]
    selected = [ore for ore in ores if float(ore.max_share_pct) > 0.0]
    if len(selected) < 2:
        errors.append("Select at least two materials with a positive maximum share.")
    if float(settings.basicity_min) > float(settings.basicity_max):
        errors.append("Minimum basicity cannot be greater than maximum basicity.")
    if (
        settings.t_basicity_min is not None
        and settings.t_basicity_max is not None
        and float(settings.t_basicity_min) > float(settings.t_basicity_max)
    ):
        errors.append(
            "Minimum total basicity cannot be greater than maximum total basicity."
        )
    total_solves = len(targets) * (int(settings.variation_scenarios) + 1)
    if total_solves > MAX_TOTAL_SOLVES:
        errors.append(
            f"This setup requests {total_solves:,} LP solves; reduce the range, "
            "scenario count, or use a larger production step "
            f"(maximum {MAX_TOTAL_SOLVES:,})."
        )
    if int(settings.variation_scenarios) < 0:
        errors.append("Variation scenario count cannot be negative.")
    if int(settings.variation_scenarios) > MAX_VARIATION_SCENARIOS:
        errors.append(
            f"Variation scenario count cannot exceed {MAX_VARIATION_SCENARIOS:,}."
        )
    return errors


def _scenario_ores(
    ores: list[OreInput],
    *,
    rng: np.random.Generator,
    settings: FrontierSettings,
    is_base: bool,
) -> tuple[list[OreInput], dict[str, float]]:
    """Create one bounded daily material scenario and its shock diagnostics."""

    if is_base:
        factors = {
            "mean_stock_factor": 1.0,
            "minimum_stock_factor": 1.0,
            "mean_price_factor": 1.0,
            "price_factor_spread": 0.0,
            "mean_fe_shift_pp": 0.0,
            "mean_gangue_factor": 1.0,
        }
        return list(ores), factors

    stock_width = max(0.0, float(settings.stock_variation_pct)) / 100.0
    price_width = max(0.0, float(settings.price_variation_pct)) / 100.0
    fe_width = max(0.0, float(settings.fe_variation_pp))
    gangue_width = max(0.0, float(settings.gangue_variation_pct)) / 100.0
    stock_factors: list[float] = []
    price_factors: list[float] = []
    fe_shifts: list[float] = []
    gangue_factors: list[float] = []
    varied: list[OreInput] = []

    for ore in ores:
        stock_factor = float(rng.uniform(1.0 - stock_width, 1.0 + stock_width))
        price_factor = float(rng.uniform(1.0 - price_width, 1.0 + price_width))
        fe_shift = float(rng.uniform(-fe_width, fe_width))
        gangue_factor = float(rng.uniform(1.0 - gangue_width, 1.0 + gangue_width))
        chemistry = replace(
            ore.chemistry,
            fe_t_pct=float(
                np.clip(float(ore.chemistry.fe_t_pct) + fe_shift, 0.0, 100.0)
            ),
            sio2_pct=max(0.0, float(ore.chemistry.sio2_pct) * gangue_factor),
            al2o3_pct=max(0.0, float(ore.chemistry.al2o3_pct) * gangue_factor),
            cao_pct=max(0.0, float(ore.chemistry.cao_pct) * gangue_factor),
            mgo_pct=max(0.0, float(ore.chemistry.mgo_pct) * gangue_factor),
        )
        varied.append(
            replace(
                ore,
                stock_mt=max(0.0, float(ore.stock_mt) * stock_factor),
                price_rs_per_mt=max(0.0, float(ore.price_rs_per_mt) * price_factor),
                chemistry=chemistry,
            )
        )
        stock_factors.append(stock_factor)
        price_factors.append(price_factor)
        fe_shifts.append(fe_shift)
        gangue_factors.append(gangue_factor)

    return varied, {
        "mean_stock_factor": float(np.mean(stock_factors)),
        "minimum_stock_factor": float(np.min(stock_factors)),
        "mean_price_factor": float(np.mean(price_factors)),
        "price_factor_spread": float(np.ptp(price_factors)),
        "mean_fe_shift_pp": float(np.mean(fe_shifts)),
        "mean_gangue_factor": float(np.mean(gangue_factors)),
    }


def identify_knee_candidates(
    curve: pd.DataFrame,
    *,
    relative_strength_threshold: float = 0.30,
    maximum_candidates: int = 4,
) -> list[dict[str, float | str]]:
    """Return distinct economic/physical knees on a feasible frontier.

    Curvature peaks identify changes in the unit-cost slope. The first point
    where charging capacity binds is included as a physical knee even when its
    numerical curvature is modest.
    """

    feasible = curve.loc[curve["feasible"]].sort_values("production_mt").copy()
    if feasible.empty:
        return []
    x = feasible["production_mt"].to_numpy(dtype=float)
    y = feasible["unit_cost_rs_per_thm"].to_numpy(dtype=float)
    candidates: list[dict[str, float | str]] = []

    if len(feasible) >= 5 and float(np.ptp(x)) > 0.0:
        slope = np.gradient(y, x)
        curvature = np.gradient(slope, x)
        interior_start = 2 if len(curvature) >= 7 else 1
        interior_stop = len(curvature) - interior_start
        interior = np.arange(interior_start, interior_stop)
        if len(interior):
            magnitude = np.abs(curvature)
            maximum = float(np.max(magnitude[interior]))
            threshold = maximum * max(
                0.0, min(1.0, float(relative_strength_threshold))
            )
            peaks: list[int] = []
            for index in interior:
                value = float(magnitude[index])
                if value + 1.0e-12 < threshold:
                    continue
                if value + 1.0e-12 < float(magnitude[index - 1]):
                    continue
                if value + 1.0e-12 < float(magnitude[index + 1]):
                    continue
                if peaks and index - peaks[-1] <= 1:
                    previous = peaks[-1]
                    if (
                        value > float(magnitude[previous]) + 1.0e-12
                        or (
                            abs(value - float(magnitude[previous])) <= 1.0e-12
                            and y[index] < y[previous]
                        )
                    ):
                        peaks[-1] = int(index)
                    continue
                peaks.append(int(index))
            if not peaks:
                peaks = [int(interior[int(np.argmax(magnitude[interior]))])]
            strongest = sorted(
                peaks,
                key=lambda index: float(magnitude[index]),
                reverse=True,
            )[: max(1, int(maximum_candidates))]
            candidates.extend(
                {
                    "production_mt": float(x[index]),
                    "unit_cost_rs_per_thm": float(y[index]),
                    "curvature_strength": float(magnitude[index]),
                    "reason": "cost curvature",
                }
                for index in strongest
            )

    capacity_rows = feasible.loc[feasible["charging_utilization_pct"] >= 99.9]
    if not capacity_rows.empty:
        capacity_row = capacity_rows.iloc[0]
        production = float(capacity_row["production_mt"])
        typical_step = (
            float(np.median(np.diff(x))) if len(x) > 1 else 0.0
        )
        if not any(
            abs(float(item["production_mt"]) - production)
            <= max(1.0e-9, typical_step * 0.5)
            for item in candidates
        ):
            candidates.append(
                {
                    "production_mt": production,
                    "unit_cost_rs_per_thm": float(
                        capacity_row["unit_cost_rs_per_thm"]
                    ),
                    "curvature_strength": 0.0,
                    "reason": "charging capacity onset",
                }
            )

    if not candidates:
        fallback = capacity_rows.iloc[0] if not capacity_rows.empty else feasible.iloc[-1]
        candidates.append(
            {
                "production_mt": float(fallback["production_mt"]),
                "unit_cost_rs_per_thm": float(fallback["unit_cost_rs_per_thm"]),
                "curvature_strength": 0.0,
                "reason": "feasible frontier endpoint",
            }
        )
    return sorted(candidates, key=lambda item: float(item["production_mt"]))


def identify_knee(
    curve: pd.DataFrame,
    *,
    tolerance_rs_per_thm: float,
) -> dict[str, Any]:
    """Locate all knees and choose the lowest-cost one as the default."""

    feasible = curve.loc[curve["feasible"]].sort_values("production_mt").copy()
    if feasible.empty:
        return {
            "knee_production_mt": None,
            "plateau_end_production_mt": None,
            "capacity_onset_production_mt": None,
            "max_feasible_production_mt": None,
            "minimum_unit_cost_rs_per_thm": None,
            "knee_unit_cost_rs_per_thm": None,
            "curvature_strength": None,
            "knee_candidates": [],
        }

    x = feasible["production_mt"].to_numpy(dtype=float)
    y = feasible["unit_cost_rs_per_thm"].to_numpy(dtype=float)
    minimum = float(np.min(y))
    near_minimum = feasible.loc[
        feasible["unit_cost_rs_per_thm"]
        <= minimum + max(0.0, float(tolerance_rs_per_thm))
    ]
    capacity_rows = feasible.loc[feasible["charging_utilization_pct"] >= 99.9]
    knee_candidates = identify_knee_candidates(feasible)
    primary = min(
        knee_candidates,
        key=lambda item: (
            float(item["unit_cost_rs_per_thm"]),
            float(item["production_mt"]),
        ),
    )
    knee = float(primary["production_mt"])
    knee_cost = float(primary["unit_cost_rs_per_thm"])
    strength = float(primary["curvature_strength"])

    return {
        "knee_production_mt": knee,
        "plateau_end_production_mt": float(near_minimum["production_mt"].max()),
        "capacity_onset_production_mt": (
            float(capacity_rows["production_mt"].min())
            if not capacity_rows.empty
            else None
        ),
        "max_feasible_production_mt": float(feasible["production_mt"].max()),
        "minimum_unit_cost_rs_per_thm": minimum,
        "knee_unit_cost_rs_per_thm": knee_cost,
        "curvature_strength": strength,
        "knee_candidates": knee_candidates,
    }


def _spearman_driver_table(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Rank scenario attributes by their association with the simulated knee."""

    labels = {
        "available_fe_stock_mt": "Available dry Fe in stock",
        "minimum_stock_factor": "Most constrained material stock",
        "stock_weighted_fe_pct": "Stock-weighted ore Fe",
        "stock_weighted_gangue_pct": "Stock-weighted SiO2 + Al2O3",
        "stock_weighted_price_rs_per_mt": "Stock-weighted material price",
        "price_spread_rs_per_mt": "Price spread between materials",
        "price_factor_spread": "Daily relative-price dispersion",
    }
    working = scenarios.loc[
        (~scenarios["is_base"]) & scenarios["knee_production_mt"].notna()
    ].copy()
    rows: list[dict[str, Any]] = []
    if len(working) < 4:
        return pd.DataFrame(columns=["driver", "knee_correlation", "direction"])
    target_rank = working["knee_production_mt"].rank(method="average")
    for key, label in labels.items():
        values = pd.to_numeric(working[key], errors="coerce")
        if values.notna().sum() < 4 or float(values.max() - values.min()) == 0.0:
            continue
        corr = float(values.rank(method="average").corr(target_rank))
        if not np.isfinite(corr):
            continue
        rows.append(
            {
                "driver": label,
                "knee_correlation": corr,
                "direction": "higher knee" if corr > 0.0 else "lower knee",
            }
        )
    return pd.DataFrame(rows).sort_values(
        "knee_correlation", key=lambda series: series.abs(), ascending=False
    )


def _material_metrics(ores: list[OreInput]) -> dict[str, float]:
    stocks = np.array([max(0.0, float(ore.stock_mt)) for ore in ores], dtype=float)
    total_stock = max(float(stocks.sum()), 1.0e-12)
    dry = np.array(
        [
            1.0 - min(100.0, max(0.0, float(ore.chemistry.moisture_pct))) / 100.0
            for ore in ores
        ],
        dtype=float,
    )
    fe = np.array([float(ore.chemistry.fe_t_pct) for ore in ores], dtype=float)
    gangue = np.array(
        [
            float(ore.chemistry.sio2_pct) + float(ore.chemistry.al2o3_pct)
            for ore in ores
        ],
        dtype=float,
    )
    prices = np.array([float(ore.price_rs_per_mt) for ore in ores], dtype=float)
    return {
        "available_fe_stock_mt": float(np.sum(stocks * dry * fe / 100.0)),
        "stock_weighted_fe_pct": float(np.sum(stocks * fe) / total_stock),
        "stock_weighted_gangue_pct": float(np.sum(stocks * gangue) / total_stock),
        "stock_weighted_price_rs_per_mt": float(np.sum(stocks * prices) / total_stock),
        "price_spread_rs_per_mt": float(np.ptp(prices)) if len(prices) else 0.0,
    }


def run_frontier_simulation(
    ores: list[OreInput],
    *,
    settings: FrontierSettings,
    hm_fe_pct: float,
    feo_in_slag_pct: float,
    model_to_plant_slag_factor: float,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
    slag_balance_settings: SlagBalanceSettings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Solve the production frontier for the base case and daily variations."""

    errors = validate_frontier_inputs(ores, settings)
    if errors:
        raise ValueError(" ".join(errors))
    targets = production_targets(settings)
    scenario_count = int(settings.variation_scenarios) + 1
    total_solves = int(len(targets) * scenario_count)
    rng = np.random.default_rng(int(settings.random_seed))
    all_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    completed = 0
    slag_factor = max(1.0e-12, float(model_to_plant_slag_factor))

    for scenario_id in range(scenario_count):
        is_base = scenario_id == 0
        scenario_ores, shock = _scenario_ores(
            ores,
            rng=rng,
            settings=settings,
            is_base=is_base,
        )
        scenario_metrics = {**shock, **_material_metrics(scenario_ores)}
        scenario_curve: list[dict[str, Any]] = []
        for target_index, production_mt in enumerate(targets):
            max_burden_mt: float | None = None
            if settings.enforce_charging_capacity:
                max_burden_mt = max_ibrm_flux_capacity_mt(
                    {
                        "max_charges_per_hour": settings.max_charges_per_hour,
                        "charge_mass_mt": settings.charge_mass_mt,
                    },
                    target_hot_metal_mt=float(production_mt),
                    nut_coke_rate_kg_per_thm=settings.nut_coke_rate_kg_per_thm,
                    nut_coke_moisture_pct=settings.nut_coke_moisture_pct,
                )
            slag_cap_mt = (
                float(settings.max_slag_rate_kg_per_thm)
                * float(production_mt)
                / 1000.0
                / slag_factor
            )
            result, solve_errors = run_lp_baseline(
                scenario_ores,
                target_production_mt=float(production_mt) * float(hm_fe_pct) / 100.0,
                target_slag_qty_mt=slag_cap_mt,
                feo_in_slag_pct=float(feo_in_slag_pct),
                target_slag_basicity_min=float(settings.basicity_min),
                target_slag_basicity_max=float(settings.basicity_max),
                target_slag_t_basicity_min=settings.t_basicity_min,
                target_slag_t_basicity_max=settings.t_basicity_max,
                target_slag_al2o3_max_pct=settings.al2o3_max_pct,
                target_slag_mgo_min_pct=settings.mgo_min_pct,
                target_slag_mgo_al2o3_ratio_min=settings.mgo_al2o3_min,
                max_burden_qty_mt=max_burden_mt,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                hot_metal_target_mt=float(production_mt),
                charge_mass_mt=float(settings.charge_mass_mt),
                _explain=False,
            )
            row: dict[str, Any] = {
                "scenario_id": scenario_id,
                "is_base": is_base,
                "production_mt": float(production_mt),
                "feasible": result is not None,
                "unit_cost_rs_per_thm": np.nan,
                "total_cost_rs": np.nan,
                "ore_cost_total_rs": np.nan,
                "ore_cost_per_thm_rs": np.nan,
                "flux_cost_total_rs": np.nan,
                "flux_cost_per_thm_rs": np.nan,
                "total_ore_qty_mt": np.nan,
                "total_burden_qty_mt": np.nan,
                "max_burden_qty_mt": max_burden_mt,
                "charging_utilization_pct": np.nan,
                "slag_mt": np.nan,
                "slag_rate_kg_per_thm": np.nan,
                "slag_basicity": np.nan,
                "slag_t_basicity": np.nan,
                "slag_al2o3_pct": np.nan,
                "slag_mgo_pct": np.nan,
                "slag_mgo_al2o3_ratio": np.nan,
                "blend": "",
                "solve_error": " | ".join(map(str, solve_errors[:2])),
            }
            if result is not None:
                flux_cost_per_thm = float(
                    result.diagnostics.get("flux_cost_per_thm_rs", 0.0) or 0.0
                )
                total_cost = float(result.ore_cost_total_rs) + (
                    flux_cost_per_thm * float(production_mt)
                )
                total_burden_mt = float(
                    result.diagnostics.get(
                        "total_burden_qty_mt",
                        result.total_qty_mt
                        + sum(
                            float(value)
                            for value in (
                                result.diagnostics.get("lp_flux_quantities_mt", {})
                                or {}
                            ).values()
                        ),
                    )
                )
                row.update(
                    {
                        "unit_cost_rs_per_thm": total_cost / float(production_mt),
                        "total_cost_rs": total_cost,
                        "ore_cost_total_rs": float(result.ore_cost_total_rs),
                        "ore_cost_per_thm_rs": (
                            float(result.ore_cost_total_rs) / float(production_mt)
                        ),
                        "flux_cost_total_rs": (
                            flux_cost_per_thm * float(production_mt)
                        ),
                        "flux_cost_per_thm_rs": flux_cost_per_thm,
                        "total_ore_qty_mt": float(result.total_qty_mt),
                        "total_burden_qty_mt": total_burden_mt,
                        "charging_utilization_pct": (
                            total_burden_mt / max_burden_mt * 100.0
                            if max_burden_mt and max_burden_mt > 0.0
                            else 0.0
                        ),
                        "slag_mt": float(result.slag_mt),
                        "slag_rate_kg_per_thm": float(result.slag_rate_kg_per_thm),
                        "slag_basicity": float(result.slag_basicity),
                        "slag_t_basicity": float(result.slag_t_basicity),
                        "slag_al2o3_pct": float(result.slag_al2o3_pct),
                        "slag_mgo_pct": float(result.slag_mgo_pct),
                        "slag_mgo_al2o3_ratio": float(
                            result.slag_mgo_al2o3_ratio
                        ),
                        "blend": " | ".join(
                            f"{ore.display_name}: {result.shares_pct.get(ore.ore_id, 0.0):.2f}%"
                            for ore in scenario_ores
                            if result.shares_pct.get(ore.ore_id, 0.0) > 0.005
                        ),
                        "solve_error": "",
                    }
                )
                for ore in scenario_ores:
                    row[f"share_pct__{ore.ore_id}"] = float(
                        result.shares_pct.get(ore.ore_id, 0.0)
                    )
                    row[f"quantity_mt__{ore.ore_id}"] = float(
                        result.quantities_mt.get(ore.ore_id, 0.0)
                    )
                for flux_id, quantity in (
                    result.diagnostics.get("lp_flux_quantities_mt", {}) or {}
                ).items():
                    row[f"flux_mt__{flux_id}"] = float(quantity)
            scenario_curve.append(row)
            all_rows.append(row)
            completed += 1
            # import time; time.sleep(0.5)  # Simulate a small delay for demonstration purposes
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "solving",
                        "completed": completed,
                        "total": total_solves,
                        "scenario": scenario_id,
                        "scenario_count": scenario_count,
                        "production_mt": float(production_mt),
                        "target_index": target_index,
                        "target_count": len(targets),
                        "feasible": result is not None,
                    }
                )

        scenario_frame = pd.DataFrame(scenario_curve)
        knee = identify_knee(
            scenario_frame,
            tolerance_rs_per_thm=settings.near_minimum_tolerance_rs_per_thm,
        )
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "is_base": is_base,
                **scenario_metrics,
                **knee,
                "feasible_points": int(scenario_frame["feasible"].sum()),
            }
        )

    frontier = pd.DataFrame(all_rows)
    scenarios = pd.DataFrame(scenario_rows)
    varied = frontier.loc[~frontier["is_base"]].copy()
    if varied.empty:
        varied = frontier.copy()
    aggregate = varied.groupby("production_mt", as_index=False).agg(
        median_unit_cost_rs_per_thm=("unit_cost_rs_per_thm", "median"),
        p10_unit_cost_rs_per_thm=(
            "unit_cost_rs_per_thm",
            lambda series: series.quantile(0.10),
        ),
        p90_unit_cost_rs_per_thm=(
            "unit_cost_rs_per_thm",
            lambda series: series.quantile(0.90),
        ),
        feasible_scenarios=("feasible", "sum"),
        scenario_count=("feasible", "count"),
    )
    aggregate["feasibility_pct"] = (
        aggregate["feasible_scenarios"] / aggregate["scenario_count"] * 100.0
    )
    drivers = _spearman_driver_table(scenarios)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "complete",
                "completed": total_solves,
                "total": total_solves,
                "scenario_count": scenario_count,
            }
        )
    return {
        "frontier": frontier,
        "scenarios": scenarios,
        "aggregate": aggregate,
        "drivers": drivers,
        "settings": settings,
        "material_names": {ore.ore_id: ore.display_name for ore in ores},
        "flux_names": {
            flux.flux_id: flux.display_name for flux in (flux_inputs or [])
        },
        "base_knees": (
            list(scenario_rows[0].get("knee_candidates", []))
            if scenario_rows
            else []
        ),
        "solve_count": total_solves,
    }


__all__ = [
    "MAX_TOTAL_SOLVES",
    "MAX_VARIATION_SCENARIOS",
    "FrontierSettings",
    "identify_knee",
    "identify_knee_candidates",
    "production_targets",
    "run_frontier_simulation",
    "validate_frontier_inputs",
]
