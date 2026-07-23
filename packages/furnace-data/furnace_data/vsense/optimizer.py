"""Deterministic legacy_v1 V-Sense optimizer."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from furnace_data.optimization_runtime.types import ObjectiveResult
from furnace_data.optimization_runtime.optimizer_runner import OptimizerRunner
from furnace_data.vsense.bounds import VSenseValidationError, validate_control_profile
from furnace_data.vsense.catalog import (
    ALGORITHM_VERSION,
    CATALOG_VERSION,
    feature_for_parameter_id,
    load_vsense_catalog,
    optimization_by_id,
    parameter_by_id,
    target_for_optimization,
)
from furnace_data.vsense.dependencies import dependent_parameters
from furnace_data.vsense.result_builder import json_safe_result


ProgressCallback = Callable[[int, float, float | None, int, float], bool]


def run_legacy_optimization(
    *,
    context: dict[str, Any],
    control_plan: list[dict[str, Any]],
    input_overrides: list[dict[str, Any]] | None = None,
    lambda_reg: float = 0.05,
    iteration_budget: dict[str, Any] | None = None,
    seed: int = 42,
    require_approved_bounds: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the deterministic, legacy-compatible advisory optimizer."""

    catalog = load_vsense_catalog()
    optimization_type_id = str(context["optimization_type_id"])
    opt = optimization_by_id(catalog).get(optimization_type_id)
    if opt is None:
        raise VSenseValidationError(
            "VSENSE_INVALID_OPTIMIZATION_TYPE",
            "Unknown V-Sense optimization type.",
        )
    lambda_value = _validate_lambda(lambda_reg, catalog)
    controls = validate_control_profile(
        optimization_type_id,
        control_plan,
        require_approved_bounds=require_approved_bounds,
        catalog=catalog,
    )
    overrides = _validate_input_overrides(input_overrides or [], catalog)
    target = target_for_optimization(optimization_type_id)
    direction = str(target["direction"])
    baseline_target = _number(context.get("target", {}).get("value"), 0.0)
    baseline_controls = {
        item["parameter_id"]: _number(item.get("value"), _midpoint_for(item["parameter_id"], catalog))
        for item in context.get("controls", [])
    }
    params = parameter_by_id(catalog)
    free_items = [item for item in controls if item["mode"] == "optimize"]
    bounds = [(item["lower_bound"], item["upper_bound"]) for item in free_items]
    baseline_solution = _candidate_record(
        controls,
        free_items,
        np.array([baseline_controls[item["parameter_id"]] for item in free_items], dtype=float),
        baseline_controls,
        baseline_target,
        direction,
        lambda_value,
    )

    if not free_items:
        best = _candidate_record(
            controls,
            free_items,
            np.array([], dtype=float),
            baseline_controls,
            baseline_target,
            direction,
            lambda_value,
        )
        optimizer_diagnostics = {
            "de_result": {
                "success": True,
                "message": "All controls fixed; optimizer skipped.",
                "nfev": 0,
                "nit": 0,
                "elapsed_s": 0.0,
            },
            "best_feasible_found": True,
            "all_controls_fixed": True,
        }
    else:
        max_iterations = int((iteration_budget or {}).get("max_iterations") or 20)
        runner = OptimizerRunner(
            {
                "strategy": str((iteration_budget or {}).get("strategy") or "best1bin"),
                "maxiter": max_iterations,
                "popsize": int((iteration_budget or {}).get("population") or 6),
                "tol": float((iteration_budget or {}).get("tolerance") or 0.01),
                "polish": bool((iteration_budget or {}).get("polish", False)),
                "seed": int(seed),
            }
        )

        def objective(x: np.ndarray) -> ObjectiveResult:
            record = _candidate_record(
                controls,
                free_items,
                x,
                baseline_controls,
                baseline_target,
                direction,
                lambda_value,
            )
            return ObjectiveResult(
                objective_value=float(record["objective"]),
                components=dict(record["components"]),
                feasible=bool(record["feasible"]),
                violations=list(record["violations"]),
                diagnostics={"candidate_controls": record["controls"]},
            )

        result = runner.run_differential_evolution(
            bounds=bounds,
            objective_fn=objective,
            baseline_solution=baseline_solution,
            progress_callback=progress_callback,
        )
        best = dict(result.best_solution)
        optimizer_diagnostics = dict(result.diagnostics)
        optimizer_diagnostics["all_controls_fixed"] = False

    recommended_controls = dict(best.get("controls") or {})
    target_result = _target_result(
        target,
        baseline_target,
        float(best["predicted_target"]),
    )
    impact_results = _impact_results(
        catalog,
        optimization_type_id,
        baseline_target,
        recommended_controls,
        baseline_controls,
    )
    dependent_results = dependent_parameters(
        baseline_controls=baseline_controls,
        recommended_controls=recommended_controls,
    )
    control_results = _control_results(
        controls,
        baseline_controls,
        recommended_controls,
        params,
    )
    feasibility = _feasibility(controls, recommended_controls, params)
    warnings = [
        *[str(item) for item in context.get("warnings") or []],
        *[f"Input override applied: {item['parameter_id']}" for item in overrides],
    ]
    result = {
        "advisory_only": True,
        "requires_operator_review": True,
        "status": "completed",
        "target": target_result,
        "controls": control_results,
        "impacts": impact_results,
        "dependent_parameters": dependent_results,
        "feasibility": feasibility,
        "diagnostics": {
            "algorithm_version": ALGORITHM_VERSION,
            "seed": int(seed),
            "lambda_reg": lambda_value,
            "optimizer": optimizer_diagnostics,
            "missing_feature_policy": "context_snapshot_defaults",
            "input_override_parameter_ids": [item["parameter_id"] for item in overrides],
        },
        "versions": {
            "catalog_version": context.get("catalog_version") or CATALOG_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "dataset_version": context.get("dataset", {}).get("version"),
            "control_profile_version": context.get("control_profile", {}).get("version"),
            "model_versions": {
                row["optimization_type_id"]: row["bundle_version"]
                for row in context.get("models") or []
            },
        },
        "warnings": list(dict.fromkeys(warnings)),
        "review": None,
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return json_safe_result(result)


def _candidate_record(
    controls: list[dict[str, Any]],
    free_items: list[dict[str, Any]],
    x: np.ndarray,
    baseline_controls: dict[str, float],
    baseline_target: float,
    direction: str,
    lambda_value: float,
) -> dict[str, Any]:
    recommended = dict(baseline_controls)
    free_values = [float(item) for item in np.asarray(x, dtype=float).tolist()]
    for item in controls:
        if item["mode"] == "fixed":
            recommended[item["parameter_id"]] = float(item["fixed_value"])
    for item, value in zip(free_items, free_values):
        recommended[item["parameter_id"]] = min(
            float(item["upper_bound"]),
            max(float(item["lower_bound"]), float(value)),
        )
    control_score = _control_score(controls, recommended, baseline_controls)
    direction_sign = 1.0 if direction == "maximize" else -1.0
    predicted_target = baseline_target + direction_sign * control_score * _target_scale(baseline_target)
    penalty = _penalty(controls, recommended, baseline_controls)
    objective = -direction_sign * predicted_target + lambda_value * penalty
    violations = _candidate_violations(controls, recommended)
    return {
        "x": free_values,
        "objective": float(objective),
        "predicted_target": float(predicted_target),
        "feasible": not violations,
        "violations": violations,
        "components": {
            "target_component": float(-direction_sign * predicted_target),
            "regularization_penalty": float(lambda_value * penalty),
            "control_score": float(control_score),
        },
        "controls": recommended,
    }


def _control_score(
    controls: list[dict[str, Any]],
    recommended: dict[str, float],
    baseline: dict[str, float],
) -> float:
    if not controls:
        return 0.0
    total = 0.0
    for item in controls:
        lower = float(item["lower_bound"])
        upper = float(item["upper_bound"])
        span = max(upper - lower, 1e-9)
        baseline_norm = (float(baseline.get(item["parameter_id"], lower)) - lower) / span
        recommended_norm = (float(recommended[item["parameter_id"]]) - lower) / span
        total += recommended_norm - baseline_norm
    return total / len(controls)


def _penalty(
    controls: list[dict[str, Any]],
    recommended: dict[str, float],
    baseline: dict[str, float],
) -> float:
    total = 0.0
    for item in controls:
        lower = float(item["lower_bound"])
        upper = float(item["upper_bound"])
        span = max(upper - lower, 1e-9)
        delta = (float(recommended[item["parameter_id"]]) - float(baseline.get(item["parameter_id"], lower))) / span
        total += delta * delta
    return total


def _candidate_violations(
    controls: list[dict[str, Any]],
    recommended: dict[str, float],
) -> list[str]:
    violations: list[str] = []
    for item in controls:
        value = float(recommended[item["parameter_id"]])
        if value < float(item["lower_bound"]) or value > float(item["upper_bound"]):
            violations.append(f"{item['parameter_id']} outside submitted bounds")
    return violations


def _target_result(
    target: dict[str, Any],
    baseline: float,
    recommended: float,
) -> dict[str, Any]:
    delta = recommended - baseline
    return {
        "parameter_id": target["id"],
        "label": target["label"],
        "unit": target["unit"],
        "direction": target["direction"],
        "baseline": float(baseline),
        "recommended": float(recommended),
        "delta": float(delta),
        "delta_pct": _percent(delta, baseline),
    }


def _control_results(
    controls: list[dict[str, Any]],
    baseline: dict[str, float],
    recommended: dict[str, float],
    params: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in controls:
        parameter_id = item["parameter_id"]
        definition = params[parameter_id]
        old = float(baseline.get(parameter_id, _midpoint_for(parameter_id, {"parameters": list(params.values())})))
        new = float(recommended[parameter_id])
        rows.append(
            {
                "parameter_id": parameter_id,
                "label": definition["label"],
                "unit": definition.get("unit"),
                "mode": item["mode"],
                "baseline": old,
                "recommended": new,
                "delta": new - old,
                "delta_pct": _percent(new - old, old),
                "lower_bound": item["lower_bound"],
                "upper_bound": item["upper_bound"],
                "at_bound": bool(
                    math.isclose(new, float(item["lower_bound"]), rel_tol=0.0, abs_tol=1e-9)
                    or math.isclose(new, float(item["upper_bound"]), rel_tol=0.0, abs_tol=1e-9)
                ),
                "approved_min": definition.get("approved_min"),
                "approved_max": definition.get("approved_max"),
            }
        )
    return rows


def _impact_results(
    catalog: dict[str, Any],
    optimization_type_id: str,
    baseline_target: float,
    recommended_controls: dict[str, float],
    baseline_controls: dict[str, float],
) -> list[dict[str, Any]]:
    opt = optimization_by_id(catalog)[optimization_type_id]
    targets = {item["id"]: item for item in (catalog.get("parameters") or []) if item.get("role") == "target"}
    score = sum(
        recommended_controls.get(key, 0.0) - baseline_controls.get(key, 0.0)
        for key in recommended_controls
    )
    impacts: list[dict[str, Any]] = []
    for target_id in opt["impact_target_ids"]:
        target = targets[target_id]
        baseline = baseline_target * (0.92 if target_id == "unit_cost" else 1.08)
        recommended = baseline + (0.001 * score)
        impacts.append(
            {
                "parameter_id": target_id,
                "label": target["label"],
                "unit": target.get("unit"),
                "baseline": float(baseline),
                "recommended": float(recommended),
                "delta": float(recommended - baseline),
                "delta_pct": _percent(recommended - baseline, baseline),
                "bundle_version": "derived-impact-legacy-v1",
            }
        )
    return impacts


def _feasibility(
    controls: list[dict[str, Any]],
    recommended: dict[str, float],
    params: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for item in controls:
        definition = params[item["parameter_id"]]
        value = float(recommended[item["parameter_id"]])
        approved_min = definition.get("approved_min")
        approved_max = definition.get("approved_max")
        if approved_min is not None and value < float(approved_min):
            violations.append({"parameter_id": item["parameter_id"], "code": "below_approved_min"})
        if approved_max is not None and value > float(approved_max):
            violations.append({"parameter_id": item["parameter_id"], "code": "above_approved_max"})
    return {
        "feasible": not violations,
        "violations": violations,
        "operator_review_required": True,
    }


def _validate_input_overrides(
    overrides: list[dict[str, Any]],
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    params = parameter_by_id(catalog)
    normalized: list[dict[str, Any]] = []
    for item in overrides:
        parameter_id = str(item.get("parameter_id") or "")
        definition = params.get(parameter_id)
        if definition is None or not definition.get("override_allowed"):
            raise VSenseValidationError(
                "VSENSE_INVALID_INPUT_OVERRIDE",
                "Unknown or non-overridable input parameter.",
            )
        value = _number(item.get("value"), None)
        if value is None:
            raise VSenseValidationError(
                "VSENSE_INVALID_INPUT_OVERRIDE",
                "Input override values must be finite numbers.",
            )
        normalized.append({"parameter_id": parameter_id, "value": value})
    return normalized


def _validate_lambda(lambda_reg: float, catalog: dict[str, Any]) -> float:
    value = _number(lambda_reg, None)
    limits = catalog["limits"]
    if value is None or value < float(limits["lambda_min"]) or value > float(limits["lambda_max"]):
        raise VSenseValidationError(
            "VSENSE_INVALID_LAMBDA",
            "lambda_reg is outside the allowed V-Sense range.",
        )
    return value


def _target_scale(baseline_target: float) -> float:
    return max(abs(float(baseline_target)), 1.0) * 0.05


def _midpoint_for(parameter_id: str, catalog: dict[str, Any]) -> float:
    params = parameter_by_id(catalog)
    definition = params.get(parameter_id)
    if not definition:
        return 0.0
    lower = definition.get("approved_min")
    upper = definition.get("approved_max")
    if lower is None or upper is None:
        return 0.0
    return (float(lower) + float(upper)) / 2.0


def _number(value: Any, default: float | None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _percent(delta: float, baseline: float) -> float | None:
    if not math.isfinite(float(delta)) or not math.isfinite(float(baseline)):
        return None
    if abs(float(baseline)) < 1e-12:
        return None
    return float(delta) / abs(float(baseline)) * 100.0
