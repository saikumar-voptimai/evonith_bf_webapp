from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from domain.optimization_runtime.types import (
    ObjectiveResult,
    OptimizationResult,
)


class OptimizerRunner:
    """Shared DE runner that keeps best feasible and best relaxed solutions."""

    def __init__(self, optimizer_cfg: dict[str, Any] | None = None) -> None:
        self.optimizer_cfg = optimizer_cfg or {}

    def run_differential_evolution(
        self,
        *,
        bounds: list[tuple[float, float]],
        objective_fn: Callable[[np.ndarray], ObjectiveResult],
        baseline_solution: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        best_feasible: dict[str, Any] | None = None
        best_relaxed: dict[str, Any] | None = None

        def wrapped(x: np.ndarray) -> float:
            nonlocal best_feasible, best_relaxed
            x_arr = np.asarray(x, dtype=float)
            result = objective_fn(x_arr)
            record = {
                "x": x_arr.tolist(),
                "objective": float(result.objective_value),
                "feasible": bool(result.feasible),
                "components": dict(result.components),
                "violations": list(result.violations),
                "diagnostics": dict(result.diagnostics),
            }

            if best_relaxed is None or record["objective"] < float(best_relaxed["objective"]):
                best_relaxed = record
            if result.feasible:
                if best_feasible is None or record["objective"] < float(best_feasible["objective"]):
                    best_feasible = record
            return float(result.objective_value)

        de_result = differential_evolution(
            func=wrapped,
            bounds=bounds,
            strategy=str(self.optimizer_cfg.get("strategy", "best1bin")),
            maxiter=int(self.optimizer_cfg.get("maxiter", 40)),
            popsize=int(self.optimizer_cfg.get("popsize", 12)),
            tol=float(self.optimizer_cfg.get("tol", 0.01)),
            polish=bool(self.optimizer_cfg.get("polish", True)),
            seed=int(self.optimizer_cfg.get("seed", 42)),
            workers=1,
        )

        best_solution = best_feasible or best_relaxed or {
            "x": [],
            "objective": float("inf"),
            "feasible": False,
            "components": {},
            "violations": ["No valid objective evaluation."],
            "diagnostics": {},
        }

        compare_metrics: dict[str, float] = {}
        if baseline_solution and "objective" in baseline_solution:
            try:
                baseline_objective = float(baseline_solution["objective"])
                compare_metrics["delta_objective_vs_baseline"] = (
                    float(best_solution["objective"]) - baseline_objective
                )
            except (TypeError, ValueError):
                pass

        diagnostics = {
            "de_result": {
                "success": bool(getattr(de_result, "success", False)),
                "message": str(getattr(de_result, "message", "")),
                "nfev": int(getattr(de_result, "nfev", 0)),
                "nit": int(getattr(de_result, "nit", 0)),
            },
            "best_feasible_found": best_feasible is not None,
        }
        return OptimizationResult(
            best_solution=best_solution,
            baseline_solution=baseline_solution,
            compare_metrics=compare_metrics,
            diagnostics=diagnostics,
        )
