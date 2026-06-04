"""Differential-evolution runner for optimization-runtime objective functions.

This module wraps SciPy differential evolution with BMO-specific bookkeeping:
it tracks the best relaxed solution, the best feasible solution, and optional
baseline comparison metrics for the Streamlit diagnostics panel.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from domain.optimization_runtime.types import ObjectiveResult, OptimizationResult


class OptimizerRunner:
    """
    Run differential evolution while retaining feasible and relaxed best states.

    The runner keeps SciPy-specific execution details out of the BMO optimizer.
    It records the best feasible candidate separately from the best penalized
    candidate so diagnostics can explain whether DE improved on the LP seed.

    Args:
         - optimizer_cfg: dict[str, Any] | None - Runtime optimizer settings.

    Returns:
         - return OptimizerRunner - Runner configured for a single optimization mode.
    """

    def __init__(self, optimizer_cfg: dict[str, Any] | None = None) -> None:
        """
        Store optimizer configuration for later DE runs.

        The configuration is intentionally kept as plain data because Streamlit
        pages and tests pass dictionaries loaded from YAML. Individual options
        are validated at the point where SciPy is called.

        Args:
             - optimizer_cfg: dict[str, Any] | None - Runtime optimizer settings.

        Returns:
             - return None - Initializes the runner configuration.
        """

        self.optimizer_cfg = optimizer_cfg or {}

    def run_differential_evolution(
        self,
        *,
        bounds: list[tuple[float, float]],
        objective_fn: Callable[[np.ndarray], ObjectiveResult],
        baseline_solution: dict[str, Any] | None = None,
        progress_callback: (
            Callable[[int, float, float | None, int, float], bool] | None
        ) = None,
    ) -> OptimizationResult:
        """
        Execute SciPy differential evolution and summarize the best solution.

        The method adapts a rich objective callback into SciPy's scalar API,
        tracks baseline comparison data, and returns diagnostics such as
        iterations and function evaluations for the UI.

        Args:
             - bounds: list[tuple[float, float]] - Per-variable lower and upper bounds.
             - objective_fn: Callable[[np.ndarray], ObjectiveResult] - Objective callback.
             - baseline_solution: dict[str, Any] | None - Optional LP baseline seed.

        Returns:
             - return OptimizationResult - Best solution, comparison metrics, and diagnostics.
        """

        best_relaxed: dict[str, Any] | None = (
            dict(baseline_solution) if baseline_solution else None
        )
        best_feasible: dict[str, Any] | None = (
            dict(baseline_solution)
            if baseline_solution and bool(baseline_solution.get("feasible", False))
            else None
        )
        nfev_state = {"count": 0}
        iteration_state = {"i": 0}
        t0 = time.monotonic()

        def wrapped(x: np.ndarray) -> float:
            """
            Adapt the rich objective result into SciPy's scalar objective API.

            SciPy only consumes a float, but BMO needs feasibility flags,
            components, violations, and diagnostics. This wrapper stores those
            details while returning the penalized scalar objective.

            Args:
                 - x: np.ndarray - Candidate vector proposed by differential evolution.

            Returns:
                 - return float - Scalar objective value including any penalties.
            """

            nonlocal best_feasible, best_relaxed
            nfev_state["count"] += 1
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

            if best_relaxed is None or record["objective"] < float(
                best_relaxed["objective"]
            ):
                best_relaxed = record
            if result.feasible:
                if best_feasible is None or record["objective"] < float(
                    best_feasible["objective"]
                ):
                    best_feasible = record
            return float(result.objective_value)

        def _scipy_callback(xk: np.ndarray, convergence: float = 0.0) -> bool:
            """
            Forward DE generation events to the optional progress callback.

            Args:
                 - xk: np.ndarray - Best candidate vector at this generation.
                 - convergence: float - SciPy convergence score in [0, 1].

            Returns:
                 - return bool - True to stop DE early, False to continue.
            """

            iteration_state["i"] += 1
            best_obj = (
                float(best_relaxed["objective"])
                if best_relaxed and "objective" in best_relaxed
                else float("inf")
            )
            best_feas = (
                float(best_feasible["objective"])
                if best_feasible and "objective" in best_feasible
                else None
            )
            elapsed_s = time.monotonic() - t0
            if progress_callback is None:
                return False
            try:
                return bool(
                    progress_callback(
                        iteration_state["i"],
                        best_obj,
                        best_feas,
                        nfev_state["count"],
                        elapsed_s,
                    )
                )
            except Exception:
                return False

        de_result = differential_evolution(
            func=wrapped,
            bounds=bounds,
            strategy=str(self.optimizer_cfg.get("strategy", "best1bin")),
            maxiter=int(self.optimizer_cfg.get("maxiter", 4)),
            popsize=int(self.optimizer_cfg.get("popsize", 4)),
            tol=float(self.optimizer_cfg.get("tol", 0.03)),
            polish=bool(self.optimizer_cfg.get("polish", False)),
            seed=int(self.optimizer_cfg.get("seed", 42)),
            workers=1,
            callback=_scipy_callback,
        )

        best_solution = (
            best_feasible
            or best_relaxed
            or {
                "x": [],
                "objective": float("inf"),
                "feasible": False,
                "components": {},
                "violations": ["No valid objective evaluation."],
                "diagnostics": {},
            }
        )

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
                "nfev": int(getattr(de_result, "nfev", nfev_state["count"])),
                "nit": int(getattr(de_result, "nit", iteration_state["i"])),
                "elapsed_s": float(time.monotonic() - t0),
            },
            "best_feasible_found": best_feasible is not None,
        }
        return OptimizationResult(
            best_solution=best_solution,
            baseline_solution=baseline_solution,
            compare_metrics=compare_metrics,
            diagnostics=diagnostics,
        )
