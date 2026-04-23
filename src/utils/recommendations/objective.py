from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from domain.optimization_runtime import ObjectiveResult


class RecommendationObjectiveEvaluator:
    """Objective evaluator for recommendation DE optimisation."""

    def __init__(
        self,
        *,
        base_row: np.ndarray,
        base_scaled: np.ndarray,
        free_idx: list[int],
        control_idx: list[int],
        scaled_prev_params: np.ndarray,
        offsets: np.ndarray,
        scales: np.ndarray,
        feature_idx: np.ndarray,
        model: Any,
        lambda_reg: float,
        maxmin: float,
    ) -> None:
        self.base_row = base_row
        self.base_scaled = base_scaled
        self.free_idx = free_idx
        self.control_idx = control_idx
        self.scaled_prev_params = scaled_prev_params
        self.offsets = offsets
        self.scales = scales
        self.feature_idx = feature_idx
        self.model = model
        self.lambda_reg = float(lambda_reg)
        self.maxmin = float(maxmin)

    def evaluate(self, trial_controls: np.ndarray) -> ObjectiveResult:
        x_raw = self.base_row.copy()
        x_raw[self.free_idx] = trial_controls
        x_scaled = self.base_scaled.copy()

        raw_vals = x_raw[self.free_idx]
        idx = self.feature_idx[self.free_idx]
        scaled_vals = (raw_vals - self.offsets[idx]) / self.scales[idx]
        x_scaled[self.free_idx] = scaled_vals

        y_pred_scaled = float(self.model.predict(x_scaled.reshape(1, -1))[0]) * self.maxmin
        penalty = float(
            np.sum((x_scaled[self.control_idx] - self.scaled_prev_params) ** 2)
        )
        objective_value = float(y_pred_scaled + self.lambda_reg * penalty)

        return ObjectiveResult(
            objective_value=objective_value,
            components={
                "predicted_target_term": y_pred_scaled,
                "regularization_term": self.lambda_reg * penalty,
                "regularization_raw": penalty,
            },
            feasible=True,
            violations=[],
            diagnostics={},
        )
