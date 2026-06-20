"""Gaussian-process style model wrappers for BMO fuel-cost inference.

The BMO runtime expects a model artifact with a ``predict(X)`` method.  The
classes below keep that contract while adding a ``predict(..., return_std=True)``
and ``predict_with_uncertainty`` path for confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_approximation import Nystroem, RBFSampler


def _as_array(X: Any) -> np.ndarray:
    """Convert supported tabular inputs to a finite float array."""

    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _feature_names(X: Any) -> list[str] | None:
    if isinstance(X, pd.DataFrame):
        return [str(c) for c in X.columns]
    return None


class RandomFeatureBayesianGp(BaseEstimator, RegressorMixin):
    """Scalable Gaussian-process approximation using random/kernel features.

    A Bayesian linear model over random Fourier or Nystroem features is the
    standard finite-dimensional approximation to a GP.  It is fast enough for
    the BMO DE optimiser and exposes predictive standard deviation.
    """

    def __init__(
        self,
        *,
        feature_map: str = "rbf_sampler",
        n_components: int = 512,
        gamma: float = 0.05,
        kernel: str = "rbf",
        ridge_alpha: float = 1.0,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        fit_intercept: bool = True,
        max_iter: int = 150,
        random_state: int = 42,
        uncertainty_scale: float = 1.0,
        residual_floor: float = 0.0,
    ) -> None:
        self.feature_map = feature_map
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.kernel = kernel
        self.ridge_alpha = float(ridge_alpha)
        self.alpha_1 = float(alpha_1)
        self.alpha_2 = float(alpha_2)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.uncertainty_scale = float(uncertainty_scale)
        self.residual_floor = float(residual_floor)

    def _make_transformer(self) -> Any:
        if self.feature_map == "nystroem":
            return Nystroem(
                kernel=self.kernel,
                gamma=self.gamma,
                n_components=self.n_components,
                random_state=self.random_state,
            )
        return RBFSampler(
            gamma=self.gamma,
            n_components=self.n_components,
            random_state=self.random_state,
        )

    def fit(self, X: Any, y: Any) -> "RandomFeatureBayesianGp":
        X_arr = _as_array(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        self.feature_names_in_ = _feature_names(X) or [f"x{i}" for i in range(X_arr.shape[1])]
        self.n_features_in_ = int(X_arr.shape[1])
        self.transformer_ = self._make_transformer()
        Z = np.asarray(self.transformer_.fit_transform(X_arr), dtype=float)
        if self.fit_intercept:
            Z_aug = np.column_stack([np.ones(Z.shape[0]), Z])
        else:
            Z_aug = Z

        reg = float(max(self.ridge_alpha, 1e-9)) * np.eye(Z_aug.shape[1])
        if self.fit_intercept and reg.shape[0] > 0:
            reg[0, 0] = 0.0
        A = Z_aug.T @ Z_aug + reg
        b = Z_aug.T @ y_arr
        try:
            self.coef_ = np.linalg.solve(A, b)
            self.posterior_precision_inv_ = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self.posterior_precision_inv_ = np.linalg.pinv(A)
            self.coef_ = self.posterior_precision_inv_ @ b

        fit_pred = Z_aug @ self.coef_
        resid = y_arr - fit_pred
        dof = max(1, Z_aug.shape[0] - min(Z_aug.shape[1], Z_aug.shape[0] - 1))
        self.residual_variance_ = float(np.sum(resid * resid) / dof)
        self.training_residual_std_ = float(np.sqrt(max(self.residual_variance_, 1e-9)))
        return self

    def _augment(self, Z: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            return np.column_stack([np.ones(Z.shape[0]), Z])
        return Z

    def _predict_raw(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        X_arr = _as_array(X)
        Z = np.asarray(self.transformer_.transform(X_arr), dtype=float)
        Z_aug = self._augment(Z)
        mean_arr = np.asarray(Z_aug @ self.coef_, dtype=float).reshape(-1)
        leverage = np.einsum("ij,jk,ik->i", Z_aug, self.posterior_precision_inv_, Z_aug, optimize=True)
        leverage = np.maximum(leverage, 0.0)
        std_arr = np.sqrt(max(float(self.residual_variance_), 1e-9) * (1.0 + leverage))
        floor = max(float(getattr(self, "residual_floor", 0.0)), 0.0)
        scale = max(float(getattr(self, "uncertainty_scale", 1.0)), 0.0)
        if floor > 0:
            std_arr = np.sqrt(np.square(std_arr) + floor * floor)
        std_arr = std_arr * scale
        return mean_arr, std_arr

    def predict(self, X: Any, return_std: bool = False) -> Any:
        mean_arr, std_arr = self._predict_raw(X)
        if return_std:
            return mean_arr, std_arr
        return mean_arr

    def predict_with_uncertainty(self, X: Any) -> dict[str, np.ndarray]:
        mean_arr, std_arr = self._predict_raw(X)
        return {
            "mean": mean_arr,
            "std": std_arr,
            "lower_95": mean_arr - 1.96 * std_arr,
            "upper_95": mean_arr + 1.96 * std_arr,
        }


@dataclass
class BmoGpResidualCostModel:
    """XGBoost mean model with a calibrated GP residual/uncertainty layer.

    The residual GP can correct the base prediction when validation data shows a
    gain.  Its correction is shrinkable through ``residual_weight``; setting the
    weight to 0 keeps the original mean prediction while still exposing GP-based
    uncertainty.  This protects deployed accuracy while adding confidence bands.
    """

    mean_model: Any
    residual_gp: Any | None = None
    residual_weight: float = 0.0
    uncertainty_scale: float = 1.0
    residual_floor: float = 0.0
    model_name: str = "xgb_mean_plus_gp_residual"
    feature_names_in_: list[str] = field(default_factory=list)
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def _predict_raw(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        base = np.asarray(self.mean_model.predict(X), dtype=float).reshape(-1)
        if self.residual_gp is None:
            correction = np.zeros_like(base)
            gp_std = np.zeros_like(base)
        else:
            try:
                correction, gp_std = self.residual_gp.predict(X, return_std=True)
            except TypeError:
                correction = self.residual_gp.predict(X)
                gp_std = np.full_like(base, float(self.residual_floor), dtype=float)
            correction = np.asarray(correction, dtype=float).reshape(-1)
            gp_std = np.asarray(gp_std, dtype=float).reshape(-1)
        mean = base + float(self.residual_weight) * correction
        floor = max(float(self.residual_floor), 0.0)
        scale = max(float(self.uncertainty_scale), 0.0)
        std = np.sqrt(np.square(gp_std) + floor * floor) * scale
        return mean, std

    def predict(self, X: Any, return_std: bool = False) -> Any:
        mean, std = self._predict_raw(X)
        if return_std:
            return mean, std
        return mean

    def predict_with_uncertainty(self, X: Any) -> dict[str, np.ndarray]:
        mean, std = self._predict_raw(X)
        return {
            "mean": mean,
            "std": std,
            "lower_95": mean - 1.96 * std,
            "upper_95": mean + 1.96 * std,
        }
