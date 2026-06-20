"""BMO fuel-cost model service.

This module loads the configured BMO model bundle, builds model-ready feature
frames, executes scaler/model inference, and falls back to a deterministic
fuel-rate formula when artifacts or predictions are unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from domain.optimization_runtime import (
    FeatureVectorBuilder,
    ModelBundleService,
    build_runtime_config,
    normalize_feature_name,
)
from utils.bmo.feature_builder import (
    PreBuiltFeatureContext,
    build_bmo_v4_feature_frame,
    build_dynamic_payload,
    max_bmo_lag_steps,
)
from utils.bmo.types import ModelPrediction


def _resolve_repo_path(path_str: str) -> Path:
    """
    Resolve a model-service path against common repository locations.

    Model artifact paths are configured in YAML and may be absolute or
    repository-relative. This helper keeps artifact lookup stable across tests,
    Streamlit runs, and command-line validation scripts.

    Args:
         - path_str: str - Absolute or repository-relative path from config.

    Returns:
         - return Path - Existing resolved path when found, otherwise first candidate.
    """

    p = Path(str(path_str or "")).expanduser()
    if p.is_absolute():
        return p

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [repo_root / p, repo_root / "src" / p, Path.cwd() / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class FuelUnitCostModelService:
    """
    Predict BMO fuel unit cost from blend and process features.

    The service owns model-bundle loading, feature construction, selected-column
    inference, and deterministic fallback behavior. Callers only need to provide
    a candidate blend payload and optional historical context.

    Args:
         - bundle_cfg: dict[str, Any] - Model bundle configuration.
         - fallback_cfg: dict[str, Any] - Deterministic fallback fuel model settings.

    Returns:
         - return FuelUnitCostModelService - Cached service for fuel-cost inference.
    """

    def __init__(
        self, bundle_cfg: dict[str, Any], fallback_cfg: dict[str, Any]
    ) -> None:
        """
        Initialize model bundle access and fallback configuration.

        The expensive model/scaler load is deferred until prediction or status
        inspection. That keeps Streamlit page construction fast while preserving
        a single cached service for repeated candidate evaluations.

        Args:
             - bundle_cfg: dict[str, Any] - Model bundle configuration.
             - fallback_cfg: dict[str, Any] - Deterministic fallback fuel model settings.

        Returns:
             - return None - Initializes lazy loading state for inference.
        """

        self.bundle_cfg = bundle_cfg or {}
        self.fallback_cfg = fallback_cfg or {}

        runtime_cfg = build_runtime_config({"model_bundle": self.bundle_cfg})
        self._bundle_service = ModelBundleService(runtime_cfg["model_bundle"])
        self._bundle_info = None
        self._feature_builder: FeatureVectorBuilder | None = None
        self.missing_feature_policy = str(
            self.bundle_cfg.get("missing_feature_policy", "default_warn")
        )
        self._selected_features: list[str] | None = None
        self._selected_features_path = (
            self.bundle_cfg.get("selected_feature_columns_path")
            or self.bundle_cfg.get("feature_columns_path")
            or self.bundle_cfg.get("selected_features_path")
        )

    def _ensure_loaded(self) -> None:
        """
        Lazily load the model bundle and selected feature list.

        This method is the single gate before inference. It populates bundle
        metadata, the shared feature builder, and optional selected feature
        columns exactly once per cached service instance.

        Args:
             - None

        Returns:
             - return None - Populates cached model bundle and feature builder.
        """

        if self._bundle_info is not None:
            return
        self._bundle_info = self._bundle_service.get_bundle()
        self._feature_builder = FeatureVectorBuilder(
            self._bundle_info,
            missing_feature_policy=self.missing_feature_policy,
        )
        self._selected_features = self._load_selected_features()

    def _load_selected_features(self) -> list[str] | None:
        """
        Load optional post-scaler feature selection used by BMO V4 artifacts.

        The training notebook exports the final XGBoost columns separately from
        the scaler's raw feature list. Loading this file lets inference mirror
        the scaler-then-column-selection flow used during training.

        Args:
             - None

        Returns:
             - return list[str] | None - Selected feature names or None when absent.
        """

        if not self._selected_features_path:
            return None

        path = _resolve_repo_path(str(self._selected_features_path))
        if not path.exists() or path.is_dir():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict):
            for key in ("feature_names", "selected_features", "columns"):
                values = data.get(key)
                if isinstance(values, list):
                    return [str(item) for item in values]
        return None

    def get_bundle_status(self) -> dict[str, Any]:
        """
        Return model/scaler load status for the BMO page header and diagnostics.

        Streamlit uses this status to show whether predictions are coming from
        the ML bundle or deterministic fallback. The status also includes
        selected-feature metadata so model asset issues are easier to diagnose.

        Args:
             - None

        Returns:
             - return dict[str, Any] - Bundle status including selected feature metadata.
        """

        self._ensure_loaded()
        if self._bundle_info is None:
            return {
                "model_loaded": False,
                "scaler_loaded": False,
                "feature_count": 0,
                "bundle_error": "bundle_not_loaded",
            }
        status = dict(self._bundle_info.status)
        status["selected_feature_columns_path"] = (
            str(_resolve_repo_path(str(self._selected_features_path)))
            if self._selected_features_path
            else None
        )
        status["selected_feature_count"] = len(self._selected_features or [])
        status["selected_features_loaded"] = bool(self._selected_features)
        return status

    def get_inference_feature_names(self) -> list[str]:
        """Return the feature names that drive the current fuel model path."""

        self._ensure_loaded()
        if self._selected_features:
            return list(self._selected_features)
        if self._bundle_info is None:
            return []
        return list(self._bundle_info.expected_features)

    def get_max_lag_steps(self) -> int:
        """Return the largest lag step required by the loaded fuel model."""

        return max_bmo_lag_steps(self.get_inference_feature_names())

    def _pick_value(
        self, payload: dict[str, float], keys: list[str], default: float
    ) -> float:
        """
        Pick the first available payload value from raw or normalized key names.

        The fallback formula accepts both model-style feature names and simpler
        normalized aliases. This helper keeps that lookup behavior consistent
        for coke, nut coke, PCI, and sinter-share inputs.

        Args:
             - payload: dict[str, float] - Feature payload supplied to prediction.
             - keys: list[str] - Candidate key names to search.
             - default: float - Value to return when no key is found.

        Returns:
             - return float - Matched payload value or default.
        """

        norm_payload = {normalize_feature_name(k): float(v) for k, v in payload.items()}
        for key in keys:
            if key in payload:
                return float(payload[key])
            norm_key = normalize_feature_name(key)
            if norm_key in norm_payload:
                return float(norm_payload[norm_key])
        return float(default)

    def _fallback_predict(self, payload: dict[str, float]) -> float:
        """
        Estimate fuel cost with the configured deterministic fallback formula.

        The fallback keeps BMO usable when model artifacts are missing or a
        prediction is invalid. It combines coke, nut coke, and PCI cost with a
        simple sinter-share adjustment.

        Args:
             - payload: dict[str, float] - Blend and process features for inference.

        Returns:
             - return float - Estimated fuel unit cost in Rs/THM.
        """

        defaults_rates = self.fallback_cfg.get("default_rates_kg_per_thm", {})
        defaults_prices = self.fallback_cfg.get("default_prices_rs_per_kg", {})

        coke_rate = self._pick_value(
            payload,
            [
                "COKE RATE KG/THM",
                "coke_rate_kg_per_thm",
                "coke_rate_kg_thm",
                "coke_rate",
            ],
            float(defaults_rates.get("coke_rate_kg_per_thm", 340.0)),
        )
        nut_rate = self._pick_value(
            payload,
            [
                "NUT COKE RATE KG/THM",
                "nut_coke_rate_kg_per_thm",
                "nut_coke_rate_kg_thm",
                "nut_coke_rate",
            ],
            float(defaults_rates.get("nut_coke_rate_kg_per_thm", 20.0)),
        )
        pci_rate = self._pick_value(
            payload,
            ["ACTUALKG/THM.", "pci_rate_kg_per_thm", "pci_rate_kg_thm", "pci_rate"],
            float(defaults_rates.get("pci_rate_kg_per_thm", 170.0)),
        )

        coke_price = float(defaults_prices.get("coke_price_rs_per_kg", 34.0))
        nut_price = float(defaults_prices.get("nut_coke_price_rs_per_kg", 32.0))
        pci_price = float(defaults_prices.get("pci_price_rs_per_kg", 22.0))

        base_cost = (
            (coke_rate * coke_price) + (nut_rate * nut_price) + (pci_rate * pci_price)
        )

        sinter_share = self._pick_value(payload, ["sinter_share_pct"], 60.0)
        target_sinter = float(self.fallback_cfg.get("target_sinter_share_pct", 60.0))
        sensitivity = float(
            self.fallback_cfg.get("sinter_deviation_sensitivity_rs_per_thm", 1200.0)
        )
        adjustment = ((target_sinter - sinter_share) / 100.0) * sensitivity
        return float(base_cost + adjustment)

    def _prediction_issue(self, value: float) -> str | None:
        """
        Return a reason when an ML fuel-cost prediction is physically invalid.

        Model predictions are still validated before being trusted in the
        optimizer. Non-finite or non-positive values fall back to deterministic
        fuel cost so DE does not chase invalid objective values.

        Args:
             - value: float - Raw model prediction in Rs/THM.

        Returns:
             - return str | None - Validation issue text, or None when valid.
        """

        if not np.isfinite(value):
            return "Model prediction is not finite."
        min_value = float(self.fallback_cfg.get("min_fuel_cost_rs_per_thm", 0.0))
        if value <= min_value:
            return (
                f"Model prediction {value:.3f} is not above the minimum "
                f"fuel cost {min_value:.3f} Rs/THM."
            )
        return None

    def _model_predict_with_details(
        self, model: Any, model_input: Any
    ) -> tuple[float, dict[str, float | str]]:
        """
        Run model prediction and collect optional GP uncertainty details.

        Legacy XGBoost artifacts expose only ``predict(X)``. GP-enabled artifacts
        can additionally expose ``predict_with_uncertainty(X)`` or
        ``predict(X, return_std=True)``. This helper supports both without
        changing the BMO inference contract: callers still receive one fuel-cost
        value, while uncertainty bands are attached to ``ModelPrediction.details``.

        Args:
             - model: Any - Loaded model artifact (XGBoost wrapper or GP model).
             - model_input: Any - Scaled, feature-ordered model input.

        Returns:
             - return tuple[float, dict[str, float | str]] - Mean prediction and
               optional uncertainty detail fields.
        """

        details: dict[str, float | str] = {}

        if hasattr(model, "predict_with_uncertainty"):
            try:
                result = model.predict_with_uncertainty(model_input)
                if isinstance(result, dict):
                    mean = result.get(
                        "mean", result.get("prediction", result.get("value"))
                    )
                    value = float(np.asarray(mean).reshape(-1)[0])
                    std = result.get("std", result.get("sigma", None))
                    if std is not None:
                        std_value = float(np.asarray(std).reshape(-1)[0])
                        if np.isfinite(std_value) and std_value >= 0:
                            details["prediction_std_rs_per_thm"] = std_value
                            details["prediction_interval_95_half_width_rs_per_thm"] = (
                                1.96 * std_value
                            )
                    lower = result.get("lower_95", result.get("lower", None))
                    upper = result.get("upper_95", result.get("upper", None))
                    if lower is not None and upper is not None:
                        details["prediction_lower_95_rs_per_thm"] = float(
                            np.asarray(lower).reshape(-1)[0]
                        )
                        details["prediction_upper_95_rs_per_thm"] = float(
                            np.asarray(upper).reshape(-1)[0]
                        )
                    details["uncertainty_source"] = type(model).__name__
                    return value, details
            except Exception as exc:  # noqa: BLE001
                details["uncertainty_warning"] = (
                    f"predict_with_uncertainty failed: {exc}"
                )

        try:
            pred = model.predict(model_input, return_std=True)
            if isinstance(pred, tuple) and len(pred) == 2:
                value = float(np.asarray(pred[0]).reshape(-1)[0])
                std_value = float(np.asarray(pred[1]).reshape(-1)[0])
                if np.isfinite(std_value) and std_value >= 0:
                    details["prediction_std_rs_per_thm"] = std_value
                    details["prediction_interval_95_half_width_rs_per_thm"] = (
                        1.96 * std_value
                    )
                    details["prediction_lower_95_rs_per_thm"] = value - 1.96 * std_value
                    details["prediction_upper_95_rs_per_thm"] = value + 1.96 * std_value
                    details["uncertainty_source"] = type(model).__name__
                return value, details
        except TypeError:
            pass

        pred = model.predict(model_input)
        return float(np.asarray(pred).reshape(-1)[0]), details

    def predict(
        self, feature_payload: dict[str, float], history_df: pd.DataFrame | None
    ) -> ModelPrediction:
        """
        Predict fuel unit cost using ML artifacts or fallback logic.

        This is the public inference entrypoint for LP enrichment and DE
        objective calls. It handles missing artifacts, feature-build failures,
        model errors, and invalid predictions with structured fallback details.

        Args:
             - feature_payload: dict[str, float] - Blend and process feature payload.
             - history_df: pd.DataFrame | None - Historical process data for lag features.

        Returns:
             - return ModelPrediction - Fuel cost value plus model/fallback diagnostics.
        """

        self._ensure_loaded()
        bundle = self._bundle_info
        builder = self._feature_builder
        if bundle is None or builder is None:
            fallback = self._fallback_predict(feature_payload)
            return ModelPrediction(
                value=float(fallback),
                model_loaded=False,
                scaler_loaded=False,
                used_fallback=True,
                details={"reason": "Model bundle unavailable."},
            )

        if not bundle.expected_features:
            fallback = self._fallback_predict(feature_payload)
            return ModelPrediction(
                value=float(fallback),
                model_loaded=bundle.model is not None,
                scaler_loaded=bundle.scaler is not None,
                used_fallback=True,
                details={
                    "reason": "No expected feature list available in model/scaler/manifest."
                },
            )

        if self._selected_features:
            return self._predict_with_selected_features(
                feature_payload=feature_payload,
                history_df=history_df,
            )

        try:
            feature_build = builder.build(
                base_sample=feature_payload,
                history_df=history_df,
                expected_features=bundle.expected_features,
            )
        except Exception as exc:
            fallback = self._fallback_predict(feature_payload)
            return ModelPrediction(
                value=float(fallback),
                model_loaded=bundle.model is not None,
                scaler_loaded=bundle.scaler is not None,
                used_fallback=True,
                details={"reason": f"Feature vector build failed: {exc}"},
            )

        x_df = feature_build.vector_df

        if bundle.model is not None:
            try:
                if bundle.scaler is not None:
                    x_in = bundle.scaler.transform(x_df)
                else:
                    x_in = x_df

                value, prediction_details = self._model_predict_with_details(
                    bundle.model, x_in
                )
                prediction_issue = self._prediction_issue(value)
                if prediction_issue:
                    fallback = self._fallback_predict(feature_payload)
                    return ModelPrediction(
                        value=float(fallback),
                        model_loaded=True,
                        scaler_loaded=bundle.scaler is not None,
                        used_fallback=True,
                        missing_features=feature_build.missing_features,
                        imputed_features=feature_build.imputed_features,
                        details={
                            "reason": prediction_issue,
                            "model_value": value,
                            "feature_sources": feature_build.source_map,
                            **prediction_details,
                        },
                    )
                return ModelPrediction(
                    value=value,
                    model_loaded=True,
                    scaler_loaded=bundle.scaler is not None,
                    used_fallback=False,
                    missing_features=feature_build.missing_features,
                    imputed_features=feature_build.imputed_features,
                    details={
                        "feature_sources": feature_build.source_map,
                        **prediction_details,
                    },
                )
            except Exception as exc:
                fallback = self._fallback_predict(feature_payload)
                return ModelPrediction(
                    value=float(fallback),
                    model_loaded=True,
                    scaler_loaded=bundle.scaler is not None,
                    used_fallback=True,
                    missing_features=feature_build.missing_features,
                    imputed_features=feature_build.imputed_features,
                    details={
                        "reason": f"Model inference failed: {exc}",
                        "feature_sources": feature_build.source_map,
                    },
                )

        fallback = self._fallback_predict(feature_payload)
        return ModelPrediction(
            value=float(fallback),
            model_loaded=False,
            scaler_loaded=bundle.scaler is not None,
            used_fallback=True,
            missing_features=feature_build.missing_features,
            imputed_features=feature_build.imputed_features,
            details={
                "reason": "Model artifact not found.",
                "feature_sources": feature_build.source_map,
            },
        )

    def _scaler_mean_defaults(self) -> dict[str, float]:
        """
        Build default raw feature values from scaler mean metadata.

        When selected-feature inference needs a full raw feature row, scaler
        means are a safer default than zero for unresolved training columns. The
        values are keyed by the scaler's expected raw feature names.

        Args:
             - None

        Returns:
             - return dict[str, float] - Expected feature names mapped to scaler means.
        """

        bundle = self._bundle_info
        if (
            bundle is None
            or bundle.scaler is None
            or not hasattr(bundle.scaler, "mean_")
        ):
            return {}
        try:
            return {
                str(feature): float(bundle.scaler.mean_[idx])
                for idx, feature in enumerate(bundle.expected_features)
            }
        except Exception:
            return {}

    def _predict_with_selected_features(
        self,
        *,
        feature_payload: dict[str, float],
        history_df: pd.DataFrame | None,
    ) -> ModelPrediction:
        """
        Run the BMO V4 scaler-then-selected-feature inference path.

        The V4 bundle scales all raw features first and then feeds only the
        selected training columns to XGBoost. This method mirrors that notebook
        flow and records feature-source diagnostics for review.

        Args:
             - feature_payload: dict[str, float] - Blend and process feature payload.
             - history_df: pd.DataFrame | None - Historical process data for lag features.

        Returns:
             - return ModelPrediction - Fuel cost prediction with feature diagnostics.
        """

        bundle = self._bundle_info
        if bundle is None or bundle.model is None or bundle.scaler is None:
            fallback = self._fallback_predict(feature_payload)
            return ModelPrediction(
                value=float(fallback),
                model_loaded=bool(bundle and bundle.model is not None),
                scaler_loaded=bool(bundle and bundle.scaler is not None),
                used_fallback=True,
                details={"reason": "Selected-feature model requires model and scaler."},
            )

        selected_features = list(self._selected_features or [])
        missing_selected = [
            feature
            for feature in selected_features
            if feature not in bundle.expected_features
        ]
        if missing_selected:
            fallback = self._fallback_predict(feature_payload)
            return ModelPrediction(
                value=float(fallback),
                model_loaded=True,
                scaler_loaded=True,
                used_fallback=True,
                missing_features=missing_selected,
                details={
                    "reason": "Selected features are not present in scaler feature list.",
                    "selected_feature_count": len(selected_features),
                },
            )

        default_values = dict(bundle.defaults)
        default_values.update(self._scaler_mean_defaults())

        try:
            feature_build = build_bmo_v4_feature_frame(
                feature_payload=feature_payload,
                history_df=history_df,
                expected_features=list(bundle.expected_features),
                default_values=default_values,
            )
            scaled = bundle.scaler.transform(feature_build.vector_df)
            scaled_df = pd.DataFrame(
                scaled,
                columns=bundle.expected_features,
                index=feature_build.vector_df.index,
            )
            model_input = scaled_df[selected_features]
            value, prediction_details = self._model_predict_with_details(
                bundle.model, model_input
            )
            prediction_issue = self._prediction_issue(value)
            if prediction_issue:
                fallback = self._fallback_predict(feature_payload)
                return ModelPrediction(
                    value=float(fallback),
                    model_loaded=True,
                    scaler_loaded=True,
                    used_fallback=True,
                    missing_features=feature_build.missing_features,
                    imputed_features=feature_build.imputed_features,
                    details={
                        "reason": prediction_issue,
                        "model_value": value,
                        "feature_sources": feature_build.source_map,
                        "raw_feature_count": len(bundle.expected_features),
                        "selected_feature_count": len(selected_features),
                        **prediction_details,
                    },
                )
            return ModelPrediction(
                value=value,
                model_loaded=True,
                scaler_loaded=True,
                used_fallback=False,
                missing_features=feature_build.missing_features,
                imputed_features=feature_build.imputed_features,
                details={
                    "feature_sources": feature_build.source_map,
                    "raw_feature_count": len(bundle.expected_features),
                    "selected_feature_count": len(selected_features),
                    "selected_features": selected_features,
                    **prediction_details,
                },
            )
        except Exception as exc:
            fallback = self._fallback_predict(feature_payload)
            return ModelPrediction(
                value=float(fallback),
                model_loaded=True,
                scaler_loaded=True,
                used_fallback=True,
                details={"reason": f"Selected-feature inference failed: {exc}"},
            )

    def build_prebuilt_context(
        self,
        *,
        ores: "list[Any]",
        process_context: dict[str, float] | None,
        history_df: "pd.DataFrame | None",
        hot_metal_target_mt: float | None = None,
    ) -> PreBuiltFeatureContext | None:
        """
        Build a prebuilt feature context for repeated inference (e.g. DE inner loop).

        The context resolves every static feature column once using process
        context and history defaults so that subsequent predictions can mutate
        only the small burden-dependent slots. The returned context is reused
        with ``predict_with_prebuilt`` for every DE candidate.

        Args:
             - ores: list[OreInput] - Ores selected for the optimization run.
             - process_context: dict[str, float] | None - Latest process variables.
             - history_df: pd.DataFrame | None - Historical process data for lag features.
             - hot_metal_target_mt: float | None - Operator HM basis for THM model fields.

        Returns:
             - return PreBuiltFeatureContext | None - Cached context or None if not buildable.
        """

        from utils.bmo.feature_builder import build_prebuilt_feature_context

        self._ensure_loaded()
        bundle = self._bundle_info
        if bundle is None or not bundle.expected_features:
            return None

        default_values = dict(bundle.defaults)
        if self._selected_features:
            default_values.update(self._scaler_mean_defaults())

        try:
            return build_prebuilt_feature_context(
                ores=ores,
                process_context=process_context,
                history_df=history_df,
                expected_features=list(bundle.expected_features),
                default_values=default_values,
                selected_features=self._selected_features,
                hot_metal_target_mt=hot_metal_target_mt,
            )
        except Exception:
            return None

    def predict_with_prebuilt(
        self,
        context: PreBuiltFeatureContext,
        quantities_mt: dict[str, float],
        history_df: "pd.DataFrame | None",
    ) -> ModelPrediction:
        """
        Predict fuel cost using a prebuilt template and per-candidate dynamic payload.

        The prebuilt template carries the resolved static feature row. This fast
        path computes only the burden-dependent payload, overwrites the matching
        template columns by index, then runs scaler.transform + selected-column
        projection + model.predict. It avoids the per-call cost of resolving
        150+ feature columns through history/lag/alias lookups.

        Args:
             - context: PreBuiltFeatureContext - Cached scaler-order template + index map.
             - quantities_mt: dict[str, float] - Candidate wet ore quantities keyed by ore id.
             - history_df: pd.DataFrame | None - History frame, used only by fallback path.

        Returns:
             - return ModelPrediction - Fuel cost prediction with diagnostics.
        """

        self._ensure_loaded()
        bundle = self._bundle_info
        if bundle is None or bundle.model is None:
            payload_fallback = build_dynamic_payload(context, quantities_mt)
            payload_fallback.update(context.static_payload)
            return self.predict(payload_fallback, history_df)

        dynamic_payload = build_dynamic_payload(context, quantities_mt)

        x = context.template_x.copy()
        for key, value in dynamic_payload.items():
            try:
                float_value = float(value)
            except (TypeError, ValueError):
                continue
            for col_idx in context.dynamic_col_indexes.get(key, ()):
                x[0, col_idx] = float_value

        try:
            # Wrap as a DataFrame with the scaler's training column names so
            # sklearn does not warn about 'X has no valid feature names'.
            x_df = pd.DataFrame(x, columns=context.expected_features)
            if bundle.scaler is not None:
                x_scaled = bundle.scaler.transform(x_df)
            else:
                x_scaled = x_df.values

            if (
                context.selected_indexes is not None
                and context.selected_features
            ):
                model_input = pd.DataFrame(
                    x_scaled[:, context.selected_indexes],
                    columns=context.selected_features,
                )
            else:
                model_input = pd.DataFrame(
                    x_scaled, columns=context.expected_features
                )

            value, prediction_details = self._model_predict_with_details(
                bundle.model, model_input
            )
        except Exception as exc:
            payload_fallback = dict(context.static_payload)
            payload_fallback.update(dynamic_payload)
            fallback = self._fallback_predict(payload_fallback)
            return ModelPrediction(
                value=float(fallback),
                model_loaded=True,
                scaler_loaded=bundle.scaler is not None,
                used_fallback=True,
                details={"reason": f"Prebuilt inference failed: {exc}"},
            )

        prediction_issue = self._prediction_issue(value)
        if prediction_issue:
            payload_fallback = dict(context.static_payload)
            payload_fallback.update(dynamic_payload)
            fallback = self._fallback_predict(payload_fallback)
            return ModelPrediction(
                value=float(fallback),
                model_loaded=True,
                scaler_loaded=bundle.scaler is not None,
                used_fallback=True,
                details={
                    "reason": prediction_issue,
                    "model_value": value,
                    **prediction_details,
                },
            )

        return ModelPrediction(
            value=value,
            model_loaded=True,
            scaler_loaded=bundle.scaler is not None,
            used_fallback=False,
            details={
                "path": "prebuilt",
                "dynamic_field_count": len(dynamic_payload),
                **prediction_details,
            },
        )
