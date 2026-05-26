from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.bmo.types import ModelPrediction
from domain.optimization_runtime import (
    FeatureVectorBuilder,
    ModelBundleService,
    build_runtime_config,
    normalize_feature_name,
)


class FuelUnitCostModelService:
    def __init__(
        self, bundle_cfg: dict[str, Any], fallback_cfg: dict[str, Any]
    ) -> None:
        self.bundle_cfg = bundle_cfg or {}
        self.fallback_cfg = fallback_cfg or {}

        runtime_cfg = build_runtime_config({"model_bundle": self.bundle_cfg})
        self._bundle_service = ModelBundleService(runtime_cfg["model_bundle"])
        self._bundle_info = None
        self._feature_builder: FeatureVectorBuilder | None = None
        self.missing_feature_policy = str(
            self.bundle_cfg.get("missing_feature_policy", "default_warn")
        )

    def _ensure_loaded(self) -> None:
        if self._bundle_info is not None:
            return
        self._bundle_info = self._bundle_service.get_bundle()
        self._feature_builder = FeatureVectorBuilder(
            self._bundle_info,
            missing_feature_policy=self.missing_feature_policy,
        )

    def get_bundle_status(self) -> dict[str, Any]:
        self._ensure_loaded()
        if self._bundle_info is None:
            return {
                "model_loaded": False,
                "scaler_loaded": False,
                "feature_count": 0,
                "bundle_error": "bundle_not_loaded",
            }
        return dict(self._bundle_info.status)

    def _pick_value(
        self, payload: dict[str, float], keys: list[str], default: float
    ) -> float:
        norm_payload = {normalize_feature_name(k): float(v) for k, v in payload.items()}
        for key in keys:
            if key in payload:
                return float(payload[key])
            norm_key = normalize_feature_name(key)
            if norm_key in norm_payload:
                return float(norm_payload[norm_key])
        return float(default)

    def _fallback_predict(self, payload: dict[str, float]) -> float:
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

    def predict(
        self, feature_payload: dict[str, float], history_df: pd.DataFrame | None
    ) -> ModelPrediction:
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

                pred = bundle.model.predict(x_in)
                value = float(np.asarray(pred).reshape(-1)[0])
                return ModelPrediction(
                    value=value,
                    model_loaded=True,
                    scaler_loaded=bundle.scaler is not None,
                    used_fallback=False,
                    missing_features=feature_build.missing_features,
                    imputed_features=feature_build.imputed_features,
                    details={"feature_sources": feature_build.source_map},
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
