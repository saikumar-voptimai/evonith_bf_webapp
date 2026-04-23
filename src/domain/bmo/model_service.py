from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from domain.bmo.feature_builder import normalize_feature_name
from domain.bmo.types import ModelPrediction


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / p,
        repo_root / "src" / p,
        Path.cwd() / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class FuelUnitCostModelService:
    def __init__(self, bundle_cfg: dict[str, Any], fallback_cfg: dict[str, Any]) -> None:
        self.bundle_cfg = bundle_cfg or {}
        self.fallback_cfg = fallback_cfg or {}

        self.model_path = _resolve_repo_path(str(self.bundle_cfg.get("model_path", "")))
        self.scaler_path = _resolve_repo_path(str(self.bundle_cfg.get("scaler_path", "")))
        self.feature_manifest_path = _resolve_repo_path(
            str(self.bundle_cfg.get("feature_manifest_path", ""))
        )
        self.lag_map_path = _resolve_repo_path(str(self.bundle_cfg.get("lag_map_path", "")))
        self.training_metrics_path = _resolve_repo_path(
            str(self.bundle_cfg.get("training_metrics_path", ""))
        )
        self.strict_loading = bool(self.bundle_cfg.get("strict_loading", False))

        self._loaded = False
        self.model = None
        self.scaler = None
        self.feature_manifest: dict[str, Any] = {}
        self.lag_map: dict[str, Any] = {}
        self.training_metrics: dict[str, Any] = {}
        self.expected_features: list[str] = []
        self.last_bundle_error: str | None = None

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists() or path.is_dir():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        self.feature_manifest = self._load_json(self.feature_manifest_path)
        self.lag_map = self._load_json(self.lag_map_path)
        self.training_metrics = self._load_json(self.training_metrics_path)

        try:
            if self.model_path.exists() and self.model_path.is_file():
                self.model = joblib.load(self.model_path)
        except Exception as exc:
            self.last_bundle_error = f"model load failed: {exc}"
            if self.strict_loading:
                raise

        try:
            if self.scaler_path.exists() and self.scaler_path.is_file():
                self.scaler = joblib.load(self.scaler_path)
        except Exception as exc:
            self.last_bundle_error = f"scaler load failed: {exc}"
            if self.strict_loading:
                raise

        if self.scaler is not None and hasattr(self.scaler, "feature_names_in_"):
            self.expected_features = list(self.scaler.feature_names_in_)
        elif self.model is not None and hasattr(self.model, "feature_names_in_"):
            self.expected_features = list(self.model.feature_names_in_)
        else:
            self.expected_features = list(self.feature_manifest.get("feature_names", []))

        target_name = str(self.feature_manifest.get("target_name", "")).strip()
        if target_name and target_name in self.expected_features:
            self.expected_features = [f for f in self.expected_features if f != target_name]

    def get_bundle_status(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "model_path": str(self.model_path),
            "scaler_path": str(self.scaler_path),
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "feature_count": len(self.expected_features),
            "bundle_error": self.last_bundle_error,
        }

    def _pick_value(self, payload: dict[str, float], keys: list[str], default: float) -> float:
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
            ["COKE RATE KG/THM", "coke_rate_kg_per_thm", "coke_rate_kg_thm", "coke_rate"],
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

        base_cost = (coke_rate * coke_price) + (nut_rate * nut_price) + (pci_rate * pci_price)

        sinter_share = self._pick_value(payload, ["sinter_share_pct"], 60.0)
        target_sinter = float(self.fallback_cfg.get("target_sinter_share_pct", 60.0))
        sensitivity = float(
            self.fallback_cfg.get("sinter_deviation_sensitivity_rs_per_thm", 1200.0)
        )
        adjustment = ((target_sinter - sinter_share) / 100.0) * sensitivity
        return float(base_cost + adjustment)

    def _parse_lag_from_name(self, feature_name: str) -> tuple[str, int] | None:
        match = re.match(r"(.+)_lag(\d+)$", str(feature_name))
        if not match:
            return None
        return match.group(1), int(match.group(2))

    def _resolve_feature_from_history(
        self, history_df: pd.DataFrame, base_feature: str, lag_steps: int
    ) -> float | None:
        if history_df is None or history_df.empty:
            return None

        columns_norm = {normalize_feature_name(c): c for c in history_df.columns}
        target_col = base_feature
        if target_col not in history_df.columns:
            norm = normalize_feature_name(base_feature)
            target_col = columns_norm.get(norm, "")
        if not target_col:
            return None

        series = pd.to_numeric(history_df[target_col], errors="coerce").dropna()
        if series.empty:
            return None

        if lag_steps < 0:
            lag_steps = 0
        idx = lag_steps + 1
        if len(series) < idx:
            return None
        return float(series.iloc[-idx])

    def _resolve_feature_vector(
        self, payload: dict[str, float], history_df: pd.DataFrame | None
    ) -> tuple[list[float], list[str], list[str]]:
        defaults = self.feature_manifest.get("defaults", {})
        lag_cfg = self.lag_map.get("lags", {})

        payload_norm = {normalize_feature_name(k): float(v) for k, v in payload.items()}
        vector: list[float] = []
        missing: list[str] = []
        imputed: list[str] = []

        for feature in self.expected_features:
            value: float | None = None

            if feature in payload:
                value = float(payload[feature])
            else:
                norm_feat = normalize_feature_name(feature)
                if norm_feat in payload_norm:
                    value = float(payload_norm[norm_feat])

            lag_conf = lag_cfg.get(feature)
            if value is None and lag_conf:
                base_feature = str(lag_conf.get("base_feature", ""))
                lag_steps = int(lag_conf.get("lag_steps", 1))
                value = self._resolve_feature_from_history(history_df, base_feature, lag_steps)

            if value is None:
                parsed = self._parse_lag_from_name(feature)
                if parsed:
                    base_feature, lag_steps = parsed
                    value = self._resolve_feature_from_history(history_df, base_feature, lag_steps)

            if value is None:
                value = self._resolve_feature_from_history(history_df, feature, lag_steps=0)

            if value is None:
                missing.append(feature)
                if feature in defaults:
                    value = float(defaults[feature])
                else:
                    value = 0.0
                imputed.append(feature)

            vector.append(float(value))

        return vector, missing, imputed

    def predict(
        self, feature_payload: dict[str, float], history_df: pd.DataFrame | None
    ) -> ModelPrediction:
        self._ensure_loaded()

        if not self.expected_features:
            fallback = self._fallback_predict(feature_payload)
            return ModelPrediction(
                value=float(fallback),
                model_loaded=self.model is not None,
                scaler_loaded=self.scaler is not None,
                used_fallback=True,
                details={"reason": "No expected feature list available in model/scaler/manifest."},
            )

        vector, missing, imputed = self._resolve_feature_vector(feature_payload, history_df)
        x_df = pd.DataFrame([vector], columns=self.expected_features, dtype=float)

        if self.model is not None:
            try:
                if self.scaler is not None:
                    x_in = self.scaler.transform(x_df)
                else:
                    x_in = x_df

                pred = self.model.predict(x_in)
                value = float(np.asarray(pred).reshape(-1)[0])
                return ModelPrediction(
                    value=value,
                    model_loaded=True,
                    scaler_loaded=self.scaler is not None,
                    used_fallback=False,
                    missing_features=missing,
                    imputed_features=imputed,
                )
            except Exception as exc:
                fallback = self._fallback_predict(feature_payload)
                return ModelPrediction(
                    value=float(fallback),
                    model_loaded=True,
                    scaler_loaded=self.scaler is not None,
                    used_fallback=True,
                    missing_features=missing,
                    imputed_features=imputed,
                    details={"reason": f"Model inference failed: {exc}"},
                )

        fallback = self._fallback_predict(feature_payload)
        return ModelPrediction(
            value=float(fallback),
            model_loaded=False,
            scaler_loaded=self.scaler is not None,
            used_fallback=True,
            missing_features=missing,
            imputed_features=imputed,
            details={"reason": "Model artifact not found."},
        )
