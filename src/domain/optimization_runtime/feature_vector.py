from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from domain.optimization_runtime.feature_utils import (
    normalize_feature_name,
    parse_lag_feature_name,
)
from domain.optimization_runtime.types import FeatureBuildResult, ModelBundleInfo


class FeatureVectorBuilder:
    """Build one-row feature vectors from expected model features + lag logic."""

    def __init__(
        self,
        bundle_info: ModelBundleInfo,
        *,
        missing_feature_policy: str = "default_warn",
    ) -> None:
        self.bundle_info = bundle_info
        self.missing_feature_policy = str(missing_feature_policy or "default_warn")

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_from_history(
        self, history_df: pd.DataFrame | None, base_feature: str, lag_steps: int
    ) -> float | None:
        if history_df is None or history_df.empty:
            return None

        columns_norm = {normalize_feature_name(str(c)): str(c) for c in history_df.columns}
        target_col = str(base_feature)
        if target_col not in history_df.columns:
            target_col = columns_norm.get(normalize_feature_name(base_feature), "")
        if not target_col:
            return None

        series = pd.to_numeric(history_df[target_col], errors="coerce").dropna()
        if series.empty:
            return None

        lag = max(0, int(lag_steps))
        idx = lag + 1
        if len(series) < idx:
            return None
        return float(series.iloc[-idx])

    def _resolve_feature(
        self,
        feature: str,
        base_sample: Mapping[str, Any],
        sample_norm: dict[str, float],
        history_df: pd.DataFrame | None,
    ) -> tuple[float | None, str]:
        if feature in base_sample:
            val = self._to_float(base_sample[feature])
            if val is not None:
                return val, "direct"

        norm_name = normalize_feature_name(feature)
        if norm_name in sample_norm:
            return sample_norm[norm_name], "normalized"

        lag_cfg = (self.bundle_info.lag_map or {}).get("lags", {}).get(feature)
        if isinstance(lag_cfg, Mapping):
            base_feature = str(lag_cfg.get("base_feature", ""))
            lag_steps = int(lag_cfg.get("lag_steps", 1))
            val = self._resolve_from_history(history_df, base_feature, lag_steps)
            if val is not None:
                return val, f"lag_map:{base_feature}:{lag_steps}"

        parsed = parse_lag_feature_name(feature)
        if parsed:
            base_feature, lag_steps = parsed
            val = self._resolve_from_history(history_df, base_feature, lag_steps)
            if val is not None:
                return val, f"lag_suffix:{base_feature}:{lag_steps}"

        val = self._resolve_from_history(history_df, feature, lag_steps=0)
        if val is not None:
            return val, "history_latest"

        if feature in self.bundle_info.defaults:
            return float(self.bundle_info.defaults[feature]), "default"

        return None, "missing"

    def build(
        self,
        *,
        base_sample: Mapping[str, Any] | pd.Series,
        history_df: pd.DataFrame | None,
        expected_features: list[str] | None = None,
    ) -> FeatureBuildResult:
        if isinstance(base_sample, pd.Series):
            sample_map = base_sample.to_dict()
        else:
            sample_map = dict(base_sample)
        sample_norm: dict[str, float] = {}
        for key, value in sample_map.items():
            numeric = self._to_float(value)
            if numeric is None:
                continue
            sample_norm[normalize_feature_name(str(key))] = float(numeric)

        feature_list = expected_features or list(self.bundle_info.expected_features)
        values: list[float] = []
        missing_features: list[str] = []
        imputed_features: list[str] = []
        source_map: dict[str, str] = {}

        for feature in feature_list:
            value, source = self._resolve_feature(
                str(feature), sample_map, sample_norm, history_df
            )
            if value is None:
                missing_features.append(str(feature))
                if self.missing_feature_policy == "strict_fail":
                    raise KeyError(
                        f"Missing feature '{feature}' and missing_feature_policy=strict_fail"
                    )
                if self.missing_feature_policy == "zero_fill":
                    value = 0.0
                    source = "zero_fill"
                else:
                    value = 0.0
                    source = "default_warn"
                imputed_features.append(str(feature))
            elif source in {"default", "zero_fill", "default_warn"}:
                imputed_features.append(str(feature))

            values.append(float(value))
            source_map[str(feature)] = source

        vector_df = pd.DataFrame([values], columns=feature_list, dtype=float)
        return FeatureBuildResult(
            vector_df=vector_df,
            missing_features=missing_features,
            imputed_features=imputed_features,
            source_map=source_map,
        )
