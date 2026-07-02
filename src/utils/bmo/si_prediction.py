"""Hot-metal Silicon (Si) prediction for BMO blends.

Display-only: predicts the Si content implied by the baseline (LP) and DE blends
so the operator can see the silicon outcome of each. Si is **not** an optimization
objective or constraint -- it never feeds the LP/DE solvers and never alters the
blend decision.

The Si model is a standalone XGBoost model with its own feature schema (198
columns), distinct from the fuel-cost model: it adds hourly-lag features
(``X__lag2h``), previous-cast Si (``CHEM_PCT_SI__lag3h/4h``) and calendar columns.
This service reuses the BMO blend payload for the blend-dependent features (so the
baseline and DE blends produce different Si), resolves lag features from their
base name, fills calendar/previous-Si context, and defaults everything else to the
scaler mean (the training average) so an unresolved feature is neutral.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from domain.optimization_runtime import ModelBundleService, build_runtime_config
from utils.bmo.feature_builder import build_feature_payload, normalize_feature_name
from utils.bmo.types import OreInput

# Matches a trailing lag suffix in either schema:
#   ``HOT BLAST VOLUMENM3/HR.__lag2h`` -> base ``HOT BLAST VOLUMENM3/HR.``
#   ``CHEM_PCT_SI__lag3h``            -> base ``CHEM_PCT_SI``
_LAG_SUFFIX_RE = re.compile(r"^(.*?)(?:__lag\d+h?|_lag\d+(?:_\([^)]+\))?)$")
_CALENDAR_FEATURES = {"month", "week_of_year", "day_of_year"}
_PREV_SI_BASE = "CHEM_PCT_SI"


def _base_feature_name(feature: str) -> str:
    """Strip a trailing lag suffix to get the underlying base feature name."""
    match = _LAG_SUFFIX_RE.match(feature)
    if match and match.group(1):
        return match.group(1)
    return feature


class SiPredictionService:
    """
    Predict hot-metal Si for a solved blend using the standalone Si model bundle.

    The expensive bundle load is deferred until first prediction. The service is
    safe to cache once per session and reused for the baseline and DE blends.

    Args:
         - bundle_cfg: dict[str, Any] - Si model bundle configuration (model/scaler/
           feature-columns paths), mirroring the fuel-cost ``model_bundle`` block.
    """

    def __init__(self, bundle_cfg: dict[str, Any]) -> None:
        self.bundle_cfg = bundle_cfg or {}
        runtime_cfg = build_runtime_config({"model_bundle": self.bundle_cfg})
        self._bundle_service = ModelBundleService(runtime_cfg["model_bundle"])
        self._bundle: Any | None = None
        self._features: list[str] = []
        self._means: dict[str, float] = {}

    def _ensure_loaded(self) -> None:
        if self._bundle is not None:
            return
        self._bundle = self._bundle_service.get_bundle()
        self._features = list(self._bundle.expected_features or [])
        scaler = getattr(self._bundle, "scaler", None)
        if scaler is not None and hasattr(scaler, "mean_"):
            self._means = {
                feature: float(scaler.mean_[idx])
                for idx, feature in enumerate(self._features)
            }

    def get_status(self) -> dict[str, Any]:
        """Return load status for the page header/diagnostics."""
        self._ensure_loaded()
        return {
            "model_loaded": self._bundle is not None and self._bundle.model is not None,
            "scaler_loaded": self._bundle is not None and self._bundle.scaler is not None,
            "feature_count": len(self._features),
        }

    def predict_blend_si(
        self,
        *,
        ores: list[OreInput],
        quantities_mt: Mapping[str, float],
        process_context: Mapping[str, Any] | None,
        prev_si: float | None = None,
        hot_metal_target_mt: float | None = None,
        now: datetime | None = None,
    ) -> float | None:
        """
        Predict Si (%) for one solved blend.

        Args:
             - ores: list[OreInput] - Ores included in the blend.
             - quantities_mt: Mapping[str, float] - Solved quantities keyed by ore id.
             - process_context: Mapping[str, Any] | None - Latest process variables
               (incl. coke strength) shared with the fuel-cost feature payload.
             - prev_si: float | None - Most recent measured cast Si, used for the
               ``CHEM_PCT_SI__lag*`` features. Defaults to the training average.
             - hot_metal_target_mt: float | None - HM basis for THM-normalised features.
             - now: datetime | None - Timestamp for calendar features (defaults to now).

        Returns:
             - return float | None - Predicted Si %, or None when the model is missing.
        """

        self._ensure_loaded()
        if (
            self._bundle is None
            or self._bundle.model is None
            or self._bundle.scaler is None
            or not self._features
        ):
            return None

        now = now or datetime.now()
        ore_name_by_id = {ore.ore_id: ore.display_name for ore in ores}
        payload = build_feature_payload(
            quantities_mt={str(k): float(v) for k, v in quantities_mt.items()},
            ore_display_name_by_id=ore_name_by_id,
            process_context=process_context,
            ores=ores,
            hot_metal_target_mt=hot_metal_target_mt,
        )
        norm_payload = {normalize_feature_name(k): v for k, v in payload.items()}
        calendar = {
            "month": float(now.month),
            "week_of_year": float(now.isocalendar().week),
            "day_of_year": float(now.timetuple().tm_yday),
        }

        values: list[float] = []
        for feature in self._features:
            default = self._means.get(feature, 0.0)
            if feature in _CALENDAR_FEATURES:
                values.append(calendar[feature])
                continue
            base = _base_feature_name(feature)
            if base == _PREV_SI_BASE:
                values.append(float(prev_si) if prev_si is not None else default)
                continue
            resolved: Any | None = None
            for key in (feature, base):
                if key in payload:
                    resolved = payload[key]
                    break
                norm_key = normalize_feature_name(key)
                if norm_key in norm_payload:
                    resolved = norm_payload[norm_key]
                    break
            try:
                values.append(float(resolved) if resolved is not None else default)
            except (TypeError, ValueError):
                values.append(default)

        frame = pd.DataFrame([values], columns=self._features)
        scaled = self._bundle.scaler.transform(frame)
        # Re-wrap with the trained feature names: the XGBoost booster validates
        # feature names on predict, so a nameless array is rejected.
        scaled_df = pd.DataFrame(scaled, columns=self._features)
        prediction = self._bundle.model.predict(scaled_df)
        value = float(np.asarray(prediction).reshape(-1)[0])
        return value if np.isfinite(value) else None
