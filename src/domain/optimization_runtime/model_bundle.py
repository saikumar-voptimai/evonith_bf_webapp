from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from domain.optimization_runtime.types import ModelBundleInfo


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(str(path_str or "")).expanduser()
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


class ModelBundleService:
    """Load model/scaler/metadata and expose the expected inference schema."""

    def __init__(
        self,
        bundle_cfg: dict[str, Any] | None = None,
        *,
        strict_loading: bool | None = None,
    ) -> None:
        self.bundle_cfg = bundle_cfg or {}
        self.strict_loading = (
            bool(strict_loading)
            if strict_loading is not None
            else bool(self.bundle_cfg.get("strict_loading", False))
        )
        self._bundle: ModelBundleInfo | None = None

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists() or path.is_dir():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_artifact(self, path: Path) -> Any | None:
        if not path.exists() or path.is_dir():
            return None
        return joblib.load(path)

    def _extract_expected_features(
        self,
        *,
        scaler: Any | None,
        model: Any | None,
        feature_manifest: dict[str, Any],
        target_name: str | None,
    ) -> list[str]:
        expected_features: list[str]
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            expected_features = list(scaler.feature_names_in_)
        elif model is not None and hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
        else:
            expected_features = list(feature_manifest.get("feature_names", []))

        if target_name and target_name in expected_features:
            expected_features = [f for f in expected_features if f != target_name]
        return expected_features

    def get_bundle(self) -> ModelBundleInfo:
        if self._bundle is not None:
            return self._bundle

        model_path = _resolve_repo_path(str(self.bundle_cfg.get("model_path", "")))
        scaler_path = _resolve_repo_path(str(self.bundle_cfg.get("scaler_path", "")))
        feature_manifest_path = _resolve_repo_path(
            str(self.bundle_cfg.get("feature_manifest_path", ""))
        )
        lag_map_path = _resolve_repo_path(str(self.bundle_cfg.get("lag_map_path", "")))
        training_metrics_path = _resolve_repo_path(
            str(self.bundle_cfg.get("training_metrics_path", ""))
        )

        model = None
        scaler = None
        bundle_error: str | None = None

        feature_manifest = self._load_json(feature_manifest_path)
        lag_map = self._load_json(lag_map_path)
        training_metrics = self._load_json(training_metrics_path)

        try:
            model = self._load_artifact(model_path)
        except Exception as exc:
            bundle_error = f"model load failed: {exc}"
            if self.strict_loading:
                raise

        try:
            scaler = self._load_artifact(scaler_path)
        except Exception as exc:
            bundle_error = f"scaler load failed: {exc}"
            if self.strict_loading:
                raise

        target_name = str(
            self.bundle_cfg.get("target_name")
            or feature_manifest.get("target_name")
            or ""
        ).strip() or None

        expected_features = self._extract_expected_features(
            scaler=scaler,
            model=model,
            feature_manifest=feature_manifest,
            target_name=target_name,
        )
        defaults = {
            str(key): float(value)
            for key, value in (feature_manifest.get("defaults", {}) or {}).items()
            if value is not None
        }

        status = {
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "feature_manifest_path": str(feature_manifest_path),
            "lag_map_path": str(lag_map_path),
            "training_metrics_path": str(training_metrics_path),
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "feature_count": len(expected_features),
            "bundle_error": bundle_error,
        }

        self._bundle = ModelBundleInfo(
            model=model,
            scaler=scaler,
            expected_features=expected_features,
            lag_map=lag_map,
            defaults=defaults,
            status=status,
            feature_manifest=feature_manifest,
            training_metrics=training_metrics,
            target_name=target_name,
        )
        return self._bundle
