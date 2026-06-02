"""Model artifact loading for optimization runtime inference.

This module resolves repository-relative model paths, loads model/scaler
artifacts, and exposes the expected feature schema used by BMO fuel-cost
prediction. It also supports native XGBoost JSON models to avoid pickle
compatibility warnings at app startup.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib

from domain.optimization_runtime.types import ModelBundleInfo


def _parse_version_parts(version: Any) -> tuple[int, int, int] | None:
    """
    Parse a package or artifact version into comparable integer parts.

    XGBoost JSON files store the writer version as a list, while the runtime
    exposes a string. Normalizing both forms lets the loader block unsafe
    older-runtime/newer-model combinations before they produce bad predictions.

    Args:
         - version: Any - Version value from package metadata or model JSON.

    Returns:
         - return tuple[int, int, int] | None - Comparable semantic version parts.
    """

    if isinstance(version, (list, tuple)):
        parts = [int(part) for part in version[:3]]
    else:
        parts = [
            int(match.group(1))
            for item in str(version).split(".")[:3]
            if (match := re.match(r"^(\d+)", item))
        ]
    if not parts:
        return None
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _read_xgboost_json_version(path: Path) -> tuple[int, int, int] | None:
    """
    Read the XGBoost writer version from a native JSON model artifact.

    Newer XGBoost JSON formats can contain metadata that older runtimes parse
    incorrectly. Reading the artifact version before loading the booster gives
    the app a chance to fail safely instead of trusting distorted predictions.

    Args:
         - path: Path - XGBoost JSON model path.

    Returns:
         - return tuple[int, int, int] | None - Writer version from the JSON file.
    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _parse_version_parts(data.get("version"))


class XGBoostJsonModel:
    """
    Wrap a native XGBoost JSON booster with a predict-compatible interface.

    The wrapper lets the app load the exported JSON model artifact instead of a
    pickled estimator. That avoids XGBoost serialization warnings while keeping
    the same ``predict`` shape expected by the model service.

    Args:
         - path: Path - Path to the JSON model exported by XGBoost.

    Returns:
         - return XGBoostJsonModel - Wrapper that can be used like a model object.
    """

    def __init__(self, path: Path) -> None:
        """
        Load the XGBoost booster from its JSON artifact.

        The native booster is cached on the wrapper and exposed with minimal
        metadata so feature-order discovery still works when XGBoost includes
        feature names in the JSON model.

        Args:
             - path: Path - Path to the JSON model exported by XGBoost.

        Returns:
             - return None - Initializes the booster and feature metadata.
        """

        import xgboost as xgb

        model_version = _read_xgboost_json_version(path)
        runtime_version = _parse_version_parts(xgb.__version__)
        if model_version and runtime_version and runtime_version < model_version:
            model_text = ".".join(str(part) for part in model_version)
            raise RuntimeError(
                f"XGBoost runtime {xgb.__version__} is older than model JSON "
                f"version {model_text}. Use a matching/newer XGBoost runtime or "
                "re-export the model with the deployed runtime version."
            )

        self.booster = xgb.Booster()
        self.booster.load_model(str(path))
        self.feature_names_in_ = list(self.booster.feature_names or [])

    def predict(self, X: Any) -> Any:
        """
        Predict with the wrapped Booster using the input column order.

        The model service passes a pandas frame or array in trained feature
        order. This method converts it to ``DMatrix`` and delegates prediction
        to the native XGBoost booster.

        Args:
             - X: Any - Tabular model input accepted by XGBoost DMatrix.

        Returns:
             - return Any - Prediction array returned by the XGBoost booster.
        """

        from xgboost import DMatrix

        return self.booster.predict(DMatrix(X))


def _resolve_repo_path(path_str: str) -> Path:
    """
    Resolve an artifact path against common repository locations.

    Model paths in YAML are usually repository-relative, while tests sometimes
    use temporary or absolute paths. This helper tries the common locations and
    returns a deterministic candidate even when the file is missing.

    Args:
         - path_str: str - Absolute or repository-relative artifact path.

    Returns:
         - return Path - Existing resolved path when found, otherwise first candidate.
    """

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
    """
    Load model, scaler, and metadata artifacts for runtime inference.

    This service centralizes artifact loading for the optimization runtime. It
    resolves paths, supports native XGBoost JSON models, extracts expected
    feature order, and records bundle status for Streamlit diagnostics.

    Args:
         - bundle_cfg: dict[str, Any] | None - Model bundle configuration.
         - strict_loading: bool | None - Whether artifact load errors should raise.

    Returns:
         - return ModelBundleService - Service with cached model bundle access.
    """

    def __init__(
        self,
        bundle_cfg: dict[str, Any] | None = None,
        *,
        strict_loading: bool | None = None,
    ) -> None:
        """
        Initialize the model bundle loader and cache state.

        Artifact loading is delayed until ``get_bundle`` is called so Streamlit
        can construct the service cheaply and cache the loaded bundle afterward.

        Args:
             - bundle_cfg: dict[str, Any] | None - Model bundle configuration.
             - strict_loading: bool | None - Whether artifact load errors should raise.

        Returns:
             - return None - Initializes service configuration and empty cache.
        """

        self.bundle_cfg = bundle_cfg or {}
        self.strict_loading = (
            bool(strict_loading)
            if strict_loading is not None
            else bool(self.bundle_cfg.get("strict_loading", False))
        )
        self._bundle: ModelBundleInfo | None = None

    def _load_json(self, path: Path) -> dict[str, Any]:
        """
        Load a JSON metadata file when it exists.

        Metadata files are optional in development and tests. Missing or invalid
        JSON therefore returns an empty dictionary instead of failing the whole
        model service when strict loading is disabled.

        Args:
             - path: Path - JSON file path to load.

        Returns:
             - return dict[str, Any] - Parsed JSON dictionary or an empty dictionary.
        """

        if not path.exists() or path.is_dir():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_artifact(self, path: Path) -> Any | None:
        """
        Load a model or scaler artifact from disk.

        JSON files are treated as native XGBoost model exports. Other artifact
        formats are loaded with joblib because scalers and legacy model bundles
        are persisted that way.

        Args:
             - path: Path - Artifact path for JSON, joblib, or pickle formats.

        Returns:
             - return Any | None - Loaded artifact, XGBoost wrapper, or None.
        """

        if not path.exists() or path.is_dir():
            return None
        if path.suffix.lower() == ".json":
            return XGBoostJsonModel(path)
        return joblib.load(path)

    def _extract_expected_features(
        self,
        *,
        scaler: Any | None,
        model: Any | None,
        feature_manifest: dict[str, Any],
        target_name: str | None,
    ) -> list[str]:
        """
        Determine the expected model input feature order.

        The scaler is treated as the source of truth when available because it
        was fit on the raw training feature matrix. Model metadata and manifest
        values are fallbacks for bundles that do not include scaler feature names.

        Args:
             - scaler: Any | None - Loaded scaler artifact, if available.
             - model: Any | None - Loaded model artifact, if available.
             - feature_manifest: dict[str, Any] - Optional feature manifest content.
             - target_name: str | None - Target column that should be excluded.

        Returns:
             - return list[str] - Ordered feature names required for inference.
        """

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
        """
        Load and cache the full model bundle used by runtime inference.

        The bundle combines model, scaler, feature metadata, lag metadata, and
        training metrics into one object. Loading happens once so repeated BMO
        predictions reuse the same artifacts and expose stable status details.

        Args:
             - None

        Returns:
             - return ModelBundleInfo - Model, scaler, feature metadata, and status.
        """

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

        target_name = (
            str(
                self.bundle_cfg.get("target_name")
                or feature_manifest.get("target_name")
                or ""
            ).strip()
            or None
        )

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
