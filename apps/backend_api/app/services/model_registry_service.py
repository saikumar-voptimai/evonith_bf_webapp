"""Lazy model registry for Phase 7 backend compute APIs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.accelerator_service import AcceleratorService
from apps.backend_api.app.services.optional_dependency_service import require_optional_module
from furnace_data.assets import model_dir_from_config
from furnace_data.optimization_runtime.model_bundle import XGBoostJsonModel

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_MODEL_EXTENSIONS = {".joblib", ".pkl", ".json", ".pth"}
_ALLOWED_MODEL_SUBDIRS = {"bmo_fuel"}


@dataclass(frozen=True)
class ModelInfo:
    name: str
    kind: str
    status: str
    loaded: bool
    optional: bool = True


class ModelRegistryService:
    """Discover model files and load them only when requested."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def _model_root(self) -> Path:
        return model_dir_from_config(self.settings.model_dir)

    def _discover(self) -> dict[str, Path]:
        root = self._model_root()
        if not root.exists():
            return {}
        models: dict[str, Path] = {}
        candidates = [path for path in root.iterdir() if path.is_file()]
        for subdir_name in _ALLOWED_MODEL_SUBDIRS:
            subdir = root / subdir_name
            if subdir.is_dir():
                candidates.extend(path for path in subdir.iterdir() if path.is_file())

        for path in candidates:
            if path.suffix.lower() not in _MODEL_EXTENSIONS:
                continue
            name = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
            if _MODEL_ID_RE.match(name):
                models[name] = path
        return models

    def list_models(self) -> list[dict[str, Any]]:
        models = self._discover()
        return [
            {
                "name": name,
                "kind": path.suffix.lower().lstrip("."),
                "status": "loaded" if name in self._cache else "available",
                "loaded": name in self._cache,
                "optional": True,
            }
            for name, path in sorted(models.items())
        ]

    def get_model_status(self, model_name: str) -> dict[str, Any]:
        self._validate_model_name(model_name)
        models = self._discover()
        if model_name not in models:
            return {
                "name": model_name,
                "kind": "unknown",
                "status": "missing",
                "loaded": False,
                "optional": True,
            }
        path = models[model_name]
        return {
            "name": model_name,
            "kind": path.suffix.lower().lstrip("."),
            "status": "loaded" if model_name in self._cache else "available",
            "loaded": model_name in self._cache,
            "optional": True,
        }

    @staticmethod
    def _validate_model_name(model_name: str) -> None:
        if not _MODEL_ID_RE.match(str(model_name or "")):
            raise ApiError(
                "MODEL_PATH_INVALID",
                "Model name is invalid.",
                status_code=400,
            )

    def _model_path(self, model_name: str) -> Path:
        self._validate_model_name(model_name)
        models = self._discover()
        path = models.get(model_name)
        if path is None:
            raise ApiError("MODEL_NOT_FOUND", "Model is not registered.", status_code=404)
        root = self._model_root()
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ApiError("MODEL_PATH_INVALID", "Model path is invalid.", status_code=400) from exc
        return resolved

    def get_model(self, model_name: str) -> Any:
        if model_name in self._cache:
            model = self._cache.pop(model_name)
            self._cache[model_name] = model
            return model

        path = self._model_path(model_name)
        try:
            if path.suffix.lower() == ".json":
                device = AcceleratorService(self.settings).resolve_xgboost_device()
                model = XGBoostJsonModel(path, device=device)
            else:
                joblib = require_optional_module("joblib", "backend-ml")

                model = joblib.load(path)
        except ModuleNotFoundError as exc:
            raise ApiError(
                "MODEL_OPTIONAL_UNAVAILABLE",
                "Optional model dependency is not installed.",
                status_code=503,
            ) from exc
        except Exception as exc:
            raise ApiError("MODEL_LOAD_FAILED", "Model could not be loaded.", status_code=500) from exc

        self._cache[model_name] = model
        while len(self._cache) > self.settings.model_cache_max_items:
            self._cache.popitem(last=False)
        return model

    def clear_model_cache(self) -> None:
        self._cache.clear()

    def predict(self, model_name: str, features: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        model = self.get_model(model_name)
        rows = features if isinstance(features, list) else [features]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ApiError("MODEL_INPUT_INVALID", "Model features must be object rows.", status_code=422)
        if not hasattr(model, "predict"):
            raise ApiError(
                "MODEL_PREDICTION_FAILED",
                "Registered model does not support prediction.",
                status_code=422,
            )
        try:
            pd = require_optional_module("pandas", "backend-base")

            frame = pd.DataFrame(rows)
            predictions = model.predict(frame)
        except Exception as exc:
            raise ApiError("MODEL_PREDICTION_FAILED", "Model prediction failed.", status_code=500) from exc
        return {
            "model_name": model_name,
            "predictions": [float(value) for value in list(predictions)],
            "device": str(getattr(model, "device", "cpu")),
            "model_status": self.get_model_status(model_name),
        }
