"""Lazy model registry for Phase 7 backend compute APIs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError
from furnace_data.runtime_paths import get_repo_root

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_MODEL_EXTENSIONS = {".joblib", ".pkl", ".json"}


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
        configured = str(self.settings.model_dir or "").strip()
        if configured:
            root = Path(configured)
            if not root.is_absolute():
                root = get_repo_root() / root
            return root.resolve()
        return (get_repo_root() / "src" / "assets" / "models").resolve()

    def _discover(self) -> dict[str, Path]:
        root = self._model_root()
        if not root.exists():
            return {}
        models: dict[str, Path] = {}
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _MODEL_EXTENSIONS:
                continue
            try:
                path.resolve().relative_to(root)
            except ValueError:
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
                model = json.loads(path.read_text(encoding="utf-8"))
            else:
                import joblib

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
            import pandas as pd

            frame = pd.DataFrame(rows)
            predictions = model.predict(frame)
        except Exception as exc:
            raise ApiError("MODEL_PREDICTION_FAILED", "Model prediction failed.", status_code=500) from exc
        return {
            "model_name": model_name,
            "predictions": [float(value) for value in list(predictions)],
            "model_status": self.get_model_status(model_name),
        }
