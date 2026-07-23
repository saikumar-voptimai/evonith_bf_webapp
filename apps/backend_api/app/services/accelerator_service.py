"""Optional accelerator checks with safe CPU fallback on portable edge hosts."""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings


_CUDA_DEVICE_PATHS = (Path("/dev/nvhost-gpu"), Path("/dev/nvidia0"))


def _enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class AcceleratorService:
    """Resolve safe CPU/CUDA choices without importing heavyweight Torch."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()

    @staticmethod
    def _version(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    @staticmethod
    def cuda_device_present() -> bool:
        return any(path.exists() for path in _CUDA_DEVICE_PATHS)

    @staticmethod
    def cuda_device_accessible() -> bool:
        return any(
            path.exists() and os.access(path, os.R_OK | os.W_OK)
            for path in _CUDA_DEVICE_PATHS
        )

    @staticmethod
    def xgboost_cuda_built() -> bool:
        try:
            import xgboost as xgb

            return _enabled(xgb.build_info().get("USE_CUDA"))
        except Exception:
            return False

    def resolve_xgboost_device(self) -> str:
        requested = self.settings.xgboost_device
        if requested == "auto":
            requested = self.settings.ml_device
        if requested == "auto":
            return (
                "cuda:0"
                if self.cuda_device_accessible() and self.xgboost_cuda_built()
                else "cpu"
            )
        if requested.startswith("cuda"):
            if self.cuda_device_accessible() and self.xgboost_cuda_built():
                return "cuda:0" if requested == "cuda" else requested
            if self.settings.cuda_required:
                raise RuntimeError(
                    "CUDA was required but a CUDA-capable XGBoost runtime is unavailable"
                )
            return "cpu"
        return "cpu"

    def status(self) -> dict[str, Any]:
        try:
            selected = self.resolve_xgboost_device()
            error = None
        except RuntimeError as exc:
            selected = "unavailable"
            error = str(exc)

        ready = not self.settings.cuda_required or selected.startswith("cuda")
        return {
            "status": "ok" if ready else "degraded",
            "cuda_required": self.settings.cuda_required,
            "cuda_device_present": self.cuda_device_present(),
            "cuda_device_accessible": self.cuda_device_accessible(),
            "requested_ml_device": self.settings.ml_device,
            "requested_xgboost_device": self.settings.xgboost_device,
            "selected_xgboost_device": selected,
            "xgboost_cuda_built": self.xgboost_cuda_built(),
            "versions": {
                "torch": self._version("torch"),
                "xgboost": self._version("xgboost"),
            },
            "error": error,
        }
