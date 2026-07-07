"""Safe optional dependency probes and lazy imports for runtime profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.util
from functools import lru_cache
from types import ModuleType
from typing import Any

from apps.backend_api.app.core.errors import ApiError


FEATURE_GROUP_MODULES: dict[str, tuple[str, ...]] = {
    "backend-ai": ("openai", "anthropic", "tiktoken"),
    "backend-ml": ("joblib", "sklearn", "scipy", "xgboost"),
    "backend-vector": ("qdrant_client", "sentence_transformers", "torch"),
    "backend-documents": ("fitz", "docx", "pptx", "pypdf"),
    "frontend": ("streamlit", "plotly", "pydeck", "pygwalker"),
}

MODULE_FEATURE_GROUP: dict[str, str] = {
    module: feature_group
    for feature_group, modules in FEATURE_GROUP_MODULES.items()
    for module in modules
}

MODULE_INSTALL_HINTS: dict[str, str] = {
    "fitz": "backend-documents",
    "docx": "backend-documents",
    "pptx": "backend-documents",
    "openai": "backend-ai",
    "anthropic": "backend-ai",
    "qdrant_client": "backend-vector",
    "sentence_transformers": "backend-vector",
    "torch": "backend-vector",
    "joblib": "backend-ml",
    "sklearn": "backend-ml",
    "scipy": "backend-ml",
    "xgboost": "backend-ml",
}


@dataclass(frozen=True)
class DependencyStatus:
    module: str
    feature_group: str
    available: bool
    status: str
    install_group: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=128)
def is_module_available(module_name: str) -> bool:
    """Return whether a module can be found without importing it."""
    module_name = str(module_name or "").strip()
    if not module_name:
        return False
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def optional_dependency_error(module_name: str, feature_group: str | None = None) -> ApiError:
    """Create a safe structured API error for a missing optional dependency."""
    module_name = str(module_name or "").strip()
    feature_group = str(feature_group or MODULE_FEATURE_GROUP.get(module_name) or "optional").strip()
    install_group = MODULE_INSTALL_HINTS.get(module_name, feature_group)
    return ApiError(
        "DEPENDENCY_OPTIONAL_NOT_INSTALLED",
        f"Optional dependency for {feature_group} is not installed.",
        status_code=503,
        details={
            "module": module_name,
            "feature_group": feature_group,
            "install_group": install_group,
            "recommendation": f"Install the {install_group} dependency profile for this feature.",
        },
    )


def require_optional_module(module_name: str, feature_group: str | None = None) -> ModuleType:
    """Import an optional module only at the call site that needs it."""
    module_name = str(module_name or "").strip()
    feature_group = str(feature_group or MODULE_FEATURE_GROUP.get(module_name) or "optional").strip()
    if not is_module_available(module_name):
        raise optional_dependency_error(module_name, feature_group)
    try:
        return importlib.import_module(module_name)
    except ApiError:
        raise
    except ModuleNotFoundError as exc:
        raise optional_dependency_error(module_name, feature_group) from exc
    except Exception as exc:
        raise ApiError(
            "DEPENDENCY_IMPORT_FAILED",
            f"Optional dependency for {feature_group} could not be imported.",
            status_code=503,
            details={
                "module": module_name,
                "feature_group": feature_group,
                "install_group": MODULE_INSTALL_HINTS.get(module_name, feature_group),
            },
        ) from exc


def get_optional_dependency_status() -> list[dict[str, Any]]:
    """Return safe status records for known optional dependency modules."""
    statuses: list[dict[str, Any]] = []
    for feature_group, modules in FEATURE_GROUP_MODULES.items():
        for module_name in modules:
            available = is_module_available(module_name)
            statuses.append(
                DependencyStatus(
                    module=module_name,
                    feature_group=feature_group,
                    available=available,
                    status="available" if available else "unavailable",
                    install_group=MODULE_INSTALL_HINTS.get(module_name, feature_group),
                    message=(
                        "Optional dependency is importable."
                        if available
                        else f"Install {MODULE_INSTALL_HINTS.get(module_name, feature_group)} to enable this feature."
                    ),
                ).to_dict()
            )
    return statuses


def clear_optional_dependency_cache() -> None:
    """Clear cached module-availability probes for tests."""
    is_module_available.cache_clear()
