"""Frontend settings for the Streamlit-to-backend migration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_BACKEND_API_BASE_URL = "http://localhost:8080/api/v1"

PAGE_API_FLAG_ENV_VARS: dict[str, str] = {
    "data_explorer": "USE_BACKEND_API_DATA_EXPLORER",
    "datasets": "USE_BACKEND_API_DATASETS",
    "feedback": "USE_BACKEND_API_FEEDBACK",
    "material_balance": "USE_BACKEND_API_MATERIAL_BALANCE",
    "recommendations": "USE_BACKEND_API_RECOMMENDATIONS",
    "blend_optimizer": "USE_BACKEND_API_BLEND_OPTIMIZER",
    "copilot": "USE_BACKEND_API_COPILOT",
    "furnacemind": "USE_BACKEND_API_FURNACEMIND",
}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class FrontendSettings:
    backend_api_base_url: str = DEFAULT_BACKEND_API_BASE_URL
    use_backend_api: bool = False
    backend_api_timeout_seconds: float = 30.0
    backend_api_connect_timeout_seconds: float = 5.0
    backend_api_max_retries: int = 1
    backend_api_verify_ssl: bool = True
    show_backend_status_badge: bool = True
    backend_api_health_path: str = "/health"
    backend_api_readiness_path: str = "/readiness"
    page_api_flags: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_api_base_url", self.backend_api_base_url.rstrip("/"))


def load_frontend_settings() -> FrontendSettings:
    """Load frontend settings from environment variables."""
    page_flags = {
        key: _env_bool(env_name, False)
        for key, env_name in PAGE_API_FLAG_ENV_VARS.items()
    }
    return FrontendSettings(
        backend_api_base_url=_env("BACKEND_API_BASE_URL", DEFAULT_BACKEND_API_BASE_URL),
        use_backend_api=_env_bool("USE_BACKEND_API", False),
        backend_api_timeout_seconds=_env_float("BACKEND_API_TIMEOUT_SECONDS", 30.0),
        backend_api_connect_timeout_seconds=_env_float("BACKEND_API_CONNECT_TIMEOUT_SECONDS", 5.0),
        backend_api_max_retries=max(0, _env_int("BACKEND_API_MAX_RETRIES", 1)),
        backend_api_verify_ssl=_env_bool("BACKEND_API_VERIFY_SSL", True),
        show_backend_status_badge=_env_bool("SHOW_BACKEND_STATUS_BADGE", True),
        backend_api_health_path=_env("BACKEND_API_HEALTH_PATH", "/health"),
        backend_api_readiness_path=_env("BACKEND_API_READINESS_PATH", "/readiness"),
        page_api_flags=page_flags,
    )


def is_backend_api_enabled(feature: str | None = None) -> bool:
    """Return whether API mode is enabled globally or for a page feature."""
    settings = load_frontend_settings()
    if feature is None:
        return settings.use_backend_api
    return settings.page_api_flags.get(feature, False)
