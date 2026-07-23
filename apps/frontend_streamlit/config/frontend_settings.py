"""Frontend settings for the Streamlit-to-backend migration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_BACKEND_API_BASE_URL = "http://localhost:8080/api/v1"

PAGE_API_FLAG_ENV_VARS: dict[str, str] = {
    "auth": "USE_BACKEND_API_AUTH",
    "admin": "USE_BACKEND_API_ADMIN",
    "welcome": "USE_BACKEND_API_WELCOME",
    "data_explorer": "USE_BACKEND_API_DATA_EXPLORER",
    "datasets": "USE_BACKEND_API_DATASETS",
    "feedback": "USE_BACKEND_API_FEEDBACK",
    "material_balance": "USE_BACKEND_API_MATERIAL_BALANCE",
    "recommendations": "USE_BACKEND_API_RECOMMENDATIONS",
    "blend_optimizer": "USE_BACKEND_API_BLEND_OPTIMIZER",
    "copilot": "USE_BACKEND_API_COPILOT",
    "furnacemind": "USE_BACKEND_API_FURNACEMIND",
    "ops": "USE_BACKEND_API_OPS",
}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_env_bool(name: str) -> bool | None:
    """Return an explicit bool, preserving an unset migration flag as None."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
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
    show_advanced_backend_status: bool = False
    backend_api_health_path: str = "/health"
    backend_api_readiness_path: str = "/readiness"
    data_api_max_preview_rows: int = 500
    data_api_max_json_rows: int = 5000
    data_api_export_format: str = "csv"
    data_api_job_ttl_hours: int = 24
    data_api_artifact_ttl_hours: int = 24
    page_api_flags: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_api_base_url", self.backend_api_base_url.rstrip("/"))


def load_frontend_settings() -> FrontendSettings:
    """Load frontend settings from environment variables."""
    page_flags = {
        key: _env_bool(env_name, False)
        for key, env_name in PAGE_API_FLAG_ENV_VARS.items()
    }
    # DATASETS was a partial migration flag. Data Explorer must use one complete
    # gateway mode, so retain it only as an alias for deployments that have not
    # yet set the new flag. An explicit new flag always wins.
    explicit_data_explorer = _optional_env_bool("USE_BACKEND_API_DATA_EXPLORER")
    if explicit_data_explorer is None:
        page_flags["data_explorer"] = page_flags["datasets"]
    else:
        page_flags["data_explorer"] = explicit_data_explorer
    return FrontendSettings(
        backend_api_base_url=_env("BACKEND_API_BASE_URL", DEFAULT_BACKEND_API_BASE_URL),
        use_backend_api=_env_bool("USE_BACKEND_API", False),
        backend_api_timeout_seconds=_env_float("BACKEND_API_TIMEOUT_SECONDS", 30.0),
        backend_api_connect_timeout_seconds=_env_float("BACKEND_API_CONNECT_TIMEOUT_SECONDS", 5.0),
        backend_api_max_retries=max(0, _env_int("BACKEND_API_MAX_RETRIES", 1)),
        backend_api_verify_ssl=_env_bool("BACKEND_API_VERIFY_SSL", True),
        show_backend_status_badge=_env_bool("SHOW_BACKEND_STATUS_BADGE", True),
        show_advanced_backend_status=_env_bool("SHOW_ADVANCED_BACKEND_STATUS", False),
        backend_api_health_path=_env("BACKEND_API_HEALTH_PATH", "/health"),
        backend_api_readiness_path=_env("BACKEND_API_READINESS_PATH", "/readiness"),
        data_api_max_preview_rows=max(1, _env_int("DATA_API_MAX_PREVIEW_ROWS", 500)),
        data_api_max_json_rows=max(1, _env_int("DATA_API_MAX_JSON_ROWS", 5000)),
        data_api_export_format=_env("DATA_API_EXPORT_FORMAT", "csv").lower(),
        data_api_job_ttl_hours=max(1, _env_int("DATA_API_JOB_TTL_HOURS", 24)),
        data_api_artifact_ttl_hours=max(1, _env_int("DATA_API_ARTIFACT_TTL_HOURS", 24)),
        page_api_flags=page_flags,
    )


def is_backend_api_enabled(feature: str | None = None) -> bool:
    """Return whether API mode is enabled globally or for a page feature."""
    settings = load_frontend_settings()
    if feature is None:
        return settings.use_backend_api
    return settings.page_api_flags.get(feature, False)
