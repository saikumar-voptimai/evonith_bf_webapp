"""Backend API settings loaded from environment variables."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_DEFAULT_CORS_ORIGINS = ("http://localhost:8501", "http://127.0.0.1:8501")


class BackendSettings(BaseSettings):
    """Settings for the independently runnable FastAPI backend."""

    api_prefix: str = Field("/api/v1", validation_alias="EVONITH_API_PREFIX")
    backend_env: str = Field("local", validation_alias="EVONITH_BACKEND_ENV")
    backend_log_level: str = Field("INFO", validation_alias="EVONITH_BACKEND_LOG_LEVEL")
    cors_origins: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CORS_ORIGINS),
        validation_alias="BACKEND_CORS_ORIGINS",
    )
    enable_legacy_routes: bool = Field(True, validation_alias="EVONITH_ENABLE_LEGACY_ROUTES")
    openapi_title: str = Field(
        "Evonith BF Backend API",
        validation_alias="EVONITH_OPENAPI_TITLE",
    )
    openapi_version: str = Field("0.1.0", validation_alias="EVONITH_OPENAPI_VERSION")
    openapi_description: str = Field(
        "Versioned backend API for Evonith BF web application",
        validation_alias="EVONITH_OPENAPI_DESCRIPTION",
    )
    auth_enabled: bool = Field(True, validation_alias="EVONITH_AUTH_ENABLED")
    auth_secret_key: str = Field("", validation_alias="EVONITH_AUTH_SECRET_KEY")
    auth_algorithm: str = Field("HS256", validation_alias="EVONITH_AUTH_ALGORITHM")
    auth_access_token_expire_minutes: int = Field(
        480,
        validation_alias="EVONITH_AUTH_ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    auth_require_secret_in_production: bool = Field(
        True,
        validation_alias="EVONITH_AUTH_REQUIRE_SECRET_IN_PRODUCTION",
    )
    auth_allow_legacy_password_hashes: bool = Field(
        True,
        validation_alias="EVONITH_AUTH_ALLOW_LEGACY_PASSWORD_HASHES",
    )
    auth_upgrade_legacy_hash_on_login: bool = Field(
        True,
        validation_alias="EVONITH_AUTH_UPGRADE_LEGACY_HASH_ON_LOGIN",
    )
    auth_password_hash_scheme: str = Field(
        "bcrypt",
        validation_alias="EVONITH_AUTH_PASSWORD_HASH_SCHEME",
    )
    auth_min_password_length: int = Field(
        8,
        validation_alias="EVONITH_AUTH_MIN_PASSWORD_LENGTH",
    )
    auth_bootstrap_admin_enabled: bool = Field(
        False,
        validation_alias="EVONITH_AUTH_BOOTSTRAP_ADMIN_ENABLED",
    )
    feedback_require_auth: bool = Field(
        True,
        validation_alias="EVONITH_FEEDBACK_REQUIRE_AUTH",
    )
    feedback_storage_backend: str = Field(
        "sqlite",
        validation_alias="EVONITH_FEEDBACK_STORAGE_BACKEND",
    )
    feedback_database_url: str = Field(
        "",
        validation_alias="EVONITH_FEEDBACK_DATABASE_URL",
    )
    feedback_max_attachment_mb: int = Field(
        10,
        validation_alias="EVONITH_FEEDBACK_MAX_ATTACHMENT_MB",
    )
    feedback_allowed_attachment_types: list[str] = Field(
        default_factory=lambda: [
            "image/png",
            "image/jpeg",
            "image/webp",
            "application/pdf",
            "text/plain",
            "text/csv",
        ],
        validation_alias="EVONITH_FEEDBACK_ALLOWED_ATTACHMENT_TYPES",
    )
    feedback_allowed_attachment_extensions: list[str] = Field(
        default_factory=lambda: [
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".pdf",
            ".txt",
            ".csv",
        ],
        validation_alias="EVONITH_FEEDBACK_ALLOWED_ATTACHMENT_EXTENSIONS",
    )
    feedback_max_attachments_per_ticket: int = Field(
        10,
        validation_alias="EVONITH_FEEDBACK_MAX_ATTACHMENTS_PER_TICKET",
    )
    feedback_default_status: str = Field(
        "open",
        validation_alias="EVONITH_FEEDBACK_DEFAULT_STATUS",
    )
    feedback_allowed_statuses: list[str] = Field(
        default_factory=lambda: ["open", "in_progress", "resolved", "closed", "rejected"],
        validation_alias="EVONITH_FEEDBACK_ALLOWED_STATUSES",
    )
    feedback_allowed_priorities: list[str] = Field(
        default_factory=lambda: ["low", "medium", "high", "critical"],
        validation_alias="EVONITH_FEEDBACK_ALLOWED_PRIORITIES",
    )
    feedback_ticket_id_prefix: str = Field(
        "FB",
        validation_alias="EVONITH_FEEDBACK_TICKET_ID_PREFIX",
    )
    feedback_enable_legacy_read_fallback: bool = Field(
        True,
        validation_alias="EVONITH_FEEDBACK_ENABLE_LEGACY_READ_FALLBACK",
    )
    compute_require_auth: bool = Field(
        True,
        validation_alias="EVONITH_COMPUTE_REQUIRE_AUTH",
    )
    compute_max_preview_rows: int = Field(
        500,
        validation_alias="EVONITH_COMPUTE_MAX_PREVIEW_ROWS",
    )
    compute_max_json_rows: int = Field(
        5000,
        validation_alias="EVONITH_COMPUTE_MAX_JSON_ROWS",
    )
    compute_max_input_rows: int = Field(
        50000,
        validation_alias="EVONITH_COMPUTE_MAX_INPUT_ROWS",
    )
    compute_job_threshold_rows: int = Field(
        5000,
        validation_alias="EVONITH_COMPUTE_JOB_THRESHOLD_ROWS",
    )
    compute_job_ttl_hours: int = Field(
        24,
        validation_alias="EVONITH_COMPUTE_JOB_TTL_HOURS",
    )
    compute_artifact_ttl_hours: int = Field(
        24,
        validation_alias="EVONITH_COMPUTE_ARTIFACT_TTL_HOURS",
    )
    compute_max_seconds: int = Field(
        180,
        validation_alias="EVONITH_COMPUTE_MAX_SECONDS",
    )
    compute_threadpool_workers: int = Field(
        1,
        validation_alias="EVONITH_COMPUTE_THREADPOOL_WORKERS",
    )
    compute_export_format: str = Field(
        "csv",
        validation_alias="EVONITH_COMPUTE_EXPORT_FORMAT",
    )
    model_dir: str = Field("", validation_alias="EVONITH_MODEL_DIR")
    model_lazy_load: bool = Field(True, validation_alias="EVONITH_MODEL_LAZY_LOAD")
    model_cache_max_items: int = Field(
        2,
        validation_alias="EVONITH_MODEL_CACHE_MAX_ITEMS",
    )
    model_load_timeout_seconds: int = Field(
        30,
        validation_alias="EVONITH_MODEL_LOAD_TIMEOUT_SECONDS",
    )
    model_allow_missing_optional_models: bool = Field(
        True,
        validation_alias="EVONITH_MODEL_ALLOW_MISSING_OPTIONAL_MODELS",
    )
    material_balance_config_source: str = Field(
        "file",
        validation_alias="EVONITH_MATERIAL_BALANCE_CONFIG_SOURCE",
    )
    material_balance_allow_runtime_config: bool = Field(
        False,
        validation_alias="EVONITH_MATERIAL_BALANCE_ALLOW_RUNTIME_CONFIG",
    )
    recommendations_enable_explanations: bool = Field(
        True,
        validation_alias="EVONITH_RECOMMENDATIONS_ENABLE_EXPLANATIONS",
    )
    recommendations_max_items: int = Field(
        50,
        validation_alias="EVONITH_RECOMMENDATIONS_MAX_ITEMS",
    )
    blend_optimizer_max_candidates: int = Field(
        100,
        validation_alias="EVONITH_BLEND_OPTIMIZER_MAX_CANDIDATES",
    )
    blend_optimizer_max_iterations: int = Field(
        1000,
        validation_alias="EVONITH_BLEND_OPTIMIZER_MAX_ITERATIONS",
    )
    blend_optimizer_timeout_seconds: int = Field(
        120,
        validation_alias="EVONITH_BLEND_OPTIMIZER_TIMEOUT_SECONDS",
    )
    blend_optimizer_enable_model_predictions: bool = Field(
        True,
        validation_alias="EVONITH_BLEND_OPTIMIZER_ENABLE_MODEL_PREDICTIONS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        prefix = (value or "/api/v1").strip()
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        return prefix.rstrip("/") or "/api/v1"

    @field_validator("backend_log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return (value or "INFO").strip().upper()

    @field_validator("auth_algorithm")
    @classmethod
    def normalize_auth_algorithm(cls, value: str) -> str:
        return (value or "HS256").strip().upper()

    @field_validator("auth_password_hash_scheme")
    @classmethod
    def normalize_password_hash_scheme(cls, value: str) -> str:
        return (value or "bcrypt").strip().lower()

    @field_validator(
        "auth_access_token_expire_minutes",
        "auth_min_password_length",
        "feedback_max_attachment_mb",
        "feedback_max_attachments_per_ticket",
        "compute_max_preview_rows",
        "compute_max_json_rows",
        "compute_max_input_rows",
        "compute_job_threshold_rows",
        "compute_job_ttl_hours",
        "compute_artifact_ttl_hours",
        "compute_max_seconds",
        "compute_threadpool_workers",
        "model_cache_max_items",
        "model_load_timeout_seconds",
        "recommendations_max_items",
        "blend_optimizer_max_candidates",
        "blend_optimizer_max_iterations",
        "blend_optimizer_timeout_seconds",
    )
    @classmethod
    def require_positive_int(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("feedback_storage_backend", "compute_export_format")
    @classmethod
    def normalize_lower_string(cls, value: str) -> str:
        return str(value or "").strip().lower()

    @field_validator(
        "feedback_default_status",
        "feedback_ticket_id_prefix",
        "model_dir",
        "material_balance_config_source",
    )
    @classmethod
    def normalize_feedback_string(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator(
        "feedback_allowed_attachment_types",
        "feedback_allowed_attachment_extensions",
        "feedback_allowed_statuses",
        "feedback_allowed_priorities",
        mode="before",
    )
    @classmethod
    def parse_feedback_csv(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return list(_DEFAULT_CORS_ORIGINS)
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(origin).strip() for origin in value if str(origin).strip()]
        return list(_DEFAULT_CORS_ORIGINS)


def load_backend_settings() -> BackendSettings:
    """Load backend settings from the current environment."""
    return BackendSettings()
