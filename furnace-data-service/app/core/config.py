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
