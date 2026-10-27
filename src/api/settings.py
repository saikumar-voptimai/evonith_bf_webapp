"""twin-api settings from the environment (prefix ``BF_API_``) and the repository ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BF_API_", env_file=REPO_ROOT / ".env", extra="ignore")

    environment: str = "dev"  # "dev" or "prod"
    git_sha: str = "unknown"  # set by the release script (#318)
    # Serve the built SPA from the API when no Caddy is in front (development preview, first Jetson boot).
    static_dir: Path | None = None
    log_level: str = "INFO"

    # Sessions (SV-0.4, #273)
    jwt_secret: SecretStr | None = Field(
        None, description="HS256 signing key, at least 32 characters; required in prod (generate: openssl rand -hex 32)"
    )
    session_hours: float = Field(12.0, gt=0, le=24, description="Lifetime of a session token")
    cookie_name: str = "bf_session"
    cookie_secure: bool = Field(True, description="Secure cookie flag; false only for plain-HTTP development access")
    trusted_proxies: str = Field(
        "127.0.0.1,::1", description="Comma-separated peer addresses whose X-Forwarded-For is trusted (Caddy)"
    )
    login_max_failures: int = Field(5, ge=1, description="Failed sign-ins allowed per client address per window")
    login_window_s: float = Field(60.0, gt=0, description="Window of the failed sign-in limit")
    identity_db_url: str | None = Field(None, description="Identity database; default DATABASE_URL")

    # Live events and jobs (SV-0.7 #276, SV-0.8 #277)
    redis_url: str | None = Field(
        None,
        validation_alias=AliasChoices("BF_REDIS_URL", "BF_API_REDIS_URL", "redis_url"),
        description="Redis shared with the worker, e.g. redis://127.0.0.1:6379/0; without it events stay in-process",
    )
    stream_heartbeat_s: float = Field(15.0, gt=0, le=60, description="Heartbeat interval of /api/stream")

    @field_validator("jwt_secret")
    @classmethod
    def _secret_length(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("BF_API_JWT_SECRET must have at least 32 characters")
        return value

    @property
    def trusted_proxy_set(self) -> frozenset[str]:
        return frozenset(p.strip() for p in self.trusted_proxies.split(",") if p.strip())


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()
