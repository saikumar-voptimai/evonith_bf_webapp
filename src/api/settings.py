"""twin-api settings from the environment (prefix ``BF_API_``) and the repository ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BF_API_", env_file=REPO_ROOT / ".env", extra="ignore")

    environment: str = "dev"  # "dev" or "prod"
    git_sha: str = "unknown"  # set by the release script (#318)
    # Serve the built SPA from the API when no Caddy is in front (development preview, first Jetson boot).
    static_dir: Path | None = None
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()
