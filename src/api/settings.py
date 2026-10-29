"""twin-api settings from the environment (prefix ``BF_API_``) and the repository ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from services.session_tokens import SessionSettings

REPO_ROOT = Path(__file__).resolve().parents[2]


class ApiSettings(SessionSettings):
    """twin-api settings; the session fields of ``SessionSettings`` are shared with the Streamlit pages."""

    model_config = SettingsConfigDict(env_prefix="BF_API_", env_file=REPO_ROOT / ".env", extra="ignore")

    environment: str = "dev"  # "dev" or "prod"
    git_sha: str = "unknown"  # fallback; a release directory's RELEASE.json (deploy/jetson/release.sh, #318) wins
    # Serve the built SPA from the API when no Caddy is in front (development preview, first Jetson boot).
    static_dir: Path | None = None
    log_level: str = "INFO"

    # Sessions (SV-0.4, #273); jwt_secret and cookie_name in SessionSettings (SV-6.6, #320)
    session_hours: float = Field(12.0, gt=0, le=24, description="Lifetime of a session token")
    cookie_secure: bool = Field(True, description="Secure cookie flag; false only for plain-HTTP development access")
    trusted_proxies: str = Field(
        "127.0.0.1,::1", description="Comma-separated peer addresses whose X-Forwarded-For is trusted (Caddy)"
    )
    login_max_failures: int = Field(5, ge=1, description="Failed sign-ins allowed per client address per window")
    login_window_s: float = Field(60.0, gt=0, description="Window of the failed sign-in limit")
    identity_db_url: str | None = Field(None, description="Identity database; default DATABASE_URL")

    # Live events and jobs (SV-0.7 #276, SV-0.8 #277); redis_url in SessionSettings
    stream_heartbeat_s: float = Field(15.0, gt=0, le=60, description="Heartbeat interval of /api/stream")
    # FurnaceMind (SV-5.2 #313)
    fm_warm_up: bool = Field(
        True, description="Load the FurnaceMind agent stack in a background thread at start (first turn without it)"
    )

    @property
    def trusted_proxy_set(self) -> frozenset[str]:
        return frozenset(p.strip() for p in self.trusted_proxies.split(",") if p.strip())


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()
