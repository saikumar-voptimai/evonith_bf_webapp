"""Liveness and version."""

from __future__ import annotations

import platform
from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api import __version__
from api.settings import get_settings
from services.release import current_release
from utils import demo

router = APIRouter(prefix="/api", tags=["system"])
STARTED_AT = datetime.now(UTC)


class Health(BaseModel):
    status: str = Field(..., description="'ok' when the process answers")
    service: str = Field("twin-api")


class Version(BaseModel):
    version: str = Field(..., description="twin-api version")
    git_sha: str = Field(..., description="Commit of the running release ('unknown' in development)")
    environment: str = Field(..., description="'dev' or 'prod'")
    python: str = Field(..., description="Python version")
    started_at: datetime = Field(..., description="Process start (UTC)")
    release: str | None = Field(
        None,
        description="Release tag of the running directory (RELEASE.json of deploy/jetson/release.sh); "
        "None in development",
    )
    installed_at: datetime | None = Field(None, description="Installation time of the running release (UTC)")
    frontend_build: str | None = Field(
        None,
        description="Version name of the SPA build installed with the release (frontend/build/_app/version.json); "
        "None without a build",
    )
    product: str = Field("", description="Product name the SPA shows (the demo's in demo mode)")
    furnace: str = Field("", description="Furnace the SPA names, e.g. BF-1 (the demo's in demo mode)")
    site: str = Field("", description="Plant site the SPA names after the furnace (the demo's in demo mode)")
    demo: bool = Field(False, description="Demo mode (BF_DEMO): names replaced, the SPA shows the demo notice")
    demo_notice: str | None = Field(None, description="Notice the SPA shows on every page in demo mode; None otherwise")


@router.get("/health", response_model=Health, summary="Liveness")
def health() -> Health:
    """The process answers. Data-source checks are in ``/api/health/deep`` (#317)."""
    return Health(status="ok")


@router.get("/version", response_model=Version, summary="Running version")
def version() -> Version:
    """The running release: its record (tag, commit, installation time) wins over ``BF_API_GIT_SHA``, which is shared
    by every release directory on the host (#318)."""
    s = get_settings()
    release = current_release()
    return Version(
        version=__version__,
        git_sha=release.commit or s.git_sha,
        environment=s.environment,
        python=platform.python_version(),
        started_at=STARTED_AT,
        release=release.tag,
        installed_at=release.installed_at,
        frontend_build=release.frontend_build,
        **demo.labels(),
        demo=demo.enabled(),
        demo_notice=demo.config().get("banner") if demo.enabled() else None,
    )
