"""Liveness and version."""

from __future__ import annotations

import platform
from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api import __version__
from api.settings import get_settings
from services.release import current_release

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
    )
