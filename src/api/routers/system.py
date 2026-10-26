"""Liveness and version."""

from __future__ import annotations

import platform
from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api import __version__
from api.settings import get_settings

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


@router.get("/health", response_model=Health, summary="Liveness")
def health() -> Health:
    """The process answers. Data-source checks are in ``/api/health/deep`` (#317)."""
    return Health(status="ok")


@router.get("/version", response_model=Version, summary="Running version")
def version() -> Version:
    s = get_settings()
    return Version(
        version=__version__,
        git_sha=s.git_sha,
        environment=s.environment,
        python=platform.python_version(),
        started_at=STARTED_AT,
    )
