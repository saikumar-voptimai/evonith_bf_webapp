"""API v1 health, readiness, and runtime status endpoints."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Request

from app.api.v1.schemas.health import HealthResponse, ReadinessResponse, RuntimeStatusResponse
from app.core.config import BackendSettings
from app.core.errors import ApiError
from app.core.responses import get_request_id
from furnace_data.runtime_paths import ensure_runtime_dirs, get_runtime_dir

router = APIRouter(tags=["health"])


def _settings(request: Request) -> BackendSettings:
    return request.app.state.backend_settings


def _runtime_checks() -> tuple[dict[str, str], dict[str, Path]]:
    paths = ensure_runtime_dirs()
    checks = {
        "runtime_dir": "ok" if paths["runtime"].exists() else "missing",
        "config": "ok",
    }
    for name, path in paths.items():
        if name == "runtime":
            continue
        checks[name] = "ok" if path.exists() and path.is_dir() else "missing"
    return checks, paths


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    settings = _settings(request)
    return HealthResponse(
        request_id=get_request_id(request),
        data={
            "status": "ok",
            "service": "evonith-backend-api",
            "api_version": "v1",
            "environment": settings.backend_env,
        },
    )


@router.get("/readiness", response_model=ReadinessResponse)
def readiness(request: Request):
    try:
        checks, _paths = _runtime_checks()
    except OSError as exc:
        raise ApiError(
            code="RUNTIME_NOT_READY",
            message="Runtime directory is not ready",
            status_code=503,
            details={"error": str(exc)},
        ) from exc

    failed = {name: status for name, status in checks.items() if status != "ok"}
    if failed:
        raise ApiError(
            code="RUNTIME_NOT_READY",
            message="Runtime directory is not ready",
            status_code=503,
            details={"checks": failed},
        )

    return ReadinessResponse(
        request_id=get_request_id(request),
        data={"status": "ready", "checks": {"runtime_dir": "ok", "config": "ok"}},
    )


@router.get("/status/runtime", response_model=RuntimeStatusResponse)
def runtime_status(request: Request):
    settings = _settings(request)
    checks, paths = _runtime_checks()
    runtime_dir = get_runtime_dir()
    disk: dict[str, int] = {}
    try:
        usage = shutil.disk_usage(runtime_dir)
        disk = {"total": usage.total, "used": usage.used, "free": usage.free}
    except OSError:
        disk = {}

    expose_path = settings.backend_env.lower() in {"local", "dev", "development", "test"}
    directories = {
        name: str(path.relative_to(runtime_dir)) if path != runtime_dir else "."
        for name, path in paths.items()
    }
    writable = "ok" if os.access(runtime_dir, os.W_OK) else "not_writable"
    checks["writable"] = writable

    return RuntimeStatusResponse(
        request_id=get_request_id(request),
        data={
            "status": "ok" if writable == "ok" else "degraded",
            "runtime_dir": str(runtime_dir) if expose_path else None,
            "checks": checks,
            "directories": directories,
            "disk": disk,
        },
    )
