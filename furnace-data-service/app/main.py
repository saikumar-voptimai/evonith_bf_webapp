"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env and source config defaults before modules read os.environ.
_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parent

load_dotenv(_SERVICE_ROOT / ".env")
os.environ.setdefault("FURNACE_CONFIG_DIR", str(_REPO_ROOT / "src" / "config"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as api_v1_router
from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestIdMiddleware
from app.routes import data as legacy_data
from app.routes import dataset as legacy_dataset
from app.routes import health as legacy_health
from furnace_data.runtime_paths import ensure_runtime_dirs, get_runtime_dir

log = logging.getLogger(__name__)


def create_app(backend_settings: BackendSettings | None = None) -> FastAPI:
    """Create the FastAPI backend app."""
    settings = backend_settings or load_backend_settings()
    configure_logging(settings)

    try:
        ensure_runtime_dirs()
    except OSError as exc:
        log.warning("Runtime directories could not be initialized: %s", exc)

    app = FastAPI(
        title=settings.openapi_title,
        description=settings.openapi_description,
        version=settings.openapi_version,
    )
    app.state.backend_settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestIdMiddleware)

    register_exception_handlers(app)
    app.include_router(api_v1_router, prefix=settings.api_prefix)

    if settings.enable_legacy_routes:
        app.include_router(legacy_health.router)
        app.include_router(legacy_data.router)
        app.include_router(legacy_dataset.router)

    log.info("Evonith backend API starting")
    log.info("API prefix: %s", settings.api_prefix)
    log.info("Runtime directory: %s", get_runtime_dir())
    log.info("CORS origins: %s", ", ".join(settings.cors_origins))
    log.info("Legacy routes enabled: %s", settings.enable_legacy_routes)
    return app


app = create_app()
