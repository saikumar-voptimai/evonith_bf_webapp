"""Canonical FastAPI application entrypoint for the Phase 12 layout."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from apps.backend_api.app.core.env import load_backend_env
from furnace_data.config import get_config_dir

# Load .env and config defaults before modules read os.environ.
_REPO_ROOT = Path(__file__).resolve().parents[3]

load_backend_env(_REPO_ROOT)
os.environ.setdefault("FURNACE_CONFIG_DIR", str(get_config_dir()))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.backend_api.app.api.v1.router import router as api_v1_router
from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import register_exception_handlers
from apps.backend_api.app.core.logging import configure_logging
from apps.backend_api.app.core.middleware import AccessLogMiddleware, RequestIdMiddleware
from apps.backend_api.app.routes import dataset as legacy_dataset
from apps.backend_api.app.routes import health as legacy_health
from apps.backend_api.app.services.audit_service import AuditService
from apps.backend_api.app.services.blend_optimizer_service import BlendOptimizerService
from apps.backend_api.app.services.copilot_service import CopilotService
from apps.backend_api.app.services.dependency_status_service import DependencyStatusService
from apps.backend_api.app.services.feedback_service import FeedbackService
from apps.backend_api.app.services.furnacemind_service import FurnaceMindService
from apps.backend_api.app.services.material_balance_service import MaterialBalanceService
from apps.backend_api.app.services.metrics_service import MetricsService
from apps.backend_api.app.services.model_registry_service import ModelRegistryService
from apps.backend_api.app.services.recommendation_service import RecommendationService
from apps.backend_api.app.services.unified_job_service import UnifiedJobService
from furnace_data.runtime_paths import ensure_runtime_dirs

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
    app.state.metrics_service = MetricsService()
    app.state.dependency_status_service = DependencyStatusService(settings)
    app.state.unified_job_service = UnifiedJobService()
    try:
        audit_service = AuditService(settings=settings)
        audit_service.ensure_storage()
        app.state.audit_service = audit_service
    except Exception as exc:
        log.warning("Audit storage could not be initialized: %s", exc)
    try:
        feedback_service = FeedbackService(settings=settings)
        feedback_service.ensure_storage()
        app.state.feedback_service = feedback_service
    except Exception as exc:
        log.warning("Feedback storage could not be initialized: %s", exc)
    try:
        model_registry_service = ModelRegistryService(settings=settings)
        app.state.model_registry_service = model_registry_service
        app.state.material_balance_service = MaterialBalanceService(settings=settings)
        app.state.recommendation_service = RecommendationService(settings=settings)
        app.state.blend_optimizer_service = BlendOptimizerService(
            settings=settings,
            model_registry=model_registry_service,
        )
        app.state.copilot_service = CopilotService(settings=settings)
        furnacemind_service = FurnaceMindService(settings=settings)
        furnacemind_service.ensure_storage()
        app.state.furnacemind_service = furnacemind_service
    except Exception as exc:
        log.warning("Compute services could not be initialized: %s", exc)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Idempotency-Key",
            "X-Request-ID",
        ],
        expose_headers=["Content-Disposition", "X-Dataset-Version", "X-Request-ID"],
    )
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIdMiddleware)

    register_exception_handlers(app)
    app.include_router(api_v1_router, prefix=settings.api_prefix)

    if settings.enable_legacy_routes:
        app.include_router(legacy_health.router)
        # The former standalone /data router accepted physical table names and
        # bypassed the v1 authorization/typed-contract boundary. Its supported
        # functionality is consolidated under /api/v1/data, so never mount it.
        app.include_router(legacy_dataset.router)

    log.info("Evonith backend API starting")
    log.info("API prefix: %s", settings.api_prefix)
    log.info("Runtime directory: [RUNTIME_DIR]")
    log.info("Backend auth enabled: %s", settings.auth_enabled)
    log.info("CORS origins: %s", ", ".join(settings.cors_origins))
    log.info("Legacy routes enabled: %s", settings.enable_legacy_routes)
    return app


app = create_app()