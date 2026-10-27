"""twin-api application factory.

    python -m uvicorn api.main:app --app-dir src --port 8100

Routes live in ``api.routers``; every error is an RFC 7807 problem (``api.errors``); every request carries an id
(``api.request_context``); every route outside ``api.auth.PUBLIC_PATHS`` requires a role (``api.auth.require_role``).
With ``BF_API_STATIC_DIR`` set, the API also serves the built SPA with the ``200.html`` fallback (development preview
or a host without Caddy).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.routing import APIRoute

from api import __version__
from api.auth import build_auth_state
from api.errors import ERROR_RESPONSES, ApiError, install_error_handlers
from api.live import LiveState
from api.request_context import RequestContextMiddleware, RequestIdFilter
from api.routers import auth, jobs, stream, system
from api.security.origin import OriginGuardMiddleware
from api.settings import ApiSettings, get_settings

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s] %(message)s"


def _configure_logging(level: str) -> None:
    logger = logging.getLogger("api")
    if any(isinstance(f, RequestIdFilter) for h in logger.handlers for f in h.filters):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False


def _operation_id(route: APIRoute) -> str:
    """``<tag>_<function>`` (e.g. ``system_health``): stable names for the generated TypeScript client."""
    return f"{route.tags[0]}_{route.name}" if route.tags else route.name


def _mount_spa(app: FastAPI, static_dir: Path) -> None:
    root = static_dir.resolve()
    fallback = root / "200.html"

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str, request: Request) -> Response:
        if path == "api" or path.startswith("api/"):
            raise ApiError(404, f"No API route {request.url.path}.")
        target = (root / path).resolve()
        if path and target.is_file() and root in target.parents:
            return FileResponse(target)
        return FileResponse(fallback, media_type="text/html")


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        live = LiveState(settings)
        await live.start()
        app.state.live = live
        try:
            yield
        finally:
            await live.stop()

    app = FastAPI(
        lifespan=lifespan,
        title="BF-02 twin-api",
        version=__version__,
        summary="Data and model API behind the BF-02 Svelte frontend (programme #262).",
        docs_url="/api/docs",
        redoc_url=None,
        swagger_ui_oauth2_redirect_url=None,
        openapi_url="/api/openapi.json",
        responses=ERROR_RESPONSES,
        generate_unique_id_function=_operation_id,
    )
    app.state.auth = build_auth_state(settings)
    # The last middleware added is the outermost: the origin guard runs inside the request context (request id).
    app.add_middleware(OriginGuardMiddleware)
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    app.include_router(system.router)
    app.include_router(auth.router)
    app.include_router(stream.router)
    app.include_router(jobs.router)
    if settings.static_dir is not None and (settings.static_dir / "200.html").is_file():
        _mount_spa(app, settings.static_dir)
    return app


app = create_app()
