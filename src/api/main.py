"""twin-api application factory (skeleton of #270; routers, error model and typed client follow in #272)."""

from __future__ import annotations

from fastapi import FastAPI

from api import __version__


def create_app() -> FastAPI:
    app = FastAPI(
        title="BF-02 twin-api",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    @app.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        """Liveness: the process answers."""
        return {"status": "ok", "service": "twin-api"}

    return app


app = create_app()
