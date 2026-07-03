"""Test fixtures for the Phase 2 backend API foundation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parent

for path in (str(SERVICE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("INFLUX_ONLINE_TOKEN", "test_online_token")
os.environ.setdefault("INFLUX_OFFLINE_TOKEN", "test_offline_token")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")

loaded_app = sys.modules.get("app")
loaded_path = str(getattr(loaded_app, "__file__", "")) if loaded_app else ""
if loaded_path.endswith("src\\app.py") or loaded_path.endswith("src/app.py"):
    del sys.modules["app"]


@pytest.fixture()
def app_factory(monkeypatch, tmp_path):
    def _factory(*, legacy_routes: bool = True, cors_origins: list[str] | None = None):
        monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

        from app.core.config import BackendSettings
        from app.main import create_app

        settings = BackendSettings(
            api_prefix="/api/v1",
            backend_env="test",
            backend_log_level="WARNING",
            cors_origins=cors_origins or ["http://localhost:8501", "http://127.0.0.1:8501"],
            enable_legacy_routes=legacy_routes,
            openapi_title="Evonith BF Backend API",
            openapi_version="0.1.0",
            openapi_description="Versioned backend API for Evonith BF web application",
        )
        return create_app(settings)

    return _factory


@pytest.fixture()
def client(app_factory):
    with TestClient(app_factory(), raise_server_exceptions=False) as test_client:
        yield test_client
