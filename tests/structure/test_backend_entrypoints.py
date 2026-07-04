"""Phase 12 backend entrypoint tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = REPO_ROOT / "furnace-data-service"


def test_canonical_backend_entrypoint_imports(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    from apps.backend_api.app.main import app

    assert app.title == "Evonith BF Backend API"


def test_canonical_backend_health_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    from apps.backend_api.app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_legacy_backend_entrypoint_reexports_canonical(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    if str(SERVICE_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVICE_ROOT))

    canonical = importlib.import_module("apps.backend_api.app.main")
    legacy = importlib.import_module("app.main")

    assert legacy.app is canonical.app
    assert legacy.create_app is canonical.create_app

