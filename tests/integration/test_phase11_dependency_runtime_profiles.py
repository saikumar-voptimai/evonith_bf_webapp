"""Integration smoke tests for Phase 11 dependency/runtime profiles."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_ROOT = REPO_ROOT / "tests" / "integration"
for path in (str(REPO_ROOT), str(INTEGRATION_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.main import create_app
from test_phase10_ops_flow import MemoryUserRepository
from apps.backend_api.app.services.admin_service import AdminService
from apps.backend_api.app.services.auth_service import AuthService
from apps.backend_api.app.services.password_service import PasswordService
from apps.backend_api.app.services.token_service import TokenService


def _run_script(name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, f"scripts/{name}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_phase11_edge_like_profile_and_phase10_regressions(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    settings = BackendSettings(
        api_prefix="/api/v1",
        backend_env="test",
        runtime_profile="edge",
        edge_mode=True,
        auth_secret_key="test-secret",
        enable_optional_ai=False,
        enable_optional_vector=False,
        feedback_require_auth=False,
        compute_require_auth=False,
        copilot_require_auth=False,
        furnacemind_require_auth=True,
        furnacemind_llm_enabled=False,
        furnacemind_memory_enabled=False,
        furnacemind_tools_enabled=False,
    )
    app = create_app(settings)
    password_service = PasswordService(settings)
    token_service = TokenService(settings)
    repository = MemoryUserRepository()
    repository.create_user(
        username="admin",
        password_hash=password_service.hash_password("adminpass"),
        role="admin",
    )
    app.state.auth_service = AuthService(
        repository=repository,
        password_service=password_service,
        token_service=token_service,
        settings=settings,
    )
    app.state.admin_service = AdminService(
        repository=repository,
        password_service=password_service,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/api/v1/health")
        readiness = client.get("/api/v1/readiness")
        login = client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass"})
        headers = {"Authorization": f"Bearer {login.json()['data']['access_token']}"}
        config = client.get("/api/v1/status/config", headers=headers)
        dependencies = client.get("/api/v1/status/dependencies", headers=headers)
        status = client.get("/api/v1/status", headers=headers)
        metrics = client.get("/api/v1/metrics", headers=headers)
        copilot = client.get("/api/v1/copilot/config")
        furnacemind = client.get("/api/v1/furnacemind/config", headers=headers)
        material_balance = client.get("/api/v1/material-balance/config")
        recommendations = client.get("/api/v1/recommendations/config")

    assert health.status_code == 200
    assert readiness.status_code == 200
    assert login.status_code == 200
    assert config.json()["data"]["runtime_profile"] == "edge"
    assert dependencies.json()["data"]["profile"]["optional_features"]["vector"] is False
    assert status.status_code == 200
    assert metrics.status_code == 200
    assert copilot.status_code == 200
    assert furnacemind.status_code == 200
    assert material_balance.status_code == 200
    assert recommendations.status_code == 200


def test_phase11_scripts_pass_integration():
    for script in (
        "check_backend_minimal_startup.py",
        "check_frontend_api_imports.py",
        "check_import_boundaries.py",
        "check_dependency_profiles.py",
    ):
        result = _run_script(script)
        assert result.returncode == 0, result.stdout + result.stderr
