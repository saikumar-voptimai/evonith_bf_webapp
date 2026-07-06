"""Integration smoke for Phase 10 operational hardening and regressions."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = REPO_ROOT / "furnace-data-service"
for path in (str(SERVICE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

loaded_app = sys.modules.get("app")
loaded_path = str(getattr(loaded_app, "__file__", "")) if loaded_app else ""
if loaded_path.endswith("src\\app.py") or loaded_path.endswith("src/app.py"):
    del sys.modules["app"]

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.main import create_app
from apps.backend_api.app.repositories.user_repository import UserRecord
from apps.backend_api.app.services.admin_service import AdminService
from apps.backend_api.app.services.auth_service import AuthService
from apps.backend_api.app.services.password_service import PasswordService
from apps.backend_api.app.services.token_service import TokenService


class MemoryUserRepository:
    def __init__(self) -> None:
        self.users: dict[str, UserRecord] = {}

    def find_by_username_or_email(self, identifier: str) -> UserRecord | None:
        return next((user for user in self.users.values() if user.username == identifier), None)

    def find_by_id(self, user_id: str) -> UserRecord | None:
        return self.users.get(str(user_id))

    def list_users(self, *, limit: int = 100, offset: int = 0) -> list[UserRecord]:
        return list(self.users.values())[offset : offset + limit]

    def count_users(self) -> int:
        return len(self.users)

    def create_user(self, *, username: str, password_hash: str, role: str, email=None, full_name=None, is_active=True):
        user = UserRecord(
            id=str(uuid4()),
            username=username,
            password_hash=password_hash,
            role=role,
            is_active=is_active,
            email=email,
            full_name=full_name,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.users[user.id] = user
        return user

    def update_user(self, user_id: str, **changes):
        return self.users.get(str(user_id))

    def update_password_hash(self, user_id: str, password_hash: str) -> None:
        user = self.users[str(user_id)]
        self.users[str(user_id)] = UserRecord(**{**user.__dict__, "password_hash": password_hash})

    def record_login(self, user_id: str) -> None:
        return None


def test_phase10_ops_and_phase9_furnacemind_regression(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    settings = BackendSettings(
        api_prefix="/api/v1",
        backend_env="test",
        auth_secret_key="test-secret",
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
        status = client.get("/api/v1/status", headers=headers)
        metrics = client.get("/api/v1/metrics", headers=headers)
        runtime = client.get("/api/v1/status/runtime/details", headers=headers)
        cleanup = client.post("/api/v1/ops/cleanup/dry-run", headers=headers)
        audit = client.get("/api/v1/ops/audit/events", headers=headers)
        errors = client.get("/api/v1/ops/error-codes", headers=headers)

        fm_config = client.get("/api/v1/furnacemind/config", headers=headers)
        conversation = client.post("/api/v1/furnacemind/conversations", json={"title": "Ops check"}, headers=headers)
        conversation_id = conversation.json()["data"]["id"]
        fm_run = client.post(
            f"/api/v1/furnacemind/conversations/{conversation_id}/runs",
            json={"message": "Check safe fallback", "allow_llm": False},
            headers=headers,
        )
        run_id = fm_run.json()["data"]["id"]
        fm_events = client.get(f"/api/v1/furnacemind/runs/{run_id}/events", headers=headers)
        copilot = client.get("/api/v1/copilot/config")
        feedback = client.get("/api/v1/feedback/config")
        data = client.get("/api/v1/data/sources")

    assert health.status_code == 200
    assert readiness.status_code == 200
    assert login.status_code == 200
    assert status.status_code == 200
    assert metrics.status_code == 200
    assert runtime.status_code == 200
    assert str(tmp_path) not in str(runtime.json())
    assert cleanup.status_code == 200
    assert cleanup.json()["data"]["dry_run"] is True
    assert audit.status_code == 200
    assert "adminpass" not in str(audit.json())
    assert errors.status_code == 200
    assert fm_config.status_code == 200
    assert conversation.status_code == 200
    assert fm_run.status_code == 200
    assert fm_run.json()["data"]["status"] == "completed"
    assert [event["sequence"] for event in fm_events.json()["data"]] == sorted(
        event["sequence"] for event in fm_events.json()["data"]
    )
    assert "RUNTIME_DIR" not in str(fm_run.json())
    assert copilot.status_code == 200
    assert feedback.status_code == 200
    assert data.status_code == 200
