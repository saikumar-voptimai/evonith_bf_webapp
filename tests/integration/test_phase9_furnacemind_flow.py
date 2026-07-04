"""Integration smoke for Phase 9 FurnaceMind API flow."""

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

from app.core.config import BackendSettings
from app.main import create_app
from app.repositories.user_repository import UserRecord
from app.services.admin_service import AdminService
from app.services.auth_service import AuthService
from app.services.password_service import PasswordService
from app.services.token_service import TokenService


class MemoryUserRepository:
    def __init__(self) -> None:
        self.users: dict[str, UserRecord] = {}

    def find_by_username_or_email(self, identifier: str) -> UserRecord | None:
        return next(
            (user for user in self.users.values() if user.username == identifier or user.email == identifier),
            None,
        )

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


def test_phase9_furnacemind_authenticated_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    settings = BackendSettings(
        api_prefix="/api/v1",
        backend_env="test",
        auth_secret_key="test-secret",
        furnacemind_require_auth=True,
        furnacemind_llm_enabled=True,
        furnacemind_enable_provider_calls=True,
        furnacemind_provider="mock",
        furnacemind_memory_enabled=True,
        furnacemind_vector_backend="fake",
        furnacemind_tools_enabled=True,
        feedback_require_auth=False,
        compute_require_auth=False,
        copilot_require_auth=False,
    )
    app = create_app(settings)
    password_service = PasswordService(settings)
    token_service = TokenService(settings)
    repository = MemoryUserRepository()
    repository.create_user(
        username="operator",
        password_hash=password_service.hash_password("operatorpass"),
        role="user",
    )
    repository.create_user(
        username="other",
        password_hash=password_service.hash_password("otherpass"),
        role="user",
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
        blocked = client.get("/api/v1/furnacemind/config")
        login = client.post("/api/v1/auth/login", json={"username": "operator", "password": "operatorpass"})
        other_login = client.post("/api/v1/auth/login", json={"username": "other", "password": "otherpass"})
        headers = {"Authorization": f"Bearer {login.json()['data']['access_token']}"}
        other_headers = {"Authorization": f"Bearer {other_login.json()['data']['access_token']}"}
        config = client.get("/api/v1/furnacemind/config", headers=headers)
        conversation = client.post("/api/v1/furnacemind/conversations", json={"title": "Shift"}, headers=headers)
        conversation_id = conversation.json()["data"]["id"]
        forbidden = client.get(f"/api/v1/furnacemind/conversations/{conversation_id}", headers=other_headers)
        upload = client.post(
            "/api/v1/furnacemind/documents",
            files={"file": ("sop.txt", b"pressure stability evidence", "text/plain")},
            headers=headers,
        )
        document_id = upload.json()["data"]["id"]
        indexed = client.post(f"/api/v1/furnacemind/documents/{document_id}/index", headers=headers)
        non_llm = client.post(
            f"/api/v1/furnacemind/conversations/{conversation_id}/runs",
            json={"message": "Summarise pressure", "document_ids": [document_id], "allow_llm": False},
            headers=headers,
        )
        mock_llm = client.post(
            f"/api/v1/furnacemind/conversations/{conversation_id}/runs",
            json={
                "message": "Summarise pressure",
                "document_ids": [document_id],
                "allow_llm": True,
                "options": {"tool_calls": [{"name": "data_summary", "input": {"rows": [{"x": 1}]}}], "export": True},
            },
            headers=headers,
        )
        run_id = mock_llm.json()["data"]["id"]
        status = client.get(f"/api/v1/furnacemind/runs/{run_id}", headers=headers)
        events = client.get(f"/api/v1/furnacemind/runs/{run_id}/events", headers=headers)
        artifact_id = status.json()["data"]["artifacts"][0]["artifact_id"]
        artifact = client.get(f"/api/v1/furnacemind/artifacts/{artifact_id}/download", headers=headers)
        assistant_id = status.json()["data"]["result_message"]["id"]
        feedback = client.post(f"/api/v1/furnacemind/messages/{assistant_id}/feedback", json={"helpful": True}, headers=headers)
        me = client.get("/api/v1/auth/me", headers=headers)
        admin_users = client.get("/api/v1/admin/users", headers=headers)
        data_sources = client.get("/api/v1/data/sources")
        datasets = client.get("/api/v1/datasets")
        feedback_config = client.get("/api/v1/feedback/config")
        mb = client.get("/api/v1/material-balance/config")
        rec = client.get("/api/v1/recommendations/config")
        bmo = client.get("/api/v1/blend-optimizer/context")
        copilot = client.get("/api/v1/copilot/config")

    assert blocked.status_code == 401
    assert login.status_code == 200
    assert other_login.status_code == 200
    assert config.status_code == 200
    assert conversation.status_code == 200
    assert forbidden.status_code == 403
    assert upload.status_code == 200
    assert indexed.json()["data"]["indexed"] is True
    assert non_llm.status_code == 200
    assert non_llm.json()["data"]["status"] == "completed"
    assert mock_llm.status_code == 200
    assert status.json()["data"]["result_message"]["metadata"]["llm_used"] is True
    assert [event["sequence"] for event in events.json()["data"]] == sorted(
        event["sequence"] for event in events.json()["data"]
    )
    assert artifact.status_code == 200
    assert feedback.status_code == 200
    assert me.status_code == 200
    assert admin_users.status_code == 403
    assert data_sources.status_code == 200
    assert datasets.status_code == 200
    assert feedback_config.status_code == 200
    assert mb.status_code == 200
    assert rec.status_code == 200
    assert bmo.status_code == 200
    assert copilot.status_code == 200
