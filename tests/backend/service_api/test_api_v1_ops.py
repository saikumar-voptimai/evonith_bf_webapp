"""Tests for Phase 10 status, metrics, jobs, audit, and cleanup APIs."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.config import BackendSettings
from app.repositories.user_repository import UserRecord
from app.services.admin_service import AdminService
from app.services.auth_service import AuthService
from app.services.compute_job_service import compute_job_service
from app.services.password_service import PasswordService
from app.services.token_service import TokenService
from furnace_data.runtime_paths import runtime_path


class MemoryUserRepository:
    def __init__(self) -> None:
        self.users: dict[str, UserRecord] = {}

    def find_by_username_or_email(self, identifier: str) -> UserRecord | None:
        return next((user for user in self.users.values() if user.username == identifier or user.email == identifier), None)

    def find_by_id(self, user_id: str) -> UserRecord | None:
        return self.users.get(str(user_id))

    def list_users(self, *, limit: int = 100, offset: int = 0) -> list[UserRecord]:
        return list(self.users.values())[offset : offset + limit]

    def count_users(self) -> int:
        return len(self.users)

    def create_user(
        self,
        *,
        username: str,
        password_hash: str,
        role: str,
        email: str | None = None,
        full_name: str | None = None,
        is_active: bool = True,
    ) -> UserRecord:
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

    def update_user(self, user_id: str, **changes) -> UserRecord | None:
        return self.users.get(str(user_id))

    def update_password_hash(self, user_id: str, password_hash: str) -> None:
        user = self.users[str(user_id)]
        self.users[str(user_id)] = UserRecord(
            id=user.id,
            username=user.username,
            password_hash=password_hash,
            role=user.role,
            is_active=user.is_active,
            email=user.email,
            full_name=user.full_name,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
        )

    def record_login(self, user_id: str) -> None:
        return None


def _client_with_auth(app_factory) -> TestClient:
    settings = BackendSettings(
        backend_env="test",
        auth_secret_key="test-secret",
        auth_password_hash_scheme="bcrypt",
        audit_log_enabled=True,
    )
    password_service = PasswordService(settings)
    token_service = TokenService(settings)
    repository = MemoryUserRepository()
    repository.create_user(
        username="admin",
        password_hash=password_service.hash_password("adminpass"),
        role="admin",
    )
    repository.create_user(
        username="operator",
        password_hash=password_service.hash_password("operatorpass"),
        role="user",
    )
    app = app_factory()
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
    return TestClient(app, raise_server_exceptions=False)


def _login(client: TestClient, username: str = "admin", password: str = "adminpass") -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return response.json()["data"]["access_token"]


def test_status_metrics_jobs_cleanup_audit_and_error_codes(app_factory, tmp_path):
    with _client_with_auth(app_factory) as client:
        no_token = client.get("/api/v1/metrics")
        admin_token = _login(client)
        headers = {"Authorization": f"Bearer {admin_token}"}
        status = client.get("/api/v1/status")
        runtime_details = client.get("/api/v1/status/runtime/details", headers=headers)
        dependencies = client.get("/api/v1/status/dependencies", headers=headers)
        metrics = client.get("/api/v1/metrics", headers=headers)

        job = compute_job_service.run_inline(
            workflow="phase10_test",
            fn=lambda: {"ok": True},
            message="test job",
        )
        jobs = client.get("/api/v1/jobs", headers=headers)
        one_job = client.get(f"/api/v1/jobs/{job.job_id}", headers=headers)

        temp_file = runtime_path("temp") / "phase10-old.tmp"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_text("old", encoding="utf-8")
        old_time = os.path.getmtime(temp_file) - 8 * 3600
        os.utime(temp_file, (old_time, old_time))
        dry_run = client.post("/api/v1/ops/cleanup/dry-run", json={"dry_run": True}, headers=headers)
        cleanup = client.post("/api/v1/ops/cleanup/run", json={"dry_run": False, "max_delete": 10}, headers=headers)
        audit = client.get("/api/v1/ops/audit/events", headers=headers)
        errors = client.get("/api/v1/ops/error-codes", headers=headers)

    assert no_token.status_code == 401
    assert status.status_code == 200
    assert runtime_details.status_code == 200
    assert runtime_details.json()["data"]["runtime"]["label"] == "runtime"
    assert str(tmp_path) not in str(runtime_details.json())
    assert dependencies.status_code == 200
    assert metrics.status_code == 200
    assert metrics.json()["data"]["requests_total"] >= 1
    assert jobs.status_code == 200
    assert any(item["job_id"] == job.job_id for item in jobs.json()["data"]["items"])
    assert one_job.status_code == 200
    assert dry_run.status_code == 200
    assert dry_run.json()["data"]["deleted"] == 0
    assert cleanup.status_code == 200
    assert cleanup.json()["data"]["deleted"] >= 1
    assert audit.status_code == 200
    assert audit.json()["data"]["total"] >= 1
    assert "adminpass" not in str(audit.json())
    assert errors.status_code == 200
    assert any(item["code"] == "FURNACEMIND_*" for item in errors.json()["data"]["items"])


def test_ops_endpoints_require_admin_role(app_factory):
    with _client_with_auth(app_factory) as client:
        user_token = _login(client, "operator", "operatorpass")
        forbidden = client.get(
            "/api/v1/status/dependencies",
            headers={"Authorization": f"Bearer {user_token}"},
        )

    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "FORBIDDEN"
