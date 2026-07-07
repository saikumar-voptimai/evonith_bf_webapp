"""Tests for Phase 5 auth and admin API endpoints."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.repositories.user_repository import UserRecord
from apps.backend_api.app.services.admin_service import AdminService
from apps.backend_api.app.services.auth_service import AuthService
from apps.backend_api.app.services.password_service import PasswordService
from apps.backend_api.app.services.token_service import TokenService


class MemoryUserRepository:
    def __init__(self) -> None:
        self.users: dict[str, UserRecord] = {}

    def find_by_username_or_email(self, identifier: str) -> UserRecord | None:
        for user in self.users.values():
            if user.username == identifier or user.email == identifier:
                return user
        return None

    def find_by_id(self, user_id: str) -> UserRecord | None:
        return self.users.get(str(user_id))

    def list_users(self, *, limit: int = 100, offset: int = 0) -> list[UserRecord]:
        rows = sorted(self.users.values(), key=lambda item: item.username)
        return rows[offset : offset + limit]

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
        if self.find_by_username_or_email(username):
            raise ValueError("User already exists")
        now = datetime.now(timezone.utc)
        user = UserRecord(
            id=str(uuid4()),
            username=username,
            password_hash=password_hash,
            role=role,
            is_active=is_active,
            email=email,
            full_name=full_name,
            created_at=now,
            updated_at=now,
        )
        self.users[user.id] = user
        return user

    def update_user(self, user_id: str, **changes) -> UserRecord | None:
        user = self.users.get(str(user_id))
        if user is None:
            return None
        updated = UserRecord(
            id=user.id,
            username=changes.get("username", user.username),
            password_hash=user.password_hash,
            role=changes.get("role", user.role),
            is_active=changes.get("is_active", user.is_active),
            email=changes.get("email", user.email),
            full_name=changes.get("full_name", user.full_name),
            created_at=user.created_at,
            updated_at=datetime.now(timezone.utc),
            last_login_at=user.last_login_at,
        )
        self.users[user.id] = updated
        return updated

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
            updated_at=datetime.now(timezone.utc),
            last_login_at=user.last_login_at,
        )

    def record_login(self, user_id: str) -> None:
        user = self.users[str(user_id)]
        self.users[str(user_id)] = UserRecord(
            id=user.id,
            username=user.username,
            password_hash=user.password_hash,
            role=user.role,
            is_active=user.is_active,
            email=user.email,
            full_name=user.full_name,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=datetime.now(timezone.utc),
        )


def _settings() -> BackendSettings:
    return BackendSettings(
        backend_env="test",
        auth_secret_key="test-secret",
        auth_password_hash_scheme="bcrypt",
        auth_min_password_length=8,
    )


def _client_with_auth(app_factory) -> TestClient:
    settings = _settings()
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


def _login(client: TestClient, username: str, password: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return response.json()["data"]["access_token"]


def test_auth_login_and_me(app_factory):
    with _client_with_auth(app_factory) as client:
        token = _login(client, "admin", "adminpass")

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json()["data"]["user"]["username"] == "admin"


def test_invalid_login_returns_structured_error(app_factory):
    with _client_with_auth(app_factory) as client:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrongpass"},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_CREDENTIALS"


def test_backend_login_upgrades_legacy_sha256_hash(app_factory):
    settings = _settings()
    password_service = PasswordService(settings)
    token_service = TokenService(settings)
    repository = MemoryUserRepository()
    legacy_hash = hashlib.sha256("legacy-pass".encode()).hexdigest()
    repository.create_user(
        username="legacy_user",
        password_hash=legacy_hash,
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

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "legacy_user", "password": "legacy-pass"},
        )

    upgraded = repository.find_by_username_or_email("legacy_user")
    assert response.status_code == 200
    assert upgraded is not None
    assert upgraded.password_hash.startswith("$2")


def test_admin_routes_require_auth_and_admin_role(app_factory):
    with _client_with_auth(app_factory) as client:
        no_token = client.get("/api/v1/admin/users")
        user_token = _login(client, "operator", "operatorpass")
        forbidden = client.get(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        admin_token = _login(client, "admin", "adminpass")
        allowed = client.get(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    assert no_token.status_code == 401
    assert no_token.json()["error"]["code"] == "AUTH_REQUIRED"
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "FORBIDDEN"
    assert allowed.status_code == 200
    assert allowed.json()["data"]["total"] == 2


def test_admin_can_create_and_deactivate_user(app_factory):
    with _client_with_auth(app_factory) as client:
        admin_token = _login(client, "admin", "adminpass")
        create_response = client.post(
            "/api/v1/admin/users",
            json={
                "username": "new_user",
                "password": "newpass123",
                "role": "supervisor",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        user_id = create_response.json()["data"]["id"]
        deactivate = client.post(
            f"/api/v1/admin/users/{user_id}/deactivate",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    assert create_response.status_code == 200
    assert create_response.json()["data"]["role"] == "supervisor"
    assert deactivate.status_code == 200
    assert deactivate.json()["data"]["is_active"] is False


def test_change_password_requires_current_password(app_factory):
    with _client_with_auth(app_factory) as client:
        token = _login(client, "operator", "operatorpass")
        response = client.post(
            "/api/v1/auth/change-password",
            json={
                "current_password": "wrongpass",
                "new_password": "operator-new-pass",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_CREDENTIALS"
