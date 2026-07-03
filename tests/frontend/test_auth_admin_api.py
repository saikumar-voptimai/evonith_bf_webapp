"""Tests for Phase 5 frontend auth/admin API adapters."""

from __future__ import annotations

from services import admin_api, auth_api


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, None, headers))
        return {"request_id": "id", "data": {"ok": True, "items": []}, "meta": {}}

    def post(self, path, json=None, params=None, headers=None):
        self.calls.append(("POST", path, params, json, headers))
        return {"request_id": "id", "data": {"ok": True, "access_token": "token"}, "meta": {}}

    def patch(self, path, json=None, params=None, headers=None):
        self.calls.append(("PATCH", path, params, json, headers))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}


def test_auth_login_calls_backend_auth_endpoint():
    client = FakeClient()

    result = auth_api.login("admin", "password", client=client)

    assert result["access_token"] == "token"
    assert client.calls == [
        (
            "POST",
            "/auth/login",
            None,
            {"username": "admin", "password": "password"},
            None,
        )
    ]


def test_auth_me_sends_bearer_token():
    client = FakeClient()

    auth_api.get_me("token-123", client=client)

    assert client.calls[0][1] == "/auth/me"
    assert client.calls[0][4] == {"Authorization": "Bearer token-123"}


def test_admin_list_users_calls_backend_endpoint_with_token():
    client = FakeClient()

    admin_api.list_users("token-123", limit=25, offset=5, client=client)

    assert client.calls[0] == (
        "GET",
        "/admin/users",
        {"limit": 25, "offset": 5},
        None,
        {"Authorization": "Bearer token-123"},
    )


def test_admin_create_user_posts_payload():
    client = FakeClient()
    payload = {"username": "new_user", "password": "newpass123", "role": "user"}

    admin_api.create_user("token-123", payload, client=client)

    assert client.calls[0] == (
        "POST",
        "/admin/users",
        None,
        payload,
        {"Authorization": "Bearer token-123"},
    )
