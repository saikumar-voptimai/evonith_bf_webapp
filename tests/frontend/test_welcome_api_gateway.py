"""Tests for Welcome API adapters and gateway selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apps.frontend_streamlit.services import dashboard_api, plant_admin_api
from apps.frontend_streamlit.services.api_errors import BackendUnavailableError
from apps.frontend_streamlit.services import welcome_gateway
from apps.frontend_streamlit.utils import session


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, None, headers))
        return {
            "request_id": "rid",
            "data": {"ok": True, "items": []},
            "meta": {"warnings": ["warn"]},
        }

    def put(self, path, json=None, params=None, headers=None):
        self.calls.append(("PUT", path, params, json, headers))
        return {"request_id": "rid", "data": {"ok": True}, "meta": {}}

    def delete(self, path, params=None, json=None, headers=None):
        self.calls.append(("DELETE", path, params, json, headers))
        return {"request_id": "rid", "data": {"deleted_count": 1}, "meta": {}}


def test_dashboard_api_calls_exact_path_query_and_bearer_header():
    client = FakeClient()

    raw = dashboard_api.get_kpis("token-123", window="1h", bucket="15m", client=client)

    assert raw["request_id"] == "rid"
    assert client.calls == [
        (
            "GET",
            "/dashboard/kpis",
            {"window": "1h", "bucket": "15m"},
            None,
            {"Authorization": "Bearer token-123"},
        )
    ]


def test_plant_admin_api_paths_methods_payloads_and_headers():
    client = FakeClient()
    payload = {
        "effective_at": "2026-07-23T10:00:00+05:30",
        "expected_snapshot_id": 123,
        "assignments": {"hopper_01": "mat-1"},
    }

    plant_admin_api.get_hopper_context("token", at="2026-07-23T04:30:00Z", client=client)
    plant_admin_api.list_hopper_history("token", limit=25, offset=5, client=client)
    plant_admin_api.update_hopper_mapping("token", payload, client=client)
    plant_admin_api.delete_hopper_history("token", [101, 102], client=client)

    assert client.calls == [
        (
            "GET",
            "/admin/hopper-mappings/context",
            {"at": "2026-07-23T04:30:00Z"},
            None,
            {"Authorization": "Bearer token"},
        ),
        (
            "GET",
            "/admin/hopper-mappings/history",
            {"limit": 25, "offset": 5},
            None,
            {"Authorization": "Bearer token"},
        ),
        (
            "PUT",
            "/admin/hopper-mappings",
            None,
            payload,
            {"Authorization": "Bearer token"},
        ),
        (
            "DELETE",
            "/admin/hopper-mappings/history",
            None,
            {"record_ids": [101, 102]},
            {"Authorization": "Bearer token"},
        ),
    ]


def test_gateway_selection_requires_backend_auth_and_token(monkeypatch):
    monkeypatch.setattr(
        welcome_gateway,
        "is_backend_api_enabled",
        lambda feature: feature in {"welcome"},
    )
    monkeypatch.setattr(welcome_gateway, "st", SimpleNamespace(session_state={}))

    with pytest.raises(BackendUnavailableError):
        welcome_gateway.get_welcome_gateway()

    monkeypatch.setattr(
        welcome_gateway,
        "is_backend_api_enabled",
        lambda feature: feature in {"welcome", "auth"},
    )
    with pytest.raises(BackendUnavailableError):
        welcome_gateway.get_welcome_gateway()

    welcome_gateway.st.session_state["auth_access_token"] = "token"
    assert isinstance(welcome_gateway.get_welcome_gateway(), welcome_gateway.ApiWelcomeGateway)


def test_gateway_selection_uses_direct_when_welcome_flag_disabled(monkeypatch):
    monkeypatch.setattr(welcome_gateway, "is_backend_api_enabled", lambda feature: False)
    monkeypatch.setattr(welcome_gateway, "st", SimpleNamespace(session_state={}))

    assert isinstance(welcome_gateway.get_welcome_gateway(), welcome_gateway.DirectWelcomeGateway)


def test_api_gateway_kpis_preserve_warnings(monkeypatch):
    monkeypatch.setattr(
        welcome_gateway.dashboard_api,
        "get_kpis",
        lambda *args, **kwargs: {
            "request_id": "rid",
            "data": {"metrics": {}},
            "meta": {"warnings": ["empty"]},
        },
    )
    gateway = welcome_gateway.ApiWelcomeGateway("token")

    result = gateway.get_kpis(window="1h", bucket="15m")

    assert result["warnings"] == ["empty"]
    assert result["request_id"] == "rid"


def test_backend_permissions_remain_in_session_state(monkeypatch):
    monkeypatch.setattr(session, "st", SimpleNamespace(session_state={}))

    session.login_user(
        "operator",
        "user",
        user_id="user-1",
        access_token="token",
        permissions=["hopper:write"],
    )

    assert session.current_permissions() == {"hopper:write"}
    assert session.has_permission("hopper:write") is True
