"""Frontend adapter for API v1 admin endpoints."""

from __future__ import annotations

from typing import Any

try:
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response


def _auth_headers(access_token: str | None) -> dict[str, str]:
    token = str(access_token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def list_users(
    access_token: str,
    *,
    limit: int = 100,
    offset: int = 0,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(
            "/admin/users",
            params={"limit": limit, "offset": offset},
            headers=_auth_headers(access_token),
        )
    )


def create_user(
    access_token: str,
    user: dict[str, Any],
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/admin/users", json=user, headers=_auth_headers(access_token))
    )


def get_user(
    access_token: str,
    user_id: str,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(f"/admin/users/{user_id}", headers=_auth_headers(access_token))
    )


def update_user(
    access_token: str,
    user_id: str,
    updates: dict[str, Any],
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.patch(
            f"/admin/users/{user_id}",
            json=updates,
            headers=_auth_headers(access_token),
        )
    )


def reset_password(
    access_token: str,
    user_id: str,
    new_password: str,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            f"/admin/users/{user_id}/reset-password",
            json={"new_password": new_password},
            headers=_auth_headers(access_token),
        )
    )


def deactivate_user(
    access_token: str,
    user_id: str,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            f"/admin/users/{user_id}/deactivate",
            json={},
            headers=_auth_headers(access_token),
        )
    )


def activate_user(
    access_token: str,
    user_id: str,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            f"/admin/users/{user_id}/activate",
            json={},
            headers=_auth_headers(access_token),
        )
    )


def list_roles(access_token: str, client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get("/admin/roles", headers=_auth_headers(access_token))
    )


def list_permissions(access_token: str, client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get("/admin/permissions", headers=_auth_headers(access_token))
    )
