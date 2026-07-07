"""Frontend adapter for API v1 auth endpoints."""

from __future__ import annotations

from typing import Any

try:
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response


def _auth_headers(access_token: str | None) -> dict[str, str]:
    token = str(access_token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def login(
    username: str,
    password: str,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            "/auth/login",
            json={"username": username, "password": password},
        )
    )


def get_me(access_token: str, client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/auth/me", headers=_auth_headers(access_token)))


def logout(access_token: str, client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/auth/logout", json={}, headers=_auth_headers(access_token))
    )


def change_password(
    access_token: str,
    current_password: str,
    new_password: str,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            "/auth/change-password",
            json={
                "current_password": current_password,
                "new_password": new_password,
            },
            headers=_auth_headers(access_token),
        )
    )
