"""Frontend adapter for API v1 operational status endpoints."""

from __future__ import annotations

from typing import Any

try:
    from services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from src.services.api_client import ApiClient, get_api_client, unwrap_api_response


def _auth_headers(access_token: str | None) -> dict[str, str]:
    token = str(access_token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_status(access_token: str | None = None, client: ApiClient | None = None) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/status", headers=_auth_headers(access_token)))


def get_runtime_status(access_token: str, client: ApiClient | None = None) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/status/runtime/details", headers=_auth_headers(access_token)))


def get_status_config(access_token: str, client: ApiClient | None = None) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/status/config", headers=_auth_headers(access_token)))


def get_dependency_status(access_token: str, client: ApiClient | None = None) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/status/dependencies", headers=_auth_headers(access_token)))


def get_metrics(access_token: str, client: ApiClient | None = None) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/metrics", headers=_auth_headers(access_token)))
