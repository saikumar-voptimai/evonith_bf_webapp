"""Frontend adapter for API v1 plant-admin endpoints."""

from __future__ import annotations

from typing import Any

from apps.frontend_streamlit.services.api_client import (
    ApiClient,
    get_api_client,
    unwrap_api_response,
)


def _auth_headers(access_token: str | None) -> dict[str, str]:
    token = str(access_token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_hopper_context(
    access_token: str,
    *,
    at: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    params = {"at": at} if at else None
    return unwrap_api_response(
        api.get(
            "/admin/hopper-mappings/context",
            params=params,
            headers=_auth_headers(access_token),
        )
    )


def list_hopper_history(
    access_token: str,
    *,
    limit: int = 50,
    offset: int = 0,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(
            "/admin/hopper-mappings/history",
            params={"limit": limit, "offset": offset},
            headers=_auth_headers(access_token),
        )
    )


def update_hopper_mapping(
    access_token: str,
    payload: dict[str, Any],
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.put(
            "/admin/hopper-mappings",
            json=payload,
            headers=_auth_headers(access_token),
        )
    )


def delete_hopper_history(
    access_token: str,
    record_ids: list[int],
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.delete(
            "/admin/hopper-mappings/history",
            json={"record_ids": record_ids},
            headers=_auth_headers(access_token),
        )
    )


def get_burden_context(
    access_token: str,
    *,
    at: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    params = {"at": at} if at else None
    return unwrap_api_response(
        api.get(
            "/admin/burden-distribution/context",
            params=params,
            headers=_auth_headers(access_token),
        )
    )


def list_burden_history(
    access_token: str,
    *,
    limit: int = 50,
    offset: int = 0,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(
            "/admin/burden-distribution/history",
            params={"limit": limit, "offset": offset},
            headers=_auth_headers(access_token),
        )
    )


def update_burden_distribution(
    access_token: str,
    payload: dict[str, Any],
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.put(
            "/admin/burden-distribution",
            json=payload,
            headers=_auth_headers(access_token),
        )
    )


def delete_burden_history(
    access_token: str,
    record_ids: list[int],
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.delete(
            "/admin/burden-distribution/history",
            json={"record_ids": record_ids},
            headers=_auth_headers(access_token),
        )
    )
