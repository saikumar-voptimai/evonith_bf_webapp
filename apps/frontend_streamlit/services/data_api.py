"""Typed frontend adapter for authenticated API v1 data endpoints.

This module intentionally contains no data-source implementation. Streamlit
uses it only to speak HTTP to backend-owned contracts.
"""

from __future__ import annotations

from typing import Any, Mapping

from apps.frontend_streamlit.services.api_client import (
    ApiClient,
    get_api_client,
    unwrap_api_response,
)
from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError


def _auth_headers(
    access_token: str | None, *, idempotency_key: str | None = None
) -> dict[str, str]:
    token = str(access_token or "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if idempotency_key:
        headers["Idempotency-Key"] = str(idempotency_key)
    return headers


def _get(
    api: ApiClient,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str],
) -> Any:
    return api.get(path, params=params, headers=headers) if headers else api.get(path, params=params)


def _post(
    api: ApiClient, path: str, *, payload: Mapping[str, Any], headers: dict[str, str]
) -> Any:
    return api.post(path, json=dict(payload), headers=headers) if headers else api.post(
        path, json=dict(payload)
    )


def _download(api: ApiClient, path: str, *, headers: dict[str, str]) -> bytes:
    return api.download(path, headers=headers) if headers else api.download(path)


# Legacy adapters retained while other pages migrate.
def list_data_sources(
    client: ApiClient | None = None,
    *,
    access_token: str | None = None,
) -> Any:
    api = client or get_api_client(access_token)
    return unwrap_api_response(_get(api, "/data/sources", headers=_auth_headers(access_token)))


def list_offline_report_types(
    client: ApiClient | None = None,
    *,
    access_token: str | None = None,
) -> Any:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _get(api, "/data/offline/report-types", headers=_auth_headers(access_token))
    )


def list_offline_tables(
    client: ApiClient | None = None,
    *,
    access_token: str | None = None,
) -> Any:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _get(api, "/data/offline/tables", headers=_auth_headers(access_token))
    )


def get_catalog(
    access_token: str | None = None, client: ApiClient | None = None
) -> dict[str, Any]:
    """Return API-owned measurements, public IDs, presets, and request limits."""

    api = client or get_api_client(access_token)
    return unwrap_api_response(_get(api, "/data/catalog", headers=_auth_headers(access_token)))


def preview_data(
    query: Mapping[str, Any],
    client: ApiClient | None = None,
    *,
    access_token: str | None = None,
) -> dict[str, Any]:
    """Request a bounded typed online/offline preview."""

    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _post(api, "/data/preview", payload=query, headers=_auth_headers(access_token))
    )


def export_data(
    query: Mapping[str, Any],
    format: str = "csv",
    client: ApiClient | None = None,
    *,
    access_token: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Create a full export from the original query, never preview rows."""

    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _post(
            api,
            "/data/export",
            payload={"query": dict(query), "format": format},
            headers=_auth_headers(access_token, idempotency_key=idempotency_key),
        )
    )


def download_artifact(
    artifact_id: str,
    access_token: str | None = None,
    client: ApiClient | None = None,
) -> bytes:
    """Download authenticated artifact bytes instead of exposing a raw URL."""

    api = client or get_api_client(access_token)
    return _download(
        api,
        f"/data/artifacts/{artifact_id}/download",
        headers=_auth_headers(access_token),
    )


def preview_hot_metal_slag(
    request: Mapping[str, Any],
    access_token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _post(
            api,
            "/data/hot-metal-slag/preview",
            payload=request,
            headers=_auth_headers(access_token),
        )
    )


def export_hot_metal_slag(
    request: Mapping[str, Any],
    access_token: str | None = None,
    client: ApiClient | None = None,
    *,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Create an HM/Slag artifact from the full specialized query."""

    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _post(
            api,
            "/data/hot-metal-slag/export",
            payload={"query": dict(request), "format": "csv"},
            headers=_auth_headers(access_token, idempotency_key=idempotency_key),
        )
    )


def get_artifact_download_url(artifact_id: str, client: ApiClient | None = None) -> str:
    """Fail safely because browser links cannot carry bearer authentication."""

    raise BackendApiHTTPError(
        "Raw artifact download URLs are disabled; use download_artifact() for authenticated bytes.",
        status_code=400,
        error_code="AUTHENTICATED_DOWNLOAD_REQUIRED",
    )
