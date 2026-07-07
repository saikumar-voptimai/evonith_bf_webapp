"""Frontend adapter for API v1 data endpoints."""

from __future__ import annotations

from typing import Any

try:
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response


def list_data_sources(client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/data/sources"))


def list_offline_report_types(client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/data/offline/report-types"))


def list_offline_tables(client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/data/offline/tables"))


def preview_data(query: dict[str, Any], client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.post("/data/preview", json=query))


def export_data(
    query: dict[str, Any],
    format: str = "csv",
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.post("/data/export", json={"query": query, "format": format}))


def get_artifact_download_url(artifact_id: str, client: ApiClient | None = None) -> str:
    api = client or get_api_client()
    return f"{api.base_url}/data/artifacts/{artifact_id}/download"
