"""Frontend adapter for API v1 Copilot endpoints."""

from __future__ import annotations

from typing import Any

try:
    from services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from src.services.api_client import ApiClient, get_api_client, unwrap_api_response


def _auth_headers(token: str | None) -> dict[str, str]:
    clean = str(token or "").strip()
    return {"Authorization": f"Bearer {clean}"} if clean else {}


def get_copilot_config(
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/copilot/config", headers=_auth_headers(token)))


def get_recent_data(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.post("/copilot/recent-data", json=payload, headers=_auth_headers(token)))


def analyze_anomaly(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.post("/copilot/anomaly", json=payload, headers=_auth_headers(token)))


def analyze_copilot(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.post("/copilot/analyze", json=payload, headers=_auth_headers(token)))


def start_copilot_job(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.post("/copilot/jobs", json=payload, headers=_auth_headers(token)))


def get_copilot_job(
    job_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/copilot/jobs/{job_id}", headers=_auth_headers(token)))


def get_copilot_artifact_download_url(
    artifact_id: str,
    client: ApiClient | None = None,
) -> str:
    api = client or get_api_client()
    return f"{api.base_url}/copilot/artifacts/{artifact_id}/download"
