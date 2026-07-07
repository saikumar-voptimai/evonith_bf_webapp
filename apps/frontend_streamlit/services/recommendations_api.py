"""Frontend adapter for API v1 Recommendations endpoints."""

from __future__ import annotations

from typing import Any

try:
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response


def _auth_headers(token: str | None) -> dict[str, str]:
    clean = str(token or "").strip()
    return {"Authorization": f"Bearer {clean}"} if clean else {}


def get_recommendations_config(
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/recommendations/config", headers=_auth_headers(token)))


def run_recommendations(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/recommendations/run", json=payload, headers=_auth_headers(token))
    )


def start_recommendations_job(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/recommendations/jobs", json=payload, headers=_auth_headers(token))
    )


def get_recommendations_job(
    job_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(f"/recommendations/jobs/{job_id}", headers=_auth_headers(token))
    )


def get_recommendations_artifact_download_url(
    artifact_id: str,
    client: ApiClient | None = None,
) -> str:
    api = client or get_api_client()
    return f"{api.base_url}/recommendations/artifacts/{artifact_id}/download"
