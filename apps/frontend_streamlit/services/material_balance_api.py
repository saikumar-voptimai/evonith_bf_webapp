"""Frontend adapter for API v1 Material Balance endpoints."""

from __future__ import annotations

from typing import Any

try:
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response


def _auth_headers(token: str | None) -> dict[str, str]:
    clean = str(token or "").strip()
    return {"Authorization": f"Bearer {clean}"} if clean else {}


def get_material_balance_config(
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/material-balance/config", headers=_auth_headers(token)))


def validate_material_balance(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/material-balance/validate", json=payload, headers=_auth_headers(token))
    )


def run_material_balance(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/material-balance/run", json=payload, headers=_auth_headers(token))
    )


def start_material_balance_job(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/material-balance/jobs", json=payload, headers=_auth_headers(token))
    )


def get_material_balance_job(
    job_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(f"/material-balance/jobs/{job_id}", headers=_auth_headers(token))
    )


def get_material_balance_artifact_download_url(
    artifact_id: str,
    client: ApiClient | None = None,
) -> str:
    api = client or get_api_client()
    return f"{api.base_url}/material-balance/artifacts/{artifact_id}/download"
