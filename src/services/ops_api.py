"""Frontend adapter for API v1 operations endpoints."""

from __future__ import annotations

from typing import Any

try:
    from services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from src.services.api_client import ApiClient, get_api_client, unwrap_api_response


def _auth_headers(access_token: str | None) -> dict[str, str]:
    token = str(access_token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def list_jobs(
    access_token: str,
    *,
    limit: int = 100,
    offset: int = 0,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(
            "/jobs",
            params={"limit": limit, "offset": offset},
            headers=_auth_headers(access_token),
        )
    )


def get_job(job_id: str, access_token: str, client: ApiClient | None = None) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/jobs/{job_id}", headers=_auth_headers(access_token)))


def dry_run_cleanup(
    access_token: str,
    payload: dict[str, Any] | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            "/ops/cleanup/dry-run",
            json=payload or {"dry_run": True},
            headers=_auth_headers(access_token),
        )
    )


def run_cleanup(
    access_token: str,
    payload: dict[str, Any] | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            "/ops/cleanup/run",
            json=payload or {"dry_run": False},
            headers=_auth_headers(access_token),
        )
    )


def list_audit_events(
    access_token: str,
    *,
    limit: int = 100,
    offset: int = 0,
    event_type: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if event_type:
        params["event_type"] = event_type
    return unwrap_api_response(
        api.get("/ops/audit/events", params=params, headers=_auth_headers(access_token))
    )


def get_error_codes(access_token: str, client: ApiClient | None = None) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/ops/error-codes", headers=_auth_headers(access_token)))

