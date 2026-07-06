"""Frontend adapter for API v1 Blend Optimizer endpoints."""

from __future__ import annotations

from typing import Any

try:
    from services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from src.services.api_client import ApiClient, get_api_client, unwrap_api_response


def _auth_headers(token: str | None) -> dict[str, str]:
    clean = str(token or "").strip()
    return {"Authorization": f"Bearer {clean}"} if clean else {}


def get_blend_optimizer_context(
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/blend-optimizer/context", headers=_auth_headers(token)))


def list_blend_optimizer_models(
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/blend-optimizer/models", headers=_auth_headers(token)))


def predict_blend_outputs(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/blend-optimizer/predict", json=payload, headers=_auth_headers(token))
    )


def optimize_blend(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/blend-optimizer/optimize", json=payload, headers=_auth_headers(token))
    )


def start_blend_optimizer_job(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/blend-optimizer/jobs", json=payload, headers=_auth_headers(token))
    )


def get_blend_optimizer_job(
    job_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(f"/blend-optimizer/jobs/{job_id}", headers=_auth_headers(token))
    )


def get_blend_optimizer_artifact_download_url(
    artifact_id: str,
    client: ApiClient | None = None,
) -> str:
    api = client or get_api_client()
    return f"{api.base_url}/blend-optimizer/artifacts/{artifact_id}/download"
