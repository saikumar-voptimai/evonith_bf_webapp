"""Frontend adapter for API v1 dataset endpoints."""

from __future__ import annotations

from typing import Any

try:
    from services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from src.services.api_client import ApiClient, get_api_client, unwrap_api_response


def list_datasets(client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/datasets"))


def preview_dataset(
    dataset_id: str,
    limit: int = 500,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/datasets/{dataset_id}/preview", params={"limit": limit}))


def refresh_dataset(request: dict[str, Any] | None = None, client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.post("/datasets/refresh", json=request or {"dataset_id": "static_ml_dataset"}))


def get_dataset_job(job_id: str, client: ApiClient | None = None) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/datasets/jobs/{job_id}"))


def get_dataset_job_download_url(job_id: str, client: ApiClient | None = None) -> str:
    api = client or get_api_client()
    return f"{api.base_url}/datasets/jobs/{job_id}/download"


def get_dataset_artifact_download_url(artifact_id: str, client: ApiClient | None = None) -> str:
    api = client or get_api_client()
    return f"{api.base_url}/datasets/artifacts/{artifact_id}/download"
