"""Typed frontend adapter for authenticated static-dataset API v1 endpoints."""

from __future__ import annotations

from typing import Any, Mapping

from apps.frontend_streamlit.services.api_client import (
    ApiClient,
    get_api_client,
    unwrap_api_response,
)


STATIC_DATASET_ID = "static_ml_dataset"

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


# Legacy adapters retained while existing pages migrate.
def list_datasets(
    client: ApiClient | None = None,
    *,
    access_token: str | None = None,
) -> Any:
    api = client or get_api_client(access_token)
    return unwrap_api_response(_get(api, "/datasets", headers=_auth_headers(access_token)))


def preview_dataset(
    dataset_id: str,
    limit: int = 500,
    client: ApiClient | None = None,
    *,
    access_token: str | None = None,
) -> Any:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _get(
            api,
            f"/datasets/{dataset_id}/preview",
            params={"limit": limit},
            headers=_auth_headers(access_token),
        )
    )


def refresh_dataset(
    request: dict[str, Any] | None = None,
    client: ApiClient | None = None,
    *,
    access_token: str | None = None,
) -> Any:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _post(
            api,
            "/datasets/refresh",
            payload=request or {"dataset_id": STATIC_DATASET_ID},
            headers=_auth_headers(access_token),
        )
    )


def get_dataset_job(
    job_id: str,
    client: ApiClient | None = None,
    *,
    access_token: str | None = None,
) -> Any:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _get(api, f"/datasets/jobs/{job_id}", headers=_auth_headers(access_token))
    )


def get_static_metadata(
    access_token: str | None = None, client: ApiClient | None = None
) -> dict[str, Any]:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _get(api, f"/datasets/{STATIC_DATASET_ID}", headers=_auth_headers(access_token))
    )


def get_scatter_analysis(
    request: Mapping[str, Any],
    access_token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _post(
            api,
            f"/datasets/{STATIC_DATASET_ID}/analyses/scatter",
            payload=request,
            headers=_auth_headers(access_token),
        )
    )


def get_timeseries(
    request: Mapping[str, Any],
    access_token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _post(
            api,
            f"/datasets/{STATIC_DATASET_ID}/timeseries",
            payload=request,
            headers=_auth_headers(access_token),
        )
    )


def create_job(
    request: Mapping[str, Any],
    access_token: str | None = None,
    client: ApiClient | None = None,
    *,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Create a persistent build, extend, or override job."""

    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _post(
            api,
            f"/datasets/{STATIC_DATASET_ID}/jobs",
            payload=request,
            headers=_auth_headers(access_token, idempotency_key=idempotency_key),
        )
    )


def get_job(
    job_id: str, access_token: str | None = None, client: ApiClient | None = None
) -> dict[str, Any]:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _get(
            api,
            f"/datasets/{STATIC_DATASET_ID}/jobs/{job_id}",
            headers=_auth_headers(access_token),
        )
    )


def get_job_events(
    job_id: str,
    *,
    after: int = 0,
    access_token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _get(
            api,
            f"/datasets/{STATIC_DATASET_ID}/jobs/{job_id}/events",
            params={"after": int(after)},
            headers=_auth_headers(access_token),
        )
    )


def cancel_job(
    job_id: str, access_token: str | None = None, client: ApiClient | None = None
) -> dict[str, Any]:
    """Cancel without automatic retry."""

    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _post(
            api,
            f"/datasets/{STATIC_DATASET_ID}/jobs/{job_id}/cancel",
            payload={},
            headers=_auth_headers(access_token),
        )
    )


def download_job_result(
    job_id: str, access_token: str | None = None, client: ApiClient | None = None
) -> bytes:
    api = client or get_api_client(access_token)
    return _download(
        api,
        f"/datasets/{STATIC_DATASET_ID}/jobs/{job_id}/download",
        headers=_auth_headers(access_token),
    )


def download_current_dataset(
    access_token: str | None = None, client: ApiClient | None = None
) -> bytes:
    api = client or get_api_client(access_token)
    return _download(
        api,
        f"/datasets/{STATIC_DATASET_ID}/download",
        headers=_auth_headers(access_token),
    )


def get_validation(
    access_token: str | None = None, client: ApiClient | None = None
) -> dict[str, Any]:
    api = client or get_api_client(access_token)
    return unwrap_api_response(
        _get(
            api,
            f"/datasets/{STATIC_DATASET_ID}/validation",
            headers=_auth_headers(access_token),
        )
    )


def get_dataset_job_download_url(job_id: str, client: ApiClient | None = None) -> str:
    """Fail safely because browser links cannot carry bearer authentication."""

    raise BackendApiHTTPError(
        "Raw dataset download URLs are disabled; use download_job_result() for authenticated bytes.",
        status_code=400,
        error_code="AUTHENTICATED_DOWNLOAD_REQUIRED",
    )


def get_dataset_artifact_download_url(
    artifact_id: str, client: ApiClient | None = None
) -> str:
    """Fail safely because browser links cannot carry bearer authentication."""

    raise BackendApiHTTPError(
        "Raw dataset artifact URLs are disabled; use download_job_result() for authenticated bytes.",
        status_code=400,
        error_code="AUTHENTICATED_DOWNLOAD_REQUIRED",
    )
