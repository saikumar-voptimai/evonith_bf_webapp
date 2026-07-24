"""Frontend adapter for API v1 Blend Optimizer endpoints."""

from __future__ import annotations

from typing import Any

try:
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response


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


def get_blend_optimizer_catalog(
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/blend-optimizer/catalog", headers=_auth_headers(token)))


def create_blend_optimizer_context(
    payload: dict[str, Any] | None = None,
    *,
    idempotency_key: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            "/blend-optimizer/contexts",
            json=payload or {"source_refresh": "use_cached"},
            headers=_auth_headers(token),
            idempotency_key=idempotency_key,
        )
    )


def get_blend_optimizer_context_by_id(
    context_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/blend-optimizer/contexts/{context_id}", headers=_auth_headers(token)))


def get_blend_optimizer_context_diagnostics(
    context_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/blend-optimizer/contexts/{context_id}/diagnostics", headers=_auth_headers(token)))


def get_blend_optimizer_preferences(
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/blend-optimizer/preferences", headers=_auth_headers(token)))


def update_blend_optimizer_preferences(
    payload: dict[str, Any],
    *,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.patch("/blend-optimizer/preferences", json=payload, headers=_auth_headers(token)))


def create_blend_optimizer_run(
    payload: dict[str, Any],
    *,
    idempotency_key: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            "/blend-optimizer/runs",
            json=payload,
            headers=_auth_headers(token),
            idempotency_key=idempotency_key,
        )
    )


def get_blend_optimizer_run(
    run_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/blend-optimizer/runs/{run_id}", headers=_auth_headers(token)))


def get_blend_optimizer_run_events(
    run_id: str,
    *,
    after: int | None = None,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    params = {"after": after} if after is not None else None
    return unwrap_api_response(api.get(f"/blend-optimizer/runs/{run_id}/events", params=params, headers=_auth_headers(token)))


def cancel_blend_optimizer_run(
    run_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(api.post(f"/blend-optimizer/runs/{run_id}/cancel", headers=_auth_headers(token)))


def create_blend_optimizer_manual_evaluation(
    run_id: str,
    payload: dict[str, Any],
    *,
    idempotency_key: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            f"/blend-optimizer/runs/{run_id}/manual-evaluations",
            json=payload,
            headers=_auth_headers(token),
            idempotency_key=idempotency_key,
        )
    )


def download_blend_optimizer_artifact(
    artifact_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> bytes:
    api = client or get_api_client()
    return api.download(f"/blend-optimizer/artifacts/{artifact_id}/download", headers=_auth_headers(token))
