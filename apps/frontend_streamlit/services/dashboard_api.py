"""Frontend adapter for API v1 dashboard endpoints."""

from __future__ import annotations

from typing import Any

from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client


def _auth_headers(access_token: str | None) -> dict[str, str]:
    token = str(access_token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_kpis(
    access_token: str,
    *,
    window: str = "1h",
    bucket: str = "15m",
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return api.get(
        "/dashboard/kpis",
        params={"window": window, "bucket": bucket},
        headers=_auth_headers(access_token),
    )
