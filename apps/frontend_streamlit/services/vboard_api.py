"""Frontend-safe V-Board API client helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from apps.frontend_streamlit.services.api_client import ApiClient, is_wrapped_api_response
from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError


JsonDict = dict[str, Any]


def bearer_headers(access_token: str) -> dict[str, str]:
    token = str(access_token or "").strip()
    if not token:
        raise BackendApiHTTPError(
            "V-Board API mode requires a backend access token.",
            status_code=401,
            error_code="AUTH_REQUIRED",
        )
    return {"Authorization": f"Bearer {token}"}


def as_gateway_payload(raw: Any, client: ApiClient | Any) -> JsonDict:
    """Expose endpoint data while preserving request ids and warnings."""

    request_id = getattr(client, "last_response_request_id", None)
    warnings: list[str] = []
    data = raw
    if is_wrapped_api_response(raw):
        request_id = raw.get("request_id") or request_id
        meta = raw.get("meta") or {}
        warnings = [str(item) for item in (meta.get("warnings") or [])]
        data = raw["data"]

    payload = dict(data) if isinstance(data, Mapping) else {"items": data}
    if warnings:
        payload["warnings"] = list(dict.fromkeys([*payload.get("warnings", []), *warnings]))
    payload["request_id"] = request_id
    return payload


class VBoardApi:
    """V-Board API wrapper backed by the central ApiClient."""

    def __init__(self, access_token: str, client: ApiClient | None = None) -> None:
        self.access_token = str(access_token or "").strip()
        self.client = client or ApiClient(access_token=self.access_token)

    def get_catalog(self) -> JsonDict:
        return as_gateway_payload(
            self.client.get("/vboard/catalog", headers=bearer_headers(self.access_token)),
            self.client,
        )

    def get_contours(self, request: JsonDict) -> JsonDict:
        return as_gateway_payload(
            self.client.post(
                "/vboard/contours",
                json=dict(request),
                headers=bearer_headers(self.access_token),
            ),
            self.client,
        )

    def get_heatload_timeseries(self, request: JsonDict) -> JsonDict:
        return as_gateway_payload(
            self.client.post(
                "/vboard/heatload-timeseries",
                json=dict(request),
                headers=bearer_headers(self.access_token),
            ),
            self.client,
        )
