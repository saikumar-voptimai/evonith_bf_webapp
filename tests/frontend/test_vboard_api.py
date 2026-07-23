from __future__ import annotations

import httpx

from apps.frontend_streamlit.services.api_client import ApiClient
from apps.frontend_streamlit.services.vboard_api import VBoardApi


def test_vboard_api_uses_exact_paths_methods_bearer_and_request_ids():
    seen: list[tuple[str, str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.method,
                request.url.path,
                request.headers["Authorization"],
                request.headers["X-Request-ID"],
            )
        )
        return httpx.Response(
            200,
            json={"request_id": "backend-id", "data": {"ok": True}, "meta": {"warnings": []}},
            headers={"X-Request-ID": "backend-id"},
            request=request,
        )

    client = ApiClient(
        base_url="http://backend.local/api/v1",
        transport=httpx.MockTransport(handler),
    )
    api = VBoardApi("token", client)

    assert api.get_catalog()["request_id"] == "backend-id"
    api.get_contours({"time_range": {"kind": "preset", "preset_id": "last_1_hour"}})
    api.get_heatload_timeseries(
        {
            "row_id": "R6",
            "time_range": {"kind": "preset", "preset_id": "last_6_hours"},
            "resolution": {"mode": "auto"},
        }
    )

    assert [item[:2] for item in seen] == [
        ("GET", "/api/v1/vboard/catalog"),
        ("POST", "/api/v1/vboard/contours"),
        ("POST", "/api/v1/vboard/heatload-timeseries"),
    ]
    assert all(item[2] == "Bearer token" for item in seen)
    assert all(item[3] for item in seen)
