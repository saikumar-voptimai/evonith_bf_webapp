from __future__ import annotations

from typing import Any

import pytest

from apps.frontend_streamlit.services import material_balance_api
from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError
from apps.frontend_streamlit.services.material_balance_gateway import ApiMaterialBalanceGateway, DirectMaterialBalanceGateway, get_material_balance_gateway


class FakeClient:
    base_url = "http://localhost:8080/api/v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any, Any, dict[str, str]]] = []

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, None, headers or {}))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}

    def post(self, path, json=None, params=None, headers=None):
        self.calls.append(("POST", path, params, json, headers or {}))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}

    def put(self, path, json=None, params=None, headers=None):
        self.calls.append(("PUT", path, params, json, headers or {}))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}

    def download(self, path, params=None, headers=None):
        self.calls.append(("DOWNLOAD", path, params, None, headers or {}))
        return b"artifact"


def test_material_balance_api_exact_paths_and_authenticated_download():
    client = FakeClient()

    material_balance_api.refresh_material_balance_cache({"scopes": ["dpr"]}, token="tok", client=client)
    material_balance_api.get_material_balance_ash_analyses(token="tok", client=client)
    material_balance_api.update_material_balance_ash_analyses({"materials": []}, token="tok", client=client)
    material_balance_api.get_material_balance_dpr_mapping("2026-07-22", token="tok", client=client)
    material_balance_api.update_material_balance_dpr_mapping({"mapping": []}, token="tok", client=client)
    assert material_balance_api.download_material_balance_artifact("abc", token="tok", client=client) == b"artifact"

    assert client.calls == [
        ("POST", "/material-balance/cache/refresh", None, {"scopes": ["dpr"]}, {"Authorization": "Bearer tok"}),
        ("GET", "/material-balance/ash-analyses", None, None, {"Authorization": "Bearer tok"}),
        ("PUT", "/material-balance/ash-analyses", None, {"materials": []}, {"Authorization": "Bearer tok"}),
        ("GET", "/material-balance/dpr-mapping", {"sample_day": "2026-07-22"}, None, {"Authorization": "Bearer tok"}),
        ("PUT", "/material-balance/dpr-mapping", None, {"mapping": []}, {"Authorization": "Bearer tok"}),
        ("DOWNLOAD", "/material-balance/artifacts/abc/download", None, None, {"Authorization": "Bearer tok"}),
    ]


def test_material_balance_gateway_selection_never_silently_falls_back(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_MATERIAL_BALANCE", raising=False)
    assert isinstance(get_material_balance_gateway(), DirectMaterialBalanceGateway)

    monkeypatch.setenv("USE_BACKEND_API_MATERIAL_BALANCE", "true")
    monkeypatch.delenv("USE_BACKEND_API_AUTH", raising=False)
    with pytest.raises(BackendApiHTTPError) as exc_info:
        get_material_balance_gateway(access_token="tok")
    assert exc_info.value.error_code == "AUTH_REQUIRED"

    monkeypatch.setenv("USE_BACKEND_API_AUTH", "true")
    with pytest.raises(BackendApiHTTPError):
        get_material_balance_gateway(access_token="")
    assert isinstance(get_material_balance_gateway(access_token="tok", client=FakeClient()), ApiMaterialBalanceGateway)