from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from apps.backend_api.app.services.vboard_service import VBoardService
from furnace_data.vboard.catalog import heatload_fields_for_row, heatload_fields_for_rows
from tests.backend.service_api.test_vboard_service import FakeRepository, NOW, _settings


class _FakeAuthService:
    users = {
        "vboard": {"id": "u1", "role": "user", "permissions": ["vboard:read"]},
        "none": {"id": "u2", "role": "user", "permissions": []},
    }

    def current_user_from_token(self, token: str):
        from apps.backend_api.app.core.errors import ApiError

        user = self.users.get(token)
        if user is None:
            raise ApiError("INVALID_TOKEN", "Invalid token.", status_code=401)
        return user


@pytest.fixture()
def vboard_client(app_factory):
    app = app_factory()
    app.state.auth_service = _FakeAuthService()
    app.state.vboard_service = VBoardService(
        settings=_settings(),
        repository=FakeRepository(),
        clock=lambda: NOW,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _headers(token: str = "vboard") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_catalog_requires_vboard_read_permission(vboard_client):
    assert vboard_client.get("/api/v1/vboard/catalog").status_code == 401
    assert vboard_client.get("/api/v1/vboard/catalog", headers=_headers("none")).status_code == 403


def test_catalog_contract_is_public_and_ordered(vboard_client):
    response = vboard_client.get("/api/v1/vboard/catalog", headers=_headers())

    assert response.status_code == 200
    data = response.json()["data"]
    assert [row["id"] for row in data["rows"]] == ["R6", "R7", "R8", "R9", "R10"]
    assert [quadrant["id"] for quadrant in data["quadrants"]] == ["Q1", "Q2", "Q3", "Q4"]
    assert next(level for level in data["temperature_levels"] if level["id"] == "6795")["elevation_m"] == 6.795
    serialized = json.dumps(data).lower()
    assert ".pkl" not in serialized
    assert "token" not in serialized
    assert "bucket" not in serialized


def test_contours_endpoint_returns_typed_data_and_uses_one_call_per_source(vboard_client):
    response = vboard_client.post(
        "/api/v1/vboard/contours",
        json={"time_range": {"kind": "preset", "preset_id": "last_1_hour"}},
        headers=_headers(),
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["resolved_range"]["start"] == "2026-07-23T03:30:00Z"
    assert data["temperature"]["status"] == "ok"
    assert data["heatload"]["status"] == "ok"
    service = vboard_client.app.state.vboard_service
    assert service.repository.temperature_calls == 1
    assert service.repository.heatload_contour_calls == 1


def test_contours_endpoint_rejects_naive_timestamps(vboard_client):
    response = vboard_client.post(
        "/api/v1/vboard/contours",
        json={
            "time_range": {
                "kind": "absolute",
                "start": "2026-07-23T08:00:00",
                "end": "2026-07-23T10:00:00+05:30",
            }
        },
        headers=_headers(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_TIME_RANGE"


def test_timeseries_endpoint_calls_only_selected_row_source(vboard_client):
    response = vboard_client.post(
        "/api/v1/vboard/heatload-timeseries",
        json={
            "row_id": "R6",
            "time_range": {"kind": "preset", "preset_id": "last_6_hours"},
            "resolution": {"mode": "fixed", "window_id": "5_minutes"},
        },
        headers=_headers(),
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["row"]["id"] == "R6"
    assert data["resolved_window_seconds"] == 300
    service = vboard_client.app.state.vboard_service
    assert service.repository.heatload_timeseries_calls == 1
    assert service.repository.last_selected_fields["heatload_delta_t"] == heatload_fields_for_row("R6")


def test_openapi_has_stable_vboard_operations_and_typed_responses(vboard_client):
    schema = vboard_client.get("/openapi.json").json()

    assert schema["paths"]["/api/v1/vboard/catalog"]["get"]["operationId"] == "get_vboard_catalog"
    assert schema["paths"]["/api/v1/vboard/contours"]["post"]["operationId"] == "query_vboard_contours"
    assert schema["paths"]["/api/v1/vboard/heatload-timeseries"]["post"]["operationId"] == "query_vboard_heatload_timeseries"
    response_schema = schema["paths"]["/api/v1/vboard/contours"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("VBoardContoursResponse")


def test_backend_vboard_sources_do_not_import_streamlit():
    import pathlib

    root = pathlib.Path("apps/backend_api/app")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "apps.frontend_streamlit" not in text
