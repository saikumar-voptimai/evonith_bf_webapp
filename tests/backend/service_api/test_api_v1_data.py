"""Tests for API v1 data endpoints."""

from __future__ import annotations

import pandas as pd


def test_data_sources_returns_wrapped_response(client):
    response = client.get("/api/v1/data/sources")

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert any(source["id"] == "online" for source in body["data"])
    assert response.headers["X-Request-ID"]


def test_data_preview_caps_rows(monkeypatch, client):
    from app.services import data_service

    monkeypatch.setenv("DATA_API_MAX_PREVIEW_ROWS", "3")
    monkeypatch.setattr(data_service, "fetch_dataframe", lambda query: pd.DataFrame({"a": range(10)}))

    response = client.post("/api/v1/data/preview", json={"source": "static_dataset", "limit": 10})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["returned_rows"] == 3
    assert data["truncated"] is True
    assert response.json()["meta"]["warnings"]


def test_data_preview_empty_dataset(monkeypatch, client):
    from app.services import data_service

    monkeypatch.setattr(data_service, "fetch_dataframe", lambda query: pd.DataFrame())

    response = client.post("/api/v1/data/preview", json={"source": "static_dataset"})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["rows"] == []
    assert data["row_count"] == 0


def test_invalid_data_source_returns_structured_error(client):
    response = client.post("/api/v1/data/preview", json={"source": "missing"})

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "DATA_SOURCE_NOT_FOUND"
    assert body["request_id"]


def test_invalid_date_range_returns_structured_error(client):
    response = client.post(
        "/api/v1/data/preview",
        json={
            "source": "offline",
            "start_time": "2026-01-02T00:00:00Z",
            "end_time": "2026-01-01T00:00:00Z",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "DATA_QUERY_INVALID"


def test_data_export_creates_and_downloads_artifact(monkeypatch, client):
    from app.services import data_service

    monkeypatch.setattr(data_service, "fetch_dataframe", lambda query: pd.DataFrame({"a": [1, 2]}))

    response = client.post(
        "/api/v1/data/export",
        json={"query": {"source": "static_dataset"}, "format": "csv"},
    )

    assert response.status_code == 200
    artifact_id = response.json()["data"]["artifact_id"]
    download = client.get(f"/api/v1/data/artifacts/{artifact_id}/download")
    assert download.status_code == 200
    assert "text/csv" in download.headers["content-type"]


def test_invalid_artifact_id_returns_structured_error(client):
    response = client.get("/api/v1/data/artifacts/not-a-valid-id/download")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "DATA_EXPORT_FAILED"


def test_openapi_includes_data_endpoint(client):
    schema = client.get("/openapi.json").json()

    assert "/api/v1/data/sources" in schema["paths"]
