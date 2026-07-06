"""Tests for API v1 dataset endpoints."""

from __future__ import annotations

import pandas as pd


def test_list_datasets_returns_wrapped_response(client):
    response = client.get("/api/v1/datasets")

    assert response.status_code == 200
    assert response.json()["request_id"]
    assert response.json()["data"][0]["id"] == "static_ml_dataset"


def test_preview_dataset_returns_capped_rows(monkeypatch, client):
    from app.services import dataset_service

    monkeypatch.setenv("DATA_API_MAX_PREVIEW_ROWS", "2")
    monkeypatch.setattr(
        dataset_service,
        "load_static_dataset_dataframe",
        lambda: pd.DataFrame({"a": [1, 2, 3]}),
    )

    response = client.get("/api/v1/datasets/static_ml_dataset/preview?limit=10")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["returned_rows"] == 2
    assert data["truncated"] is True


def test_missing_dataset_returns_structured_error(client):
    response = client.get("/api/v1/datasets/missing/preview")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_refresh_dataset_returns_job_id(monkeypatch, client):
    from app.services import dataset_service

    monkeypatch.setattr(dataset_service.job_service, "run_background", lambda job, fn: None)

    response = client.post("/api/v1/datasets/refresh", json={"dataset_id": "static_ml_dataset"})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["job_id"]
    assert data["status"] == "pending"


def test_get_dataset_job_status(monkeypatch, client):
    from app.services import dataset_service

    job = dataset_service.job_service.create_job("Queued")
    dataset_service.job_service.update_job(job.job_id, status="failed", error_code="DATASET_JOB_FAILED")

    response = client.get(f"/api/v1/datasets/jobs/{job.job_id}")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "failed"
    assert data["error_code"] == "DATASET_JOB_FAILED"


def test_missing_dataset_job_returns_structured_error(client):
    response = client.get("/api/v1/datasets/jobs/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_JOB_NOT_FOUND"


def test_legacy_dataset_route_still_available(client):
    response = client.get("/dataset/cache-info")

    assert response.status_code in {200, 404}


def test_openapi_includes_datasets_endpoint(client):
    schema = client.get("/openapi.json").json()

    assert "/api/v1/datasets" in schema["paths"]
