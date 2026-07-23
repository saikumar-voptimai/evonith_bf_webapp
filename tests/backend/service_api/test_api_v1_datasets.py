"""Tests for API v1 dataset endpoints and their compatibility authorization."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient


class _FakeAuthService:
    _users = {
        "reader": {
            "id": "reader-1",
            "role": "user",
            "permissions": ["data:read", "data:export"],
        },
        "supervisor": {
            "id": "supervisor-2",
            "role": "supervisor",
            "permissions": ["data:read", "data:export", "datasets:build", "datasets:refresh"],
        },
        "other": {
            "id": "other-3",
            "role": "user",
            "permissions": ["data:read", "data:export"],
        },
        "export_any": {
            "id": "export-any-4",
            "role": "user",
            "permissions": ["data:read", "data:export", "data:export:any"],
        },
        "admin": {
            "id": "admin-4",
            "role": "admin",
            "permissions": [
                "data:read",
                "data:export",
                "data:export:any",
                "datasets:build",
                "datasets:refresh",
                "datasets:override",
            ],
        },
        "none": {"id": "none-5", "role": "user", "permissions": []},
    }

    def current_user_from_token(self, token: str):
        user = self._users.get(token)
        if user is None:
            from apps.backend_api.app.core.errors import ApiError

            raise ApiError("INVALID_TOKEN", "Invalid token.", status_code=401)
        return user


@pytest.fixture()
def dataset_client(app_factory) -> TestClient:
    app = app_factory()
    app.state.auth_service = _FakeAuthService()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _headers(token: str = "reader") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_dataset_compatibility_reads_require_data_read(dataset_client):
    assert dataset_client.get("/api/v1/datasets").status_code == 401
    assert dataset_client.get("/api/v1/datasets", headers=_headers("none")).status_code == 403


def test_list_datasets_returns_wrapped_response(dataset_client):
    response = dataset_client.get("/api/v1/datasets", headers=_headers())

    assert response.status_code == 200
    assert response.json()["request_id"]
    assert response.json()["data"][0]["id"] == "static_ml_dataset"


def test_preview_dataset_returns_capped_rows(monkeypatch, dataset_client):
    from apps.backend_api.app.services import dataset_service

    monkeypatch.setenv("DATA_API_MAX_PREVIEW_ROWS", "2")
    monkeypatch.setattr(
        dataset_service,
        "load_static_dataset_dataframe",
        lambda: pd.DataFrame({"a": [1, 2, 3]}),
    )

    response = dataset_client.get(
        "/api/v1/datasets/static_ml_dataset/preview?limit=10",
        headers=_headers(),
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["returned_rows"] == 2
    assert data["truncated"] is True


def test_missing_dataset_returns_structured_error(dataset_client):
    response = dataset_client.get(
        "/api/v1/datasets/missing/preview",
        headers=_headers(),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_refresh_dataset_is_a_versioned_canonical_job_adapter(monkeypatch, dataset_client):
    from apps.backend_api.app.services import dataset_service

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        dataset_service,
        "get_static_metadata",
        lambda: SimpleNamespace(
            version="canonical-version",
            range=SimpleNamespace(end=now - timedelta(days=1)),
        ),
    )
    scheduled: list[str] = []
    monkeypatch.setattr(
        dataset_service.job_service,
        "run_background",
        lambda job, _fn: scheduled.append(job.job_id),
    )

    forbidden = dataset_client.post(
        "/api/v1/datasets/refresh",
        json={"dataset_id": "static_ml_dataset"},
        headers=_headers(),
    )
    headers = {**_headers("supervisor"), "Idempotency-Key": "legacy-refresh-adapter"}
    response = dataset_client.post(
        "/api/v1/datasets/refresh",
        json={"dataset_id": "static_ml_dataset"},
        headers=headers,
    )
    replay = dataset_client.post(
        "/api/v1/datasets/refresh",
        json={"dataset_id": "static_ml_dataset"},
        headers=headers,
    )

    assert forbidden.status_code == 403
    assert response.status_code == replay.status_code == 200
    data = response.json()["data"]
    assert data["job_id"] == replay.json()["data"]["job_id"]
    job = dataset_service.job_service.get_job(data["job_id"])
    assert job is not None
    assert job.owner_user_id == "supervisor-2"
    assert job.operation == "extend"
    assert scheduled == [job.job_id]


def test_get_dataset_job_status_is_owner_scoped(dataset_client):
    from apps.backend_api.app.services import dataset_service

    job = dataset_service.job_service.create_job("Queued", owner_user_id="reader-1")
    dataset_service.job_service.update_job(job.job_id, status="failed", error_code="DATASET_JOB_FAILED")

    forbidden = dataset_client.get(f"/api/v1/datasets/jobs/{job.job_id}", headers=_headers("other"))
    response = dataset_client.get(f"/api/v1/datasets/jobs/{job.job_id}", headers=_headers())

    assert forbidden.status_code == 403
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "failed"
    assert data["error_code"] == "DATASET_JOB_FAILED"


def test_export_any_does_not_grant_elevated_dataset_job_access(dataset_client):
    from apps.backend_api.app.services import dataset_service

    job = dataset_service.job_service.create_job(
        "Queued", operation="build_range", owner_user_id="reader-1"
    )
    response = dataset_client.get(
        f"/api/v1/datasets/static_ml_dataset/jobs/{job.job_id}",
        headers=_headers("export_any"),
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"


def test_missing_dataset_job_returns_structured_error(dataset_client):
    response = dataset_client.get("/api/v1/datasets/jobs/missing", headers=_headers())

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_JOB_NOT_FOUND"


def test_legacy_dataset_route_still_available_and_authorized(dataset_client):
    assert dataset_client.get("/dataset/cache-info").status_code == 401
    response = dataset_client.get("/dataset/cache-info", headers=_headers())
    assert response.status_code in {200, 404}


def test_legacy_artifact_download_is_owner_or_elevated(monkeypatch, tmp_path, dataset_client):
    from apps.backend_api.app.services.artifact_service import create_csv_artifact

    artifact = create_csv_artifact(
        pd.DataFrame({"a": [1]}),
        "owner_scoped",
        owner_user_id="reader-1",
    )
    url = f"/api/v1/datasets/artifacts/{artifact.artifact_id}/download"

    assert dataset_client.get(url).status_code == 401
    assert dataset_client.get(url, headers=_headers("other")).status_code == 403
    assert dataset_client.get(url, headers=_headers()).status_code == 200
    assert dataset_client.get(url, headers=_headers("admin")).status_code == 200


def test_openapi_includes_dataset_contract_and_csv_docs(dataset_client):
    schema = dataset_client.get("/openapi.json").json()

    assert "/api/v1/datasets" in schema["paths"]
    assert "/api/v1/datasets/static_ml_dataset" in schema["paths"]
    download = schema["paths"]["/api/v1/datasets/static_ml_dataset/download"]["get"]
    assert "text/csv" in download["responses"]["200"]["content"]