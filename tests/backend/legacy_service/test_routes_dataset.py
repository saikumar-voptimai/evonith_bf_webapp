"""Compatibility tests for standalone ``/dataset`` adapters.

The deprecated route family must stay authorized and map its historical task
shapes onto the canonical durable static-dataset service.
"""

from __future__ import annotations

from datetime import datetime, timezone

from apps.backend_api.app.api.v1.schemas.datasets import (
    DatasetJobCreated,
    DatasetJobStatus,
    DatasetRange,
    StaticDatasetMetadata,
    StaticDatasetTimeColumn,
)
from apps.backend_api.app.core.errors import ApiError


def _metadata() -> StaticDatasetMetadata:
    return StaticDatasetMetadata(
        version="a" * 64,
        etag='"sha256-' + ("a" * 64) + '"',
        row_count=19,
        column_count=3,
        columns=[],
        time_column=StaticDatasetTimeColumn(id="timestamp"),
        range=DatasetRange(
            start=datetime(2026, 3, 1, tzinfo=timezone.utc),
            end=datetime(2026, 3, 31, tzinfo=timezone.utc),
        ),
        last_built_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )


def _created() -> DatasetJobCreated:
    return DatasetJobCreated(
        job_id="job-123",
        status="pending",
        operation="build_range",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )


def test_cache_info_projects_canonical_metadata(monkeypatch, client):
    from apps.backend_api.app.services import dataset_service

    monkeypatch.setattr(dataset_service, "get_static_metadata", _metadata)
    response = client.get("/dataset/cache-info")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data_start"] == "2026-03-01"
    assert body["raw_end"] == "2026-03-31"
    assert body["rows"] == 19


def test_cache_info_reports_no_cache_without_a_canonical_dataset(monkeypatch, client):
    from apps.backend_api.app.services import dataset_service

    monkeypatch.setattr(
        dataset_service,
        "get_static_metadata",
        lambda: (_ for _ in ()).throw(ApiError("DATASET_NOT_AVAILABLE", "missing", 404)),
    )

    response = client.get("/dataset/cache-info")

    assert response.status_code == 200
    assert response.json()["status"] == "no_cache"


def test_static_download_uses_canonical_download_service(monkeypatch, client, tmp_path):
    from apps.backend_api.app.services import dataset_service

    csv_path = tmp_path / "canonical.csv"
    csv_path.write_text("timestamp,fuel_rate\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")
    monkeypatch.setattr(dataset_service, "current_dataset_download", lambda: (csv_path, "a" * 64))

    response = client.get("/dataset/static")

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert response.headers["x-dataset-version"] == "a" * 64


def test_fetch_rejects_bad_date_order_before_creating_job(client):
    response = client.post(
        "/dataset/fetch",
        json={"start_date": "2026-03-31", "end_date": "2026-03-01", "rm_choice": "charge"},
    )

    assert response.status_code == 400
    assert "start_date" in response.json()["detail"].lower()


def test_fetch_adapts_to_canonical_build_job(monkeypatch, client):
    from apps.backend_api.app.services import dataset_service

    captured = {}

    def submit(payload, *, current_user, idempotency_key):
        captured.update(payload=payload, current_user=current_user, idempotency_key=idempotency_key)
        return _created()

    monkeypatch.setattr(dataset_service, "submit_static_dataset_job", submit)
    response = client.post(
        "/dataset/fetch",
        json={"start_date": "2026-01-01", "end_date": "2026-01-07", "rm_choice": "charge"},
    )

    assert response.status_code == 200
    assert response.json()["task_id"] == "job-123"
    assert captured["payload"].operation == "build_range"
    assert captured["payload"].options.produce_download is True
    assert captured["idempotency_key"].startswith("legacy-")


def test_fetch_rejects_noncanonical_rm_choice(client):
    response = client.post(
        "/dataset/fetch",
        json={"start_date": "2026-01-01", "end_date": "2026-01-07", "rm_choice": "dpr"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_DATASET_OPTION"


def test_task_status_maps_durable_job_shape(monkeypatch, client):
    from apps.backend_api.app.services import dataset_service

    monkeypatch.setattr(
        dataset_service,
        "get_job",
        lambda _job_id, _current_user: DatasetJobStatus(
            job_id="job-123",
            status="running",
            progress=45,
            message="Running",
            created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        ),
    )

    response = client.get("/dataset/status/job-123")

    assert response.status_code == 200
    assert response.json()["task_id"] == "job-123"
    assert response.json()["status"] == "running"
    assert response.json()["progress"] == "45%"


def test_unknown_task_is_structured_not_found(monkeypatch, client):
    from apps.backend_api.app.services import dataset_service

    monkeypatch.setattr(
        dataset_service,
        "get_job",
        lambda _job_id, _current_user: (_ for _ in ()).throw(
            ApiError("DATASET_JOB_NOT_FOUND", "missing", 404)
        ),
    )

    response = client.get("/dataset/status/nonexistent")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_JOB_NOT_FOUND"


def test_pending_task_download_remains_conflict(monkeypatch, client):
    from apps.backend_api.app.services import dataset_service

    monkeypatch.setattr(
        dataset_service,
        "get_job",
        lambda _job_id, _current_user: DatasetJobStatus(
            job_id="job-123",
            status="pending",
            progress=0,
            created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        ),
    )

    response = client.get("/dataset/download/job-123")

    assert response.status_code == 409