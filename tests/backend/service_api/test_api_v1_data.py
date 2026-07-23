"""Contract tests for API v1 Data Explorer data endpoints."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient


class _FakeAuthService:
    _users = {
        "reader": {"id": "reader-1", "role": "user", "permissions": ["data:read"]},
        "exporter": {"id": "reader-1", "role": "user", "permissions": ["data:read", "data:export"]},
        "other": {"id": "other-2", "role": "user", "permissions": ["data:read", "data:export"]},
        "export_any": {"id": "export-any-5", "role": "user", "permissions": ["data:export:any"]},
        "admin": {"id": "admin-3", "role": "admin", "permissions": ["data:read", "data:export", "data:export:any"]},
        "none": {"id": "none-4", "role": "user", "permissions": []},
    }

    def current_user_from_token(self, token: str):
        user = self._users.get(token)
        if user is None:
            from apps.backend_api.app.core.errors import ApiError

            raise ApiError("INVALID_TOKEN", "Invalid token.", status_code=401)
        return user


@pytest.fixture()
def data_client(app_factory) -> TestClient:
    app = app_factory()
    app.state.auth_service = _FakeAuthService()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _headers(token: str = "reader") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _online_query(**changes):
    payload = {
        "source": "online",
        "measurements": ["process_params"],
        "time_range": {"kind": "preset", "preset_id": "last_15_minutes"},
        "aggregation": None,
        "fields": ["production_per_hour"],
        "limit": 500,
        "offset": 0,
    }
    payload.update(changes)
    return payload


def _offline_report_query(**changes):
    payload = {
        "source": "offline",
        "selection": {"kind": "report", "report_id": "rm_composition"},
        "time_range": {
            "kind": "absolute",
            "start": "2026-07-01T00:00:00+05:30",
            "end": "2026-07-01T01:00:00+05:30",
        },
        "aggregation": None,
        "fields": None,
        "limit": 500,
        "offset": 0,
    }
    payload.update(changes)
    return payload


def test_catalog_requires_data_read_permission(data_client):
    assert data_client.get("/api/v1/data/catalog").status_code == 401
    assert data_client.get("/api/v1/data/catalog", headers=_headers("none")).status_code == 403


def test_catalog_uses_public_ids_and_exact_fifteen_minutes(data_client):
    response = data_client.get("/api/v1/data/catalog", headers=_headers())

    assert response.status_code == 200
    data = response.json()["data"]
    preset = next(item for item in data["time_presets"] if item["id"] == "last_15_minutes")
    assert preset["duration_seconds"] == 900
    serialized = json.dumps(data)
    assert "offline_feed." not in serialized
    assert "plant_master." not in serialized
    assert "lab_sample_id" not in serialized
    assert "cast_no_ladle_spec" not in serialized
    assert data["offline_tables"]
    assert all(item["id"].startswith("offline-table-") for item in data["offline_tables"])
    assert all(item["label"].startswith("Offline table ") for item in data["offline_tables"])


def test_preview_validates_typed_query_and_returns_utc_rows(monkeypatch, data_client):
    from apps.backend_api.app.services import data_service

    frame = pd.DataFrame(
        {"production_per_hour": [100.0]},
        index=pd.DatetimeIndex(["2026-07-01T00:00:00"], name="time"),
    )
    monkeypatch.setattr(data_service, "fetch_online", lambda **_: frame)

    response = data_client.post("/api/v1/data/preview", json=_online_query(), headers=_headers())

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["source"] == "online"
    assert body["columns"][0]["id"] == "time"
    assert body["rows"][0]["time"].endswith("Z")
    assert body["resolved_range"]["start"].endswith("Z")


def test_preview_rejects_naive_time_and_invalid_aggregation(data_client):
    naive = _online_query(
        time_range={
            "kind": "absolute",
            "start": "2026-07-01T00:00:00",
            "end": "2026-07-01T01:00:00+05:30",
        }
    )
    assert data_client.post("/api/v1/data/preview", json=naive, headers=_headers()).status_code == 422

    too_large_window = _online_query(
        time_range={
            "kind": "absolute",
            "start": "2026-07-01T00:00:00+05:30",
            "end": "2026-07-01T00:01:00+05:30",
        },
        aggregation={"mode": "mean", "window_id": "1_hour"},
    )
    response = data_client.post("/api/v1/data/preview", json=too_large_window, headers=_headers())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_AGGREGATION"


def test_preview_rejects_a_time_preset_unsupported_by_its_source(data_client):
    query = {
        "source": "offline",
        "selection": {"kind": "report", "report_id": "hm_slag"},
        "time_range": {"kind": "preset", "preset_id": "last_1_minute"},
        "aggregation": None,
        "fields": None,
        "limit": 1,
        "offset": 0,
    }

    response = data_client.post("/api/v1/data/preview", json=query, headers=_headers())

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_TIME_PRESET"


def test_catalog_analysis_limits_follow_static_dataset_policy(monkeypatch, data_client):
    from apps.backend_api.app.services import dataset_service

    monkeypatch.setattr(dataset_service, "max_scatter_points", lambda: 37)
    monkeypatch.setattr(dataset_service, "max_timeseries_points_per_field", lambda: 19)

    response = data_client.get("/api/v1/data/catalog", headers=_headers())

    assert response.status_code == 200
    limits = response.json()["data"]["limits"]
    assert limits["max_scatter_points"] == 37
    assert limits["max_timeseries_points_per_field"] == 19

def test_preview_partial_and_complete_online_source_failures(monkeypatch, data_client):
    from apps.backend_api.app.services import data_service

    def partial(**kwargs):
        if kwargs["measurements"] == ["temperature_profile"]:
            raise OSError("unavailable")
        return pd.DataFrame(
            {"production_per_hour": [101.0]},
            index=pd.DatetimeIndex(["2026-07-01T00:00:00Z"], name="time"),
        )

    monkeypatch.setattr(data_service, "fetch_online", partial)
    response = data_client.post(
        "/api/v1/data/preview",
        json=_online_query(measurements=["process_params", "temperature_profile"], fields=None),
        headers=_headers(),
    )
    assert response.status_code == 200
    assert response.json()["meta"]["warnings"]

    monkeypatch.setattr(data_service, "fetch_online", lambda **_: (_ for _ in ()).throw(OSError("down")))
    response = data_client.post("/api/v1/data/preview", json=_online_query(), headers=_headers())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATA_SOURCE_UNAVAILABLE"


def test_preview_reports_503_when_every_underlying_online_fetch_fails(monkeypatch, data_client):
    """The core wrapper must not turn a complete outage into an empty 200."""

    from apps.backend_api.app.core import online_fetcher

    class FailingFetcher:
        def __init__(self, **_kwargs):
            pass

        def fetch_averaged_data(self, **_kwargs):
            raise OSError("connection unavailable")

    monkeypatch.setattr(online_fetcher, "BaseDataFetcher", FailingFetcher)
    response = data_client.post("/api/v1/data/preview", json=_online_query(), headers=_headers())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATA_SOURCE_UNAVAILABLE"


def test_preview_caps_rows_and_never_accepts_raw_table_names(monkeypatch, data_client):
    from apps.backend_api.app.services import data_service

    monkeypatch.setattr(data_service, "max_preview_rows", lambda settings=None: 3)
    frame = pd.DataFrame(
        {"production_per_hour": range(10)},
        index=pd.date_range("2026-07-01", periods=10, tz="UTC", name="time"),
    )
    monkeypatch.setattr(data_service, "fetch_online", lambda **_: frame)
    response = data_client.post("/api/v1/data/preview", json=_online_query(limit=10), headers=_headers())
    assert response.status_code == 200
    assert response.json()["data"]["returned_rows"] == 3
    assert response.json()["data"]["truncated"] is True

    raw_table = {
        "source": "offline",
        "selection": {"kind": "table", "table_id": "offline_feed.charge_data"},
        "time_range": {"kind": "preset", "preset_id": "last_1_day"},
        "aggregation": None,
        "fields": None,
        "limit": 1,
        "offset": 0,
    }
    response = data_client.post("/api/v1/data/preview", json=raw_table, headers=_headers())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_TABLE"


def test_export_is_authenticated_idempotent_and_owner_scoped(monkeypatch, data_client):
    from apps.backend_api.app.services import data_service

    calls = {"count": 0}

    def fetch(**_):
        calls["count"] += 1
        return pd.DataFrame(
            {"production_per_hour": [100.0, 101.0]},
            index=pd.date_range("2026-07-01", periods=2, tz="UTC", name="time"),
        )

    monkeypatch.setattr(data_service, "fetch_online", fetch)
    headers = {**_headers("exporter"), "Idempotency-Key": "export-1"}
    response = data_client.post(
        "/api/v1/data/export",
        json={"query": _online_query(limit=1), "format": "csv"},
        headers=headers,
    )
    assert response.status_code == 200
    artifact = response.json()["data"]
    assert artifact["download_path"].startswith("/api/v1/data/artifacts/")
    assert "runtime" not in json.dumps(artifact).lower()
    assert artifact["row_count"] == 2

    duplicate = data_client.post(
        "/api/v1/data/export",
        json={"query": _online_query(limit=1), "format": "csv"},
        headers=headers,
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["data"]["artifact_id"] == artifact["artifact_id"]
    assert calls["count"] == 1

    changed_payload = _online_query(limit=2)
    reused = data_client.post(
        "/api/v1/data/export",
        json={"query": changed_payload, "format": "csv"},
        headers=headers,
    )
    assert reused.status_code == 409
    assert reused.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"

    path = artifact["download_path"]
    download = data_client.get(path, headers=_headers("exporter"))
    assert download.status_code == 200
    assert len(download.text.strip().splitlines()) == 3
    assert data_client.get(path, headers=_headers("other")).status_code == 403
    # data:export:any is sufficient for an authenticated elevated cross-owner download.
    assert data_client.get(path, headers=_headers("export_any")).status_code == 200
    assert data_client.get(path, headers=_headers("admin")).status_code == 200
    missing_key = data_client.post(
        "/api/v1/data/export",
        json={"query": _online_query(limit=1), "format": "csv"},
        headers=_headers("exporter"),
    )
    assert missing_key.status_code == 422


def test_export_reports_a_typed_413_when_the_row_limit_is_exceeded(monkeypatch, data_client):
    from apps.backend_api.app.services import data_service

    frame = pd.DataFrame(
        {"production_per_hour": [100.0, 101.0]},
        index=pd.date_range("2026-07-01", periods=2, tz="UTC", name="time"),
    )
    monkeypatch.setattr(data_service, "fetch_online", lambda **_: frame)
    monkeypatch.setattr(data_service, "max_export_rows", lambda _settings=None: 1)

    response = data_client.post(
        "/api/v1/data/export",
        json={"query": _online_query(), "format": "csv"},
        headers={**_headers("exporter"), "Idempotency-Key": "row-limit-413"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "EXPORT_LIMIT_EXCEEDED"

def test_offline_report_hides_physical_source_table_in_preview_and_export(monkeypatch, data_client):
    """Legacy offline provenance must not cross either public data boundary."""

    from apps.backend_api.app.services import data_service

    def fetch_offline(**_):
        return pd.DataFrame(
            {
                "source_table": ["offline_feed.ore_chemistry"],
                "fe_t": [62.5],
            },
            index=pd.DatetimeIndex(["2026-06-30T18:30:00Z"], name="time"),
        )

    monkeypatch.setattr(data_service, "fetch_database_offline", fetch_offline)
    query = _offline_report_query()

    preview = data_client.post("/api/v1/data/preview", json=query, headers=_headers())
    assert preview.status_code == 200
    preview_payload = preview.json()["data"]
    assert "source_table" not in json.dumps(preview_payload)
    assert "offline_feed.ore_chemistry" not in json.dumps(preview_payload)
    assert {column["id"] for column in preview_payload["columns"]} == {"time", "fe_t"}

    export = data_client.post(
        "/api/v1/data/export",
        json={"query": query, "format": "csv"},
        headers={**_headers("exporter"), "Idempotency-Key": "offline-report-public-shape"},
    )
    assert export.status_code == 200
    download = data_client.get(export.json()["data"]["download_path"], headers=_headers("exporter"))
    assert download.status_code == 200
    assert "source_table" not in download.text
    assert "offline_feed.ore_chemistry" not in download.text
    assert "fe_t" in download.text


def test_hot_metal_slag_preview_and_export_hide_sensitive_columns(monkeypatch, data_client):
    from furnace_data.dataset.service import DatasetService

    def hot_metal(*_, **__):
        return pd.DataFrame(
            {
                "id": [99],
                "lab_sample_id": ["secret"],
                "cast_no_ladle_spec": ["secret"],
                "import_batch_id": ["batch-secret"],
                "source_row_number": [7],
                "chem_pct_si": [0.7],
            },
            index=pd.DatetimeIndex(["2026-07-01T00:00:00"], name="time"),
        )

    monkeypatch.setattr(DatasetService, "fetch_hotmetal_hourly", hot_metal)
    response = data_client.post(
        "/api/v1/data/hot-metal-slag/preview",
        json={
            "start": "2026-07-01T00:00:00+05:30",
            "end": "2026-07-01T02:00:00+05:30",
            "interval_minutes": 60,
            "limit": 1,
            "offset": 0,
        },
        headers=_headers(),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    private_columns = {"id", "cast_no_ladle_spec", "lab_sample_id", "import_batch_id", "source_row_number"}
    assert private_columns.isdisjoint(data["rows"][0])
    assert all(column["id"] not in private_columns for column in data["columns"])
    assert data["synthetic_row_count"] == 0
    assert data["interpolated_columns"] == []

    export = data_client.post(
        "/api/v1/data/hot-metal-slag/export",
        json={
            "query": {
                "start": "2026-07-01T00:00:00+05:30",
                "end": "2026-07-01T02:00:00+05:30",
                "interval_minutes": 60,
                "limit": 1,
                "offset": 0,
            },
            "format": "csv",
        },
        headers={**_headers("exporter"), "Idempotency-Key": "hm-public-shape"},
    )
    assert export.status_code == 200
    csv_columns = data_client.get(
        export.json()["data"]["download_path"], headers=_headers("exporter")
    ).text.splitlines()[0].split(",")
    assert private_columns.isdisjoint(csv_columns)


def test_hot_metal_slag_preview_reads_precise_provenance(monkeypatch, data_client):
    from furnace_data.dataset.service import DatasetService

    def hot_metal(*_, **__):
        frame = pd.DataFrame(
            {"lab_sample_id": ["secret"], "chem_pct_si": [0.7]},
            index=pd.DatetimeIndex(["2026-07-01T00:00:00"], name="time"),
        )
        frame.attrs["synthetic_row_count"] = 1
        frame.attrs["synthetic_timestamps"] = ("2026-07-01T00:00:00",)
        frame.attrs["interpolated_columns"] = ("chem_pct_si", "lab_sample_id")
        return frame

    monkeypatch.setattr(DatasetService, "fetch_hotmetal_hourly", hot_metal)
    response = data_client.post(
        "/api/v1/data/hot-metal-slag/preview",
        json={
            "start": "2026-07-01T00:00:00+05:30",
            "end": "2026-07-01T02:00:00+05:30",
            "interval_minutes": 60,
            "limit": 1,
            "offset": 0,
        },
        headers=_headers(),
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["synthetic_row_count"] == 1
    assert data["interpolated_columns"] == ["chem_pct_si"]

def test_shared_hot_metal_service_records_interpolation_provenance(monkeypatch):
    from furnace_data.dataset import service as dataset_service_module
    from furnace_data.dataset.service import DatasetService

    source = pd.DataFrame(
        {"chem_pct_si": [0.6, 0.8], "sample_id": ["a", "b"]},
        index=pd.DatetimeIndex(
            ["2026-06-30T18:30:00Z", "2026-06-30T20:30:00Z"], name="time"
        ),
    )
    monkeypatch.setattr(
        dataset_service_module,
        "fetch_database_offline_data",
        lambda **_: source,
    )

    result = DatasetService(local_tz="Asia/Kolkata").fetch_hotmetal_hourly(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 1),
        interval_minutes=60,
    )

    assert 0 < result.attrs["synthetic_row_count"] < len(result)
    assert len(result.attrs["synthetic_timestamps"]) == result.attrs["synthetic_row_count"]
    assert "chem_pct_si" in result.attrs["interpolated_columns"]
    assert "sample_id" in result.attrs["interpolated_columns"]

def test_openapi_has_typed_data_contracts(data_client):
    schema = data_client.get("/openapi.json").json()
    assert schema["paths"]["/api/v1/data/catalog"]["get"]["operationId"] == "dataCatalog"
    assert schema["paths"]["/api/v1/data/preview"]["post"]["operationId"] == "dataPreview"
    preview_schema = schema["paths"]["/api/v1/data/preview"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert preview_schema["discriminator"]["propertyName"] == "source"
    download_schema = schema["paths"]["/api/v1/data/artifacts/{artifact_id}/download"]["get"]["responses"]["200"]["content"]["text/csv"]["schema"]
    assert download_schema == {"type": "string", "format": "binary"}
