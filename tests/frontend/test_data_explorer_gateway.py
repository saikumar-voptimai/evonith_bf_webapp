"""Focused contract tests for the API-first Data Explorer frontend boundary."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pandas as pd
import pytest

from apps.frontend_streamlit.services.api_client import ApiClient
from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError
from apps.frontend_streamlit.services.data_explorer_gateway import (
    ApiDataQueryGateway,
    ApiDatasetGateway,
    DirectDataQueryGateway,
    DirectDatasetGateway,
    _DirectArtifactStore,
    get_data_explorer_gateways,
)


class RecordingClient:
    base_url = "http://backend.local/api/v1"

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.last_response_request_id = "transport-request-id"

    @staticmethod
    def _response() -> dict:
        return {
            "request_id": "backend-request-id",
            "data": {"ok": True},
            "meta": {"api_version": "v1", "warnings": ["backend warning"]},
        }

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, headers))
        return self._response()

    def post(self, path, json=None, params=None, headers=None):
        self.calls.append(("POST", path, json, params, headers))
        return self._response()

    def download(self, path, params=None, headers=None):
        self.calls.append(("DOWNLOAD", path, params, headers))
        return b"authenticated-csv"


def _online_request() -> dict:
    return {
        "source": "online",
        "measurements": ["process_params"],
        "time_range": {"kind": "preset", "preset_id": "last_15_minutes"},
        "aggregation": {"mode": "mean", "window_id": "15_minutes"},
        "fields": ["production_per_hour"],
        "limit": 500,
        "offset": 0,
    }


def test_api_data_gateway_uses_bearer_paths_typed_payloads_and_artifact_download():
    client = RecordingClient()
    gateway = ApiDataQueryGateway("access-token", client=client)
    request = _online_request()

    catalog = gateway.get_catalog()
    preview = gateway.preview(request)
    export = gateway.create_export(
        {"query": request, "format": "csv"}, idempotency_key="export-key"
    )
    hm_request = {
        "start": "2026-07-01T00:00:00+05:30",
        "end": "2026-07-01T23:59:59+05:30",
        "interval_minutes": 60,
        "limit": 500,
        "offset": 0,
    }
    gateway.preview_hot_metal_slag(hm_request)
    gateway.export_hot_metal_slag(hm_request, idempotency_key="hm-key")
    assert gateway.download_artifact("artifact-1") == b"authenticated-csv"

    assert catalog["request_id"] == "backend-request-id"
    assert preview["warnings"] == ["backend warning"]
    assert export["ok"] is True
    assert client.calls[0] == (
        "GET",
        "/data/catalog",
        None,
        {"Authorization": "Bearer access-token"},
    )
    assert client.calls[1] == (
        "POST",
        "/data/preview",
        request,
        None,
        {"Authorization": "Bearer access-token"},
    )
    assert client.calls[2][1:3] == (
        "/data/export",
        {"query": request, "format": "csv"},
    )
    assert client.calls[2][4] == {
        "Authorization": "Bearer access-token",
        "Idempotency-Key": "export-key",
    }
    assert client.calls[3][1:3] == ("/data/hot-metal-slag/preview", hm_request)
    assert client.calls[4][1:3] == (
        "/data/hot-metal-slag/export",
        {"query": hm_request, "format": "csv"},
    )
    assert client.calls[4][4]["Idempotency-Key"] == "hm-key"
    assert client.calls[5] == (
        "DOWNLOAD",
        "/data/artifacts/artifact-1/download",
        None,
        {"Authorization": "Bearer access-token"},
    )


def test_api_dataset_gateway_uses_canonical_paths_bearer_and_job_idempotency():
    client = RecordingClient()
    gateway = ApiDatasetGateway("access-token", client=client)
    scatter = {
        "dataset_version": "version-1",
        "x_field": "fuel_rate",
        "y_field": "production_per_hour",
        "max_points": 5000,
    }
    series = {
        "dataset_version": "version-1",
        "fields": ["fuel_rate"],
        "time_range": {
            "start": "2026-07-01T00:00:00+05:30",
            "end": "2026-07-01T23:59:59+05:30",
        },
        "max_points_per_field": 5000,
    }
    job = {
        "operation": "extend",
        "end": "2026-07-23T23:59:59+05:30",
        "expected_dataset_version": "version-1",
        "options": {"validate": True},
    }

    gateway.get_static_metadata()
    gateway.get_scatter_analysis(scatter)
    gateway.get_timeseries(series)
    gateway.create_job(job, idempotency_key="job-key")
    gateway.get_job("job-1")
    gateway.get_job_events("job-1", after=4)
    gateway.cancel_job("job-1")
    assert gateway.download_job_result("job-1") == b"authenticated-csv"
    assert gateway.download_current_dataset() == b"authenticated-csv"
    gateway.get_validation()

    paths = [(call[0], call[1]) for call in client.calls]
    assert paths == [
        ("GET", "/datasets/static_ml_dataset"),
        ("POST", "/datasets/static_ml_dataset/analyses/scatter"),
        ("POST", "/datasets/static_ml_dataset/timeseries"),
        ("POST", "/datasets/static_ml_dataset/jobs"),
        ("GET", "/datasets/static_ml_dataset/jobs/job-1"),
        ("GET", "/datasets/static_ml_dataset/jobs/job-1/events"),
        ("POST", "/datasets/static_ml_dataset/jobs/job-1/cancel"),
        ("DOWNLOAD", "/datasets/static_ml_dataset/jobs/job-1/download"),
        ("DOWNLOAD", "/datasets/static_ml_dataset/download"),
        ("GET", "/datasets/static_ml_dataset/validation"),
    ]
    assert client.calls[3][4] == {
        "Authorization": "Bearer access-token",
        "Idempotency-Key": "job-key",
    }
    assert client.calls[5][2] == {"after": 4}
    assert client.calls[6][4] == {"Authorization": "Bearer access-token"}


def test_api_client_injects_bearer_and_idempotency_key_without_post_retries():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    client = ApiClient(
        base_url="http://backend.local/api/v1",
        access_token="access-token",
        max_retries=3,
        transport=httpx.MockTransport(handler),
    )

    assert client.post("/data/export", json={}, idempotency_key="export-key") == {"ok": True}
    assert len(seen) == 1
    assert seen[0].headers["Authorization"] == "Bearer access-token"
    assert seen[0].headers["Idempotency-Key"] == "export-key"


def test_data_explorer_factory_uses_complete_switch_and_datasets_compatibility_alias(monkeypatch):
    fake_client = RecordingClient()
    monkeypatch.setenv("USE_BACKEND_API_DATA_EXPLORER", "true")
    data, dataset = get_data_explorer_gateways(access_token="token", client=fake_client)
    assert isinstance(data, ApiDataQueryGateway)
    assert isinstance(dataset, ApiDatasetGateway)

    monkeypatch.delenv("USE_BACKEND_API_DATA_EXPLORER", raising=False)
    monkeypatch.setenv("USE_BACKEND_API_DATASETS", "true")
    data, dataset = get_data_explorer_gateways(access_token="token", client=fake_client)
    assert isinstance(data, ApiDataQueryGateway)
    assert isinstance(dataset, ApiDatasetGateway)

    monkeypatch.setenv("USE_BACKEND_API_DATA_EXPLORER", "false")
    data, dataset = get_data_explorer_gateways()
    assert isinstance(data, DirectDataQueryGateway)
    assert isinstance(dataset, DirectDatasetGateway)


def test_api_mode_missing_token_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_DATA_EXPLORER", "true")

    with pytest.raises(BackendApiHTTPError) as exc_info:
        get_data_explorer_gateways(access_token="")

    assert exc_info.value.status_code == 401
    assert exc_info.value.error_code == "AUTHENTICATION_REQUIRED"


def test_api_gateway_module_and_page_keep_direct_data_access_out_of_api_boundary():
    root = Path(__file__).resolve().parents[2]
    gateway_text = (
        root / "apps" / "frontend_streamlit" / "services" / "data_explorer_gateway.py"
    ).read_text(encoding="utf-8")
    api_section = gateway_text.split("class DirectDataQueryGateway", 1)[0]
    assert "furnace_data" not in api_section
    assert "direct://data-artifacts" not in gateway_text

    page_text = (
        root / "apps" / "frontend_streamlit" / "custom_pages" / "2_Data_Explorer.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "furnace_data",
        "maybe_refresh",
        "StaticDatasetManager",
        "get_static_dataset_path",
        "load_static_dataset",
        "DatasetFetcher",
        "validate_dataset",
        "fetch_online_df",
        "fetch_offline_data",
        "to_csv(",
    ):
        assert forbidden not in page_text
    for session_key in (
        "data_explorer.scatter_result",
        "data_explorer.timeseries_result",
        "data_explorer.online_preview",
        "data_explorer.offline_preview",
        "data_explorer.hm_slag_preview",
        "data_explorer.active_job",
    ):
        assert session_key in page_text


def test_page_uses_catalog_limits_marks_stale_results_and_refreshes_completed_jobs():
    root = Path(__file__).resolve().parents[2]
    page_text = (
        root / "apps" / "frontend_streamlit" / "custom_pages" / "2_Data_Explorer.py"
    ).read_text(encoding="utf-8")

    for expected in (
        "_show_stale_notice",
        "max_scatter_points",
        "max_timeseries_points_per_field",
        "max_selected_fields",
        "post_completion_refreshed",
        "gateway.get_static_metadata()",
        "gateway.get_validation()",
        "TypeError",
        "No current canonical static dataset is available",
        "data_explorer_extend_unavailable",
        "data_explorer_override_unavailable",
    ):
        assert expected in page_text

def test_direct_catalog_uses_api_compatible_opaque_offline_table_contract(monkeypatch):
    furnace_data = ModuleType("furnace_data")
    furnace_data.__path__ = []
    influx = ModuleType("furnace_data.influx")
    influx.__path__ = []
    query = ModuleType("furnace_data.influx.query")
    offline = ModuleType("furnace_data.offline")

    query.TIMEDELTAS = {}
    query.WINDOWING = {}
    query.field_labels = lambda _measurement: {}
    query.measurement_label = lambda measurement: measurement
    offline.OFFLINE_REPORT_MAP = {}
    offline.OFFLINE_TABLES = {
        "plant.hidden_beta": {"field_b"},
        "plant.hidden_alpha": {"field_a"},
    }
    furnace_data.influx = influx
    furnace_data.offline = offline
    influx.query = query
    monkeypatch.setitem(sys.modules, "furnace_data", furnace_data)
    monkeypatch.setitem(sys.modules, "furnace_data.influx", influx)
    monkeypatch.setitem(sys.modules, "furnace_data.influx.query", query)
    monkeypatch.setitem(sys.modules, "furnace_data.offline", offline)

    catalog = DirectDataQueryGateway().get_catalog()
    tables = catalog["offline_tables"]
    expected_ids = sorted(
        f"offline-table-{hashlib.sha256(table.encode('utf-8')).hexdigest()[:16]}"
        for table in offline.OFFLINE_TABLES
    )

    assert [table["id"] for table in tables] == expected_ids
    assert [table["label"] for table in tables] == ["Offline table 1", "Offline table 2"]
    assert all(re.fullmatch(r"offline-table-[0-9a-f]{16}", table["id"]) for table in tables)
    serialized = json.dumps(catalog)
    assert "plant.hidden_alpha" not in serialized
    assert "plant.hidden_beta" not in serialized

def test_direct_offline_report_hides_physical_source_table_in_preview_and_export(monkeypatch):
    """The deprecated rollback path must enforce the same public shape as v1."""

    furnace_data = ModuleType("furnace_data")
    furnace_data.__path__ = []
    offline = ModuleType("furnace_data.offline")
    source_table = "offline_feed.ore_chemistry"
    frame = pd.DataFrame(
        {"source_table": [source_table], "fe_t": [62.5]},
        index=pd.DatetimeIndex(["2026-06-30T18:30:00Z"], name="time"),
    )
    offline.OFFLINE_REPORT_MAP = {"RM_COMPOSITION": (source_table,)}
    offline.OFFLINE_TABLES = {}
    offline.fetch_offline_data = lambda *_args, **_kwargs: frame.copy()
    offline.fetch_offline_report = lambda *_args, **_kwargs: frame.copy()
    furnace_data.offline = offline
    monkeypatch.setitem(sys.modules, "furnace_data", furnace_data)
    monkeypatch.setitem(sys.modules, "furnace_data.offline", offline)

    gateway = DirectDataQueryGateway(artifacts=_DirectArtifactStore())
    query = {
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

    preview = gateway.preview(query)
    assert "source_table" not in json.dumps(preview)
    assert source_table not in json.dumps(preview)
    assert {column["id"] for column in preview["columns"]} == {"time", "fe_t"}

    export = gateway.create_export(
        {"query": query, "format": "csv"}, idempotency_key="direct-offline-public-shape"
    )
    content = gateway.download_artifact(export["artifact_id"]).decode("utf-8")
    assert "source_table" not in content
    assert source_table not in content
    assert "fe_t" in content

def test_direct_gateway_intentionally_rejects_canonical_mutation_operations():
    gateway = DirectDatasetGateway()

    with pytest.raises(BackendApiHTTPError) as exc_info:
        gateway.create_job(
            {"operation": "extend", "end": "2026-07-23T23:59:59+05:30"},
            idempotency_key="direct-mutation",
        )

    assert exc_info.value.error_code == "DATASET_NOT_READY"