"""Fixed-fixture parity contracts for the Data Explorer API migration.

These tests deliberately exercise the public gateway DTOs rather than the
Streamlit page.  Direct mode is a temporary rollback path, so it must render
the same successful read-only result shapes as API mode for a known fixture.
Dataset mutations are intentionally different: canonical extend/override work
is API-only and the direct gateway must reject it.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import ModuleType
from typing import Any, Mapping

import pandas as pd
import pytest

from apps.backend_api.app.api.v1.schemas.data import (
    HotMetalSlagPreviewRequest,
    OfflineDataQuery,
    OnlineDataQuery,
    ResolvedTimeRange,
)
from apps.backend_api.app.api.v1.schemas.datasets import (
    ScatterAnalysisRequest as BackendScatterAnalysisRequest,
    TimeSeriesRequest as BackendTimeSeriesRequest,
)
from apps.backend_api.app.services import data_service, dataset_service
from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError
from apps.frontend_streamlit.services.data_explorer_gateway import (
    ApiDataQueryGateway,
    ApiDatasetGateway,
    DirectDataQueryGateway,
    DirectDatasetGateway,
    _DirectArtifactStore,
)


_START = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
_END = datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc)
_VERSION = "fixture-version-1"


class _FixtureApiClient:
    """Small deterministic API client that returns real service DTO fixtures."""

    base_url = "http://backend.example/api/v1"

    def __init__(self, responses: Mapping[tuple[str, str], Any]) -> None:
        self.responses = dict(responses)
        self.calls: list[tuple[str, str, Any, Any, dict[str, str] | None]] = []
        self.last_response_request_id = "transport-request-id"

    def get(self, path: str, params: Any = None, headers: dict[str, str] | None = None) -> Any:
        self.calls.append(("GET", path, None, params, headers))
        return self._response("GET", path)

    def post(
        self,
        path: str,
        json: Any = None,
        params: Any = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        self.calls.append(("POST", path, json, params, headers))
        return self._response("POST", path)

    def _response(self, method: str, path: str) -> dict[str, Any]:
        return {
            "request_id": "fixture-request-id",
            "data": self.responses[(method, path)],
            "meta": {"api_version": "v1", "warnings": []},
        }


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _resolved_range() -> dict[str, str]:
    return {"start": _iso(_START), "end": _iso(_END)}


def _preview_contract(result: Mapping[str, Any]) -> dict[str, Any]:
    """Drop transport/deprecation noise, retaining the browser-visible DTO."""

    return {
        "columns": [
            {
                "id": column["id"],
                "label": column["label"],
                "dtype": column["dtype"],
                "unit": column.get("unit"),
            }
            for column in result["columns"]
        ],
        "rows": result["rows"],
        "returned_rows": result["returned_rows"],
        "total_rows": result["total_rows"],
        # Generic data previews retain the backwards-compatible row_count;
        # HM/Slag has the newer total_rows-only response contract.
        "row_count": result.get("row_count"),
        "offset": result.get("offset"),
        "truncated": result["truncated"],
        "source": result["source"],
        "resolved_range": result["resolved_range"],
    }


def _metadata_contract(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": result["dataset_id"],
        "version": result["version"],
        # Generic data previews retain the backwards-compatible row_count;
        # HM/Slag has the newer total_rows-only response contract.
        "row_count": result.get("row_count"),
        "column_count": result["column_count"],
        "columns": [
            {
                "id": column["id"],
                "label": column["label"],
                "dtype": column["dtype"],
                "unit": column.get("unit"),
                "plottable": column["plottable"],
                "filterable": column["filterable"],
            }
            for column in result["columns"]
        ],
        "time_column": result["time_column"],
        "range": result["range"],
        "download_available": result["download_available"],
    }


def _static_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
    """Return canonical API and display-name direct frames for one fixture."""

    index = pd.date_range(_START, periods=5, freq="15min", tz="UTC", name="timestamp")
    api_frame = pd.DataFrame(
        {
            "fuel_rate": [100.0, 101.0, 102.0, 103.0, 104.0],
            "production_per_hour": [201.0, 203.0, 205.0, 207.0, 209.0],
        },
        index=index,
    )
    direct_frame = api_frame.rename(
        columns={"fuel_rate": "Fuel Rate", "production_per_hour": "Production per Hour"}
    )
    direct_fields = {
        "Fuel Rate": "fuel_rate",
        "Production per Hour": "production_per_hour",
    }
    labels = {
        "fuel_rate": "Fuel Rate",
        "production_per_hour": "Production per Hour",
    }
    return api_frame, direct_frame, direct_fields, labels


def _install_static_fixture(
    monkeypatch: pytest.MonkeyPatch,
    direct: DirectDatasetGateway,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    api_frame, direct_frame, direct_fields, labels = _static_frames()
    monkeypatch.setattr(
        direct,
        "_load_static_dataset",
        lambda: (direct_frame.copy(), dict(direct_fields), _VERSION),
    )
    monkeypatch.setattr(
        dataset_service,
        "_dataset_context",
        lambda: (api_frame.copy(), dict(labels), None, _VERSION),
    )
    return api_frame, direct_frame


def _api_data_gateway(payload: Mapping[str, Any]) -> ApiDataQueryGateway:
    return ApiDataQueryGateway(
        "fixture-token",
        client=_FixtureApiClient({("POST", "/data/preview"): dict(payload)}),
    )


def _api_dataset_gateway(path: str, payload: Mapping[str, Any]) -> tuple[ApiDatasetGateway, _FixtureApiClient]:
    client = _FixtureApiClient({("POST", path): dict(payload)})
    return ApiDatasetGateway("fixture-token", client=client), client


def test_online_preview_direct_and_api_gateways_have_identical_fixed_fixture_dto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {"production_per_hour": [100.0, 102.0]},
        index=pd.DatetimeIndex([_START, _END], name="time"),
    )
    request = {
        "source": "online",
        "measurements": ["process_params"],
        "time_range": {"kind": "absolute", **_resolved_range()},
        "aggregation": None,
        "fields": ["production_per_hour"],
        "limit": 10,
        "offset": 0,
    }

    # Keep the service serializer real while replacing only external I/O.
    resolved = ResolvedTimeRange(start=_START, end=_END)
    monkeypatch.setattr(
        data_service,
        "fetch_query_result",
        lambda *_args, **_kwargs: data_service.QueryFetchResult(frame.copy(), resolved, []),
    )
    backend = data_service.preview_data(OnlineDataQuery.model_validate(request)).model_dump(mode="json")

    direct = DirectDataQueryGateway()
    monkeypatch.setattr(
        direct,
        "_fetch_online",
        lambda _request, *, apply_page: (
            frame.copy(),
            {"production_per_hour": "production_per_hour"},
            _resolved_range(),
        ),
    )

    api_result = _api_data_gateway(backend).preview(request)
    direct_result = direct.preview(request)

    assert _preview_contract(direct_result) == _preview_contract(api_result)
    assert direct_result["columns"][0] == {
        "id": "time",
        "label": "Time",
        "dtype": "datetime",
        "unit": None,
    }
    assert direct_result["rows"][0]["time"] == _iso(_START)


def test_offline_preview_direct_and_api_gateways_have_identical_fixed_fixture_dto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {"fuel_rate": [470.0, 472.0]},
        index=pd.DatetimeIndex([_START, _END], name="time"),
    )
    request = {
        "source": "offline",
        "selection": {"kind": "table", "table_id": "offline-table-fixture"},
        "time_range": {"kind": "absolute", **_resolved_range()},
        "aggregation": None,
        "fields": ["fuel_rate"],
        "limit": 10,
        "offset": 0,
    }
    resolved = ResolvedTimeRange(start=_START, end=_END)
    monkeypatch.setattr(
        data_service,
        "fetch_query_result",
        lambda *_args, **_kwargs: data_service.QueryFetchResult(frame.copy(), resolved, []),
    )
    backend = data_service.preview_data(OfflineDataQuery.model_validate(request)).model_dump(mode="json")

    direct = DirectDataQueryGateway()
    monkeypatch.setattr(
        direct,
        "_fetch_offline",
        lambda _request, *, apply_page: (
            frame.copy(),
            {"fuel_rate": "fuel_rate"},
            _resolved_range(),
        ),
    )

    assert _preview_contract(direct.preview(request)) == _preview_contract(
        _api_data_gateway(backend).preview(request)
    )


def test_static_metadata_uses_the_same_canonical_column_labels_and_types_in_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = DirectDatasetGateway()
    _install_static_fixture(monkeypatch, direct)
    monkeypatch.setattr(dataset_service, "_make_manager", lambda: type("Manager", (), {"get_meta": lambda self: None})())

    backend = dataset_service.get_static_metadata().model_dump(mode="json")
    api_client = _FixtureApiClient(
        {("GET", "/datasets/static_ml_dataset"): backend}
    )
    api_result = ApiDatasetGateway("fixture-token", client=api_client).get_static_metadata()
    direct_result = direct.get_static_metadata()

    assert _metadata_contract(direct_result) == _metadata_contract(api_result)
    assert [column["label"] for column in direct_result["columns"]] == [
        "Fuel Rate",
        "Production per Hour",
    ]


def test_scatter_filter_and_regression_have_the_same_read_only_semantics_in_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = DirectDatasetGateway()
    _install_static_fixture(monkeypatch, direct)
    request = {
        "dataset_version": _VERSION,
        "x_field": "fuel_rate",
        "y_field": "production_per_hour",
        "filter": {"field": "fuel_rate", "mode": "inside", "minimum": 101.0, "maximum": 104.0},
        "regression": {"enabled": True, "degree": 1},
        "max_points": 3,
    }
    backend = dataset_service.get_scatter_analysis(
        BackendScatterAnalysisRequest.model_validate(request)
    ).model_dump(mode="json")
    api_gateway, _client = _api_dataset_gateway(
        "/datasets/static_ml_dataset/analyses/scatter", backend
    )

    direct_result = direct.get_scatter_analysis(request)
    api_result = api_gateway.get_scatter_analysis(request)

    for key in (
        "dataset_version",
        "x",
        "y",
        "total_matching_rows",
        "returned_points",
        "downsampled",
        "dropped_rows",
    ):
        assert direct_result[key] == api_result[key]
    assert direct_result["regression"] is not None
    assert api_result["regression"] is not None
    for key in ("degree", "coefficients", "r_squared", "line_x", "line_y"):
        assert direct_result["regression"][key] == pytest.approx(api_result["regression"][key])


def test_timeseries_outside_filter_and_resampling_have_the_same_labels_and_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = DirectDatasetGateway()
    api_frame, direct_frame = _install_static_fixture(monkeypatch, direct)
    # Make the outside filter meaningful and retain two rows in one one-hour bin.
    api_frame.loc[:, "fuel_rate"] = [100.0, 105.0, 120.0, 130.0, 135.0]
    api_frame.loc[:, "production_per_hour"] = [200.0, 210.0, 240.0, 260.0, 270.0]
    direct_frame.loc[:, "Fuel Rate"] = api_frame["fuel_rate"].to_numpy()
    direct_frame.loc[:, "Production per Hour"] = api_frame["production_per_hour"].to_numpy()
    direct_fields = {"Fuel Rate": "fuel_rate", "Production per Hour": "production_per_hour"}
    labels = {"fuel_rate": "Fuel Rate", "production_per_hour": "Production per Hour"}
    monkeypatch.setattr(
        direct,
        "_load_static_dataset",
        lambda: (direct_frame.copy(), dict(direct_fields), _VERSION),
    )
    monkeypatch.setattr(
        dataset_service,
        "_dataset_context",
        lambda: (api_frame.copy(), dict(labels), None, _VERSION),
    )
    request = {
        "dataset_version": _VERSION,
        "fields": ["fuel_rate", "production_per_hour"],
        "time_range": {"start": _iso(_START), "end": _iso(_END)},
        "filter": {"field": "fuel_rate", "mode": "outside", "minimum": 104.0, "maximum": 131.0},
        "resample": {"mode": "mean", "window": "1h"},
        "max_points_per_field": 10,
    }
    backend = dataset_service.get_timeseries(
        BackendTimeSeriesRequest.model_validate(request)
    ).model_dump(mode="json")
    api_gateway, _client = _api_dataset_gateway(
        "/datasets/static_ml_dataset/timeseries", backend
    )

    direct_result = direct.get_timeseries(request)
    api_result = api_gateway.get_timeseries(request)

    assert direct_result["dataset_version"] == api_result["dataset_version"]
    assert direct_result["resolved_range"] == api_result["resolved_range"]
    assert direct_result["downsampled"] is api_result["downsampled"]
    assert direct_result["series"] == api_result["series"]
    assert [series["label"] for series in direct_result["series"]] == [
        "Fuel Rate",
        "Production per Hour",
    ]


def test_validation_summary_and_named_checks_have_the_same_fixed_fixture_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = DirectDatasetGateway()
    _install_static_fixture(monkeypatch, direct)
    monkeypatch.setattr(dataset_service, "_store_validation", lambda _validation: None)

    validator = ModuleType("furnace_data.dataset.validator")
    validator.validate_dataset = lambda _frame: {
        "errors": [],
        "warnings": [],
        "checks": {
            "timestamp_monotonic": True,
            "duplicate_timestamps": True,
            "numeric_finite": True,
        },
    }
    monkeypatch.setitem(sys.modules, "furnace_data.dataset.validator", validator)

    backend = dataset_service.get_static_validation().model_dump(mode="json")
    api_client = _FixtureApiClient(
        {("GET", "/datasets/static_ml_dataset/validation"): backend}
    )
    api_result = ApiDatasetGateway("fixture-token", client=api_client).get_validation()
    direct_result = direct.get_validation()

    assert direct_result["dataset_version"] == api_result["dataset_version"]
    assert direct_result["status"] == api_result["status"]
    assert direct_result["summary"] == api_result["summary"]
    assert [(check["id"], check["status"]) for check in direct_result["checks"]] == [
        (check["id"], check["status"]) for check in api_result["checks"]
    ]


def test_hot_metal_slag_interpolation_provenance_and_visible_rows_match_fixture_api_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from furnace_data.dataset.service import DatasetService

    source = pd.DataFrame(
        {
            "chem_pct_si": [0.65, 0.70],
            "synthetic": [False, True],
            "id": [101, 102],
            "lab_sample_id": ["private-1", "private-2"],
            "import_batch_id": ["batch-1", "batch-1"],
            "source_row_number": [1, 2],
        },
        index=pd.DatetimeIndex([_START, _END], name="time"),
    )
    source.attrs["synthetic_timestamps"] = [_END]
    source.attrs["interpolated_columns"] = ["chem_pct_si", "synthetic", "id", "lab_sample_id", "import_batch_id", "source_row_number"]

    def _fetch_hot_metal(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        result = source.copy()
        result.attrs.update(source.attrs)
        return result

    monkeypatch.setattr(DatasetService, "fetch_hotmetal_hourly", _fetch_hot_metal)
    request = {
        "start": _iso(_START),
        "end": _iso(_END),
        "interval_minutes": 60,
        "interpolation": {"numeric": "time", "metadata": "forward_backward_fill"},
        "limit": 10,
        "offset": 0,
    }
    backend = data_service.preview_hot_metal_slag(
        HotMetalSlagPreviewRequest.model_validate(request)
    ).model_dump(mode="json")
    client = _FixtureApiClient(
        {("POST", "/data/hot-metal-slag/preview"): backend}
    )
    api_result = ApiDataQueryGateway("fixture-token", client=client).preview_hot_metal_slag(request)

    direct = DirectDataQueryGateway(artifacts=_DirectArtifactStore())
    # Supply the raw service fixture so the direct public boundary itself must
    # scrub every private HM/Slag column, including defensive future callers.
    monkeypatch.setattr(direct, "_fetch_hot_metal_slag", lambda _request: (source.copy(), _resolved_range()))
    direct_result = direct.preview_hot_metal_slag(request)

    assert _preview_contract({**direct_result, "source": "online"}) == _preview_contract(
        {**api_result, "source": "online"}
    )
    assert direct_result["interval_minutes"] == api_result["interval_minutes"] == 60
    assert direct_result["synthetic_row_count"] == api_result["synthetic_row_count"] == 1
    assert direct_result["interpolated_columns"] == api_result["interpolated_columns"] == [
        "chem_pct_si",
        "synthetic",
    ]
    private_columns = {"id", "lab_sample_id", "import_batch_id", "source_row_number"}
    assert all(private_columns.isdisjoint(row) for row in direct_result["rows"])
    assert all(private_columns.isdisjoint(row) for row in api_result["rows"])
    assert all(column["id"] not in private_columns for column in direct_result["columns"])
    assert all(column["id"] not in private_columns for column in api_result["columns"])

    export = direct.export_hot_metal_slag(request, idempotency_key="direct-hm-private-columns")
    csv_columns = direct.download_artifact(export["artifact_id"]).decode("utf-8").splitlines()[0].split(",")
    assert private_columns.isdisjoint(csv_columns)


@pytest.mark.parametrize("operation", ["extend", "override"])
def test_extend_and_override_are_intentionally_api_only_canonical_mutations(
    operation: str,
) -> None:
    """Direct mode rejects writes; the canonical API jobs endpoint owns them."""

    request: dict[str, Any] = {
        "operation": operation,
        "expected_dataset_version": _VERSION,
        "options": {"validate": True, "produce_download": False},
        "end": _iso(_END),
    }
    if operation == "override":
        request["start"] = _iso(_START)
    direct = DirectDatasetGateway()
    with pytest.raises(BackendApiHTTPError) as exc_info:
        direct.create_job(request, idempotency_key=f"{operation}-fixture-key")
    assert exc_info.value.error_code == "DATASET_NOT_READY"

    job_payload = {
        "job_id": f"{operation}-job",
        "status": "pending",
        "operation": operation,
        "idempotent_replay": False,
        "created_at": _iso(_START),
    }
    api_gateway, client = _api_dataset_gateway("/datasets/static_ml_dataset/jobs", job_payload)
    assert api_gateway.create_job(request, idempotency_key=f"{operation}-fixture-key")["job_id"] == f"{operation}-job"
    assert client.calls == [
        (
            "POST",
            "/datasets/static_ml_dataset/jobs",
            request,
            None,
            {
                "Authorization": "Bearer fixture-token",
                "Idempotency-Key": f"{operation}-fixture-key",
            },
        )
    ]
