"""Data Explorer gateways for the temporary direct-to-API migration.

The Streamlit page talks only to the protocols in this module.  This keeps the
API implementation free of Streamlit and, importantly, ensures that enabling
``USE_BACKEND_API_DATA_EXPLORER`` cannot accidentally initialise a local
Influx/PostgreSQL/static-dataset dependency.

The direct gateways are a deprecated rollback path.  Their furnace-data
imports deliberately live inside direct gateway methods; do not move them to
module scope.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, NotRequired, Protocol, TypedDict, runtime_checkable
from zoneinfo import ZoneInfo

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.services.api_client import ApiClient, is_wrapped_api_response
from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError, FrontendApiError


JsonDict = dict[str, Any]
DISPLAY_TIMEZONE = "Asia/Kolkata"
STATIC_DATASET_ID = "static_ml_dataset"
_SENSITIVE_HM_SLAG_COLUMNS = frozenset(
    {"id", "cast_no_ladle_spec", "lab_sample_id", "import_batch_id", "source_row_number"}
)
_STATIC_FIELD_LABEL_OVERRIDES = {
    "fuel_rate": "Fuel Rate",
    "production_per_hour": "Production per Hour",
    "eta_co": "Eta CO",
    "coke_rate": "Coke Rate",
    "actual_kg_thm": "PCI Rate",
    "unit_cost_lakhs_per_thm": "Unit Cost (lakhs/tHM)",
}
_STATIC_FIELD_UNITS = {
    "fuel_rate": "kg/tHM",
    "coke_rate": "kg/tHM",
    "actual_kg_thm": "kg/tHM",
    "production_per_hour": "t/h",
    "eta_co": "%",
    "unit_cost_lakhs_per_thm": "lakhs/tHM",
}


class PresetTimeRangeRequest(TypedDict):
    kind: Literal["preset"]
    preset_id: str


class AbsoluteTimeRangeRequest(TypedDict):
    kind: Literal["absolute"]
    start: str
    end: str


TimeRangeRequest = PresetTimeRangeRequest | AbsoluteTimeRangeRequest


class AggregationRequest(TypedDict):
    mode: str
    window_id: str


class OnlineDataPreviewRequest(TypedDict):
    source: Literal["online"]
    measurements: list[str]
    time_range: TimeRangeRequest
    aggregation: AggregationRequest | None
    fields: list[str] | None
    limit: int
    offset: int


class OfflineReportSelection(TypedDict):
    kind: Literal["report"]
    report_id: str


class OfflineTableSelection(TypedDict):
    kind: Literal["table"]
    table_id: str


class OfflineDataPreviewRequest(TypedDict):
    source: Literal["offline"]
    selection: OfflineReportSelection | OfflineTableSelection
    time_range: TimeRangeRequest
    aggregation: None
    fields: list[str] | None
    limit: int
    offset: int


DataPreviewRequest = OnlineDataPreviewRequest | OfflineDataPreviewRequest


class DataExportRequest(TypedDict):
    query: DataPreviewRequest
    format: Literal["csv"]


class NumericFilterRequest(TypedDict):
    field: str
    mode: Literal["inside", "outside"]
    minimum: float
    maximum: float


class RegressionRequest(TypedDict):
    enabled: bool
    degree: int


class ScatterAnalysisRequest(TypedDict):
    dataset_version: str
    x_field: str
    y_field: str
    filter: NotRequired[NumericFilterRequest | None]
    regression: NotRequired[RegressionRequest | None]
    max_points: int


class TimeseriesRangeRequest(TypedDict):
    start: str
    end: str


class ResampleRequest(TypedDict):
    mode: str
    window: str


class TimeseriesRequest(TypedDict):
    dataset_version: str
    fields: list[str]
    time_range: TimeseriesRangeRequest
    filter: NotRequired[NumericFilterRequest | None]
    resample: NotRequired[ResampleRequest | None]
    max_points_per_field: int


class DatasetJobOptions(TypedDict, total=False):
    validate: bool
    produce_download: bool
    rm_choice: str


class DatasetJobRequest(TypedDict, total=False):
    operation: Literal["build_range", "extend", "override"]
    start: str
    end: str
    expected_dataset_version: str
    options: DatasetJobOptions


class HotMetalSlagRequest(TypedDict):
    start: str
    end: str
    interval_minutes: int
    interpolation: NotRequired[JsonDict]
    limit: NotRequired[int]
    offset: NotRequired[int]


@runtime_checkable
class DataQueryGateway(Protocol):
    """Frontend-neutral data-query operations used by Data Explorer."""

    def get_catalog(self) -> JsonDict: ...

    def preview(self, request: DataPreviewRequest) -> JsonDict: ...

    def create_export(self, request: DataExportRequest, *, idempotency_key: str) -> JsonDict: ...

    def download_artifact(self, artifact_id: str) -> bytes: ...

    def preview_hot_metal_slag(self, request: HotMetalSlagRequest) -> JsonDict: ...

    def export_hot_metal_slag(
        self, request: HotMetalSlagRequest, *, idempotency_key: str
    ) -> JsonDict: ...


@runtime_checkable
class DatasetGateway(Protocol):
    """Frontend-neutral static-dataset operations used by Data Explorer."""

    def get_static_metadata(self) -> JsonDict: ...

    def get_scatter_analysis(self, request: ScatterAnalysisRequest) -> JsonDict: ...

    def get_timeseries(self, request: TimeseriesRequest) -> JsonDict: ...

    def create_job(self, request: DatasetJobRequest, *, idempotency_key: str) -> JsonDict: ...

    def get_job(self, job_id: str) -> JsonDict: ...

    def get_job_events(self, job_id: str, *, after: int) -> JsonDict: ...

    def cancel_job(self, job_id: str, *, idempotency_key: str | None = None) -> JsonDict: ...

    def download_job_result(self, job_id: str) -> bytes: ...

    def download_current_dataset(self) -> bytes: ...

    def get_validation(self) -> JsonDict: ...


def new_idempotency_key() -> str:
    """Return a UUID suitable for one deliberate export or job submission."""

    return str(uuid.uuid4())


def _bearer_headers(access_token: str, *, idempotency_key: str | None = None) -> dict[str, str]:
    token = str(access_token or "").strip()
    if not token:
        raise BackendApiHTTPError(
            "Data Explorer API mode requires a backend access token.",
            status_code=401,
            error_code="AUTHENTICATION_REQUIRED",
        )
    headers = {"Authorization": f"Bearer {token}"}
    if idempotency_key:
        headers["Idempotency-Key"] = str(idempotency_key)
    return headers


def _as_gateway_payload(raw: Any, client: ApiClient | Any) -> JsonDict:
    """Preserve envelope request ids and warnings while exposing endpoint data."""

    request_id = getattr(client, "last_response_request_id", None)
    warnings: list[str] = []
    data = raw
    if is_wrapped_api_response(raw):
        request_id = raw.get("request_id") or request_id
        meta = raw.get("meta") or {}
        warnings = [str(item) for item in (meta.get("warnings") or [])]
        data = raw["data"]

    if isinstance(data, Mapping):
        payload = dict(data)
    else:
        payload = {"items": data}

    data_warnings = payload.get("warnings")
    if isinstance(data_warnings, list):
        warnings = [*warnings, *(str(item) for item in data_warnings)]
    if warnings:
        payload["warnings"] = list(dict.fromkeys(warnings))
    payload["request_id"] = request_id
    return payload


class ApiDataQueryGateway:
    """Data Explorer data gateway backed exclusively by API v1."""

    def __init__(self, access_token: str, client: ApiClient | None = None) -> None:
        self.access_token = str(access_token or "").strip()
        self.client = client or ApiClient(access_token=self.access_token)

    def _headers(self, *, idempotency_key: str | None = None) -> dict[str, str]:
        return _bearer_headers(self.access_token, idempotency_key=idempotency_key)

    def get_catalog(self) -> JsonDict:
        return _as_gateway_payload(self.client.get("/data/catalog", headers=self._headers()), self.client)

    def preview(self, request: DataPreviewRequest) -> JsonDict:
        return _as_gateway_payload(
            self.client.post("/data/preview", json=dict(request), headers=self._headers()),
            self.client,
        )

    def create_export(self, request: DataExportRequest, *, idempotency_key: str) -> JsonDict:
        return _as_gateway_payload(
            self.client.post(
                "/data/export",
                json=dict(request),
                headers=self._headers(idempotency_key=idempotency_key),
            ),
            self.client,
        )

    def download_artifact(self, artifact_id: str) -> bytes:
        return self.client.download(
            f"/data/artifacts/{artifact_id}/download", headers=self._headers()
        )

    def preview_hot_metal_slag(self, request: HotMetalSlagRequest) -> JsonDict:
        return _as_gateway_payload(
            self.client.post(
                "/data/hot-metal-slag/preview",
                json=dict(request),
                headers=self._headers(),
            ),
            self.client,
        )

    def export_hot_metal_slag(
        self, request: HotMetalSlagRequest, *, idempotency_key: str
    ) -> JsonDict:
        return _as_gateway_payload(
            self.client.post(
                "/data/hot-metal-slag/export",
                json={"query": dict(request), "format": "csv"},
                headers=self._headers(idempotency_key=idempotency_key),
            ),
            self.client,
        )


class ApiDatasetGateway:
    """Data Explorer static-dataset gateway backed exclusively by API v1."""

    def __init__(self, access_token: str, client: ApiClient | None = None) -> None:
        self.access_token = str(access_token or "").strip()
        self.client = client or ApiClient(access_token=self.access_token)

    def _headers(self, *, idempotency_key: str | None = None) -> dict[str, str]:
        return _bearer_headers(self.access_token, idempotency_key=idempotency_key)

    def get_static_metadata(self) -> JsonDict:
        return _as_gateway_payload(
            self.client.get(f"/datasets/{STATIC_DATASET_ID}", headers=self._headers()), self.client
        )

    def get_scatter_analysis(self, request: ScatterAnalysisRequest) -> JsonDict:
        return _as_gateway_payload(
            self.client.post(
                f"/datasets/{STATIC_DATASET_ID}/analyses/scatter",
                json=dict(request),
                headers=self._headers(),
            ),
            self.client,
        )

    def get_timeseries(self, request: TimeseriesRequest) -> JsonDict:
        return _as_gateway_payload(
            self.client.post(
                f"/datasets/{STATIC_DATASET_ID}/timeseries",
                json=dict(request),
                headers=self._headers(),
            ),
            self.client,
        )

    def create_job(self, request: DatasetJobRequest, *, idempotency_key: str) -> JsonDict:
        return _as_gateway_payload(
            self.client.post(
                f"/datasets/{STATIC_DATASET_ID}/jobs",
                json=dict(request),
                headers=self._headers(idempotency_key=idempotency_key),
            ),
            self.client,
        )

    def get_job(self, job_id: str) -> JsonDict:
        return _as_gateway_payload(
            self.client.get(
                f"/datasets/{STATIC_DATASET_ID}/jobs/{job_id}", headers=self._headers()
            ),
            self.client,
        )

    def get_job_events(self, job_id: str, *, after: int) -> JsonDict:
        return _as_gateway_payload(
            self.client.get(
                f"/datasets/{STATIC_DATASET_ID}/jobs/{job_id}/events",
                params={"after": int(after)},
                headers=self._headers(),
            ),
            self.client,
        )

    def cancel_job(self, job_id: str, *, idempotency_key: str | None = None) -> JsonDict:
        # Cancellation deliberately has no retry policy.  A supplied key is only
        # forwarded for backend deployments that support cancellation idempotency.
        return _as_gateway_payload(
            self.client.post(
                f"/datasets/{STATIC_DATASET_ID}/jobs/{job_id}/cancel",
                json={},
                headers=self._headers(idempotency_key=idempotency_key),
            ),
            self.client,
        )

    def download_job_result(self, job_id: str) -> bytes:
        return self.client.download(
            f"/datasets/{STATIC_DATASET_ID}/jobs/{job_id}/download", headers=self._headers()
        )

    def download_current_dataset(self) -> bytes:
        return self.client.download(
            f"/datasets/{STATIC_DATASET_ID}/download", headers=self._headers()
        )

    def get_validation(self) -> JsonDict:
        return _as_gateway_payload(
            self.client.get(
                f"/datasets/{STATIC_DATASET_ID}/validation", headers=self._headers()
            ),
            self.client,
        )


@dataclass(frozen=True)
class _DirectArtifact:
    content: bytes
    filename: str
    content_type: str
    created_at: datetime
    row_count: int


class _DirectArtifactStore:
    """Small in-process artifact store used only by the deprecated direct mode."""

    def __init__(self) -> None:
        self._artifacts: dict[str, _DirectArtifact] = {}
        self._idempotency: dict[str, str] = {}
        self._lock = threading.Lock()

    def put(
        self,
        content: bytes,
        *,
        filename: str,
        row_count: int,
        idempotency_key: str,
    ) -> JsonDict:
        key = str(idempotency_key or "").strip()
        if not key:
            raise ValueError("An idempotency key is required for exports.")
        with self._lock:
            existing_id = self._idempotency.get(key)
            if existing_id and existing_id in self._artifacts:
                artifact = self._artifacts[existing_id]
                return self._response(existing_id, artifact)

            artifact_id = str(uuid.uuid4())
            artifact = _DirectArtifact(
                content=content,
                filename=_safe_filename(filename),
                content_type="text/csv",
                created_at=datetime.now(timezone.utc),
                row_count=row_count,
            )
            self._artifacts[artifact_id] = artifact
            self._idempotency[key] = artifact_id
            return self._response(artifact_id, artifact)

    @staticmethod
    def _response(artifact_id: str, artifact: _DirectArtifact) -> JsonDict:
        return {
            "artifact_id": artifact_id,
            "filename": artifact.filename,
            "content_type": artifact.content_type,
            "row_count": artifact.row_count,
            "expires_at": None,
            "warnings": [
                "Direct mode is deprecated; the download is held in this Streamlit process only."
            ],
            "request_id": None,
        }

    def get(self, artifact_id: str) -> bytes:
        with self._lock:
            artifact = self._artifacts.get(str(artifact_id))
        if artifact is None:
            raise BackendApiHTTPError(
                "Direct-mode artifact was not found. Create the export again.",
                status_code=404,
                error_code="ARTIFACT_NOT_FOUND",
            )
        return artifact.content


_DIRECT_ARTIFACTS = _DirectArtifactStore()
_DIRECT_JOBS: dict[str, JsonDict] = {}
_DIRECT_JOBS_LOCK = threading.Lock()


def _safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "export.csv")).strip("._")
    if not name:
        name = "export.csv"
    return name if name.lower().endswith(".csv") else f"{name}.csv"


def _public_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _table_public_id(table_name: str) -> str:
    """Return the API-compatible opaque ID for an allowlisted offline table."""

    digest = hashlib.sha256(str(table_name).encode("utf-8")).hexdigest()[:16]
    return f"offline-table-{digest}"
def _canonical_field_id(value: object) -> str:
    raw = str(value)
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    aliases = {
        "unitcost_lakhs_thm": "unit_cost_lakhs_per_thm",
        "unitcost_lakhs_per_thm": "unit_cost_lakhs_per_thm",
        "unit_cost_lakhs_thm": "unit_cost_lakhs_per_thm",
    }
    return aliases.get(normalized, normalized or "field")


def _label_for_id(value: str) -> str:
    if value == "unit_cost_lakhs_per_thm":
        return "Unit Cost (lakhs/tHM)"
    return str(value).replace("_", " ").title()


def _catalog_dtype(identifier: str) -> str:
    """Mirror the v1 catalog's conservative offline-field type contract."""

    name = identifier.lower()
    if name in {"time", "timestamp", "date_time", "created_at", "updated_at"}:
        return "datetime"
    if name.startswith("is_"):
        return "boolean"
    if any(token in name for token in ("id", "code", "name", "notes", "description", "type")):
        return "string"
    return "number"


def _static_label_for_id(value: str) -> str:
    """Use the canonical static-dataset labels in both gateway modes."""

    return _STATIC_FIELD_LABEL_OVERRIDES.get(value, _label_for_id(value))


def _parse_aware_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BackendApiHTTPError(
            "Timestamp must include a timezone offset.",
            status_code=422,
            error_code="INVALID_TIME_RANGE",
        )
    return parsed.astimezone(timezone.utc)


def _utc_iso(value: Any, pd: Any) -> str | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(DISPLAY_TIMEZONE)
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _json_value(value: Any, pd: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        if isinstance(value, datetime):
            return _utc_iso(value, pd)
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_value(value.item(), pd)
        except (TypeError, ValueError):
            pass
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_rows(frame: Any, pd: Any, *, field_ids: Mapping[str, str] | None = None) -> list[JsonDict]:
    """Convert a DataFrame to JSON-native API rows with UTC timestamps."""

    output = frame.copy()
    if isinstance(output.index, pd.DatetimeIndex):
        # API v1 normalises all query and HM/Slag preview indexes to ``time``.
        # The direct rollback DTO must retain that public contract too.
        output.index.name = "time"
        output = output.reset_index()
    if field_ids:
        output = output.rename(columns=dict(field_ids))
    rows: list[JsonDict] = []
    for record in output.to_dict(orient="records"):
        item: JsonDict = {}
        for key, value in record.items():
            field = "time" if str(key) in {"time", "timestamp", "date_time", "time (IST)"} else str(key)
            if field == "time":
                item[field] = _utc_iso(value, pd)
            else:
                item[field] = _json_value(value, pd)
        rows.append(item)
    return rows


def _column_metadata(
    frame: Any,
    pd: Any,
    *,
    field_ids: Mapping[str, str] | None = None,
    include_index: bool = False,
) -> list[JsonDict]:
    columns: list[JsonDict] = []
    if include_index and isinstance(frame.index, pd.DatetimeIndex):
        columns.append({"id": "time", "label": "Time", "dtype": "datetime", "unit": None})
    for column in frame.columns:
        public_id = (field_ids or {}).get(str(column), _canonical_field_id(column))
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            dtype = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            dtype = "boolean"
        elif pd.api.types.is_integer_dtype(series):
            dtype = "integer"
        elif pd.api.types.is_numeric_dtype(series):
            dtype = "number"
        else:
            dtype = "string"
        columns.append(
            {
                "id": public_id,
                "label": _label_for_id(public_id),
                "dtype": dtype,
                "unit": None,
            }
        )
    return columns


def _hm_timestamp_utc(value: Any, pd: Any) -> Any | None:
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(DISPLAY_TIMEZONE)
        return stamp.tz_convert("UTC")
    except (TypeError, ValueError):
        return None


def _hot_metal_slag_provenance(frame: Any, pd: Any) -> tuple[int, list[str]]:
    """Expose only explicit, public-safe shared-service interpolation provenance."""

    raw_columns = frame.attrs.get("interpolated_columns")
    interpolated_columns: list[str] = []
    if isinstance(raw_columns, (list, tuple, set, frozenset)):
        seen: set[str] = set()
        for raw_column in raw_columns:
            column = str(raw_column)
            if (
                column not in seen
                and column in frame.columns
                and column not in _SENSITIVE_HM_SLAG_COLUMNS
            ):
                seen.add(column)
                interpolated_columns.append(column)

    raw_timestamps = frame.attrs.get("synthetic_timestamps")
    if isinstance(raw_timestamps, (list, tuple, set, frozenset)) and isinstance(frame.index, pd.DatetimeIndex):
        synthetic_timestamps = {
            stamp
            for raw_timestamp in raw_timestamps
            if (stamp := _hm_timestamp_utc(raw_timestamp, pd)) is not None
        }
        return (
            sum(
                (stamp := _hm_timestamp_utc(timestamp, pd)) is not None and stamp in synthetic_timestamps
                for timestamp in frame.index
            ),
            interpolated_columns,
        )

    # Count-only provenance cannot identify filtered rows.  Preserve the API's
    # conservative compatibility rule instead of deriving a count from columns.
    try:
        synthetic_row_count = max(0, int(frame.attrs.get("synthetic_row_count", 0)))
    except (TypeError, ValueError):
        synthetic_row_count = 0
    return min(synthetic_row_count, len(frame.index)), interpolated_columns


class DirectDataQueryGateway:
    """Deprecated direct rollback gateway.

    Removal criterion: remove this class after API parity has been demonstrated
    in production and ``USE_BACKEND_API_DATA_EXPLORER`` is the only supported
    page mode.  Keep all furnace-data imports local to this class.
    """

    def __init__(self, artifacts: _DirectArtifactStore | None = None) -> None:
        self._artifacts = artifacts or _DIRECT_ARTIFACTS

    def get_catalog(self) -> JsonDict:
        # Direct-only imports: do not move these to module scope.
        from furnace_data.influx.query import TIMEDELTAS, WINDOWING, field_labels, measurement_label
        from furnace_data.offline import OFFLINE_REPORT_MAP, OFFLINE_TABLES

        measurements = sorted({*field_labels.__globals__.get("_config", {}).get("data_mapping", {}).keys()})
        online_measurements = [
            {
                "id": measurement,
                "label": measurement_label(measurement),
                "fields": [
                    {
                        "id": field,
                        "label": label,
                        "dtype": "number",
                        "unit": None,
                    }
                    for field, label in field_labels(measurement).items()
                ],
            }
            for measurement in measurements
        ]
        private_offline_columns = {"source_table"}
        hm_slag_tables = set(OFFLINE_REPORT_MAP.get("HM_SLAG", []))

        def offline_fields(columns: Any, *, hide_hm_slag_columns: bool) -> tuple[list[JsonDict], bool]:
            if columns is None:
                return [], False
            names = set(columns)
            names.difference_update(private_offline_columns)
            if hide_hm_slag_columns:
                names.difference_update(_SENSITIVE_HM_SLAG_COLUMNS)
            return [
                {
                    "id": _canonical_field_id(column),
                    "label": _label_for_id(_canonical_field_id(column)),
                    "dtype": _catalog_dtype(_canonical_field_id(column)),
                    "unit": None,
                }
                for column in sorted(names)
            ], True

        def report_fields(report: str) -> tuple[list[JsonDict], bool]:
            columns: set[str] = set()
            for table in OFFLINE_REPORT_MAP.get(report, []):
                table_columns = OFFLINE_TABLES.get(table)
                if table_columns is None:
                    return [], False
                columns.update(table_columns)
            return offline_fields(columns, hide_hm_slag_columns=report == "HM_SLAG")

        offline_reports = [
            {
                "id": _public_id(report),
                "label": report.replace("_", " ").title(),
                "fields": report_columns,
                "supports_field_selection": selectable,
            }
            for report in sorted(OFFLINE_REPORT_MAP)
            for report_columns, selectable in [report_fields(report)]
        ]
        offline_tables = [
            {
                "id": public_id,
                # Keep direct fallback labels opaque too: physical table
                # names must not leak through a display label.
                "label": f"Offline table {table_number}",
                "fields": table_fields,
                "supports_field_selection": selectable,
                "supports_aggregation": False,
            }
            for table_number, (public_id, table, columns) in enumerate(
                sorted(
                    ((_table_public_id(table), table, columns) for table, columns in OFFLINE_TABLES.items()),
                    key=lambda item: item[0],
                ),
                start=1,
            )
            for table_fields, selectable in [
                offline_fields(columns, hide_hm_slag_columns=table in hm_slag_tables)
            ]
        ]

        return {
            "display_timezone": DISPLAY_TIMEZONE,
            "online_measurements": online_measurements,
            "time_presets": [
                {
                    "id": _public_id(label),
                    "label": label.title(),
                    "duration_seconds": int(duration.total_seconds()),
                    "sources": ["online", "offline"],
                }
                for label, duration in TIMEDELTAS.items()
            ],
            "aggregation_windows": [
                {"id": "none", "label": "None", "duration_seconds": None},
                *[
                    {
                        "id": _public_id(label),
                        "label": label.title(),
                        "duration_seconds": _window_seconds(value),
                    }
                    for label, value in WINDOWING.items()
                ],
            ],
            "offline_reports": offline_reports,
            "offline_tables": offline_tables,
            "limits": {
                "max_preview_rows": 500,
                "max_selected_fields": 20,
                "max_scatter_points": 5000,
                "max_timeseries_points_per_field": 5000,
                "max_hm_slag_interval_minutes": 600,
            },
            "warnings": ["Direct data access is deprecated; enable the Data Explorer API."],
            "request_id": None,
        }

    def preview(self, request: DataPreviewRequest) -> JsonDict:
        frame, field_ids, resolved_range = self._fetch_query(dict(request), apply_page=True)
        limit = max(1, int(request.get("limit", 500)))
        offset = max(0, int(request.get("offset", 0)))
        total_rows = int(len(frame.index))
        page = frame.iloc[offset : offset + limit]
        pd = self._pd()
        return {
            "columns": _column_metadata(frame, pd, field_ids=field_ids, include_index=True),
            "rows": _json_rows(page, pd, field_ids=field_ids),
            "returned_rows": int(len(page.index)),
            "total_rows": total_rows,
            "row_count": total_rows,
            "offset": offset,
            "truncated": offset + len(page.index) < total_rows,
            "resolved_range": resolved_range,
            "source": request.get("source"),
            "warnings": ["Direct data access is deprecated."],
            "request_id": None,
        }

    def create_export(self, request: DataExportRequest, *, idempotency_key: str) -> JsonDict:
        query = dict(request["query"])
        frame, _field_ids, _range = self._fetch_query(query, apply_page=False)
        filename = f"{query.get('source', 'data')}_data.csv"
        return self._store_frame_export(frame, filename=filename, idempotency_key=idempotency_key)

    def download_artifact(self, artifact_id: str) -> bytes:
        return self._artifacts.get(artifact_id)

    def preview_hot_metal_slag(self, request: HotMetalSlagRequest) -> JsonDict:
        frame, resolved_range = self._fetch_hot_metal_slag(dict(request))
        # Keep the public boundary defensive even if a rollback implementation
        # or test double returns an unsanitized HM/Slag frame.
        frame = frame.drop(columns=list(_SENSITIVE_HM_SLAG_COLUMNS), errors="ignore")
        limit = max(1, int(request.get("limit", 500)))
        offset = max(0, int(request.get("offset", 0)))
        total_rows = int(len(frame.index))
        page = frame.iloc[offset : offset + limit]
        pd = self._pd()
        synthetic, interpolated_columns = _hot_metal_slag_provenance(frame, pd)
        return {
            "columns": _column_metadata(frame, pd, include_index=True),
            "rows": _json_rows(page, pd),
            "returned_rows": int(len(page.index)),
            "total_rows": total_rows,
            "offset": offset,
            "truncated": offset + len(page.index) < total_rows,
            "resolved_range": resolved_range,
            "interval_minutes": int(request["interval_minutes"]),
            "synthetic_row_count": synthetic,
            "interpolated_columns": interpolated_columns,
            "warnings": ["Direct data access is deprecated."],
            "request_id": None,
        }

    def export_hot_metal_slag(
        self, request: HotMetalSlagRequest, *, idempotency_key: str
    ) -> JsonDict:
        frame, _range = self._fetch_hot_metal_slag(dict(request))
        frame = frame.drop(columns=list(_SENSITIVE_HM_SLAG_COLUMNS), errors="ignore")
        return self._store_frame_export(
            frame,
            filename="hot_metal_slag.csv",
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _pd() -> Any:
        import pandas as pd

        return pd

    def _fetch_query(
        self, request: JsonDict, *, apply_page: bool
    ) -> tuple[Any, dict[str, str], JsonDict]:
        source = str(request.get("source") or "").lower()
        if source == "online":
            return self._fetch_online(request, apply_page=apply_page)
        if source == "offline":
            return self._fetch_offline(request, apply_page=apply_page)
        raise BackendApiHTTPError(
            "Unknown data source.", status_code=422, error_code="INVALID_SOURCE"
        )

    def _fetch_online(
        self, request: JsonDict, *, apply_page: bool
    ) -> tuple[Any, dict[str, str], JsonDict]:
        # Direct-only import.
        from furnace_data.influx.online import fetch_online_df

        measurements = [str(item) for item in request.get("measurements") or []]
        if not measurements:
            raise BackendApiHTTPError(
                "Select at least one online measurement.",
                status_code=422,
                error_code="INVALID_MEASUREMENT",
            )
        legacy_range, start, end = self._legacy_time_range(request.get("time_range") or {})
        aggregation = request.get("aggregation")
        window = "None"
        request_type = "ts"
        if isinstance(aggregation, Mapping):
            window = self._legacy_window(str(aggregation.get("window_id") or "none"))
            request_type = "windowed-average" if window != "None" else "ts"
        frame = fetch_online_df(
            measurements,
            legacy_range,
            request_type=request_type,
            window_by=window,
            start_time_override=start,
            end_time_override=end,
            column_naming="field",
        )
        fields = [str(item) for item in request.get("fields") or []]
        if fields:
            selected = [field for field in fields if field in frame.columns]
            frame = frame.loc[:, selected]
        field_ids = {str(column): _canonical_field_id(column) for column in frame.columns}
        return frame, field_ids, self._resolved_range(frame, start=start, end=end)

    def _fetch_offline(
        self, request: JsonDict, *, apply_page: bool
    ) -> tuple[Any, dict[str, str], JsonDict]:
        # Direct-only import.
        from furnace_data.offline import OFFLINE_REPORT_MAP, OFFLINE_TABLES, fetch_offline_data, fetch_offline_report

        selection = request.get("selection") or {}
        if not isinstance(selection, Mapping):
            raise BackendApiHTTPError("Offline selection is required.", status_code=422, error_code="INVALID_FILTER")
        legacy_range, start, end = self._legacy_time_range(request.get("time_range") or {})
        if start is not None and end is not None:
            direct_range: Any = (start, end)
        else:
            direct_range = legacy_range
        kind = str(selection.get("kind") or "")
        private_columns = {"source_table"}
        if kind == "report":
            report_id = str(selection.get("report_id") or "")
            report = next((name for name in OFFLINE_REPORT_MAP if _public_id(name) == report_id), None)
            if report is None:
                raise BackendApiHTTPError("Unknown offline report.", status_code=422, error_code="INVALID_REPORT")
            frame = fetch_offline_report(report, direct_range, query_type="ts")
            if report == "HM_SLAG":
                private_columns.update(_SENSITIVE_HM_SLAG_COLUMNS)
        elif kind == "table":
            public_id = str(selection.get("table_id") or "")
            table = next((name for name in OFFLINE_TABLES if _table_public_id(name) == public_id), None)
            if table is None:
                raise BackendApiHTTPError("Unknown offline table.", status_code=422, error_code="INVALID_TABLE")
            columns_by_id = {_canonical_field_id(column): str(column) for column in OFFLINE_TABLES[table] or []}
            fields = [str(item) for item in request.get("fields") or []]
            columns = [columns_by_id[field] for field in fields if field in columns_by_id] or None
            frame = fetch_offline_data(table, direct_range, query_type="ts", columns=columns)
            if table in set(OFFLINE_REPORT_MAP.get("HM_SLAG", [])):
                private_columns.update(_SENSITIVE_HM_SLAG_COLUMNS)
        else:
            raise BackendApiHTTPError("Unknown offline selection.", status_code=422, error_code="INVALID_FILTER")
        # The legacy report helper annotates rows with a physical database
        # table name.  That is internal provenance, never public data.
        frame = frame.drop(columns=list(private_columns), errors="ignore")
        field_ids = {str(column): _canonical_field_id(column) for column in frame.columns}
        return frame, field_ids, self._resolved_range(frame, start=start, end=end)

    def _fetch_hot_metal_slag(self, request: JsonDict) -> tuple[Any, JsonDict]:
        # Direct-only imports.  This operation is only invoked after a form
        # submission, never while constructing the page or gateway.
        from furnace_data.dataset.fetcher import DatasetFetcher

        start = _parse_aware_datetime(request["start"])
        end = _parse_aware_datetime(request["end"])
        if start > end:
            raise BackendApiHTTPError("Start must not be after end.", status_code=422, error_code="INVALID_TIME_RANGE")
        interval = int(request.get("interval_minutes") or 0)
        if not 1 <= interval <= 600:
            raise BackendApiHTTPError("Interval must be between 1 and 600 minutes.", status_code=422, error_code="INVALID_FILTER")
        frame = DatasetFetcher().service.fetch_hotmetal_hourly(
            start_date=start.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).date(),
            end_date=end.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).date(),
            interval_minutes=interval,
        )
        # Match v1 public data semantics: normalise timestamps, apply the
        # requested UTC bounds, and remove private fields before serialising
        # either previews or exports.  Pandas preserves the service attrs used
        # for interpolation provenance through these operations.
        frame = self._utc_index(frame)
        frame = frame.loc[(frame.index >= self._pd().Timestamp(start)) & (frame.index <= self._pd().Timestamp(end))]
        frame = frame.drop(columns=[column for column in _SENSITIVE_HM_SLAG_COLUMNS if column in frame], errors="ignore")
        return frame, {"start": _utc_iso(start, self._pd()), "end": _utc_iso(end, self._pd())}

    def _legacy_time_range(self, request: Mapping[str, Any]) -> tuple[str, datetime | None, datetime | None]:
        kind = str(request.get("kind") or "")
        if kind == "preset":
            preset_id = str(request.get("preset_id") or "")
            legacy = preset_id.replace("_", " ").lower()
            return legacy, None, None
        if kind == "absolute":
            start = _parse_aware_datetime(str(request.get("start") or ""))
            end = _parse_aware_datetime(str(request.get("end") or ""))
            if start > end:
                raise BackendApiHTTPError("Start must not be after end.", status_code=422, error_code="INVALID_TIME_RANGE")
            return "last 1 hour", start, end
        raise BackendApiHTTPError("Unknown time range.", status_code=422, error_code="INVALID_TIME_RANGE")

    @staticmethod
    def _legacy_window(window_id: str) -> str:
        if window_id == "none":
            return "None"
        return str(window_id).replace("_", " ")

    def _resolved_range(
        self, frame: Any, *, start: datetime | None, end: datetime | None
    ) -> JsonDict:
        pd = self._pd()
        # Absolute ranges are caller intent and match the resolved API range
        # even when the source has no samples at either boundary.
        if start is not None and end is not None:
            return {"start": _utc_iso(start, pd), "end": _utc_iso(end, pd)}
        if isinstance(frame.index, pd.DatetimeIndex) and not frame.empty:
            return {"start": _utc_iso(frame.index.min(), pd), "end": _utc_iso(frame.index.max(), pd)}
        return {"start": _utc_iso(start, pd), "end": _utc_iso(end, pd)}

    def _store_frame_export(self, frame: Any, *, filename: str, idempotency_key: str) -> JsonDict:
        content = frame.to_csv(index=True, date_format="%Y-%m-%dT%H:%M:%SZ").encode("utf-8")
        return self._artifacts.put(
            content,
            filename=filename,
            row_count=int(len(frame.index)),
            idempotency_key=idempotency_key,
        )


class DirectDatasetGateway:
    """Deprecated direct static-dataset rollback gateway.

    Its only build operation is an explicit, non-mutating ``build_range``
    candidate query.  Direct extend/override are intentionally disabled so this
    fallback cannot modify the packaged YAML or a canonical dataset on rerun; the API job service is the sole canonical mutation path.
    """

    def __init__(self, artifacts: _DirectArtifactStore | None = None) -> None:
        self._artifacts = artifacts or _DIRECT_ARTIFACTS

    def get_static_metadata(self) -> JsonDict:
        frame, fields, version = self._load_static_dataset()
        pd = self._pd()
        return {
            "dataset_id": STATIC_DATASET_ID,
            "version": version,
            "etag": version,
            "status": "ready",
            "row_count": int(len(frame.index)),
            "column_count": int(len(frame.columns)),
            "columns": [
                {
                    "id": public_id,
                    "label": _static_label_for_id(public_id),
                    "dtype": self._static_dtype(frame[column]),
                    "unit": _STATIC_FIELD_UNITS.get(public_id),
                    "plottable": bool(pd.api.types.is_numeric_dtype(frame[column])),
                    "filterable": bool(pd.api.types.is_numeric_dtype(frame[column])),
                }
                for column, public_id in fields.items()
            ],
            "time_column": {"id": "timestamp", "timezone": "UTC"},
            "range": self._range(frame),
            "last_built_at": None,
            "validation_status": "not_run",
            "download_available": True,
            "warnings": ["Direct dataset access is deprecated."],
            "request_id": None,
        }

    def get_scatter_analysis(self, request: ScatterAnalysisRequest) -> JsonDict:
        frame, fields, version = self._load_static_dataset()
        self._assert_version(request["dataset_version"], version)
        pd, np = self._pd(), self._np()
        actual_by_id = {public: actual for actual, public in fields.items()}
        x_name = actual_by_id.get(str(request["x_field"]))
        y_name = actual_by_id.get(str(request["y_field"]))
        if not x_name or not y_name:
            raise BackendApiHTTPError("Scatter field is unavailable.", status_code=422, error_code="INVALID_FIELD")
        if not pd.api.types.is_numeric_dtype(frame[x_name]) or not pd.api.types.is_numeric_dtype(frame[y_name]):
            raise BackendApiHTTPError("Scatter fields must be numeric.", status_code=422, error_code="INVALID_FIELD")

        filtered = frame.copy()
        filter_request = request.get("filter")
        if filter_request:
            filtered = self._apply_numeric_filter(filtered, filter_request, actual_by_id)

        x = pd.to_numeric(filtered[x_name], errors="coerce")
        y = pd.to_numeric(filtered[y_name], errors="coerce")
        null_rows = int((x.isna() | y.isna()).sum())
        finite = np.isfinite(x.fillna(np.nan)) & np.isfinite(y.fillna(np.nan))
        valid = filtered.loc[finite].copy()
        x_valid = pd.to_numeric(valid[x_name], errors="coerce").astype(float)
        y_valid = pd.to_numeric(valid[y_name], errors="coerce").astype(float)
        non_finite = int(len(filtered.index) - null_rows - len(valid.index))
        total = int(len(valid.index))
        max_points = max(1, int(request.get("max_points", 5000)))
        sampled = self._deterministic_sample(valid, max_points)
        regression = self._regression(x_valid, y_valid, request.get("regression"))
        return {
            "dataset_version": version,
            "x": [_json_value(value, pd) for value in sampled[x_name].tolist()],
            "y": [_json_value(value, pd) for value in sampled[y_name].tolist()],
            "total_matching_rows": total,
            "returned_points": int(len(sampled.index)),
            "downsampled": total > len(sampled.index),
            "regression": regression,
            "dropped_rows": {"null": null_rows, "non_numeric": 0, "non_finite": non_finite},
            "warnings": ["Direct dataset access is deprecated."],
            "request_id": None,
        }

    def get_timeseries(self, request: TimeseriesRequest) -> JsonDict:
        frame, fields, version = self._load_static_dataset()
        self._assert_version(request["dataset_version"], version)
        pd = self._pd()
        actual_by_id = {public: actual for actual, public in fields.items()}
        selected = [(field_id, actual_by_id.get(field_id)) for field_id in request.get("fields", [])]
        if not selected or any(actual is None for _field_id, actual in selected):
            raise BackendApiHTTPError("Select available time-series fields.", status_code=422, error_code="INVALID_FIELD")
        start = _parse_aware_datetime(request["time_range"]["start"])
        end = _parse_aware_datetime(request["time_range"]["end"])
        if start > end:
            raise BackendApiHTTPError("Start must not be after end.", status_code=422, error_code="INVALID_TIME_RANGE")
        indexed = self._utc_index(frame)
        indexed = indexed.loc[(indexed.index >= pd.Timestamp(start)) & (indexed.index <= pd.Timestamp(end))]
        filter_request = request.get("filter")
        if filter_request:
            indexed = self._apply_numeric_filter(indexed, filter_request, actual_by_id)
        resample = request.get("resample")
        if resample and str(resample.get("mode") or "").lower() != "none":
            window = str(resample.get("window") or "")
            if not window:
                raise BackendApiHTTPError("A resampling window is required.", status_code=422, error_code="INVALID_FILTER")
            indexed = indexed.resample(window).mean(numeric_only=True)
        indexed = indexed[~indexed.index.duplicated(keep="last")].sort_index()
        max_points = max(1, int(request.get("max_points_per_field", 5000)))
        series: list[JsonDict] = []
        downsampled = False
        for field_id, actual in selected:
            assert actual is not None
            values = pd.to_numeric(indexed[actual], errors="coerce")
            field_frame = pd.DataFrame({"value": values}).dropna()
            sampled = self._deterministic_sample(field_frame, max_points)
            downsampled = downsampled or len(sampled.index) < len(field_frame.index)
            series.append(
                {
                    "field": field_id,
                    "label": _static_label_for_id(field_id),
                    "unit": _STATIC_FIELD_UNITS.get(field_id),
                    "points": [
                        {"timestamp": _utc_iso(stamp, pd), "value": _json_value(value, pd)}
                        for stamp, value in sampled["value"].items()
                    ],
                }
            )
        return {
            "dataset_version": version,
            "series": series,
            "resolved_range": {"start": _utc_iso(start, pd), "end": _utc_iso(end, pd)},
            "downsampled": downsampled,
            "warnings": ["Direct dataset access is deprecated."],
            "request_id": None,
        }

    def create_job(self, request: DatasetJobRequest, *, idempotency_key: str) -> JsonDict:
        operation = str(request.get("operation") or "")
        if operation != "build_range":
            raise BackendApiHTTPError(
                "Direct mode supports only non-mutating build-range candidates. Enable the Data Explorer API for canonical dataset mutations.",
                status_code=409,
                error_code="DATASET_NOT_READY",
            )
        start = _parse_aware_datetime(str(request.get("start") or ""))
        end = _parse_aware_datetime(str(request.get("end") or ""))
        if start > end:
            raise BackendApiHTTPError("Start must not be after end.", status_code=422, error_code="INVALID_TIME_RANGE")
        # Direct-only import.  Use the canonical get_dataset() name; never use
        # the removed get_ml_dataset() compatibility alias.
        from furnace_data.dataset.fetcher import DatasetFetcher

        options = request.get("options") or {}
        rm_choice = str(options.get("rm_choice") or "RM Charge")
        job_id = str(uuid.uuid4())
        job: JsonDict = {
            "job_id": job_id,
            "status": "running",
            "operation": operation,
            "events": [{"sequence": 1, "stage": "fetch", "percent": 10, "message": "Fetching selected range."}],
            "request_id": None,
        }
        with _DIRECT_JOBS_LOCK:
            _DIRECT_JOBS[job_id] = job
        try:
            frame = DatasetFetcher().get_dataset(
                start_date=start.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).date(),
                end_date=end.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).date(),
                rm_choice=rm_choice,
                cache_override=True,
            )
            artifact_id = None
            if bool(options.get("produce_download", True)):
                artifact = self._store_candidate(frame, idempotency_key=idempotency_key)
                artifact_id = artifact["artifact_id"]
            job.update(
                {
                    "status": "completed",
                    "progress": 100,
                    "result": {"row_count": int(len(frame.index)), "artifact_id": artifact_id},
                    "artifact_id": artifact_id,
                    "events": [
                        *job["events"],
                        {"sequence": 2, "stage": "complete", "percent": 100, "message": "Candidate dataset is ready."},
                    ],
                }
            )
        except Exception as exc:  # direct compatibility errors are surfaced, not hidden
            job.update(
                {
                    "status": "failed",
                    "error_code": "DATA_SOURCE_UNAVAILABLE",
                    "error_message": "The direct candidate build failed. Check backend connectivity and try again.",
                    "events": [
                        *job["events"],
                        {"sequence": 2, "stage": "failed", "percent": 100, "message": "Candidate build failed."},
                    ],
                }
            )
        return {key: value for key, value in job.items() if key != "events"}

    def get_job(self, job_id: str) -> JsonDict:
        job = self._job(job_id)
        return {key: value for key, value in job.items() if key != "events"}

    def get_job_events(self, job_id: str, *, after: int) -> JsonDict:
        job = self._job(job_id)
        events = [event for event in job.get("events", []) if int(event["sequence"]) > int(after)]
        return {"job_id": job_id, "events": events, "request_id": None}

    def cancel_job(self, job_id: str, *, idempotency_key: str | None = None) -> JsonDict:
        job = self._job(job_id)
        if job.get("status") not in {"queued", "running"}:
            raise BackendApiHTTPError(
                "This direct-mode job is no longer cancellable.",
                status_code=409,
                error_code="DATASET_JOB_NOT_CANCELLABLE",
            )
        job["status"] = "cancelled"
        events = job.setdefault("events", [])
        events.append(
            {"sequence": len(events) + 1, "stage": "cancelled", "percent": 100, "message": "Cancelled."}
        )
        return self.get_job(job_id)

    def download_job_result(self, job_id: str) -> bytes:
        job = self._job(job_id)
        artifact_id = job.get("artifact_id")
        if not artifact_id:
            raise BackendApiHTTPError("Job has no downloadable result.", status_code=404, error_code="ARTIFACT_NOT_FOUND")
        return self._artifacts.get(str(artifact_id))

    def download_current_dataset(self) -> bytes:
        frame, _fields, _version = self._load_static_dataset()
        return frame.to_csv(index=True, date_format="%Y-%m-%dT%H:%M:%SZ").encode("utf-8")

    def get_validation(self) -> JsonDict:
        # Direct-only import.  Validation is never performed while loading the
        # page; callers invoke this method only after an explicit request.
        from furnace_data.dataset.validator import validate_dataset

        frame, _fields, version = self._load_static_dataset()
        report = validate_dataset(frame)
        errors = list(report.get("errors") or [])
        warnings = list(report.get("warnings") or [])
        checks = [
            {"id": str(key), "status": "passed", "message": "Checked in direct compatibility mode.", "details": {}}
            for key in (report.get("checks") or {})
        ]
        return {
            "dataset_version": version,
            "status": "failed" if errors else "passed",
            "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "summary": {"errors": len(errors), "warnings": len(warnings)},
            "checks": checks,
            "warnings": [*warnings, "Direct dataset validation is deprecated."],
            "request_id": None,
        }

    @staticmethod
    def _pd() -> Any:
        import pandas as pd

        return pd

    @staticmethod
    def _np() -> Any:
        import numpy as np

        return np

    def _static_dtype(self, series: Any) -> str:
        pd = self._pd()
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series) or pd.to_numeric(series, errors="coerce").notna().any():
            return "number"
        return "string"

    def _load_static_dataset(self) -> tuple[Any, dict[str, str], str]:
        # Direct-only imports.  Do not call load_static_dataset without an
        # existing path: its compatibility fallback may rebuild a local file.
        from furnace_data.config import load_config
        from furnace_data.dataset.static_csv import get_static_dataset_path, load_static_dataset

        config = load_config("setting_ds_dv.yml")
        configured_path = Path(str(config.get("DATA") or ""))
        runtime_path = get_static_dataset_path(config.get("DATA"))
        candidates = [runtime_path]
        if configured_path:
            candidates.append(configured_path if configured_path.is_absolute() else Path.cwd() / configured_path)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise BackendApiHTTPError(
                "Static ML dataset is not available in direct mode.",
                status_code=404,
                error_code="DATASET_NOT_AVAILABLE",
            )
        frame = load_static_dataset(path)
        if frame is None or frame.empty:
            raise BackendApiHTTPError(
                "Static ML dataset has no rows.", status_code=404, error_code="DATASET_NOT_AVAILABLE"
            )
        frame = self._ensure_unit_cost(frame)
        field_map = self._static_field_map(frame, config)
        version = self._dataset_version(frame, path)
        return frame, field_map, version

    def _ensure_unit_cost(self, frame: Any) -> Any:
        output = frame.copy()
        target = "UNITCOST LAKHS/THM"
        existing = next((column for column in output.columns if _canonical_field_id(column) == "unit_cost_lakhs_per_thm"), None)
        if existing:
            return output
        coke = next((column for column in output.columns if _canonical_field_id(column) in {"coke_rate", "coke_rate_kg_thm"}), None)
        pci = next((column for column in output.columns if _canonical_field_id(column) in {"pci_kg_thm", "actual_kg_thm"}), None)
        if coke and pci:
            pd = self._pd()
            output[target] = (pd.to_numeric(output[coke], errors="coerce") + (13250 / 25000) * pd.to_numeric(output[pci], errors="coerce")) * 0.25
        return output

    @staticmethod
    def _static_field_map(frame: Any, config: Mapping[str, Any]) -> dict[str, str]:
        reverse_rename = {
            str(value): str(key)
            for key, value in (config.get("rename_dict") or {}).items()
            if value is not None
        }
        mapped: dict[str, str] = {}
        used: set[str] = set()
        for column in frame.columns:
            raw_id = reverse_rename.get(str(column), _canonical_field_id(column))
            public_id = _canonical_field_id(raw_id)
            if public_id == "unit_cost_lakhs_thm":
                public_id = "unit_cost_lakhs_per_thm"
            if public_id in used:
                public_id = f"{public_id}_{len(used)}"
            mapped[str(column)] = public_id
            used.add(public_id)
        return mapped

    def _dataset_version(self, frame: Any, path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(path.stat().st_mtime_ns).encode("ascii"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update("|".join(str(column) for column in frame.columns).encode("utf-8"))
        digest.update(str(len(frame.index)).encode("ascii"))
        return f"direct-{digest.hexdigest()[:20]}"

    def _assert_version(self, requested: str, actual: str) -> None:
        if str(requested) != actual:
            raise BackendApiHTTPError(
                "The static dataset changed; refresh metadata and retry.",
                status_code=409,
                error_code="DATASET_VERSION_CONFLICT",
            )

    def _apply_numeric_filter(
        self, frame: Any, request: Mapping[str, Any], actual_by_id: Mapping[str, str]
    ) -> Any:
        pd = self._pd()
        field = actual_by_id.get(str(request.get("field") or ""))
        if not field:
            raise BackendApiHTTPError("Filter field is unavailable.", status_code=422, error_code="INVALID_FILTER")
        minimum = float(request.get("minimum"))
        maximum = float(request.get("maximum"))
        if minimum > maximum:
            raise BackendApiHTTPError("Filter minimum must not exceed maximum.", status_code=422, error_code="INVALID_FILTER")
        series = pd.to_numeric(frame[field], errors="coerce")
        mode = str(request.get("mode") or "inside")
        if mode == "inside":
            mask = (series >= minimum) & (series <= maximum)
        elif mode == "outside":
            mask = (series < minimum) | (series > maximum)
        else:
            raise BackendApiHTTPError("Unknown filter mode.", status_code=422, error_code="INVALID_FILTER")
        return frame.loc[mask]

    def _regression(self, x: Any, y: Any, request: Any) -> JsonDict | None:
        if not isinstance(request, Mapping) or not bool(request.get("enabled")):
            return None
        np = self._np()
        degree = int(request.get("degree") or 1)
        if not 1 <= degree <= 5 or len(x) < degree + 1 or x.nunique() < degree + 1:
            raise BackendApiHTTPError(
                "Insufficient distinct numeric rows for regression.",
                status_code=422,
                error_code="INSUFFICIENT_REGRESSION_DATA",
            )
        coefficients = np.polyfit(x.to_numpy(), y.to_numpy(), degree)
        predicted = np.polyval(coefficients, x.to_numpy())
        ss_res = float(np.sum((y.to_numpy() - predicted) ** 2))
        ss_tot = float(np.sum((y.to_numpy() - float(y.mean())) ** 2))
        line_x = np.linspace(float(x.min()), float(x.max()), min(500, max(2, len(x))))
        return {
            "degree": degree,
            "coefficients": [float(value) for value in coefficients.tolist()],
            "r_squared": None if ss_tot == 0 else float(1 - ss_res / ss_tot),
            "line_x": [float(value) for value in line_x.tolist()],
            "line_y": [float(value) for value in np.polyval(coefficients, line_x).tolist()],
        }

    def _deterministic_sample(self, frame: Any, max_points: int) -> Any:
        if len(frame.index) <= max_points:
            return frame
        np = self._np()
        positions = np.linspace(0, len(frame.index) - 1, num=max_points, dtype=int)
        return frame.iloc[positions]

    def _utc_index(self, frame: Any) -> Any:
        pd = self._pd()
        output = frame.copy()
        index = pd.to_datetime(output.index, errors="coerce")
        if getattr(index, "tz", None) is None:
            index = index.tz_localize(DISPLAY_TIMEZONE).tz_convert("UTC")
        else:
            index = index.tz_convert("UTC")
        output.index = index
        return output.loc[~output.index.isna()]

    def _range(self, frame: Any) -> JsonDict:
        pd = self._pd()
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
            return {"start": None, "end": None}
        return {"start": _utc_iso(frame.index.min(), pd), "end": _utc_iso(frame.index.max(), pd)}

    def _store_candidate(self, frame: Any, *, idempotency_key: str) -> JsonDict:
        return self._artifacts.put(
            frame.to_csv(index=True, date_format="%Y-%m-%dT%H:%M:%SZ").encode("utf-8"),
            filename="static_ml_dataset_candidate.csv",
            row_count=int(len(frame.index)),
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _job(job_id: str) -> JsonDict:
        with _DIRECT_JOBS_LOCK:
            job = _DIRECT_JOBS.get(str(job_id))
        if job is None:
            raise BackendApiHTTPError("Dataset job was not found.", status_code=404, error_code="DATASET_JOB_NOT_FOUND")
        return job


def _window_seconds(value: str) -> int | None:
    match = re.fullmatch(r"(\d+)\s*([mhd])", str(value).strip().lower())
    if not match:
        return None
    amount = int(match.group(1))
    return amount * {"m": 60, "h": 3600, "d": 86400}[match.group(2)]


def get_data_explorer_gateways(
    *,
    access_token: str | None = None,
    client: ApiClient | None = None,
) -> tuple[DataQueryGateway, DatasetGateway]:
    """Return the single complete Data Explorer mode selected by configuration.

    API mode is intentionally all-or-nothing.  A backend error or missing token
    is propagated to the page for a section-level error; this factory never
    silently falls back to direct data access.
    """

    if is_backend_api_enabled("data_explorer"):
        token = str(access_token or "").strip()
        if not token:
            raise BackendApiHTTPError(
                "Data Explorer API mode requires a backend access token.",
                status_code=401,
                error_code="AUTHENTICATION_REQUIRED",
            )
        return ApiDataQueryGateway(token, client), ApiDatasetGateway(token, client)
    return DirectDataQueryGateway(_DIRECT_ARTIFACTS), DirectDatasetGateway(_DIRECT_ARTIFACTS)


__all__ = [
    "AbsoluteTimeRangeRequest",
    "AggregationRequest",
    "ApiDataQueryGateway",
    "ApiDatasetGateway",
    "DataExportRequest",
    "DataPreviewRequest",
    "DataQueryGateway",
    "DatasetGateway",
    "DatasetJobRequest",
    "DirectDataQueryGateway",
    "DirectDatasetGateway",
    "HotMetalSlagRequest",
    "OfflineDataPreviewRequest",
    "OnlineDataPreviewRequest",
    "ScatterAnalysisRequest",
    "TimeseriesRequest",
    "get_data_explorer_gateways",
    "new_idempotency_key",
]
