"""Service layer for frontend-neutral API v1 Data Explorer access."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd

from apps.backend_api.app.api.v1.schemas.data import (
    AbsoluteTimeRange,
    AggregationWindow,
    DataCatalogLimits,
    DataCatalogResponse,
    DataColumnInfo,
    DataExportRequest,
    DataExportResponse,
    DataPreviewResponse,
    DataQueryRequest,
    DataSourceInfo,
    HotMetalSlagExportRequest,
    HotMetalSlagPreviewRequest,
    HotMetalSlagPreviewResponse,
    OfflineDataQuery,
    OfflineReportInfo,
    OfflineReportSelection,
    OfflineTableInfo,
    OfflineTableSelection,
    OnlineDataQuery,
    PresetTimeRange,
    ResolvedTimeRange,
    TimePreset,
)
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.offline_fetcher import fetch_database_offline
from apps.backend_api.app.core.online_fetcher import (
    ONLINE_MEASUREMENTS,
    config as _online_config,
    fetch_online,
    list_measurements,
)
from apps.backend_api.app.services.artifact_service import (
    ArtifactIdempotencyConflictError,
    ArtifactMetadata,
    create_csv_artifact,
    find_idempotent_artifact,
)
from apps.backend_api.app.services.serialization import dataframe_to_preview
from furnace_data.offline import OFFLINE_REPORT_MAP, OFFLINE_TABLES

if TYPE_CHECKING:  # pragma: no cover
    from apps.backend_api.app.core.config import BackendSettings


DISPLAY_TIMEZONE = "Asia/Kolkata"
_DISPLAY_ZONE = ZoneInfo(DISPLAY_TIMEZONE)
_MAX_SCATTER_POINTS = 5_000
_MAX_TIMESERIES_POINTS_PER_FIELD = 5_000
_SENSITIVE_HM_SLAG_COLUMNS = frozenset(
    {"id", "cast_no_ladle_spec", "lab_sample_id", "import_batch_id", "source_row_number"}
)
# ``fetch_offline_report`` may add this implementation-detail column at
# runtime.  It is never a public field, even if a future source whitelist
# happens to contain the same name.
_INTERNAL_OFFLINE_PROVENANCE_COLUMNS = frozenset({"source_table"})

_HM_SLAG_TABLES = frozenset(OFFLINE_REPORT_MAP.get("HM_SLAG", ()))

# These public IDs are independent of legacy fetcher display strings. Every
# preset is resolved to an absolute UTC range before it reaches a source.
_TIME_PRESET_DEFINITIONS: tuple[tuple[str, str, int, tuple[str, ...]], ...] = (
    ("last_1_minute", "Last 1 minute", 60, ("online",)),
    ("last_5_minutes", "Last 5 minutes", 300, ("online",)),
    ("last_10_minutes", "Last 10 minutes", 600, ("online",)),
    ("last_15_minutes", "Last 15 minutes", 900, ("online", "offline")),
    ("last_30_minutes", "Last 30 minutes", 1_800, ("online", "offline")),
    ("last_1_hour", "Last 1 hour", 3_600, ("online", "offline")),
    ("last_3_hours", "Last 3 hours", 10_800, ("online", "offline")),
    ("last_6_hours", "Last 6 hours", 21_600, ("online", "offline")),
    ("last_8_hours", "Last 8 hours", 28_800, ("online", "offline")),
    ("last_12_hours", "Last 12 hours", 43_200, ("online", "offline")),
    ("last_1_day", "Last 1 day", 86_400, ("online", "offline")),
    ("last_3_days", "Last 3 days", 259_200, ("online", "offline")),
    ("last_1_week", "Last 1 week", 604_800, ("online", "offline")),
    ("last_2_weeks", "Last 2 weeks", 1_209_600, ("online", "offline")),
    ("last_1_month", "Last 1 month", 2_592_000, ("online", "offline")),
    ("last_2_months", "Last 2 months", 5_184_000, ("online", "offline")),
    ("last_3_months", "Last 3 months", 7_776_000, ("online", "offline")),
    ("last_6_months", "Last 6 months", 15_552_000, ("offline",)),
    ("last_1_year", "Last 1 year", 31_536_000, ("offline",)),
)
_PRESET_SECONDS = {item[0]: item[2] for item in _TIME_PRESET_DEFINITIONS}
_PRESET_SOURCES = {item[0]: frozenset(item[3]) for item in _TIME_PRESET_DEFINITIONS}
_AGGREGATION_WINDOWS: tuple[tuple[str, str, int | None, str | None], ...] = (
    ("none", "None", None, None),
    ("1_minute", "1 minute", 60, "1m"),
    ("5_minutes", "5 minutes", 300, "5m"),
    ("15_minutes", "15 minutes", 900, "15m"),
    ("30_minutes", "30 minutes", 1_800, "30m"),
    ("1_hour", "1 hour", 3_600, "1h"),
    ("3_hours", "3 hours", 10_800, "3h"),
    ("6_hours", "6 hours", 21_600, "6h"),
    ("1_day", "1 day", 86_400, "1d"),
)
_AGGREGATION_BY_ID = {item[0]: item for item in _AGGREGATION_WINDOWS}
_MEASUREMENT_LABELS = {
    "process_params": "Process Params",
    "temperature_profile": "Temperature Profile",
    "heatload_delta_t": "Heatload Delta T",
    "cooling_water": "Cooling Water",
    "delta_t": "Delta T",
    "miscellaneous": "Miscellaneous",
}
_FIELD_LABELS = {
    "production_per_hour": "Production per Hour",
    "fuel_rate": "Fuel Rate",
    "coke_rate": "Coke Rate",
    "actual_kg_thm": "Actual Coal Rate",
    "body_etaco": "Eta CO",
}
_FIELD_UNITS = {
    "production_per_hour": "t/h",
    "fuel_rate": "kg/tHM",
    "coke_rate": "kg/tHM",
    "actual_kg_thm": "kg/tHM",
    "body_etaco": "%",
}
_REPORT_LABELS = {
    "HM_SLAG": "HM & Slag",
    "RM_COMPOSITION": "Raw Material Composition",
    "RAW_MATERIAL_STRENGTH": "Raw Material Strength",
    "BURDEN_DISTRIBUTION": "Burden Distribution",
    "HOPPER_MANAGEMENT": "Hopper Management",
}


@dataclass(frozen=True)
class QueryFetchResult:
    dataframe: pd.DataFrame
    resolved_range: ResolvedTimeRange | None
    warnings: list[str]


@dataclass(frozen=True)
class ExportResult:
    response: DataExportResponse
    warnings: list[str]


@dataclass(frozen=True)
class HotMetalSlagFetchResult:
    dataframe: pd.DataFrame
    resolved_range: ResolvedTimeRange
    synthetic_row_count: int
    interpolated_columns: list[str]
    warnings: list[str]


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    return default


def _setting_int(settings: BackendSettings | None, attribute: str, env_names: tuple[str, ...], default: int) -> int:
    value = getattr(settings, attribute, None) if settings is not None else None
    try:
        return max(1, int(value)) if value is not None else max(1, _env_int(env_names, default))
    except (TypeError, ValueError):
        return max(1, _env_int(env_names, default))


def max_preview_rows(settings: BackendSettings | None = None) -> int:
    return _setting_int(settings, "data_preview_max_rows", ("DATA_PREVIEW_MAX_ROWS", "DATA_API_MAX_PREVIEW_ROWS"), 500)


def max_selected_fields(settings: BackendSettings | None = None) -> int:
    return _setting_int(settings, "data_max_selected_fields", ("DATA_MAX_SELECTED_FIELDS",), 20)


def max_export_rows(settings: BackendSettings | None = None) -> int:
    return _setting_int(settings, "data_export_max_rows", ("DATA_EXPORT_MAX_ROWS",), 100_000)


def artifact_ttl_hours(settings: BackendSettings | None = None) -> int:
    return _setting_int(settings, "data_export_ttl_hours", ("DATA_EXPORT_TTL_HOURS", "DATA_API_ARTIFACT_TTL_HOURS"), 24)


def hot_metal_slag_max_preview_days(settings: BackendSettings | None = None) -> int:
    return _setting_int(settings, "hot_metal_slag_max_preview_days", ("HOT_METAL_SLAG_MAX_PREVIEW_DAYS",), 31)


def hot_metal_slag_max_interval_minutes(settings: BackendSettings | None = None) -> int:
    return _setting_int(settings, "hot_metal_slag_max_interval_minutes", ("HOT_METAL_SLAG_MAX_INTERVAL_MINUTES",), 600)


def _label_from_id(identifier: str) -> str:
    return " ".join(
        part.upper() if part in {"hm", "rm", "dpr"} else part.title()
        for part in identifier.replace("-", "_").split("_")
        if part
    )


def _column_dtype(identifier: str) -> str:
    name = identifier.lower()
    if name in {"time", "timestamp", "date_time", "created_at", "updated_at"}:
        return "datetime"
    if name.startswith("is_"):
        return "boolean"
    if any(token in name for token in ("id", "code", "name", "notes", "description", "type")):
        return "string"
    return "number"


def _column_info(identifier: str) -> DataColumnInfo:
    return DataColumnInfo(
        id=identifier,
        label=_FIELD_LABELS.get(identifier, _label_from_id(identifier)),
        dtype=_column_dtype(identifier),
        unit=_FIELD_UNITS.get(identifier),
    )


def _report_id(internal_report: str) -> str:
    return internal_report.strip().lower()


def _report_map() -> dict[str, str]:
    return {_report_id(report): report for report in OFFLINE_REPORT_MAP}


def _public_table_id(internal_table: str) -> str:
    """A deterministic opaque public ID that does not disclose a physical name."""
    digest = hashlib.sha256(internal_table.encode("utf-8")).hexdigest()[:16]
    return f"offline-table-{digest}"


def _table_map() -> dict[str, str]:
    return {_public_table_id(table): table for table in OFFLINE_TABLES}


def _allowed_columns_for_table(internal_table: str) -> set[str] | None:
    columns = OFFLINE_TABLES.get(internal_table)
    return None if columns is None else set(columns)


def _report_column_contract(internal_report: str) -> tuple[set[str] | None, bool]:
    allowed: set[str] = set()
    for table in OFFLINE_REPORT_MAP.get(internal_report, []):
        columns = _allowed_columns_for_table(table)
        if columns is None:
            return None, False
        allowed.update(columns)
    allowed.difference_update(_INTERNAL_OFFLINE_PROVENANCE_COLUMNS)
    if internal_report == "HM_SLAG":
        allowed.difference_update(_SENSITIVE_HM_SLAG_COLUMNS)
    return allowed, True


def _is_hm_slag_scope(internal_report: str | None, internal_table: str | None) -> bool:
    return internal_report == "HM_SLAG" or internal_table in _HM_SLAG_TABLES


def _table_column_contract(internal_table: str) -> tuple[set[str] | None, bool]:
    allowed = _allowed_columns_for_table(internal_table)
    if allowed is not None:
        allowed.difference_update(_INTERNAL_OFFLINE_PROVENANCE_COLUMNS)
        if internal_table in _HM_SLAG_TABLES:
            allowed.difference_update(_SENSITIVE_HM_SLAG_COLUMNS)
    return allowed, allowed is not None


def _online_field_contract(measurement: str) -> list[DataColumnInfo]:
    mapping = (_online_config.get("data_mapping") or {}).get(measurement) or {}
    labels_by_id = {str(field): str(label) for label, field in mapping.items()}
    field_ids = list(dict.fromkeys(str(field) for field in list_measurements().get(measurement, [])))
    return [
        DataColumnInfo(
            id=field_id,
            label=_FIELD_LABELS.get(field_id, labels_by_id.get(field_id, _label_from_id(field_id))),
            dtype="number",
            unit=_FIELD_UNITS.get(field_id),
        )
        for field_id in field_ids
    ]


@lru_cache(maxsize=1)
def _base_catalog() -> DataCatalogResponse:
    online_measurements = [
        {
            "id": measurement,
            "label": _MEASUREMENT_LABELS.get(measurement, _label_from_id(measurement)),
            "fields": _online_field_contract(measurement),
        }
        for measurement in ONLINE_MEASUREMENTS
    ]
    reports: list[OfflineReportInfo] = []
    for public_id, internal_report in sorted(_report_map().items()):
        columns, selectable = _report_column_contract(internal_report)
        reports.append(
            OfflineReportInfo(
                id=public_id,
                label=_REPORT_LABELS.get(internal_report, _label_from_id(public_id)),
                fields=[_column_info(name) for name in sorted(columns or set())],
                supports_field_selection=selectable,
            )
        )
    tables: list[OfflineTableInfo] = []
    for table_number, (public_id, internal_table) in enumerate(
        sorted(_table_map().items()), start=1
    ):
        columns, selectable = _table_column_contract(internal_table)
        tables.append(
            OfflineTableInfo(
                id=public_id,
                # The allowlisted internal table name is intentionally never
                # serialized, including as a human-readable label.
                label=f"Offline table {table_number}",
                fields=[_column_info(name) for name in sorted(columns or set())],
                supports_field_selection=selectable,
                supports_aggregation=False,
            )
        )
    return DataCatalogResponse(
        display_timezone=DISPLAY_TIMEZONE,
        online_measurements=online_measurements,
        time_presets=[
            TimePreset(id=item[0], label=item[1], duration_seconds=item[2], sources=list(item[3]))
            for item in _TIME_PRESET_DEFINITIONS
        ],
        aggregation_windows=[
            AggregationWindow(id=item[0], label=item[1], duration_seconds=item[2])
            for item in _AGGREGATION_WINDOWS
        ],
        offline_reports=reports,
        offline_tables=tables,
        limits=DataCatalogLimits(
            max_preview_rows=500,
            max_selected_fields=20,
            max_scatter_points=_MAX_SCATTER_POINTS,
            max_timeseries_points_per_field=_MAX_TIMESERIES_POINTS_PER_FIELD,
            max_hm_slag_interval_minutes=600,
        ),
    )


def clear_data_catalog_cache() -> None:
    """Controlled cache reset used by tests and configuration reloads."""
    _base_catalog.cache_clear()


def get_data_catalog(settings: BackendSettings | None = None) -> DataCatalogResponse:
    base = _base_catalog()
    # Keep the Explorer's analysis controls aligned with the server-side static
    # dataset policies. The import is lazy to avoid coupling the core data
    # catalogue to static-dataset initialization at application startup.
    from apps.backend_api.app.services import dataset_service

    return base.model_copy(
        update={
            "limits": DataCatalogLimits(
                max_preview_rows=max_preview_rows(settings),
                max_selected_fields=max_selected_fields(settings),
                max_scatter_points=dataset_service.max_scatter_points(),
                max_timeseries_points_per_field=dataset_service.max_timeseries_points_per_field(),
                max_hm_slag_interval_minutes=hot_metal_slag_max_interval_minutes(settings),
            )
        },
        deep=True,
    )


def list_data_sources() -> list[DataSourceInfo]:
    """Compatibility list; API-first clients should call ``get_data_catalog``."""
    return [
        DataSourceInfo(id="online", name="Online process data", kind="online", description="InfluxDB-backed online process measurements"),
        DataSourceInfo(id="offline", name="Offline reports", kind="offline", description="PostgreSQL-backed offline operational reports"),
    ]


def list_offline_report_types() -> dict[str, str]:
    return {
        public_id: _REPORT_LABELS.get(internal, _label_from_id(public_id))
        for public_id, internal in sorted(_report_map().items())
    }


def list_offline_tables() -> dict[str, object]:
    catalog = get_data_catalog()
    return {
        "reports": [report.model_dump(mode="json") for report in catalog.offline_reports],
        "tables": [table.model_dump(mode="json") for table in catalog.offline_tables],
    }


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def resolve_time_range(
    time_range: PresetTimeRange | AbsoluteTimeRange,
    *,
    source: str | None = None,
) -> ResolvedTimeRange:
    if isinstance(time_range, PresetTimeRange):
        seconds = _PRESET_SECONDS.get(time_range.preset_id)
        if seconds is None:
            raise ApiError("INVALID_TIME_PRESET", "Unknown time preset.", status_code=400)
        if source is not None and source not in _PRESET_SOURCES[time_range.preset_id]:
            raise ApiError(
                "INVALID_TIME_PRESET",
                "The selected time preset is not supported for this data source.",
                status_code=400,
            )
        end = datetime.now(timezone.utc)
        return ResolvedTimeRange(start=end - timedelta(seconds=seconds), end=end)
    return ResolvedTimeRange(start=_to_utc(time_range.start), end=_to_utc(time_range.end))


def _aggregation_window(query: OnlineDataQuery, resolved: ResolvedTimeRange) -> str | None:
    if query.aggregation is None:
        return None
    definition = _AGGREGATION_BY_ID.get(query.aggregation.window_id)
    if definition is None or definition[2] is None or definition[3] is None:
        raise ApiError("INVALID_AGGREGATION", "Unknown aggregation window.", status_code=400)
    if definition[2] > (resolved.end - resolved.start).total_seconds():
        raise ApiError(
            "INVALID_AGGREGATION",
            "Aggregation window cannot exceed the selected time range.",
            status_code=400,
        )
    return definition[3]


def _validate_field_selection(
    fields: list[str] | None,
    *,
    allowed: set[str] | None,
    selectable: bool,
    settings: BackendSettings | None,
) -> None:
    if fields is not None and len(fields) > max_selected_fields(settings):
        raise ApiError("INVALID_FIELD", "Too many fields were selected.", status_code=400)
    if fields is None:
        return
    if not selectable or allowed is None:
        raise ApiError("INVALID_FIELD", "Field selection is not supported for this source.", status_code=400)
    if set(fields).difference(allowed):
        raise ApiError("INVALID_FIELD", "One or more fields are not available for this source.", status_code=400)


def _validate_query(
    query: DataQueryRequest,
    settings: BackendSettings | None = None,
) -> tuple[ResolvedTimeRange | None, str | None, str | None]:
    if isinstance(query, OnlineDataQuery):
        if set(query.measurements).difference(ONLINE_MEASUREMENTS):
            raise ApiError("INVALID_MEASUREMENT", "Unknown online measurement.", status_code=400)
        fields = {
            field.id
            for measurement in query.measurements
            for field in _online_field_contract(measurement)
        }
        _validate_field_selection(query.fields, allowed=fields, selectable=True, settings=settings)
        resolved = resolve_time_range(query.time_range, source="online")
        _aggregation_window(query, resolved)
        return resolved, None, None
    if isinstance(query, OfflineDataQuery):
        resolved = resolve_time_range(query.time_range, source="offline")
        if isinstance(query.selection, OfflineReportSelection):
            report = _report_map().get(query.selection.report_id)
            if report is None:
                raise ApiError("INVALID_REPORT", "Unknown offline report.", status_code=400)
            allowed, selectable = _report_column_contract(report)
            _validate_field_selection(query.fields, allowed=allowed, selectable=selectable, settings=settings)
            return resolved, report, None
        if isinstance(query.selection, OfflineTableSelection):
            table = _table_map().get(query.selection.table_id)
            if table is None:
                raise ApiError("INVALID_TABLE", "Unknown offline table.", status_code=400)
            allowed, selectable = _table_column_contract(table)
            _validate_field_selection(query.fields, allowed=allowed, selectable=selectable, settings=settings)
            return resolved, None, table
    raise ApiError("INVALID_FIELD", "Unsupported data query.", status_code=400)


def _normalize_dataframe_index(
    dataframe: pd.DataFrame,
    *,
    naive_timezone: str = DISPLAY_TIMEZONE,
) -> pd.DataFrame:
    if dataframe is None:
        return pd.DataFrame()
    frame = dataframe.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        return frame
    index = pd.to_datetime(frame.index, errors="coerce")
    if index.tz is None:
        index = index.tz_localize(naive_timezone)
    frame.index = index.tz_convert("UTC")
    frame.index.name = "time"
    return frame.sort_index()


def _select_returned_fields(dataframe: pd.DataFrame, fields: list[str] | None, warnings: list[str]) -> pd.DataFrame:
    if fields is None:
        return dataframe
    available = [field for field in fields if field in dataframe.columns]
    if set(fields).difference(available):
        warnings.append("Some requested fields were unavailable in the returned data.")
    return dataframe.loc[:, available]


def _fetch_online_dataframe(query: OnlineDataQuery, resolved: ResolvedTimeRange) -> tuple[pd.DataFrame, list[str]]:
    window = _aggregation_window(query, resolved)
    query_type = "windowed-average" if window else "ts"
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    failures = 0
    for measurement in query.measurements:
        try:
            frame = fetch_online(
                measurements=[measurement],
                query_type=query_type,
                window=window,
                start_time=resolved.start,
                end_time=resolved.end,
                preset=None,
            )
        except Exception:
            failures += 1
            warnings.append(f"Online measurement '{measurement}' was unavailable.")
            continue
        if frame is not None and not frame.empty:
            frames.append(_normalize_dataframe_index(frame))
    if failures == len(query.measurements):
        raise ApiError("DATA_SOURCE_UNAVAILABLE", "Online data source is unavailable.", status_code=503)
    if not frames:
        return pd.DataFrame(), warnings
    combined = pd.concat(frames, axis=1, join="outer")
    combined = combined.loc[:, ~combined.columns.duplicated()].sort_index()
    return _select_returned_fields(combined, query.fields, warnings), warnings


def _fetch_offline_dataframe(
    query: OfflineDataQuery,
    resolved: ResolvedTimeRange,
    *,
    internal_report: str | None,
    internal_table: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    try:
        frame = fetch_database_offline(
            report_type=internal_report or "HM_SLAG",
            start_time=resolved.start,
            end_time=resolved.end,
            preset=None,
            table_name=internal_table,
            query_type="ts",
            window=None,
        )
    except ValueError as exc:
        raise ApiError("DATA_SOURCE_UNAVAILABLE", "Offline data source could not process the request.", status_code=503) from exc
    except Exception as exc:
        raise ApiError("DATA_SOURCE_UNAVAILABLE", "Offline data source is unavailable.", status_code=503) from exc
    warnings: list[str] = []
    normalized = _normalize_dataframe_index(frame, naive_timezone="UTC")
    # ``fetch_offline_report`` adds a physical source table as internal
    # provenance.  It must never reach public previews or exports.
    normalized = normalized.drop(columns=list(_INTERNAL_OFFLINE_PROVENANCE_COLUMNS), errors="ignore")
    if _is_hm_slag_scope(internal_report, internal_table):
        normalized = normalized.drop(
            columns=[column for column in _SENSITIVE_HM_SLAG_COLUMNS if column in normalized.columns],
            errors="ignore",
        )
    return _select_returned_fields(normalized, query.fields, warnings), warnings



def fetch_query_result(query: DataQueryRequest, *, settings: BackendSettings | None = None) -> QueryFetchResult:
    resolved, internal_report, internal_table = _validate_query(query, settings)
    if isinstance(query, OnlineDataQuery):
        dataframe, warnings = _fetch_online_dataframe(query, resolved)
        return QueryFetchResult(dataframe, resolved, warnings)
    if isinstance(query, OfflineDataQuery):
        dataframe, warnings = _fetch_offline_dataframe(
            query,
            resolved,
            internal_report=internal_report,
            internal_table=internal_table,
        )
        return QueryFetchResult(dataframe, resolved, warnings)
    raise ApiError("INVALID_FIELD", "Unsupported data query.", status_code=400)


def fetch_dataframe(query: DataQueryRequest, *, settings: BackendSettings | None = None) -> pd.DataFrame:
    """Compatibility convenience wrapper for callers that only need a frame."""
    return fetch_query_result(query, settings=settings).dataframe


def preview_data(query: DataQueryRequest, *, settings: BackendSettings | None = None) -> DataPreviewResponse:
    result = fetch_query_result(query, settings=settings)
    limit = min(query.limit, max_preview_rows(settings))
    warnings = list(result.warnings)
    if query.limit > limit:
        warnings.append(f"Requested limit capped to {limit} rows.")
    columns, rows, total_rows, truncated = dataframe_to_preview(
        result.dataframe,
        limit=limit,
        offset=query.offset,
        include_index=True,
    )
    if result.dataframe.empty:
        warnings.append("No data was returned for the selected range.")
    return DataPreviewResponse(
        columns=columns,
        rows=rows,
        total_rows=total_rows,
        row_count=total_rows,
        returned_rows=len(rows),
        offset=query.offset,
        truncated=truncated,
        source=query.source,
        resolved_range=result.resolved_range,
        warnings=warnings,
    )


def _query_fingerprint(query: object) -> str:
    payload = query.model_dump(mode="json") if hasattr(query, "model_dump") else query
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_response(metadata: ArtifactMetadata) -> DataExportResponse:
    expires_at = datetime.fromisoformat(str(metadata.expires_at).replace("Z", "+00:00"))
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return DataExportResponse(
        artifact_id=metadata.artifact_id,
        filename=metadata.filename,
        content_type="text/csv",
        row_count=max(0, int(metadata.row_count or 0)),
        download_path=f"/api/v1/data/artifacts/{metadata.artifact_id}/download",
        expires_at=expires_at.astimezone(timezone.utc),
    )


def _require_idempotency_key(idempotency_key: str | None) -> str:
    value = str(idempotency_key or "").strip()
    if not value:
        raise ApiError("IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required for export creation.", status_code=400)
    if len(value) > 200:
        raise ApiError("INVALID_IDEMPOTENCY_KEY", "Idempotency-Key is too long.", status_code=400)
    return value


def export_data(
    request: DataExportRequest,
    *,
    owner_user_id: str,
    idempotency_key: str | None,
    settings: BackendSettings | None = None,
) -> ExportResult:
    key = _require_idempotency_key(idempotency_key)
    fingerprint = _query_fingerprint(request.query)
    try:
        existing = find_idempotent_artifact(
            owner_user_id=str(owner_user_id),
            query_fingerprint=fingerprint,
            idempotency_key=key,
        )
    except ArtifactIdempotencyConflictError as exc:
        raise ApiError(
            "IDEMPOTENCY_KEY_REUSED",
            "Idempotency-Key was already used for a different export request.",
            status_code=409,
        ) from exc
    if existing is not None:
        return ExportResult(response=_artifact_response(existing), warnings=[])
    result = fetch_query_result(request.query, settings=settings)
    if len(result.dataframe) > max_export_rows(settings):
        raise ApiError("EXPORT_LIMIT_EXCEEDED", "The selected export exceeds the configured row limit.", status_code=413)
    try:
        artifact = create_csv_artifact(
            result.dataframe,
            f"{request.query.source}_data_{datetime.now(timezone.utc):%Y%m%d_%H%M%SZ}",
            ttl_hours=artifact_ttl_hours(settings),
            owner_user_id=str(owner_user_id),
            query_fingerprint=fingerprint,
            idempotency_key=key,
            artifact_kind="data_export",
        )
    except ArtifactIdempotencyConflictError as exc:
        raise ApiError(
            "IDEMPOTENCY_KEY_REUSED",
            "Idempotency-Key was already used for a different export request.",
            status_code=409,
        ) from exc
    return ExportResult(response=_artifact_response(artifact), warnings=result.warnings)


def _validate_hot_metal_slag_request(
    request: HotMetalSlagPreviewRequest,
    settings: BackendSettings | None = None,
) -> ResolvedTimeRange:
    if request.interval_minutes > hot_metal_slag_max_interval_minutes(settings):
        raise ApiError("INVALID_TIME_RANGE", "The requested interval exceeds the configured limit.", status_code=400)
    resolved = ResolvedTimeRange(start=_to_utc(request.start), end=_to_utc(request.end))
    if resolved.end - resolved.start > timedelta(days=hot_metal_slag_max_preview_days(settings)):
        raise ApiError(
            "INVALID_TIME_RANGE",
            "The selected range exceeds the configured Hot Metal & Slag limit.",
            status_code=400,
        )
    return resolved

def _hot_metal_slag_provenance(
    dataframe: pd.DataFrame,
    normalized: pd.DataFrame,
) -> tuple[int, list[str]]:
    """Read shared-service interpolation provenance without overclaiming it.

    Older shared-service releases do not expose provenance attrs.  In that
    case the API reports zero synthetic rows and no interpolated columns rather
    than assuming every grid row was synthesized.
    """

    raw_columns = dataframe.attrs.get("interpolated_columns")
    interpolated_columns: list[str] = []
    if isinstance(raw_columns, (list, tuple, set, frozenset)):
        seen: set[str] = set()
        for raw_column in raw_columns:
            column = str(raw_column)
            if (
                column not in seen
                and column in normalized.columns
                and column not in _SENSITIVE_HM_SLAG_COLUMNS
            ):
                seen.add(column)
                interpolated_columns.append(column)

    raw_timestamps = dataframe.attrs.get("synthetic_timestamps")
    has_timestamp_provenance = isinstance(raw_timestamps, (list, tuple, set, frozenset))
    if has_timestamp_provenance and isinstance(normalized.index, pd.DatetimeIndex):
        synthetic_timestamps: set[pd.Timestamp] = set()
        for raw_timestamp in raw_timestamps:
            try:
                timestamp = pd.Timestamp(raw_timestamp)
                if pd.isna(timestamp):
                    continue
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(DISPLAY_TIMEZONE)
                synthetic_timestamps.add(timestamp.tz_convert("UTC"))
            except (TypeError, ValueError):
                continue
        synthetic_row_count = sum(
            pd.Timestamp(timestamp) in synthetic_timestamps
            for timestamp in normalized.index
        )
        return synthetic_row_count, interpolated_columns

    # A count-only attribute is retained for backwards compatibility with an
    # intermediate shared-service release.  It cannot identify a filtered row,
    # so clamp it to the public result instead of assuming all rows are synthetic.
    try:
        synthetic_row_count = max(0, int(dataframe.attrs.get("synthetic_row_count", 0)))
    except (TypeError, ValueError):
        synthetic_row_count = 0
    return min(synthetic_row_count, len(normalized)), interpolated_columns


def fetch_hot_metal_slag_result(
    request: HotMetalSlagPreviewRequest,
    *,
    settings: BackendSettings | None = None,
) -> HotMetalSlagFetchResult:
    resolved = _validate_hot_metal_slag_request(request, settings)
    # The shared service owns the hourly/interpolation algorithm. This API owns
    # UTC range semantics and public-safe output filtering.
    from furnace_data.dataset.service import DatasetService

    try:
        dataframe = DatasetService().fetch_hotmetal_hourly(
            start_date=resolved.start.astimezone(_DISPLAY_ZONE).date(),
            end_date=resolved.end.astimezone(_DISPLAY_ZONE).date(),
            interval_minutes=request.interval_minutes,
        )
    except Exception as exc:
        raise ApiError("DATA_SOURCE_UNAVAILABLE", "Hot Metal & Slag data source is unavailable.", status_code=503) from exc
    normalized = _normalize_dataframe_index(dataframe, naive_timezone=DISPLAY_TIMEZONE)
    if isinstance(normalized.index, pd.DatetimeIndex):
        normalized = normalized.loc[
            (normalized.index >= pd.Timestamp(resolved.start))
            & (normalized.index <= pd.Timestamp(resolved.end))
        ]
    normalized = normalized.drop(
        columns=[column for column in _SENSITIVE_HM_SLAG_COLUMNS if column in normalized.columns],
        errors="ignore",
    )
    synthetic_row_count, interpolated_columns = _hot_metal_slag_provenance(
        dataframe,
        normalized,
    )
    warnings: list[str] = []
    if normalized.empty:
        warnings.append("No Hot Metal & Slag data was returned for the selected range.")
    elif synthetic_row_count:
        warnings.append(f"Interpolation produced {synthetic_row_count} synthetic grid rows.")
    return HotMetalSlagFetchResult(
        dataframe=normalized,
        resolved_range=resolved,
        synthetic_row_count=synthetic_row_count,
        interpolated_columns=interpolated_columns,
        warnings=warnings,
    )


def preview_hot_metal_slag(
    request: HotMetalSlagPreviewRequest,
    *,
    settings: BackendSettings | None = None,
) -> HotMetalSlagPreviewResponse:
    result = fetch_hot_metal_slag_result(request, settings=settings)
    limit = min(request.limit, max_preview_rows(settings))
    warnings = list(result.warnings)
    if request.limit > limit:
        warnings.append(f"Requested limit capped to {limit} rows.")
    columns, rows, total_rows, truncated = dataframe_to_preview(
        result.dataframe,
        limit=limit,
        offset=request.offset,
        include_index=True,
    )
    return HotMetalSlagPreviewResponse(
        columns=columns,
        rows=rows,
        returned_rows=len(rows),
        total_rows=total_rows,
        offset=request.offset,
        truncated=truncated,
        resolved_range=result.resolved_range,
        interval_minutes=request.interval_minutes,
        synthetic_row_count=result.synthetic_row_count,
        interpolated_columns=result.interpolated_columns,
        warnings=warnings,
    )


def export_hot_metal_slag(
    request: HotMetalSlagExportRequest,
    *,
    owner_user_id: str,
    idempotency_key: str | None,
    settings: BackendSettings | None = None,
) -> ExportResult:
    key = _require_idempotency_key(idempotency_key)
    fingerprint = f"hm-slag:{_query_fingerprint(request.query)}"
    try:
        existing = find_idempotent_artifact(
            owner_user_id=str(owner_user_id),
            query_fingerprint=fingerprint,
            idempotency_key=key,
        )
    except ArtifactIdempotencyConflictError as exc:
        raise ApiError(
            "IDEMPOTENCY_KEY_REUSED",
            "Idempotency-Key was already used for a different export request.",
            status_code=409,
        ) from exc
    if existing is not None:
        return ExportResult(response=_artifact_response(existing), warnings=[])
    result = fetch_hot_metal_slag_result(request.query, settings=settings)
    if len(result.dataframe) > max_export_rows(settings):
        raise ApiError("EXPORT_LIMIT_EXCEEDED", "The selected export exceeds the configured row limit.", status_code=413)
    try:
        artifact = create_csv_artifact(
            result.dataframe,
            f"hot_metal_slag_{datetime.now(timezone.utc):%Y%m%d_%H%M%SZ}",
            ttl_hours=artifact_ttl_hours(settings),
            owner_user_id=str(owner_user_id),
            query_fingerprint=fingerprint,
            idempotency_key=key,
            artifact_kind="hot_metal_slag_export",
        )
    except ArtifactIdempotencyConflictError as exc:
        raise ApiError(
            "IDEMPOTENCY_KEY_REUSED",
            "Idempotency-Key was already used for a different export request.",
            status_code=409,
        ) from exc
    return ExportResult(response=_artifact_response(artifact), warnings=result.warnings)
