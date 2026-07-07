"""Service layer for API v1 data access."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd

from apps.backend_api.app.api.v1.schemas.data import (
    DataExportRequest,
    DataExportResponse,
    DataPreviewResponse,
    DataQueryRequest,
    DataSourceInfo,
)
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.offline_fetcher import fetch_database_offline
from apps.backend_api.app.core.online_fetcher import ONLINE_MEASUREMENTS, fetch_online, list_measurements
from apps.backend_api.app.services.artifact_service import create_csv_artifact
from apps.backend_api.app.services.serialization import dataframe_to_preview
from furnace_data.offline import OFFLINE_REPORT_MAP, list_offline_tables as _list_offline_tables


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def max_preview_rows() -> int:
    return max(1, _env_int("DATA_API_MAX_PREVIEW_ROWS", 500))


def max_json_rows() -> int:
    return max(max_preview_rows(), _env_int("DATA_API_MAX_JSON_ROWS", 5000))


def artifact_ttl_hours() -> int:
    return max(1, _env_int("DATA_API_ARTIFACT_TTL_HOURS", 24))


def list_data_sources() -> list[DataSourceInfo]:
    return [
        DataSourceInfo(
            id="online",
            name="Online process data",
            kind="online",
            description="InfluxDB-backed online process measurements",
        ),
        DataSourceInfo(
            id="offline",
            name="Offline reports",
            kind="offline",
            description="PostgreSQL-backed offline operational reports",
        ),
        DataSourceInfo(
            id="static_dataset",
            name="Static ML dataset",
            kind="dataset",
            description="Runtime cached ML dataset CSV",
        ),
    ]


def list_offline_report_types() -> dict[str, str]:
    return {key: ",".join(value) for key, value in OFFLINE_REPORT_MAP.items()}


def list_offline_tables() -> dict[str, Any]:
    return _list_offline_tables()


def validate_query(query: DataQueryRequest) -> None:
    if query.start_time and query.end_time and query.start_time > query.end_time:
        raise ApiError(
            code="DATA_QUERY_INVALID",
            message="start_time must be before or equal to end_time",
            status_code=400,
        )
    if query.limit is not None and query.limit > max_json_rows():
        raise ApiError(
            code="DATA_QUERY_TOO_LARGE",
            message=f"limit must be <= {max_json_rows()}",
            status_code=413,
            details={"max_json_rows": max_json_rows()},
        )


def _fetch_online_dataframe(query: DataQueryRequest) -> pd.DataFrame:
    filters = query.filters or {}
    measurements = filters.get("measurements") or query.columns or ["process_params"]
    if isinstance(measurements, str):
        measurements = [measurements]
    unknown = [item for item in measurements if item not in ONLINE_MEASUREMENTS]
    if unknown:
        raise ApiError(
            code="DATA_SOURCE_NOT_FOUND",
            message=f"Unknown online measurement(s): {unknown}",
            status_code=404,
            details={"valid_measurements": ONLINE_MEASUREMENTS},
        )
    try:
        return fetch_online(
            measurements=measurements,
            query_type=query.mode or filters.get("query_type") or "windowed-average",
            window=filters.get("window") or filters.get("window_by") or "1h",
            start_time=query.start_time,
            end_time=query.end_time,
            preset=filters.get("preset"),
        )
    except ValueError as exc:
        raise ApiError("DATA_QUERY_INVALID", str(exc), status_code=400) from exc
    except Exception as exc:
        raise ApiError("DATA_SOURCE_UNAVAILABLE", "Online data source unavailable", status_code=503) from exc


def _fetch_offline_dataframe(query: DataQueryRequest) -> pd.DataFrame:
    try:
        return fetch_database_offline(
            report_type=query.report_type or "HM_SLAG",
            start_time=query.start_time,
            end_time=query.end_time,
            preset=(query.filters or {}).get("preset"),
            table_name=query.table_name,
            query_type=query.mode or "ts",
            window=(query.filters or {}).get("window"),
        )
    except ValueError as exc:
        raise ApiError("DATA_QUERY_INVALID", str(exc), status_code=400) from exc
    except Exception as exc:
        raise ApiError("DATA_SOURCE_UNAVAILABLE", "Offline data source unavailable", status_code=503) from exc


def _fetch_static_dataset_dataframe() -> pd.DataFrame:
    from apps.backend_api.app.services.dataset_service import load_static_dataset_dataframe

    return load_static_dataset_dataframe()


def fetch_dataframe(query: DataQueryRequest) -> pd.DataFrame:
    validate_query(query)
    if query.source == "online":
        return _fetch_online_dataframe(query)
    if query.source == "offline":
        return _fetch_offline_dataframe(query)
    if query.source == "static_dataset":
        return _fetch_static_dataset_dataframe()
    raise ApiError(
        code="DATA_SOURCE_NOT_FOUND",
        message=f"Unknown data source: {query.source}",
        status_code=404,
    )


def preview_data(query: DataQueryRequest) -> DataPreviewResponse:
    limit = min(query.limit or max_preview_rows(), max_preview_rows())
    warnings: list[str] = []
    if query.limit and query.limit > max_preview_rows():
        warnings.append(f"Requested limit capped to {max_preview_rows()} rows")

    df = fetch_dataframe(query)
    if query.columns and query.source != "online":
        missing = [column for column in query.columns if column not in df.columns]
        if missing:
            raise ApiError(
                code="DATA_QUERY_INVALID",
                message=f"Unknown column(s): {missing}",
                status_code=400,
            )
        df = df[query.columns]

    columns, rows, row_count, truncated = dataframe_to_preview(
        df,
        limit=limit,
        offset=query.offset or 0,
        include_index=True,
    )
    return DataPreviewResponse(
        columns=columns,
        rows=rows,
        row_count=row_count,
        returned_rows=len(rows),
        truncated=truncated,
        source=query.source,
        warnings=warnings,
    )


def export_data(request: DataExportRequest) -> DataExportResponse:
    df = fetch_dataframe(request.query)
    prefix = f"{request.query.source}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    artifact = create_csv_artifact(df, prefix, ttl_hours=artifact_ttl_hours())
    return DataExportResponse(
        artifact_id=artifact.artifact_id,
        filename=artifact.filename,
        content_type=artifact.content_type,
        row_count=artifact.row_count,
        download_url=f"/api/v1/data/artifacts/{artifact.artifact_id}/download",
        expires_at=datetime.fromisoformat(artifact.expires_at) if artifact.expires_at else None,
    )
