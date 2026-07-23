"""Canonical backend service for the versioned static ML dataset.

This module is the only backend entry point for static dataset metadata,
analyses, validation and dataset-job orchestration.  It deliberately keeps
pandas/numpy and filesystem details behind typed API contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from apps.backend_api.app.api.v1.schemas.datasets import (
    BuildRangeJobRequest,
    DatasetInfo,
    DatasetJobCreated,
    DatasetJobEvent,
    DatasetJobEventsResponse,
    DatasetJobOptions,
    DatasetJobResponse,
    DatasetJobResult,
    DatasetJobStatus,
    DatasetPreviewResponse,
    DatasetRefreshRequest,
    DatasetRange,
    DroppedRows,
    ExtendJobRequest,
    NumericRangeFilter,
    OverrideJobRequest,
    ScatterAnalysisRequest,
    ScatterAnalysisResponse,
    ScatterRegression,
    StaticDatasetColumn,
    StaticDatasetJobStatus,
    StaticDatasetMetadata,
    StaticDatasetTimeColumn,
    StaticDatasetValidation,
    TimeSeries,
    TimeSeriesPoint,
    TimeSeriesRequest,
    TimeSeriesResponse,
    ValidationCheck,
    ValidationSummary,
)
from apps.backend_api.app.config import settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.artifact_service import create_csv_artifact
from apps.backend_api.app.services.job_service import JobState, job_service
from apps.backend_api.app.services.serialization import dataframe_to_preview
from furnace_data.dataset.static import StaticDatasetManager
from furnace_data.runtime_paths import runtime_path


STATIC_DATASET_ID = "static_ml_dataset"
DISPLAY_TIMEZONE = "Asia/Kolkata"
_UNIT_COST_FIELD = "unit_cost_lakhs_per_thm"

_FIELD_ID_OVERRIDES = {
    "body_etaco": "eta_co",
    "eta_co": "eta_co",
    "unit_cost_lakhs_thm": _UNIT_COST_FIELD,
    "unitcost_lakhs_thm": _UNIT_COST_FIELD,
    "unit_cost_lakhs_per_thm": _UNIT_COST_FIELD,
}
_FIELD_LABEL_OVERRIDES = {
    "fuel_rate": "Fuel Rate",
    "production_per_hour": "Production per Hour",
    "eta_co": "Eta CO",
    "coke_rate": "Coke Rate",
    "actual_kg_thm": "PCI Rate",
    _UNIT_COST_FIELD: "Unit Cost (lakhs/tHM)",
}
_FIELD_UNIT_OVERRIDES = {
    "fuel_rate": "kg/tHM",
    "coke_rate": "kg/tHM",
    "actual_kg_thm": "kg/tHM",
    "production_per_hour": "t/h",
    "eta_co": "%",
    _UNIT_COST_FIELD: "lakhs/tHM",
}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def max_preview_rows() -> int:
    return _env_int("DATA_PREVIEW_MAX_ROWS", _env_int("DATA_API_MAX_PREVIEW_ROWS", 500))


def max_scatter_points() -> int:
    return _env_int("DATASET_MAX_SCATTER_POINTS", 5_000)


def max_timeseries_points_per_field() -> int:
    return _env_int("DATASET_MAX_TIMESERIES_POINTS_PER_FIELD", 5_000)


def max_timeseries_fields() -> int:
    return _env_int("DATASET_MAX_TIMESERIES_FIELDS", 20)


def max_build_range_days() -> int:
    configured = getattr(settings, "dataset_max_build_range_days", None)
    if configured is not None:
        try:
            return max(1, int(configured))
        except (TypeError, ValueError):
            pass
    return _env_int("DATASET_MAX_BUILD_RANGE_DAYS", 366)

def artifact_ttl_hours() -> int:
    return _env_int("DATA_EXPORT_TTL_HOURS", _env_int("DATA_API_ARTIFACT_TTL_HOURS", 24))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc_datetime(value: datetime | pd.Timestamp) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _parse_metadata_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _make_manager() -> StaticDatasetManager:
    return StaticDatasetManager(
        static_dir=settings.static_dir,
        offline_lag_days=settings.offline_lag_days,
        max_versions=settings.static_max_versions,
        legacy_csv_path=settings.legacy_csv_path or None,
    )


def _current_dataset_path() -> Path | None:
    path = _make_manager().current_csv_path()
    return path if path and path.exists() else None


def _normalise_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a timestamp-indexed frame using UTC without changing source fields."""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    candidate = pd.to_datetime(result.index, errors="coerce", utc=True)
    valid = ~pd.isna(candidate)
    result = result.loc[valid].copy()
    result.index = pd.DatetimeIndex(candidate[valid], name="timestamp")
    return result.sort_index(kind="stable")


def load_static_dataset_dataframe() -> pd.DataFrame:
    """Load the current canonical CSV; never create or refresh it on a read."""
    csv_path = _current_dataset_path()
    if csv_path is None:
        raise ApiError("DATASET_NOT_AVAILABLE", "The static ML dataset is not available.", 404)
    try:
        source = pd.read_csv(csv_path, parse_dates=[0])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ApiError("DATASET_NOT_AVAILABLE", "The static ML dataset cannot be read.", 404) from exc
    if source.empty:
        return pd.DataFrame()
    timestamp_column = source.columns[0]
    timestamps = pd.to_datetime(source.pop(timestamp_column), errors="coerce", utc=True)
    source.index = pd.DatetimeIndex(timestamps, name="timestamp")
    source = source.loc[~source.index.isna()]
    return source.sort_index(kind="stable")


def _configured_aliases() -> dict[str, str]:
    try:
        from furnace_data.config import load_config

        rename_dict = load_config("setting_ds_dv.yml").get("rename_dict", {}) or {}
    except Exception:
        return {}
    return {str(display): str(alias) for alias, display in rename_dict.items()}


def _slugify(value: str) -> str:
    candidate = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return candidate[:112] or "field"


def _canonical_field_id(column: object, aliases: dict[str, str], used: set[str]) -> str:
    original = str(column)
    candidate = aliases.get(original, original)
    candidate = _FIELD_ID_OVERRIDES.get(candidate, candidate)
    candidate = _FIELD_ID_OVERRIDES.get(_slugify(candidate), candidate)
    candidate = _slugify(candidate)
    candidate = _FIELD_ID_OVERRIDES.get(candidate, candidate)
    if candidate not in used:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    return f"{candidate}_{suffix}"


def _is_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    return bool(pd.to_numeric(series, errors="coerce").notna().any())


def _dtype_for(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if _is_numeric(series):
        return "number"
    return "string"


def _default_label(field_id: str, source_label: str) -> str:
    if field_id in _FIELD_LABEL_OVERRIDES:
        return _FIELD_LABEL_OVERRIDES[field_id]
    if source_label and source_label != field_id:
        return source_label
    return field_id.replace("_", " ").title()


def _canonicalise_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename CSV fields to stable public IDs and derive Unit Cost once."""
    result = _normalise_index(frame)
    aliases = _configured_aliases()
    used: set[str] = set()
    rename: dict[object, str] = {}
    labels: dict[str, str] = {}
    for column in result.columns:
        field_id = _canonical_field_id(column, aliases, used)
        used.add(field_id)
        rename[column] = field_id
        labels[field_id] = _default_label(field_id, str(column))
    result = result.rename(columns=rename)
    if _UNIT_COST_FIELD not in result.columns:
        coke_field = "coke_rate" if "coke_rate" in result.columns else None
        pci_field = "actual_kg_thm" if "actual_kg_thm" in result.columns else None
        if coke_field and pci_field:
            coke = pd.to_numeric(result[coke_field], errors="coerce")
            pci = pd.to_numeric(result[pci_field], errors="coerce")
            result[_UNIT_COST_FIELD] = (coke + (0.53 * pci)) * 0.25
            labels[_UNIT_COST_FIELD] = _FIELD_LABEL_OVERRIDES[_UNIT_COST_FIELD]
    else:
        labels[_UNIT_COST_FIELD] = _FIELD_LABEL_OVERRIDES[_UNIT_COST_FIELD]
    return result, labels


def _column_models(frame: pd.DataFrame, labels: dict[str, str]) -> list[StaticDatasetColumn]:
    return [
        StaticDatasetColumn(
            id=str(column),
            label=_default_label(str(column), labels.get(str(column), str(column))),
            dtype=_dtype_for(frame[column]),
            unit=_FIELD_UNIT_OVERRIDES.get(str(column)),
            plottable=_is_numeric(frame[column]),
            filterable=_is_numeric(frame[column]),
        )
        for column in frame.columns
    ]


def _version_from_path(path: Path | None, frame: pd.DataFrame) -> str:
    """Hash the public canonical dataframe, never the raw CSV representation."""
    _ = path  # Retained for call-site compatibility while versions are canonical.
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update("|".join(map(str, frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _validation_cache_path(version: str) -> Path:
    directory = Path(settings.static_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"validation_{version}.json"


def _stored_validation_status(version: str) -> str:
    try:
        value = json.loads(_validation_cache_path(version).read_text(encoding="utf-8"))
        status = str(value.get("status") or "")
        return status if status in {"passed", "warning", "failed"} else "not_run"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "not_run"


def _store_validation(validation: StaticDatasetValidation) -> None:
    try:
        _validation_cache_path(validation.dataset_version).write_text(
            validation.model_dump_json(), encoding="utf-8"
        )
    except OSError:
        # Validation remains available to the caller even if runtime cache storage
        # is temporarily unavailable.
        return

def _dataset_context() -> tuple[pd.DataFrame, dict[str, str], Path | None, str]:
    path = _current_dataset_path()
    raw = load_static_dataset_dataframe()
    canonical, labels = _canonicalise_dataframe(raw)
    return canonical, labels, path, _version_from_path(path, canonical)


def _validation_for_frame(frame: pd.DataFrame, version: str) -> StaticDatasetValidation:
    checks: list[ValidationCheck] = []
    errors = 0
    warnings = 0
    monotonic = bool(frame.index.is_monotonic_increasing)
    checks.append(
        ValidationCheck(
            id="timestamp_monotonic",
            status="passed" if monotonic else "failed",
            message="Timestamps are sorted in ascending order." if monotonic else "Timestamps are not sorted.",
        )
    )
    if not monotonic:
        errors += 1
    duplicate_count = int(frame.index.duplicated(keep=False).sum())
    duplicate_status = "passed" if duplicate_count == 0 else "warning"
    checks.append(
        ValidationCheck(
            id="duplicate_timestamps",
            status=duplicate_status,
            message="No duplicate timestamps found." if duplicate_count == 0 else "Duplicate timestamps are resolved deterministically in analyses.",
            details={"count": duplicate_count},
        )
    )
    if duplicate_count:
        warnings += 1
    non_finite = 0
    for column in frame.columns:
        if not _is_numeric(frame[column]):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        non_finite += int((values.notna() & ~np.isfinite(values)).sum())
    finite_status = "passed" if non_finite == 0 else "warning"
    checks.append(
        ValidationCheck(
            id="numeric_finite",
            status=finite_status,
            message="Numeric values are finite." if non_finite == 0 else "Non-finite numeric values were found and will be excluded from analyses.",
            details={"count": non_finite},
        )
    )
    if non_finite:
        warnings += 1
    status = "failed" if errors else "warning" if warnings else "passed"
    return StaticDatasetValidation(
        dataset_version=version,
        status=status,
        checked_at=_utc_now(),
        summary=ValidationSummary(errors=errors, warnings=warnings),
        checks=checks,
    )


def get_static_metadata() -> StaticDatasetMetadata:
    frame, labels, _path, version = _dataset_context()
    if frame.empty:
        raise ApiError("DATASET_NOT_AVAILABLE", "The static ML dataset is empty.", 404)
    manager = _make_manager()
    meta = manager.get_meta()
    validation_status = _stored_validation_status(version)
    return StaticDatasetMetadata(
        dataset_id=STATIC_DATASET_ID,
        version=version,
        etag=f'"sha256-{version}"',
        status="ready",
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=_column_models(frame, labels),
        time_column=StaticDatasetTimeColumn(id="timestamp", timezone="UTC"),
        range=DatasetRange(start=_as_utc_datetime(frame.index.min()), end=_as_utc_datetime(frame.index.max())),
        last_built_at=_parse_metadata_datetime(meta.last_updated) if meta else None,
        validation_status=validation_status,
        download_available=True,
    )


def _assert_version(requested: str, actual: str) -> None:
    if requested != actual:
        raise ApiError("DATASET_VERSION_CONFLICT", "The static dataset changed; refresh metadata and retry.", 409)


def _field_lookup(frame: pd.DataFrame, labels: dict[str, str]) -> dict[str, StaticDatasetColumn]:
    return {model.id: model for model in _column_models(frame, labels)}


def _require_numeric_field(field_id: str, lookup: dict[str, StaticDatasetColumn]) -> None:
    field = lookup.get(field_id)
    if field is None or not field.plottable or field.dtype != "number":
        raise ApiError("INVALID_FIELD", "The requested field is not a numeric plottable dataset field.", 422)


def _apply_numeric_filter(frame: pd.DataFrame, filter_request: NumericRangeFilter | None) -> pd.DataFrame:
    if filter_request is None:
        return frame
    if not math.isfinite(filter_request.minimum) or not math.isfinite(filter_request.maximum):
        raise ApiError("INVALID_FILTER", "Filter bounds must be finite numbers.", 422)
    if filter_request.field not in frame.columns or not _is_numeric(frame[filter_request.field]):
        raise ApiError("INVALID_FILTER", "The requested filter field is not numeric.", 422)
    values = pd.to_numeric(frame[filter_request.field], errors="coerce")
    inside = (values >= filter_request.minimum) & (values <= filter_request.maximum)
    mask = inside if filter_request.mode == "inside" else ((values < filter_request.minimum) | (values > filter_request.maximum))
    return frame.loc[mask.fillna(False)]


def _deterministic_downsample(frame: pd.DataFrame, maximum: int) -> tuple[pd.DataFrame, bool]:
    if len(frame) <= maximum:
        return frame, False
    positions = np.linspace(0, len(frame) - 1, num=maximum, dtype=int)
    return frame.iloc[np.unique(positions)], True


def get_scatter_analysis(request: ScatterAnalysisRequest) -> ScatterAnalysisResponse:
    frame, labels, _path, version = _dataset_context()
    _assert_version(request.dataset_version, version)
    lookup = _field_lookup(frame, labels)
    _require_numeric_field(request.x_field, lookup)
    _require_numeric_field(request.y_field, lookup)
    if request.filter is not None:
        _require_numeric_field(request.filter.field, lookup)
    filtered = _apply_numeric_filter(frame, request.filter)
    x_raw = filtered[request.x_field]
    y_raw = filtered[request.y_field]
    x_numeric = pd.to_numeric(x_raw, errors="coerce")
    y_numeric = pd.to_numeric(y_raw, errors="coerce")
    null_mask = x_raw.isna() | y_raw.isna()
    non_numeric_mask = (~null_mask) & (x_numeric.isna() | y_numeric.isna())
    finite_mask = np.isfinite(x_numeric.fillna(np.nan)) & np.isfinite(y_numeric.fillna(np.nan))
    non_finite_mask = (~null_mask) & (~non_numeric_mask) & ~finite_mask
    valid_mask = ~(null_mask | non_numeric_mask | non_finite_mask)
    valid = pd.DataFrame({"x": x_numeric[valid_mask], "y": y_numeric[valid_mask]}).sort_index(kind="stable")
    if len(valid) < 2 or len(valid.drop_duplicates()) < 2:
        raise ApiError(
            "INSUFFICIENT_REGRESSION_DATA",
            "At least two distinct valid numeric rows are required for scatter analysis.",
            422,
        )
    regression: ScatterRegression | None = None
    if request.regression.enabled:
        required = request.regression.degree + 1
        if len(valid) < required or int(valid["x"].nunique()) < required:
            raise ApiError("INSUFFICIENT_REGRESSION_DATA", "There are not enough distinct numeric rows for the requested regression degree.", 422)
        try:
            coefficients_array = np.polyfit(valid["x"].to_numpy(dtype=float), valid["y"].to_numpy(dtype=float), request.regression.degree)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            raise ApiError("INSUFFICIENT_REGRESSION_DATA", "Regression could not be calculated from the selected data.", 422) from exc
        polynomial = np.poly1d(coefficients_array)
        predicted = polynomial(valid["x"].to_numpy(dtype=float))
        residual = float(np.sum((valid["y"].to_numpy(dtype=float) - predicted) ** 2))
        total = float(np.sum((valid["y"].to_numpy(dtype=float) - valid["y"].mean()) ** 2))
        r_squared = None if total == 0 else float(1 - residual / total)
        line_x = np.linspace(float(valid["x"].min()), float(valid["x"].max()), num=min(250, max(required, len(valid))))
        line_y = polynomial(line_x)
        regression = ScatterRegression(
            degree=request.regression.degree,
            coefficients=[float(value) for value in coefficients_array.tolist()],
            r_squared=r_squared,
            line_x=[float(value) for value in line_x.tolist()],
            line_y=[float(value) for value in line_y.tolist()],
        )
    displayed, downsampled = _deterministic_downsample(valid, min(request.max_points, max_scatter_points()))
    return ScatterAnalysisResponse(
        dataset_version=version,
        x=[float(value) for value in displayed["x"].tolist()],
        y=[float(value) for value in displayed["y"].tolist()],
        total_matching_rows=len(valid),
        returned_points=len(displayed),
        downsampled=downsampled,
        regression=regression,
        dropped_rows=DroppedRows(null=int(null_mask.sum()), non_numeric=int(non_numeric_mask.sum()), non_finite=int(non_finite_mask.sum())),
    )


def get_timeseries(request: TimeSeriesRequest) -> TimeSeriesResponse:
    frame, labels, _path, version = _dataset_context()
    _assert_version(request.dataset_version, version)
    if len(request.fields) > max_timeseries_fields():
        raise ApiError("INVALID_FIELD", "Too many time-series fields were requested.", 422)
    lookup = _field_lookup(frame, labels)
    for field_id in request.fields:
        _require_numeric_field(field_id, lookup)
    if request.filter is not None:
        _require_numeric_field(request.filter.field, lookup)
    start = pd.Timestamp(request.time_range.start).tz_convert("UTC")
    end = pd.Timestamp(request.time_range.end).tz_convert("UTC")
    selected = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
    selected = _apply_numeric_filter(selected, request.filter)
    selected = selected.sort_index(kind="stable")
    selected = selected.loc[~selected.index.duplicated(keep="last")]
    numeric = selected.loc[:, request.fields].apply(pd.to_numeric, errors="coerce")
    if request.resample is not None:
        try:
            numeric = numeric.resample(request.resample.window).mean()
        except ValueError as exc:
            raise ApiError("INVALID_TIME_RANGE", "The requested resampling window is invalid.", 422) from exc
    series: list[TimeSeries] = []
    downsampled = False
    point_limit = min(request.max_points_per_field, max_timeseries_points_per_field())
    for field_id in request.fields:
        values = numeric[field_id].dropna()
        values = values[np.isfinite(values)]
        if len(values) > point_limit:
            positions = np.linspace(0, len(values) - 1, num=point_limit, dtype=int)
            values = values.iloc[np.unique(positions)]
            downsampled = True
        points = [
            TimeSeriesPoint(timestamp=_as_utc_datetime(timestamp), value=float(value))
            for timestamp, value in values.items()
        ]
        metadata = lookup[field_id]
        series.append(TimeSeries(field=field_id, label=metadata.label, unit=metadata.unit, points=points))
    return TimeSeriesResponse(
        dataset_version=version,
        series=series,
        resolved_range=DatasetRange(start=_as_utc_datetime(start), end=_as_utc_datetime(end)),
        downsampled=downsampled,
    )


def get_static_validation() -> StaticDatasetValidation:
    frame, _labels, _path, version = _dataset_context()
    validation = _validation_for_frame(frame, version)
    _store_validation(validation)
    return validation


def _canonical_download_path(version: str) -> Path:
    return runtime_path(
        "datasets",
        "downloads",
        f"static_ml_dataset_{version}.csv",
        create_parent=True,
    )


def _write_canonical_download(path: Path, frame: pd.DataFrame) -> None:
    """Atomically materialize the public canonical-column CSV for streaming."""
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(
            temporary_path,
            index=True,
            date_format="%Y-%m-%dT%H:%M:%SZ",
        )
        with temporary_path.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def current_dataset_download() -> tuple[Path, str]:
    """Return a server-only canonical CSV path and immutable content version."""
    source_path = _current_dataset_path()
    if source_path is None:
        raise ApiError("DATASET_NOT_AVAILABLE", "The static ML dataset is not available.", 404)
    frame = load_static_dataset_dataframe()
    canonical, _labels = _canonicalise_dataframe(frame)
    version = _version_from_path(source_path, canonical)
    download_path = _canonical_download_path(version)
    _write_canonical_download(download_path, canonical)
    return download_path, version


# ---------------------------------------------------------------------------
# Persistent static dataset jobs.
# ---------------------------------------------------------------------------


def _request_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _owner_id(current_user: dict[str, Any]) -> str:
    value = str(current_user.get("id") or "").strip()
    if not value:
        raise ApiError("FORBIDDEN", "Authenticated user identity is required.", 403)
    return value


def _is_elevated_operator(current_user: dict[str, Any]) -> bool:
    """Dataset-job elevation is deliberately separate from export artifacts."""

    permissions = {str(value) for value in current_user.get("permissions") or []}
    return (
        "datasets:override" in permissions
        or str(current_user.get("role") or "").lower() == "admin"
    )


def assert_job_access(job: JobState, current_user: dict[str, Any]) -> None:
    if job.owner_user_id == _owner_id(current_user) or _is_elevated_operator(current_user):
        return
    raise ApiError("FORBIDDEN", "You do not have access to this dataset job.", 403)


def assert_job_operation_permission(operation: str, current_user: dict[str, Any]) -> None:
    permissions = {str(value) for value in current_user.get("permissions") or []}
    required = {
        "build_range": "datasets:build",
        "extend": "datasets:refresh",
        "override": "datasets:override",
    }.get(operation)
    if required is None or required not in permissions:
        raise ApiError("FORBIDDEN", "Insufficient permissions for this dataset operation.", 403)


def _job_result(job: JobState) -> DatasetJobResult | None:
    if not job.result:
        return None
    payload = dict(job.result)
    requested = payload.get("requested_range")
    if isinstance(requested, dict) and requested.get("start") and requested.get("end"):
        payload["requested_range"] = DatasetRange(
            start=_as_utc_datetime(datetime.fromisoformat(str(requested["start"]).replace("Z", "+00:00"))),
            end=_as_utc_datetime(datetime.fromisoformat(str(requested["end"]).replace("Z", "+00:00"))),
        )
    return DatasetJobResult.model_validate(payload)


def static_job_status(job: JobState) -> StaticDatasetJobStatus:
    operation = job.operation if job.operation in {"build_range", "extend", "override"} else "build_range"
    return StaticDatasetJobStatus(
        job_id=job.job_id,
        operation=operation,
        status=job.status,
        progress=float(job.progress or 0),
        message=job.message,
        error_code=job.error_code,
        error_message=job.error_message,
        artifact_id=job.artifact_id,
        cancel_requested=job.cancel_requested,
        created_at=job.created_at or _utc_now(),
        updated_at=job.updated_at or job.created_at or _utc_now(),
        completed_at=job.completed_at,
        result=_job_result(job),
    )


def submit_static_dataset_job(
    request: BuildRangeJobRequest | ExtendJobRequest | OverrideJobRequest,
    *,
    current_user: dict[str, Any],
    idempotency_key: str,
) -> DatasetJobCreated:
    assert_job_operation_permission(request.operation, current_user)
    payload = request.model_dump(mode="json")
    before_version: str | None = None
    metadata: StaticDatasetMetadata | None = None
    if request.operation in {"extend", "override"}:
        metadata = get_static_metadata()
        _assert_version(request.expected_dataset_version, metadata.version)
        before_version = metadata.version
    owner_id = _owner_id(current_user)
    requested_start = getattr(request, "start", None)
    if request.operation == "extend" and requested_start is None:
        if metadata is None or metadata.range is None:
            raise ApiError("DATASET_NOT_AVAILABLE", "The static ML dataset range is unavailable.", 404)
        requested_start = metadata.range.end
    if requested_start is not None:
        local_start = requested_start.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).date()
        local_end = request.end.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).date()
        requested_days = (local_end - local_start).days + 1
        if requested_days > max_build_range_days():
            raise ApiError(
                "INVALID_TIME_RANGE",
                "The requested static dataset range exceeds the configured maximum.",
                422,
            )
    job, replayed = job_service.create_or_get_dataset_job(
        operation=request.operation,
        owner_user_id=owner_id,
        owner_username=str(current_user.get("username") or "") or None,
        idempotency_key=idempotency_key,
        request_fingerprint=_request_fingerprint(payload),
        request_payload=payload,
        requested_start=requested_start,
        requested_end=request.end,
        expected_dataset_version=getattr(request, "expected_dataset_version", None),
        message=f"{request.operation.replace('_', ' ').title()} queued",
    )
    if not replayed:
        job_service.run_background(job, lambda state: _run_static_dataset_job(state, request, before_version))
    return DatasetJobCreated(
        job_id=job.job_id,
        status=job.status,
        operation=request.operation,
        idempotent_replay=replayed,
        created_at=job.created_at or _utc_now(),
    )


def get_static_job(job_id: str, current_user: dict[str, Any]) -> StaticDatasetJobStatus:
    job = job_service.get_job(job_id)
    if job is None or job.operation not in {"build_range", "extend", "override"}:
        raise ApiError("DATASET_JOB_NOT_FOUND", "Dataset job not found.", 404)
    assert_job_access(job, current_user)
    return static_job_status(job)


def get_static_job_events(job_id: str, current_user: dict[str, Any], *, after: int = 0) -> DatasetJobEventsResponse:
    job = job_service.get_job(job_id)
    if job is None or job.operation not in {"build_range", "extend", "override"}:
        raise ApiError("DATASET_JOB_NOT_FOUND", "Dataset job not found.", 404)
    assert_job_access(job, current_user)
    events = job_service.get_events(job_id, after=after)
    return DatasetJobEventsResponse(
        job_id=job_id,
        events=[DatasetJobEvent(sequence=e.sequence, stage=e.stage, percent=e.percent, message=e.message, created_at=e.created_at) for e in events],
        last_sequence=events[-1].sequence if events else max(0, int(after)),
    )


def cancel_static_job(job_id: str, current_user: dict[str, Any]) -> StaticDatasetJobStatus:
    job = job_service.get_job(job_id)
    if job is None or job.operation not in {"build_range", "extend", "override"}:
        raise ApiError("DATASET_JOB_NOT_FOUND", "Dataset job not found.", 404)
    assert_job_access(job, current_user)
    if job.status not in {"pending", "running"}:
        raise ApiError("DATASET_JOB_NOT_CANCELLABLE", "Dataset job cannot be cancelled.", 409)
    cancelled = job_service.request_cancel(job_id)
    return static_job_status(cancelled or job)


def _check_cancelled(job_id: str) -> None:
    if job_service.is_cancel_requested(job_id):
        raise ApiError("DATASET_JOB_CANCELLED", "Dataset job cancelled.", 409)


def _clean_fetched_range(manager: StaticDatasetManager, *, start: datetime, end: datetime) -> pd.DataFrame:
    raw = manager.fetcher.get_dataset(
        start_date=start.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).date(),
        end_date=end.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).date(),
        rm_choice="RM Charge",
        cache_override=True,
    )
    if raw.empty:
        raise ApiError("DATASET_NOT_AVAILABLE", "No rows were returned for the requested dataset range.", 422)
    raw = _normalise_index(raw)
    try:
        cleaned = manager.cleaner.clean(raw)
    except Exception as exc:
        raise ApiError("DATASET_BUILD_FAILED", "The requested dataset range could not be cleaned.", 500) from exc
    cleaned = _normalise_index(cleaned)
    start_utc = pd.Timestamp(start).tz_convert("UTC")
    end_utc = pd.Timestamp(end).tz_convert("UTC")
    cleaned = cleaned.loc[(cleaned.index >= start_utc) & (cleaned.index <= end_utc)]
    if cleaned.empty:
        raise ApiError("DATASET_NOT_AVAILABLE", "No usable rows were returned for the requested dataset range.", 422)
    return cleaned


def _combine_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, axis=0, sort=False).sort_index(kind="stable")
    return combined.loc[~combined.index.duplicated(keep="last")]


def _validation_status_for_promotion(
    frame: pd.DataFrame, *, validate: bool
) -> tuple[pd.DataFrame, str, StaticDatasetValidation | None]:
    canonical, _labels = _canonicalise_dataframe(frame)
    version = _version_from_path(None, canonical)
    if not validate:
        return canonical, version, None
    validation = _validation_for_frame(canonical, version)
    _store_validation(validation)
    if validation.status == "failed":
        raise ApiError("DATASET_VALIDATION_FAILED", "Staged static dataset validation failed.", 422)
    return canonical, version, validation


def _result_payload(
    *,
    before: str | None,
    after: str | None,
    row_count: int,
    validation_status: str,
    started: float,
    start: datetime | None,
    end: datetime,
) -> dict[str, Any]:
    requested_range = None
    if start is not None:
        requested_range = {
            "start": _as_utc_datetime(start).isoformat(),
            "end": _as_utc_datetime(end).isoformat(),
        }
    return {
        "dataset_version_before": before,
        "dataset_version_after": after,
        "row_count": int(row_count),
        "validation_status": validation_status,
        "duration_seconds": round(max(0.0, monotonic() - started), 6),
        "requested_range": requested_range,
    }


def _run_static_dataset_job(
    job: JobState,
    request: BuildRangeJobRequest | ExtendJobRequest | OverrideJobRequest,
    before_version: str | None,
) -> None:
    """Build staging data, validate it, then safely promote only when requested."""
    started = monotonic()
    manager = _make_manager()
    job_service.append_event(job.job_id, stage="fetching", percent=10, message="Fetching dataset range")
    _check_cancelled(job.job_id)

    if request.operation == "build_range":
        staged = _clean_fetched_range(manager, start=request.start, end=request.end)
        job_service.append_event(job.job_id, stage="validating", percent=65, message="Validating candidate dataset")
        _check_cancelled(job.job_id)
        canonical, candidate_version, validation = _validation_status_for_promotion(
            staged, validate=request.options.validate_dataset
        )
        artifact_id: str | None = None
        if request.options.produce_download:
            job_service.append_event(job.job_id, stage="exporting", percent=85, message="Creating candidate dataset download")
            artifact = create_csv_artifact(
                canonical,
                f"static_ml_dataset_{candidate_version[:12]}",
                ttl_hours=artifact_ttl_hours(),
                owner_user_id=job.owner_user_id,
                query_fingerprint=job.request_fingerprint,
                idempotency_key=job.idempotency_key,
                artifact_kind="dataset_job_result",
            )
            artifact_id = artifact.artifact_id
        job_service.update_job(
            job.job_id,
            progress=95,
            artifact_id=artifact_id,
            result=_result_payload(
                before=before_version,
                after=candidate_version,
                row_count=len(canonical),
                validation_status=validation.status if validation else "not_run",
                started=started,
                start=request.start,
                end=request.end,
            ),
            message="Candidate dataset built",
        )
        return

    existing_raw = load_static_dataset_dataframe()
    if existing_raw.empty:
        raise ApiError("DATASET_NOT_AVAILABLE", "The static ML dataset is not available.", 404)
    existing_raw = _normalise_index(existing_raw)
    existing_canonical, _existing_labels = _canonicalise_dataframe(existing_raw)
    existing_version = _version_from_path(_current_dataset_path(), existing_canonical)
    _assert_version(request.expected_dataset_version, existing_version)

    if request.operation == "extend":
        latest = _as_utc_datetime(existing_raw.index.max())
        if request.end <= latest:
            raise ApiError("INVALID_TIME_RANGE", "Extend end must be after the current canonical dataset cutoff.", 422)
        fetch_start = latest + pd.Timedelta(nanoseconds=1).to_pytimedelta()
        replacement = _clean_fetched_range(manager, start=fetch_start, end=request.end)
        staged = _combine_frames(existing_raw, replacement)
        range_start: datetime | None = fetch_start
    else:
        replacement = _clean_fetched_range(manager, start=request.start, end=request.end)
        start_utc = pd.Timestamp(request.start).tz_convert("UTC")
        end_utc = pd.Timestamp(request.end).tz_convert("UTC")
        preserved = existing_raw.loc[(existing_raw.index < start_utc) | (existing_raw.index > end_utc)]
        staged = _combine_frames(preserved, replacement)
        range_start = request.start

    job_service.append_event(job.job_id, stage="validating", percent=65, message="Validating staged canonical dataset")
    _check_cancelled(job.job_id)
    _canonical, _staged_version, validation = _validation_status_for_promotion(
        staged, validate=True
    )
    _check_cancelled(job.job_id)

    with job_service.canonical_mutation_lock():
        current_raw = load_static_dataset_dataframe()
        current_canonical, _current_labels = _canonicalise_dataframe(current_raw)
        current_version = _version_from_path(_current_dataset_path(), current_canonical)
        _assert_version(request.expected_dataset_version, current_version)
        _check_cancelled(job.job_id)
        job_service.append_event(job.job_id, stage="promoting", percent=85, message="Promoting validated canonical dataset")
        manager.save(staged, "RM Charge")

    # Promotion is now durable and irreversible. A cancellation received after
    # save is deliberately treated as too late, so the job cannot report a
    # cancelled result for a dataset version it already published.
    promoted_frame, _labels, _promoted_path, promoted_version = _dataset_context()
    if validation is not None:
        validation = validation.model_copy(update={"dataset_version": promoted_version})
        _store_validation(validation)
    artifact_id: str | None = None
    if request.options.produce_download:
        artifact = create_csv_artifact(
            promoted_frame,
            f"static_ml_dataset_{promoted_version[:12]}",
            ttl_hours=artifact_ttl_hours(),
            owner_user_id=job.owner_user_id,
            query_fingerprint=job.request_fingerprint,
            idempotency_key=job.idempotency_key,
            artifact_kind="dataset_job_result",
        )
        artifact_id = artifact.artifact_id
    job_service.update_job(
        job.job_id,
        status="completed",
        progress=100,
        artifact_id=artifact_id,
        cancel_requested=False,
        result=_result_payload(
            before=before_version or existing_version,
            after=promoted_version,
            row_count=len(promoted_frame),
            validation_status=validation.status if validation else "not_run",
            started=started,
            start=range_start,
            end=request.end,
        ),
        message="Canonical dataset promoted",
    )
    job_service.append_event(
        job.job_id,
        stage="completed",
        percent=100,
        message="Canonical dataset promoted",
    )


# ---------------------------------------------------------------------------
# Existing small v1 routes retain their compatibility response shapes.
# ---------------------------------------------------------------------------


def list_datasets() -> list[DatasetInfo]:
    path = _current_dataset_path()
    if path is None:
        return [DatasetInfo(id=STATIC_DATASET_ID, name="Static ML dataset", description="Runtime cached furnace ML dataset", available=False)]
    try:
        metadata = get_static_metadata()
    except ApiError:
        metadata = None
    return [
        DatasetInfo(
            id=STATIC_DATASET_ID,
            name="Static ML dataset",
            description="Runtime cached furnace ML dataset",
            available=True,
            source="runtime",
            row_count=metadata.row_count if metadata else None,
            last_updated=metadata.last_built_at if metadata else None,
            columns=[column.id for column in metadata.columns] if metadata else None,
        )
    ]


def preview_dataset(dataset_id: str, limit: int = 500) -> DatasetPreviewResponse:
    if dataset_id != STATIC_DATASET_ID:
        raise ApiError("DATASET_NOT_FOUND", "Unknown dataset.", status_code=404)
    capped_limit = min(max(limit, 0), max_preview_rows())
    frame = load_static_dataset_dataframe()
    columns, rows, row_count, truncated = dataframe_to_preview(frame, limit=capped_limit, include_index=True)
    return DatasetPreviewResponse(
        dataset_id=dataset_id,
        columns=columns,
        rows=rows,
        row_count=row_count,
        returned_rows=len(rows),
        truncated=truncated,
    )


def _job_download_url(job: JobState) -> str | None:
    return f"/api/v1/datasets/static_ml_dataset/jobs/{job.job_id}/download" if job.artifact_id else None


def job_to_status(job: JobState) -> DatasetJobStatus:
    return DatasetJobStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        error_code=job.error_code,
        error_message=job.error_message,
        artifact_id=job.artifact_id,
        download_url=_job_download_url(job),
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
    )


def _legacy_refresh_options(request: DatasetRefreshRequest) -> DatasetJobOptions:
    """Translate the pre-v1 options without retaining a second refresh workflow."""

    options = request.options or {}
    return DatasetJobOptions(
        validate_dataset=bool(options.get("validate", options.get("apply_cleaning", True))),
        produce_download=bool(options.get("produce_download", True)),
    )


def _legacy_refresh_datetime(value: datetime | None) -> datetime | None:
    """Keep old refresh payloads usable while canonical jobs use UTC instants."""

    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def refresh_dataset(
    request: DatasetRefreshRequest,
    request_id: str | None = None,
    *,
    current_user: dict[str, Any],
    idempotency_key: str | None = None,
) -> DatasetJobResponse:
    """Compatibility adapter onto the sole canonical static-dataset job path.

    ``/datasets/refresh`` remains only while legacy callers migrate.  It never
    invokes ``StaticDatasetManager.update_static`` directly: every mutation is
    version-checked, durable, cancellable, and globally serialized by
    ``submit_static_dataset_job``.
    """

    dataset_id = request.dataset_id or STATIC_DATASET_ID
    if dataset_id != STATIC_DATASET_ID:
        raise ApiError("DATASET_NOT_FOUND", "Unknown dataset.", status_code=404)

    start = _legacy_refresh_datetime(request.start_time)
    explicit_end = _legacy_refresh_datetime(request.end_time)
    if explicit_end is None:
        # Legacy refresh omitted an end boundary. Resolve that shorthand once
        # per UTC day so a retry with the same Idempotency-Key has a stable
        # canonical payload rather than a different wall-clock timestamp.
        now = _utc_now()
        end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        end = explicit_end
    if start is not None and end < start:
        raise ApiError("INVALID_TIME_RANGE", "start_time must be before or equal to end_time.", 422)
    options = _legacy_refresh_options(request)
    key = str(idempotency_key or "").strip() or f"legacy-refresh-{uuid4().hex}"

    try:
        metadata = get_static_metadata()
    except ApiError as exc:
        if exc.code != "DATASET_NOT_AVAILABLE":
            raise
        if start is None:
            raise ApiError(
                "DATASET_NOT_AVAILABLE",
                "A start_time is required to bootstrap a static dataset through the legacy refresh adapter.",
                404,
            ) from exc
        payload: BuildRangeJobRequest | ExtendJobRequest | OverrideJobRequest = BuildRangeJobRequest(
            operation="build_range",
            start=start,
            end=end,
            options=options,
        )
    else:
        if request.force:
            if start is None:
                raise ApiError(
                    "INVALID_TIME_RANGE",
                    "A start_time is required when force is true on the legacy refresh adapter.",
                    422,
                )
            payload = OverrideJobRequest(
                operation="override",
                start=start,
                end=end,
                expected_dataset_version=metadata.version,
                options=options,
            )
        else:
            payload = ExtendJobRequest(
                operation="extend",
                end=end,
                expected_dataset_version=metadata.version,
                options=options,
            )

    created = submit_static_dataset_job(
        payload,
        current_user=current_user,
        idempotency_key=key,
    )
    job = job_service.get_job(created.job_id)
    return DatasetJobResponse(
        job_id=created.job_id,
        status=created.status,
        message=(job.message if job is not None else "Dataset refresh queued"),
        request_id=request_id,
        created_at=created.created_at,
        updated_at=job.updated_at if job is not None else created.created_at,
        artifact_id=job.artifact_id if job is not None else None,
        download_url=_job_download_url(job) if job is not None else None,
    )


def get_job(job_id: str, current_user: dict[str, Any]) -> DatasetJobStatus:
    job = job_service.get_job(job_id)
    if not job:
        raise ApiError("DATASET_JOB_NOT_FOUND", "Dataset job not found.", status_code=404)
    assert_job_access(job, current_user)
    return job_to_status(job)
