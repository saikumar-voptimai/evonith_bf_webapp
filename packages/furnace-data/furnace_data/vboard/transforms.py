"""Canonical V-Board source transformations."""

from __future__ import annotations

import math
from datetime import timezone
from typing import Any

import numpy as np
import pandas as pd

from furnace_data.vboard.catalog import (
    PROCESSING_POLICY_ID,
    QUADRANTS,
    ROWS,
    heatload_field,
    load_vboard_catalog,
)


_SENSOR_SUFFIXES = tuple("abcdefghijklmn")
_QUADRANT_ANGLES = (45, 135, 225, 315)


def transform_temperature_contour(
    frame: pd.DataFrame,
    *,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transform one temperature source result into per-level quadrant stats."""

    catalog = catalog or load_vboard_catalog()
    output_levels: list[dict[str, Any]] = []
    missing_level_ids: list[str] = []
    frame = _single_row_frame(frame)

    for level in catalog["temperature_levels"]:
        level_id = str(level["id"])
        n_sensors = int(level["sensor_count"])
        level_quadrants: list[dict[str, Any]] = []
        for q_index, quadrant in enumerate(QUADRANTS):
            weights = _weights_for_angle(_QUADRANT_ANGLES[q_index], n_sensors)
            sensor_indices = [idx for idx, weight in enumerate(weights) if weight > 0]
            stats = {
                suffix: _weighted_sensor_value(
                    frame,
                    level_id=level_id,
                    suffix=suffix,
                    sensor_indices=sensor_indices,
                    weights=weights,
                )
                for suffix in ("mean", "min", "max")
            }
            valid_count = _valid_sensor_count(
                frame,
                level_id=level_id,
                suffix="mean",
                sensor_indices=sensor_indices,
            )
            level_quadrants.append(
                {
                    "quadrant_id": quadrant["id"],
                    "mean": stats["mean"],
                    "minimum": stats["min"],
                    "maximum": stats["max"],
                    "valid_sensor_count": valid_count,
                }
            )
        if all(item["mean"] is None for item in level_quadrants):
            missing_level_ids.append(level_id)
        output_levels.append(
            {
                "level_id": level_id,
                "elevation_m": _json_number(level["elevation_m"]),
                "quadrants": level_quadrants,
            }
        )

    warnings: list[str] = []
    status = _section_status(
        total_items=len(output_levels),
        missing_items=len(missing_level_ids),
        empty_message="No temperature data was returned for the selected range.",
        partial_message="Some temperature levels have no valid sensor values.",
        warnings=warnings,
    )
    return {
        "status": status,
        "unit": catalog["display"]["temperature_unit"],
        "levels": output_levels,
        "missing_level_ids": missing_level_ids,
        "warnings": warnings,
    }


def transform_heatload_contour(
    frame: pd.DataFrame,
    *,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transform one heat-load source result into R6-R10 quadrant stats."""

    catalog = catalog or load_vboard_catalog()
    frame = _single_row_frame(frame)
    rows: list[dict[str, Any]] = []
    missing_row_ids: list[str] = []

    for row in ROWS:
        row_id = row["id"]
        raw_stats = {suffix: [] for suffix in ("mean", "min", "max")}
        for quadrant in QUADRANTS:
            field = heatload_field(row_id, quadrant["id"])
            for suffix in raw_stats:
                raw_stats[suffix].append(_series_value(frame, f"{field}_{suffix}"))

        cleaned_stats = {
            suffix: _interpolate_quadrants(
                [_normalize_heatload_contour_value(value) for value in values]
            )
            for suffix, values in raw_stats.items()
        }
        quadrants = []
        for idx, quadrant in enumerate(QUADRANTS):
            quadrants.append(
                {
                    "quadrant_id": quadrant["id"],
                    "mean": cleaned_stats["mean"][idx],
                    "minimum": cleaned_stats["min"][idx],
                    "maximum": cleaned_stats["max"][idx],
                }
            )
        if all(item["mean"] is None for item in quadrants):
            missing_row_ids.append(row_id)
        rows.append({"row_id": row_id, "quadrants": quadrants})

    warnings: list[str] = []
    status = _section_status(
        total_items=len(rows),
        missing_items=len(missing_row_ids),
        empty_message="No heat-load contour data was returned for the selected range.",
        partial_message="Some heat-load rows have no valid values.",
        warnings=warnings,
    )
    return {
        "status": status,
        "unit": catalog["display"]["heatload_unit"],
        "display_label": catalog["display"]["heatload_label"],
        "rows": rows,
        "missing_row_ids": missing_row_ids,
        "warnings": warnings,
    }


def transform_heatload_timeseries(
    frame: pd.DataFrame,
    *,
    row_id: str,
    resolved_window_seconds: int,
    max_points_per_quadrant: int,
    catalog: dict[str, Any] | None = None,
    processing_policy_id: str = PROCESSING_POLICY_ID,
) -> dict[str, Any]:
    """Transform one selected-row heat-load result into four quadrant series."""

    catalog = catalog or load_vboard_catalog()
    row_id = str(row_id).strip().upper()
    prepared = _prepare_time_index(frame)
    source_guard = max(1, int(max_points_per_quadrant)) * 20
    if len(prepared.index) > source_guard:
        prepared = _deterministic_sample(prepared, source_guard)

    series: list[dict[str, Any]] = []
    downsampled = False
    for quadrant in QUADRANTS:
        field = heatload_field(row_id, quadrant["id"])
        values = (
            pd.to_numeric(prepared[field], errors="coerce")
            if field in prepared.columns
            else pd.Series(dtype="float64", index=prepared.index)
        )
        values = values.map(_normalize_heatload_timeseries_value)
        values = values.rolling("3600s", min_periods=1).mean()
        field_frame = pd.DataFrame({"value": values})
        sampled = _deterministic_sample(field_frame, max(1, int(max_points_per_quadrant)))
        downsampled = downsampled or len(sampled.index) < len(field_frame.index)
        points = [
            {"timestamp": _iso_z(timestamp), "value": _json_number(value)}
            for timestamp, value in sampled["value"].items()
        ]
        series.append(
            {
                "quadrant_id": quadrant["id"],
                "points": points,
                "returned_points": len(points),
                "missing_points": sum(1 for point in points if point["value"] is None),
            }
        )

    warnings: list[str] = []
    if prepared.empty:
        warnings.append("No heat-load time-series data was returned for the selected range.")
    return {
        "unit": catalog["display"]["heatload_unit"],
        "display_label": catalog["display"]["heatload_label"],
        "resolved_window_seconds": int(resolved_window_seconds),
        "processing": {
            "policy_id": processing_policy_id,
            "smoothing_kind": "time_based_moving_average",
            "smoothing_window_seconds": 3600,
            "normalization": "legacy_compatible",
        },
        "series": series,
        "downsampled": downsampled,
        "warnings": warnings,
    }


def build_empty_heatload_timeseries(
    *,
    catalog: dict[str, Any] | None = None,
    resolved_window_seconds: int,
    processing_policy_id: str = PROCESSING_POLICY_ID,
) -> dict[str, Any]:
    catalog = catalog or load_vboard_catalog()
    return {
        "unit": catalog["display"]["heatload_unit"],
        "display_label": catalog["display"]["heatload_label"],
        "resolved_window_seconds": int(resolved_window_seconds),
        "processing": {
            "policy_id": processing_policy_id,
            "smoothing_kind": "time_based_moving_average",
            "smoothing_window_seconds": 3600,
            "normalization": "legacy_compatible",
        },
        "series": [
            {
                "quadrant_id": quadrant["id"],
                "points": [],
                "returned_points": 0,
                "missing_points": 0,
            }
            for quadrant in QUADRANTS
        ],
        "downsampled": False,
        "warnings": ["No heat-load time-series data was returned for the selected range."],
    }


def _single_row_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    if frame.empty:
        return frame
    return frame.tail(1)


def _weights_for_angle(angle: float, n_sensors: int) -> list[float]:
    return [
        max(1 - abs((angle - idx * 360 / n_sensors + 180) % 360 - 180) / (360 / n_sensors), 0)
        for idx in range(n_sensors)
    ]


def _weighted_sensor_value(
    frame: pd.DataFrame,
    *,
    level_id: str,
    suffix: str,
    sensor_indices: list[int],
    weights: list[float],
) -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    for idx in sensor_indices:
        field = f"temp_{level_id}_{_SENSOR_SUFFIXES[idx]}_{suffix}"
        value = _series_value(frame, field)
        if value is None:
            continue
        weight = float(weights[idx])
        weighted_sum += float(value) * weight
        weight_sum += weight
    if weight_sum == 0:
        return None
    return _json_number(weighted_sum / weight_sum)


def _valid_sensor_count(
    frame: pd.DataFrame,
    *,
    level_id: str,
    suffix: str,
    sensor_indices: list[int],
) -> int:
    count = 0
    for idx in sensor_indices:
        field = f"temp_{level_id}_{_SENSOR_SUFFIXES[idx]}_{suffix}"
        if _series_value(frame, field) is not None:
            count += 1
    return count


def _series_value(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    value = frame[column].iloc[-1]
    return _json_number(value)


def _json_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _normalize_heatload_contour_value(value: Any) -> float | None:
    number = _json_number(value)
    if number is None or number < 0:
        return None
    if number > 1:
        return 1.0
    return number


def _normalize_heatload_timeseries_value(value: Any) -> float | None:
    number = _json_number(value)
    if number is None:
        return None
    if number < 0.1:
        return 0.0
    if number > 1:
        return 1.0
    return number


def _interpolate_quadrants(values: list[float | None]) -> list[float | None]:
    series = pd.Series(values, dtype="float64")
    if series.notna().sum() >= 2:
        series = series.interpolate(method="linear", limit_area="inside")
    return [_json_number(value) for value in series.tolist()]


def _prepare_time_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="time"))
    output = frame.copy()
    if "time" in output.columns:
        timestamps = pd.to_datetime(output["time"], utc=True, errors="coerce")
        output = output.drop(columns=["time"])
    else:
        timestamps = pd.to_datetime(output.index, utc=True, errors="coerce")
    output.index = timestamps
    output.index.name = "time"
    output = output.loc[~output.index.isna()]
    output = output[~output.index.duplicated(keep="last")].sort_index()
    return output


def _deterministic_sample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    max_points = max(1, int(max_points))
    if len(frame.index) <= max_points:
        return frame
    positions = np.linspace(0, len(frame.index) - 1, num=max_points, dtype=int)
    return frame.iloc[positions]


def _iso_z(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _section_status(
    *,
    total_items: int,
    missing_items: int,
    empty_message: str,
    partial_message: str,
    warnings: list[str],
) -> str:
    if total_items == 0 or missing_items == total_items:
        warnings.append(empty_message)
        return "empty"
    if missing_items:
        warnings.append(partial_message)
        return "partial"
    return "ok"
