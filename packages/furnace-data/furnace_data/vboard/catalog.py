"""Canonical V-Board catalog and geometry metadata."""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any

from furnace_data.config import load_config
from furnace_data.influx.query import influx_fields
from furnace_data.vboard.models import ResolutionWindow


CATALOG_VERSION = "vboard-catalog-v1"
DISPLAY_TIMEZONE = "Asia/Kolkata"
PROCESSING_POLICY_ID = "legacy_v1"
PROCESSING_POLICY_DESCRIPTION = (
    "Legacy-compatible V-Board source quality, clipping, interpolation, "
    "and smoothing policy."
)

ROWS = tuple({"id": f"R{idx}", "label": f"Row {idx}", "order": idx} for idx in range(6, 11))
QUADRANTS = tuple({"id": f"Q{idx}", "label": f"Q{idx}", "order": idx} for idx in range(1, 5))

PRESETS = (
    ("last_5_minutes", "Last 5 minutes", 5 * 60),
    ("last_15_minutes", "Last 15 minutes", 15 * 60),
    ("last_30_minutes", "Last 30 minutes", 30 * 60),
    ("last_1_hour", "Last 1 hour", 60 * 60),
    ("last_6_hours", "Last 6 hours", 6 * 60 * 60),
    ("last_12_hours", "Last 12 hours", 12 * 60 * 60),
    ("last_1_day", "Last 1 day", 24 * 60 * 60),
    ("last_3_days", "Last 3 days", 3 * 24 * 60 * 60),
    ("last_1_week", "Last 1 week", 7 * 24 * 60 * 60),
    ("last_2_weeks", "Last 2 weeks", 14 * 24 * 60 * 60),
    ("last_1_month", "Last 1 month", 30 * 24 * 60 * 60),
)

RESOLUTION_WINDOWS = (
    ResolutionWindow("1_minute", "1 minute", 60),
    ResolutionWindow("5_minutes", "5 minutes", 5 * 60),
    ResolutionWindow("15_minutes", "15 minutes", 15 * 60),
    ResolutionWindow("30_minutes", "30 minutes", 30 * 60),
    ResolutionWindow("1_hour", "1 hour", 60 * 60),
)

CIRCUMFERENTIAL_TEMPERATURE_GROUPS = (
    {
        "id": "lower_furnace",
        "title": "Lower Furnace Temperature",
        "level_ids": ["4373", "5411", "5757", "6103"],
    },
    {
        "id": "mid_furnace",
        "title": "Mid Furnace Temperature",
        "level_ids": ["6795", "7565", "8335"],
    },
    {
        "id": "upper_furnace",
        "title": "Upper Furnace Temperature",
        "level_ids": ["9105", "12975", "15162", "18660"],
    },
)

DEFAULT_LIMITS = {
    "max_absolute_range_days": 31,
    "max_timeseries_points_per_quadrant": 2000,
    "max_source_rows": 200000,
}


def preset_by_id() -> dict[str, dict[str, Any]]:
    return {preset["id"]: preset for preset in _preset_dicts()}


def rows_by_id() -> dict[str, dict[str, Any]]:
    return {row["id"]: dict(row) for row in ROWS}


def quadrants_by_id() -> dict[str, dict[str, Any]]:
    return {quadrant["id"]: dict(quadrant) for quadrant in QUADRANTS}


def resolution_windows_by_id() -> dict[str, ResolutionWindow]:
    return {window.id: window for window in RESOLUTION_WINDOWS}


def heatload_field(row_id: str, quadrant_id: str) -> str:
    row = _validate_row_id(row_id)
    quadrant = _validate_quadrant_id(quadrant_id)
    return f"heat_load_{row.lower()}_{quadrant.lower()}"


def heatload_fields_for_row(row_id: str) -> list[str]:
    row = _validate_row_id(row_id)
    return [heatload_field(row, quadrant["id"]) for quadrant in QUADRANTS]


def heatload_fields_for_rows(row_ids: list[str] | tuple[str, ...] | None = None) -> list[str]:
    rows = row_ids or [row["id"] for row in ROWS]
    fields: list[str] = []
    for row in rows:
        fields.extend(heatload_fields_for_row(row))
    return fields


def temperature_fields() -> list[str]:
    return influx_fields("temperature_profile")


def auto_resolution_window_id(duration_seconds: int) -> str:
    """Resolve a bounded default aggregation window for a time-series range."""

    duration = max(1, int(duration_seconds))
    if duration <= 6 * 60 * 60:
        return "1_minute"
    if duration <= 24 * 60 * 60:
        return "5_minutes"
    if duration <= 7 * 24 * 60 * 60:
        return "15_minutes"
    return "1_hour"


def query_window_for_window_id(window_id: str) -> str:
    mapping = {
        "1_minute": "1m",
        "5_minutes": "5m",
        "15_minutes": "15m",
        "30_minutes": "30m",
        "1_hour": "1h",
    }
    if window_id not in mapping:
        raise ValueError(f"Unknown V-Board resolution window: {window_id}")
    return mapping[window_id]


def load_vboard_catalog(
    *,
    max_absolute_range_days: int | None = None,
    max_timeseries_points_per_quadrant: int | None = None,
    max_source_rows: int | None = None,
    processing_policy_id: str | None = None,
    display_timezone: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-native catalog with caller-provided deployment limits."""

    catalog = copy.deepcopy(_base_catalog())
    limits = catalog["limits"]
    if max_absolute_range_days is not None:
        limits["max_absolute_range_days"] = int(max_absolute_range_days)
    if max_timeseries_points_per_quadrant is not None:
        limits["max_timeseries_points_per_quadrant"] = int(max_timeseries_points_per_quadrant)
    if max_source_rows is not None:
        limits["max_source_rows"] = int(max_source_rows)
    if processing_policy_id:
        catalog["processing_policy"]["id"] = str(processing_policy_id)
    if display_timezone:
        catalog["display_timezone"] = str(display_timezone)
    return catalog


@lru_cache(maxsize=1)
def _base_catalog() -> dict[str, Any]:
    config = load_config("setting_ds_dv.yml")
    geometry = config["plot"]["geometry"]
    contour = config["plot"]["contour"]
    heights_dict = geometry["heights_dict"]
    temperature_levels = _temperature_levels(heights_dict)
    _validate_catalog(temperature_levels)

    return {
        "catalog_version": CATALOG_VERSION,
        "display_timezone": DISPLAY_TIMEZONE,
        "presets": _preset_dicts(),
        "rows": [dict(row) for row in ROWS],
        "quadrants": [dict(quadrant) for quadrant in QUADRANTS],
        "temperature_levels": temperature_levels,
        "circumferential_temperature_groups": copy.deepcopy(
            list(CIRCUMFERENTIAL_TEMPERATURE_GROUPS)
        ),
        "longitudinal_geometry": {
            "profile_points": [
                {"x": float(point[0]), "y": float(point[1])}
                for point in geometry["geometry_points"]
            ],
            "regions": [
                {"id": _public_id(label), "label": str(label), "elevation_m": float(elevation)}
                for label, elevation in geometry["regions"]
            ],
            "x_range": [
                float(contour["furnace_grid_X_low"]),
                float(contour["furnace_grid_X_high"]),
            ],
            "y_range": [4.0, float(contour["furnace_grid_Y_high"])],
        },
        "display": {
            "temperature_unit": "\u00b0C",
            "heatload_unit": None,
            "heatload_label": "Heat-load index",
        },
        "processing_policy": {
            "id": PROCESSING_POLICY_ID,
            "description": PROCESSING_POLICY_DESCRIPTION,
        },
        "resolution_windows": [
            {"id": item.id, "label": item.label, "seconds": item.seconds}
            for item in RESOLUTION_WINDOWS
        ],
        "limits": dict(DEFAULT_LIMITS),
    }


def _preset_dicts() -> list[dict[str, Any]]:
    return [
        {
            "id": preset_id,
            "label": label,
            "duration_seconds": duration_seconds,
            "supported_for": ["contours", "heatload_timeseries"],
        }
        for preset_id, label, duration_seconds in PRESETS
    ]


def _temperature_levels(heights_dict: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, (level_id, payload) in enumerate(
        sorted(heights_dict.items(), key=lambda item: float(item[1]["level"])),
        start=1,
    ):
        elevation_m = float(payload["level"])
        rows.append(
            {
                "id": str(level_id),
                "elevation_m": elevation_m,
                "label": f"{elevation_m:.3f} m",
                "sensor_count": int(payload["n_sensors"]),
                "order": order,
            }
        )
    return rows


def _validate_catalog(temperature_levels: list[dict[str, Any]]) -> None:
    level_ids = [str(item["id"]) for item in temperature_levels]
    elevations = [float(item["elevation_m"]) for item in temperature_levels]
    orders = [int(item["order"]) for item in temperature_levels]
    if len(level_ids) != len(set(level_ids)):
        raise ValueError("V-Board temperature level IDs must be unique.")
    if len(elevations) != len(set(elevations)):
        raise ValueError("V-Board temperature elevations must be unique.")
    if len(orders) != len(set(orders)):
        raise ValueError("V-Board temperature level order values must be unique.")
    known_levels = set(level_ids)
    for group in CIRCUMFERENTIAL_TEMPERATURE_GROUPS:
        unknown = [level_id for level_id in group["level_ids"] if level_id not in known_levels]
        if unknown:
            raise ValueError(
                f"V-Board group {group['id']} references unknown levels: {unknown}"
            )


def _validate_row_id(row_id: str) -> str:
    normalized = str(row_id or "").strip().upper()
    if normalized not in {row["id"] for row in ROWS}:
        raise ValueError(f"Unknown V-Board row ID: {row_id}")
    return normalized


def _validate_quadrant_id(quadrant_id: str) -> str:
    normalized = str(quadrant_id or "").strip().upper()
    if normalized not in {quadrant["id"] for quadrant in QUADRANTS}:
        raise ValueError(f"Unknown V-Board quadrant ID: {quadrant_id}")
    return normalized


def _public_id(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")
