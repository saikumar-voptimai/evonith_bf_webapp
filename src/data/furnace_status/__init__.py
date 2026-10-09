"""Stable public API for the refactored Furnace Status feature."""

from data.furnace_status.catalogue import (
    CONFIG,
    CONFIG_PATH,
    PARAMETERS,
    PARAMETERS_BY_KEY,
    SECTION_BLAST,
    SECTION_HEARTH,
    SECTION_HEAT_LOAD,
    SECTION_INJECTION,
    SECTION_PERFORMANCE,
    SECTION_PRODUCTION,
    SECTION_UPTAKE,
    SECTIONS,
    SETTINGS,
    CatalogueError,
    FurnaceStatusConfig,
    FurnaceStatusSettings,
    load_catalogue,
    measurements_for,
    parameters_in_section,
    source_fields_for_measurement,
)
from data.furnace_status.service import (
    _current_reading,
    _trend_result,
    clear_cache,
    load_status_snapshot,
    load_trend_view,
)
from domain.furnace_status.formatting import (
    NOT_AVAILABLE,
    format_ist,
    format_reading,
    format_value,
    has_display_value,
)
from domain.furnace_status.readings import (
    IST,
    finite_series,
    latest_finite_point,
    normalize_timestamps,
    resolve_reading,
    series_for_spec,
)
from domain.furnace_status.status import build_status_snapshot, compute_plant_status
from domain.furnace_status.time_ranges import (
    CUSTOM_INTERVAL,
    DEFAULT_INTERVAL,
    FIXED_INTERVALS,
    INTERVAL_OPTIONS,
    MAX_CUSTOM_RANGE,
    RangeError,
    choose_window,
    default_custom_range,
    floor_to_minute,
    resolve_custom_range,
    resolve_fixed_range,
)
from domain.furnace_status.trends import build_trend_data, compute_trend_stats
from domain.furnace_status.types import (
    ParameterReading,
    ParameterSpec,
    PlantStatus,
    StatusSnapshot,
    TrendData,
    TrendStats,
    TrendViewData,
    ViewState,
)
from domain.furnace_status.view_state import (
    PARAMETER_QUERY_KEY,
    VIEW_QUERY_KEY,
    VIEW_STATUS,
    VIEW_TREND,
    parse_view_state,
)

STATUS_LOOKBACK = SETTINGS.status_lookback
SELECTED_RANGE = SETTINGS.selected_range_label
CACHE_TTL_SECONDS = SETTINGS.cache_ttl_seconds
STALE_AFTER = SETTINGS.stale_after
LIVE_MAX_AGE = SETTINGS.live_max_age
LIVE_MIN_COVERAGE = SETTINGS.live_min_coverage

__all__ = [name for name in globals() if not name.startswith("__")]
