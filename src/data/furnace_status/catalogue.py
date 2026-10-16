"""Typed loader and ordered catalogue for ``config/furnace_status.yml``."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from domain.furnace_status.types import ParameterSpec
from furnace_data.influx.query import influx_fields

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "furnace_status.yml"
_AGGREGATES = frozenset({"sum", "mean", "std"})
_COLOUR_KEYS = (
    "surface",
    "inset",
    "hover",
    "grid",
    "border",
    "segment",
    "text_muted",
    "text",
    "accent",
    "warm",
    "warm_text",
    "glow",
    "outline",
    "leader",
    "body_top",
    "body_upper",
    "body_mid",
    "body_lower",
    "body_bottom",
    "lining",
    "zone_divider",
    "live",
    "partial",
)


class CatalogueError(ValueError):
    """The Furnace Status configuration is internally inconsistent."""


@dataclass(frozen=True)
class FurnaceStatusSettings:
    timezone_name: str
    status_lookback: str
    selected_range_label: str
    cache_ttl_seconds: int
    refresh_choices: tuple[tuple[str, int], ...]
    refresh_default: str
    stale_after: timedelta
    live_max_age: timedelta
    live_min_coverage: float
    fixed_intervals: tuple[tuple[str, timedelta], ...]
    custom_interval: str
    default_interval: str
    max_custom_range: timedelta
    raw_gap_break_seconds: float

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)


@dataclass(frozen=True)
class FurnaceZone:
    name: str
    lower_m: float
    upper_m: float
    readings: tuple[str, ...]


@dataclass(frozen=True)
class FurnaceStatusUI:
    colours: tuple[tuple[str, str], ...]
    zones: tuple[FurnaceZone, ...]


@dataclass(frozen=True)
class FurnaceStatusConfig:
    settings: FurnaceStatusSettings
    sections: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]
    ui: FurnaceStatusUI


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CatalogueError(f"{location} must be a mapping")
    return value


def _positive_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogueError(f"{location} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise CatalogueError(f"{location} must be a positive number")
    return result


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CatalogueError(f"{location} must be a finite number")
    return result


def _nonempty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogueError(f"{location} must be a non-empty string")
    return value


def _hex_colour(value: object, location: str) -> str:
    colour = _nonempty_string(value, location)
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", colour) is None:
        raise CatalogueError(f"{location} must be a six-digit hex colour")
    return colour


def _load_settings(raw: object) -> FurnaceStatusSettings:
    settings = _mapping(raw, "settings")
    trend = _mapping(settings.get("trend"), "settings.trend")
    raw_fixed = _mapping(trend.get("fixed_intervals"), "fixed_intervals")
    fixed = tuple(
        (
            _nonempty_string(label, "fixed interval label"),
            timedelta(hours=_positive_number(hours, f"fixed interval {label}")),
        )
        for label, hours in raw_fixed.items()
    )
    default = _nonempty_string(trend.get("default_interval"), "default_interval")
    if default not in {label for label, _ in fixed}:
        raise CatalogueError("default_interval must name a fixed interval")

    timezone_name = _nonempty_string(settings.get("timezone"), "timezone")
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise CatalogueError(f"unknown timezone: {timezone_name}") from exc

    cache_ttl = settings.get("cache_ttl_seconds")
    if isinstance(cache_ttl, bool) or not isinstance(cache_ttl, int) or cache_ttl <= 0:
        raise CatalogueError("cache_ttl_seconds must be a positive integer")
    refresh = _mapping(settings.get("refresh"), "settings.refresh")
    raw_refresh_choices = _mapping(
        refresh.get("choices"), "settings.refresh.choices"
    )
    refresh_choices: list[tuple[str, int]] = []
    for label, seconds in raw_refresh_choices.items():
        name = _nonempty_string(label, "refresh choice label")
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
            raise CatalogueError("refresh choice seconds must be positive integers")
        refresh_choices.append((name, seconds))
    if not refresh_choices:
        raise CatalogueError("settings.refresh.choices must not be empty")
    refresh_default = _nonempty_string(
        refresh.get("default"), "settings.refresh.default"
    )
    if refresh_default not in dict(refresh_choices):
        raise CatalogueError("settings.refresh.default must name a refresh choice")
    coverage = settings.get("live_min_coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        raise CatalogueError("live_min_coverage must be numeric")
    coverage = float(coverage)
    if not 0 <= coverage <= 1:
        raise CatalogueError("live_min_coverage must be between 0 and 1")

    return FurnaceStatusSettings(
        timezone_name=timezone_name,
        status_lookback=_nonempty_string(
            settings.get("status_lookback"), "status_lookback"
        ),
        selected_range_label=_nonempty_string(
            settings.get("selected_range_label"), "selected_range_label"
        ),
        cache_ttl_seconds=cache_ttl,
        refresh_choices=tuple(refresh_choices),
        refresh_default=refresh_default,
        stale_after=timedelta(
            minutes=_positive_number(settings.get("stale_after_minutes"), "stale_after")
        ),
        live_max_age=timedelta(
            minutes=_positive_number(
                settings.get("live_max_age_minutes"), "live_max_age"
            )
        ),
        live_min_coverage=coverage,
        fixed_intervals=fixed,
        custom_interval=_nonempty_string(
            trend.get("custom_interval"), "custom_interval"
        ),
        default_interval=default,
        max_custom_range=timedelta(
            days=_positive_number(
                trend.get("max_custom_range_days"), "max_custom_range"
            )
        ),
        raw_gap_break_seconds=_positive_number(
            trend.get("raw_gap_break_seconds", 120), "raw_gap_break_seconds"
        ),
    )


def _optional_string(raw: Mapping[str, Any], name: str, key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CatalogueError(f"parameter {name!r} {key} must be a non-empty string")
    return value


def _load_parameter(raw_value: object, sections: frozenset[str]) -> ParameterSpec:
    raw = _mapping(raw_value, "parameter")
    key = _nonempty_string(raw.get("key"), "parameter key")
    label = _nonempty_string(raw.get("label"), f"parameter {key!r} label")
    section = _nonempty_string(raw.get("section"), f"parameter {key!r} section")
    if section not in sections:
        raise CatalogueError(f"parameter {key!r} uses unknown section {section!r}")

    measurement = _optional_string(raw, key, "measurement")
    field = _optional_string(raw, key, "field")
    if (measurement is None) != (field is None):
        raise CatalogueError(
            f"parameter {key!r} must define measurement and field together"
        )

    components_value = raw.get("components", [])
    if not isinstance(components_value, list) or not all(
        isinstance(item, str) and item for item in components_value
    ):
        raise CatalogueError(f"parameter {key!r} components must be a list of strings")
    components = tuple(components_value)
    if len(components) != len(set(components)):
        raise CatalogueError(f"parameter {key!r} has duplicate components")
    if components and measurement is None:
        raise CatalogueError(f"derived parameter {key!r} must define a source")
    if components and len(components) < 2:
        raise CatalogueError(
            f"derived parameter {key!r} must define at least two components"
        )
    if field is not None and field in components:
        raise CatalogueError(
            f"derived parameter {key!r} field must not duplicate a component"
        )

    aggregate = raw.get("aggregate", "sum")
    if aggregate not in _AGGREGATES:
        raise CatalogueError(f"parameter {key!r} has invalid aggregate {aggregate!r}")

    setpoint = _optional_string(raw, key, "setpoint_field")
    if setpoint is not None and measurement is None:
        raise CatalogueError(f"parameter {key!r} has a setpoint without a source")

    unavailable = raw.get("unavailable_reason", "")
    if not isinstance(unavailable, str):
        raise CatalogueError(f"parameter {key!r} unavailable_reason must be a string")
    if measurement is None and not unavailable.strip():
        raise CatalogueError(
            f"unsourced parameter {key!r} must define unavailable_reason"
        )

    decimals = raw.get("decimals", 1)
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise CatalogueError(
            f"parameter {key!r} decimals must be a non-negative integer"
        )
    scale = raw.get("scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise CatalogueError(f"parameter {key!r} scale must be numeric")
    scale = float(scale)
    if not math.isfinite(scale) or scale == 0:
        raise CatalogueError(f"parameter {key!r} scale must be finite and non-zero")

    spec = ParameterSpec(
        key=key,
        label=label,
        section=section,
        measurement=measurement,
        field=field,
        unit=str(raw.get("unit", "")),
        decimals=decimals,
        scale=scale,
        trend_label=_optional_string(raw, key, "trend_label"),
        setpoint_field=setpoint,
        components=components,
        aggregate=aggregate,
        source_note=str(raw.get("source_note", "")),
        unavailable_reason=unavailable,
    )
    if spec.has_source:
        configured_fields = frozenset(influx_fields(spec.measurement))
        invalid = [
            source for source in spec.source_fields if source not in configured_fields
        ]
        if invalid:
            raise CatalogueError(
                f"parameter {key!r} uses fields not configured for {measurement!r}: {invalid}"
            )
    return spec


def _load_ui(raw_value: object, parameter_keys: frozenset[str]) -> FurnaceStatusUI:
    raw = _mapping(raw_value, "ui")
    raw_colours = _mapping(raw.get("colours"), "ui.colours")
    if set(raw_colours) != set(_COLOUR_KEYS):
        raise CatalogueError(
            f"ui.colours must define exactly: {', '.join(_COLOUR_KEYS)}"
        )
    colours = tuple(
        (key, _hex_colour(raw_colours[key], f"ui.colours.{key}"))
        for key in _COLOUR_KEYS
    )

    raw_zones = raw.get("zones")
    if not isinstance(raw_zones, list) or not raw_zones:
        raise CatalogueError("ui.zones must be a non-empty list")
    zones: list[FurnaceZone] = []
    for index, raw_zone in enumerate(raw_zones):
        zone = _mapping(raw_zone, f"ui.zones[{index}]")
        name = _nonempty_string(zone.get("name"), f"ui.zones[{index}].name")
        bounds = zone.get("range_m")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise CatalogueError(f"zone {name!r} range_m must contain two numbers")
        lower = _finite_number(bounds[0], f"zone {name!r} lower bound")
        upper = _finite_number(bounds[1], f"zone {name!r} upper bound")
        if lower < 0 or lower >= upper:
            raise CatalogueError(f"zone {name!r} must have 0 <= lower < upper")
        readings = zone.get("readings")
        if not isinstance(readings, list) or not readings or not all(
            isinstance(key, str) and key for key in readings
        ):
            raise CatalogueError(f"zone {name!r} readings must be a non-empty list")
        if len(readings) != len(set(readings)):
            raise CatalogueError(f"zone {name!r} has duplicate readings")
        unknown = sorted(set(readings) - parameter_keys)
        if unknown:
            raise CatalogueError(f"zone {name!r} uses unknown parameters: {unknown}")
        zones.append(FurnaceZone(name, lower, upper, tuple(readings)))

    if len({zone.name for zone in zones}) != len(zones):
        raise CatalogueError("ui.zones names must be unique")
    for upper_zone, lower_zone in zip(zones, zones[1:], strict=False):
        if not math.isclose(upper_zone.lower_m, lower_zone.upper_m):
            raise CatalogueError("ui.zones must be ordered and contiguous")
    return FurnaceStatusUI(colours, tuple(zones))


def load_catalogue(path: str | Path = CONFIG_PATH) -> FurnaceStatusConfig:
    """Load and fully validate a Furnace Status YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        root = _mapping(yaml.safe_load(handle) or {}, str(config_path))

    raw_sections = root.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise CatalogueError("sections must be a non-empty list")
    sections = tuple(_nonempty_string(item, "section") for item in raw_sections)
    if len(sections) != len(set(sections)):
        raise CatalogueError("sections must be unique")

    raw_parameters = root.get("parameters")
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise CatalogueError("parameters must be a non-empty list")
    parameters = tuple(
        _load_parameter(item, frozenset(sections)) for item in raw_parameters
    )
    keys = [spec.key for spec in parameters]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise CatalogueError(f"duplicate parameter keys: {duplicates}")
    return FurnaceStatusConfig(
        _load_settings(root.get("settings")),
        sections,
        parameters,
        _load_ui(root.get("ui"), frozenset(keys)),
    )


CONFIG = load_catalogue()
SETTINGS = CONFIG.settings
SECTIONS = CONFIG.sections
PARAMETERS = CONFIG.parameters
PARAMETERS_BY_KEY = {spec.key: spec for spec in PARAMETERS}
UI = CONFIG.ui

(
    SECTION_PRODUCTION,
    SECTION_BLAST,
    SECTION_UPTAKE,
    SECTION_INJECTION,
    SECTION_PERFORMANCE,
    SECTION_HEAT_LOAD,
    SECTION_HEARTH,
) = SECTIONS


def parameters_in_section(
    section: str, specs: Sequence[ParameterSpec] = PARAMETERS
) -> tuple[ParameterSpec, ...]:
    return tuple(spec for spec in specs if spec.section == section)


def measurements_for(specs: Sequence[ParameterSpec] = PARAMETERS) -> tuple[str, ...]:
    return tuple(dict.fromkeys(spec.measurement for spec in specs if spec.measurement))


def source_fields_for_measurement(
    measurement: str, specs: Sequence[ParameterSpec] = PARAMETERS
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            field
            for spec in specs
            if spec.measurement == measurement
            for field in spec.source_fields
        )
    )
