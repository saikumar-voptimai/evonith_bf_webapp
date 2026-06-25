"""InfluxQL query building utilities.

Provides
--------
TIMEDELTAS   Preset lookback string → :class:`datetime.timedelta` map.
WINDOWING    Human-readable window string → InfluxQL window string map.
query_builder Build an InfluxQL SELECT statement for a given measurement.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from furnace_data.config import load_config

# Loaded once at import time — YAML is cheap to parse.
_config = load_config("setting_ds_dv.yml")

# ---------------------------------------------------------------------------
# Public constants / mapping helpers
# ---------------------------------------------------------------------------

TIMEDELTAS: dict[str, timedelta] = {
    "last 1 minute":  timedelta(minutes=1),
    "last 5 minutes": timedelta(minutes=5),
    "last 15 minutes": timedelta(minutes=15),
    "last 30 minutes": timedelta(minutes=30),
    "last 1 hour":    timedelta(hours=1),
    "last 6 hours":   timedelta(hours=6),
    "last 8 hours":   timedelta(hours=8),
    "last 12 hours":  timedelta(hours=12),
    "last 1 day":     timedelta(days=1),
    "last 3 days":    timedelta(days=3),
    "last 1 week":    timedelta(weeks=1),
    "last 2 weeks":   timedelta(weeks=2),
    "last 1 month":   timedelta(days=30),
    "last 2 months":  timedelta(days=60),
    "last 3 months":  timedelta(days=90),
}

WINDOWING: dict[str, str] = {
    "1 minute":  "1m",
    "5 minutes": "5m",
    "10 minutes": "10m",
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour":    "1h",
    "6 hours":   "6h",
    "12 hours":  "12h",
    "1 day":     "1d",
}


def mapping_for_measurement(measurement: str) -> dict[str, str]:
    """Return the human-label to Influx-field mapping for a measurement."""
    return _config.get("data_mapping", {}).get(measurement, {})


def measurement_label(measurement: str) -> str:
    """Return a user-facing measurement label derived from the config key."""
    return measurement.replace("_", " ").title()


def human_labels(measurement: str) -> list[str]:
    """Return human-readable labels derived from data_mapping keys."""
    return list(mapping_for_measurement(measurement).keys())


def influx_fields(measurement: str) -> list[str]:
    """Return de-duplicated Influx fields derived from data_mapping values."""
    return list(dict.fromkeys(mapping_for_measurement(measurement).values()))


def field_labels(measurement: str) -> dict[str, str]:
    """Return the Influx-field to human-label mapping for a measurement."""
    return {
        str(influx_field): str(human_label)
        for human_label, influx_field in mapping_for_measurement(measurement).items()
    }


def field_label(measurement: str, field: str) -> str:
    """Return the configured user-facing label for an Influx field."""
    return field_labels(measurement).get(field, field)


def measurements_for_field(field: str) -> list[str]:
    """Return measurements containing an Influx field in data_mapping."""
    matches: list[str] = []
    for measurement, mapping in (_config.get("data_mapping") or {}).items():
        if field in {str(value) for value in (mapping or {}).values()}:
            matches.append(str(measurement))
    return list(dict.fromkeys(matches))


def measurement_for_field(field: str) -> str | None:
    """Return the first measurement containing an Influx field, if configured."""
    matches = measurements_for_field(field)
    return matches[0] if matches else None


def display_column_name(
    measurement: str,
    field: str,
    *,
    measurement_labels: dict[str, str] | None = None,
    field_label_overrides: dict[str, str] | None = None,
) -> str:
    """Return the legacy display-prefixed DataFrame column name for a field."""
    measurement_labels = measurement_labels or {}
    field_label_overrides = field_label_overrides or {}
    return (
        f"{measurement_labels.get(measurement, measurement_label(measurement))} - "
        f"{field_label_overrides.get(field, field_label(measurement, field))}"
    )


def online_column_aliases(
    field: str,
    *,
    measurement: str | None = None,
    measurement_labels: dict[str, str] | None = None,
    field_label_overrides: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Return canonical and legacy online DataFrame column names for a field."""
    aliases = [field]
    measurements = [measurement] if measurement else measurements_for_field(field)
    for item in measurements:
        aliases.append(
            display_column_name(
                item,
                field,
                measurement_labels=measurement_labels,
                field_label_overrides=field_label_overrides,
            )
        )
        aliases.append(f"{measurement_label(item)} - {field}")
    return tuple(dict.fromkeys(aliases))


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


def query_builder(
    measurement: str,
    start: datetime,
    stop: datetime,
    type: str = "average",
    window_by: str | None = "1h",
) -> str:
    """Build an InfluxQL SELECT query for *measurement* over [start, stop].

    Args:
        measurement: InfluxDB measurement name (e.g. ``"process_params"``).
        start:       UTC-aware start datetime.
        stop:        UTC-aware stop datetime.
        type:        One of ``"ts"``, ``"average"``, ``"avg-min-max"``,
                     ``"windowed-average"``.
        window_by:   Aggregation window for ``"windowed-average"`` type.
                     Accepts human-readable strings (``"15 minutes"``) or
                     InfluxQL strings (``"15m"``).  Pass ``None`` to fall
                     back to ``"ts"`` automatically.

    Returns:
        InfluxQL query string.

    Raises:
        ValueError: If *type* is not recognised.
    """
    # Normalise window_by
    if window_by is not None:
        window_by = WINDOWING.get(window_by, window_by)
        if window_by.replace(" ", "").lower() == "none":
            window_by = None

    # If windowed-average was requested but window_by resolved to None → fall back to ts
    if type == "windowed-average" and window_by is None:
        type = "ts"

    start_iso = start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    end_iso = stop.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    fields = influx_fields(measurement)

    if type == "ts":
        return (
            f"SELECT * FROM {measurement} "
            f"WHERE time >= '{start_iso}' AND time <= '{end_iso}'"
        )
    elif type == "average":
        avg_str = [f"MEAN({col}) AS {col}" for col in fields]
        return (
            f"SELECT {', '.join(avg_str)} FROM {measurement} "
            f"WHERE time >= '{start_iso}' AND time < '{end_iso}'"
        )
    elif type == "avg-min-max":
        parts = [
            f'MEAN({col}) AS "{col}_mean", MIN({col}) AS "{col}_min", MAX({col}) AS "{col}_max"'
            for col in fields
        ]
        return (
            f"SELECT {', '.join(parts)} FROM {measurement} "
            f"WHERE time >= '{start_iso}' AND time < '{end_iso}'"
        )
    elif type == "windowed-average":
        avg_str = [f"MEAN({col}) AS {col}" for col in fields]
        return (
            f"SELECT {', '.join(avg_str)} FROM {measurement} "
            f"WHERE time >= '{start_iso}' AND time < '{end_iso}' "
            f"GROUP BY time({window_by}) fill(null)"
        )
    else:
        raise ValueError(
            f"Invalid query type: {type!r}. "
            "Use 'ts', 'average', 'avg-min-max', or 'windowed-average'."
        )
