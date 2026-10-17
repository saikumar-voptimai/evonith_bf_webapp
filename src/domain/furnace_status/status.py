"""Plant telemetry classification and status snapshot construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import pandas as pd

from data.furnace_status.catalogue import PARAMETERS, SETTINGS, measurements_for
from domain.furnace_status.formatting import format_value
from domain.furnace_status.readings import resolve_reading
from domain.furnace_status.types import (
    OperatingState,
    ParameterReading,
    ParameterSpec,
    PlantStatus,
    StatusSnapshot,
)


def compute_operating_state(
    readings: Sequence[ParameterReading],
) -> OperatingState:
    """Classify furnace operation from the configured unrounded live value."""
    policy = SETTINGS.operating_state
    reading = next(
        (item for item in readings if item.spec.key == policy.parameter), None
    )
    if reading is None:
        return OperatingState(
            "unknown", "Unknown", "Operating-state parameter is unavailable."
        )

    spec = reading.spec
    unit = f" {spec.unit}" if spec.unit else ""
    threshold = format_value(policy.normal_min, spec.decimals)
    rule = f"Normal when {spec.label} ≥ {threshold}{unit}"
    if reading.value is None:
        return OperatingState(
            "unknown", "Unknown", f"{rule} · now no fresh value"
        )

    current = format_value(reading.value, spec.decimals)
    if reading.value >= policy.normal_min:
        return OperatingState(
            "normal", "Normal", f"{rule} · now {current}{unit}"
        )
    return OperatingState(
        "abnormal", "Abnormal", f"{rule} · now {current}{unit}"
    )


def compute_plant_status(
    readings: Sequence[ParameterReading], now: datetime
) -> PlantStatus:
    backed = [reading for reading in readings if reading.spec.has_source]
    with_value = [reading for reading in backed if reading.value is not None]
    coverage = len(with_value) / len(backed) if backed else 0.0
    detail = f"{len(with_value)} of {len(backed)} mapped parameters reporting"
    timestamps = [
        reading.timestamp for reading in with_value if reading.timestamp is not None
    ]
    if not timestamps:
        return PlantStatus("offline", "Offline", detail, coverage, None)
    age = max(now - max(timestamps).to_pydatetime(), timedelta(0))
    if age > SETTINGS.stale_after:
        return PlantStatus("offline", "Offline", detail, coverage, age)
    if age <= SETTINGS.live_max_age and coverage >= SETTINGS.live_min_coverage:
        return PlantStatus("live", "Live", detail, coverage, age)
    return PlantStatus("partial", "Partial", detail, coverage, age)


def build_status_snapshot(
    frames: Mapping[str, pd.DataFrame | None],
    now: datetime,
    specs: Sequence[ParameterSpec] = PARAMETERS,
) -> StatusSnapshot:
    readings = tuple(resolve_reading(spec, frames, now) for spec in specs)
    timestamps = [
        timestamp
        for reading in readings
        for timestamp in (reading.timestamp, reading.setpoint_timestamp)
        if timestamp is not None
    ]
    failed = tuple(
        measurement
        for measurement in measurements_for(specs)
        if frames.get(measurement) is None
    )
    return StatusSnapshot(
        readings=readings,
        status=compute_plant_status(readings, now),
        operating_state=compute_operating_state(readings),
        last_updated=max(timestamps) if timestamps else None,
        failed_measurements=failed,
    )
