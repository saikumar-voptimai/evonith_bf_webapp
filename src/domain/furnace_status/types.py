"""Immutable values shared across the Furnace Status feature."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import pandas as pd

Aggregate = Literal["sum", "mean"]
ReadingIssue = Literal["no_source", "fetch_failed", "no_data", "stale"]
TrendStatus = Literal["ok", "no_source", "no_data", "error"]


@dataclass(frozen=True)
class ParameterSpec:
    """One display parameter and its optional live source."""

    key: str
    label: str
    section: str
    measurement: str | None = None
    field: str | None = None
    unit: str = ""
    decimals: int = 1
    scale: float = 1.0
    trend_label: str | None = None
    setpoint_field: str | None = None
    components: tuple[str, ...] = ()
    aggregate: Aggregate = "sum"
    source_note: str = ""
    unavailable_reason: str = ""

    @property
    def has_source(self) -> bool:
        return self.measurement is not None and self.field is not None

    @property
    def source_fields(self) -> tuple[str, ...]:
        actual = self.components or (self.field,)
        return tuple(field for field in (*actual, self.setpoint_field) if field)

    @property
    def title(self) -> str:
        return self.trend_label or self.label


@dataclass(frozen=True)
class ParameterReading:
    """Resolved and already-scaled dashboard value for one parameter."""

    spec: ParameterSpec
    value: float | None = None
    timestamp: pd.Timestamp | None = None
    issue: ReadingIssue | None = None
    setpoint: float | None = None
    setpoint_timestamp: pd.Timestamp | None = None


@dataclass(frozen=True)
class PlantStatus:
    """Telemetry availability, not furnace safety or process state."""

    level: Literal["live", "partial", "offline"]
    label: str
    detail: str
    coverage: float
    age: timedelta | None


@dataclass(frozen=True)
class StatusSnapshot:
    readings: tuple[ParameterReading, ...]
    status: PlantStatus
    last_updated: pd.Timestamp | None
    failed_measurements: tuple[str, ...]


@dataclass(frozen=True)
class TrendData:
    spec: ParameterSpec
    status: TrendStatus
    series: pd.Series | None
    message: str = ""
    setpoint: pd.Series | None = None


@dataclass(frozen=True)
class TrendStats:
    minimum: float
    maximum: float
    mean: float


@dataclass(frozen=True)
class TrendViewData:
    current: ParameterReading
    trend: TrendData
    stats: TrendStats | None


@dataclass(frozen=True)
class ViewState:
    view: Literal["status", "trend"]
    spec: ParameterSpec | None = None
    needs_reset: bool = False
