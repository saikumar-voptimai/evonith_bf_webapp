"""Load and validate all Scheduled Tasks catalog options from configuration.

The catalog is sourced from ``scheduled_jobs.yml`` and acts as the single source
of truth for the operator-facing controls shown in the page. UI code reads stable
identifiers and labels from these models, while definition builders use the same
objects for consistency and validation.

This module is configuration-only: it validates values and returns typed
objects, but does not connect to devices or perform scheduling side effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config.config_loader import load_config

_JOB_TYPE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class JobTypeOption:
    """Catalog entry describing one operator-visible task type."""

    id: str
    label: str
    description: str
    default_analysis_level: str
    input_profile: str


@dataclass(frozen=True, slots=True)
class TargetDeviceOption:
    """Catalog entry identifying a configured execution target."""

    device_id: str
    device_type: str


@dataclass(frozen=True, slots=True)
class CatalogOption:
    """General-purpose selectable option (id/label) from configuration."""

    id: str
    label: str
    minutes: int | None = None


@dataclass(frozen=True, slots=True)
class TimezoneOption:
    """Timezone choice with system identifier plus UI label."""

    name: str
    label: str


@dataclass(frozen=True, slots=True)
class ScheduledJobCatalog:
    """Validated in-memory snapshot of catalog options and defaults."""

    job_types: tuple[JobTypeOption, ...]
    target_devices: tuple[TargetDeviceOption, ...]
    data_sources: tuple[CatalogOption, ...]
    eta_co_signals: tuple[CatalogOption, ...]
    aggregation_intervals: tuple[CatalogOption, ...]
    timezones: tuple[TimezoneOption, ...]
    defaults: dict[str, Any]

    def job_type(self, job_type_id: str) -> JobTypeOption | None:
        """Return the task type matching a stable identifier."""

        return next((item for item in self.job_types if item.id == job_type_id), None)

    def target_device(self, device_id: str) -> TargetDeviceOption | None:
        """Return the target device matching a stable identifier."""

        return next(
            (item for item in self.target_devices if item.device_id == device_id),
            None,
        )

    def data_source(self, source_id: str) -> CatalogOption | None:
        """Return the configured furnace data source matching an identifier."""

        return next(
            (item for item in self.data_sources if item.id == source_id),
            None,
        )

    def eta_co_signal(self, signal_id: str) -> CatalogOption | None:
        """Return the configured ETA CO signal matching an identifier."""

        return next(
            (item for item in self.eta_co_signals if item.id == signal_id),
            None,
        )

    def aggregation_interval(self, value: str) -> CatalogOption | None:
        """Return the configured aggregation interval matching a stored value."""

        return next(
            (item for item in self.aggregation_intervals if item.id == value),
            None,
        )

    def timezone(self, timezone_name: str) -> TimezoneOption | None:
        """Return the configured option for an IANA timezone name."""

        return next(
            (item for item in self.timezones if item.name == timezone_name),
            None,
        )

    @property
    def default_timezone(self) -> str:
        """Return the default IANA timezone name."""

        return str(self.defaults["timezone"])

    @property
    def default_target_device(self) -> TargetDeviceOption:
        """Return the configured default device after catalog validation."""

        target = self.target_device(str(self.defaults["target_device_id"]))
        if target is None:  # Configuration validation prevents this at runtime.
            raise ValueError("The configured default target device is unavailable.")
        return target


def _required_text(item: dict[str, Any], key: str, section: str) -> str:
    """Extract a required non-empty string value from a catalog mapping."""

    raw_value = item.get(key)
    value = raw_value.strip() if isinstance(raw_value, str) else ""
    if not value:
        raise ValueError(f"scheduled_jobs.yml: {section}.{key} is required")
    return value


def _ensure_unique(values: list[str], section: str) -> None:
    """Assert all values in a catalog section are unique."""

    if len(values) != len(set(values)):
        raise ValueError(f"scheduled_jobs.yml: duplicate identifiers in {section}")


def _required_mapping(item: dict[str, Any], key: str, section: str) -> dict[str, Any]:
    """Extract a required nested mapping and return a copy."""

    value = item.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"scheduled_jobs.yml: {section}.{key} is required")
    return dict(value)


@lru_cache(maxsize=1)
def get_scheduled_job_catalog() -> ScheduledJobCatalog:
    """Load and validate the non-sensitive Scheduled Tasks choice catalog."""

    raw = load_config("scheduled_jobs.yml") or {}
    if raw.get("schema_version") != "scheduled-jobs-catalog/v1":
        raise ValueError("scheduled_jobs.yml has an unsupported schema_version")

    job_types = tuple(
        JobTypeOption(
            id=_required_text(item, "id", "job_types"),
            label=_required_text(item, "label", "job_types"),
            description=_required_text(item, "description", "job_types"),
            default_analysis_level=_required_text(
                item, "default_analysis_level", "job_types"
            ),
            input_profile=_required_text(item, "input_profile", "job_types"),
        )
        for item in raw.get("job_types", [])
    )
    target_devices = tuple(
        TargetDeviceOption(
            device_id=_required_text(item, "device_id", "target_devices"),
            device_type=_required_text(item, "device_type", "target_devices"),
        )
        for item in raw.get("target_devices", [])
    )
    data_sources = tuple(
        CatalogOption(
            id=_required_text(item, "id", "data_sources"),
            label=_required_text(item, "label", "data_sources"),
        )
        for item in raw.get("data_sources", [])
    )
    eta_co_signals = tuple(
        CatalogOption(
            id=_required_text(item, "id", "eta_co_signals"),
            label=_required_text(item, "label", "eta_co_signals"),
        )
        for item in raw.get("eta_co_signals", [])
    )
    aggregation_intervals = tuple(
        CatalogOption(
            id=_required_text(item, "value", "aggregation_intervals"),
            label=_required_text(item, "label", "aggregation_intervals"),
            minutes=int(item.get("minutes", 0)),
        )
        for item in raw.get("aggregation_intervals", [])
    )
    timezones = tuple(
        TimezoneOption(
            name=_required_text(item, "name", "timezones"),
            label=_required_text(item, "label", "timezones"),
        )
        for item in raw.get("timezones", [])
    )

    if not all(
        (
            job_types,
            target_devices,
            data_sources,
            eta_co_signals,
            aggregation_intervals,
            timezones,
        )
    ):
        raise ValueError("scheduled_jobs.yml must define every operator choice catalog")

    _ensure_unique([item.id for item in job_types], "job_types")
    _ensure_unique([item.device_id for item in target_devices], "target_devices")
    _ensure_unique([item.id for item in data_sources], "data_sources")
    _ensure_unique([item.id for item in eta_co_signals], "eta_co_signals")
    _ensure_unique([item.id for item in aggregation_intervals], "aggregation_intervals")
    _ensure_unique([item.name for item in timezones], "timezones")

    allowed_analysis_levels = {"none", "low", "medium", "high"}
    if any(not _JOB_TYPE_ID_PATTERN.fullmatch(item.id) for item in job_types):
        raise ValueError("scheduled_jobs.yml contains an invalid job type identifier")
    if any(
        item.default_analysis_level not in allowed_analysis_levels for item in job_types
    ):
        raise ValueError(
            "scheduled_jobs.yml contains an unsupported default analysis level"
        )
    if any(item.input_profile not in {"eta_co", "common"} for item in job_types):
        raise ValueError("scheduled_jobs.yml contains an unsupported input profile")
    if any(
        item.device_type not in {"vm", "jetson", "raspberry_pi"}
        for item in target_devices
    ):
        raise ValueError("scheduled_jobs.yml contains an unsupported device type")
    if any(item.minutes is None or item.minutes < 1 for item in aggregation_intervals):
        raise ValueError("scheduled_jobs.yml aggregation minutes must be positive")
    for option in timezones:
        try:
            ZoneInfo(option.name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                "scheduled_jobs.yml contains an invalid IANA timezone"
            ) from exc

    defaults = dict(raw.get("defaults") or {})
    if not defaults:
        raise ValueError("scheduled_jobs.yml defaults are required")

    timezone_name = str(defaults.get("timezone", "")).strip()
    if timezone_name != "Asia/Kolkata":
        raise ValueError("scheduled_jobs.yml defaults.timezone must be Asia/Kolkata")
    if not any(item.name == timezone_name for item in timezones):
        raise ValueError(
            "scheduled_jobs.yml defaults.timezone must reference a configured timezone"
        )

    target_device_id = str(defaults.get("target_device_id", "")).strip()
    if not any(item.device_id == target_device_id for item in target_devices):
        raise ValueError(
            "scheduled_jobs.yml defaults.target_device_id must reference a configured target device"
        )

    eta_defaults = _required_mapping(defaults, "eta_co", "defaults")
    if not any(item.id == eta_defaults.get("signal") for item in eta_co_signals):
        raise ValueError("scheduled_jobs.yml ETA default signal is not configured")
    default_aggregation = next(
        (
            item
            for item in aggregation_intervals
            if item.id == eta_defaults.get("aggregation_interval")
        ),
        None,
    )
    if default_aggregation is None:
        raise ValueError(
            "scheduled_jobs.yml ETA default aggregation interval is not configured"
        )
    if not (
        0
        <= float(eta_defaults.get("critical_threshold", -1))
        < float(eta_defaults.get("warning_threshold", -1))
        <= 100
    ):
        raise ValueError("scheduled_jobs.yml ETA default thresholds are invalid")
    default_duration = int(eta_defaults.get("report_duration_minutes", 0))
    if not 5 <= default_duration <= 1440:
        raise ValueError("scheduled_jobs.yml ETA default duration is invalid")
    if (
        default_aggregation.minutes is not None
        and default_aggregation.minutes > default_duration
    ):
        raise ValueError(
            "scheduled_jobs.yml ETA default aggregation exceeds its duration"
        )
    if not isinstance(eta_defaults.get("include_graph"), bool) or not isinstance(
        eta_defaults.get("include_ai_summary"), bool
    ):
        raise ValueError("scheduled_jobs.yml ETA output defaults must be booleans")

    reliability = _required_mapping(defaults, "reliability", "defaults")
    if not 1 <= int(reliability.get("maximum_attempts", 0)) <= 10:
        raise ValueError("scheduled_jobs.yml default maximum attempts are invalid")
    if not 1 <= int(reliability.get("retry_interval_seconds", 0)) <= 3600:
        raise ValueError("scheduled_jobs.yml default retry interval is invalid")
    if not 60 <= int(reliability.get("timeout_seconds", 0)) <= 86400:
        raise ValueError("scheduled_jobs.yml default timeout is invalid")
    if not isinstance(reliability.get("notify_on_failure"), bool):
        raise ValueError("scheduled_jobs.yml failure notification default is invalid")
    if reliability.get("overlap_policy") != "skip":
        raise ValueError("scheduled_jobs.yml default overlap policy is invalid")
    if reliability.get("misfire_policy") != "fire_once_latest":
        raise ValueError("scheduled_jobs.yml default misfire policy is invalid")
    if defaults.get("policy_profile") != "bf_operator_read_only_v1":
        raise ValueError("scheduled_jobs.yml default policy profile is invalid")

    return ScheduledJobCatalog(
        job_types=job_types,
        target_devices=target_devices,
        data_sources=data_sources,
        eta_co_signals=eta_co_signals,
        aggregation_intervals=aggregation_intervals,
        timezones=timezones,
        defaults=defaults,
    )


__all__ = [
    "CatalogOption",
    "JobTypeOption",
    "ScheduledJobCatalog",
    "TargetDeviceOption",
    "TimezoneOption",
    "get_scheduled_job_catalog",
]
