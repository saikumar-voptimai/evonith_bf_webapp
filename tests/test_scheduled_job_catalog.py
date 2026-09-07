"""Focused tests for the configuration-backed scheduled-job catalog."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

import utils.scheduled_tasks.scheduled_job_catalog as catalog_module
from utils.scheduled_tasks.scheduled_job_catalog import get_scheduled_job_catalog


@pytest.fixture(autouse=True)
def _clear_catalog_cache() -> None:
    """Keep patched catalog inputs isolated despite the production LRU cache."""

    get_scheduled_job_catalog.cache_clear()
    yield
    get_scheduled_job_catalog.cache_clear()


def _configured_catalog() -> dict[str, Any]:
    """Return a mutable copy of the configured Scheduled Tasks catalog."""

    return deepcopy(catalog_module.load_config("scheduled_jobs.yml"))


def test_catalog_exposes_stable_operator_choices_and_resolves_ids() -> None:
    """Verify stable operator choices and identifier lookup behavior."""

    catalog = get_scheduled_job_catalog()

    assert [item.id for item in catalog.job_types] == [
        "eta_co_report",
        "shift_report",
        "daily_report",
        "furnace_summary",
        "custom_report",
    ]
    assert [item.label for item in catalog.job_types] == [
        "ETA CO Report",
        "Shift Handover Summary",
        "Daily Furnace Summary",
        "Furnace Performance Summary",
        "Custom Report",
    ]
    eta_job = catalog.job_type("eta_co_report")
    assert eta_job is not None
    assert eta_job.input_profile == "eta_co"
    assert catalog.job_type("missing") is None

    target = catalog.target_device("bf2-jetson-01")
    assert target is not None
    assert target.device_type == "jetson"
    assert catalog.target_device("missing") is None
    assert catalog.default_target_device.device_id == "bf2-jetson-01"

    assert catalog.data_source("online_process_data") is not None
    assert catalog.eta_co_signal("body_etaco") is not None
    one_minute = catalog.aggregation_interval("1min")
    assert one_minute is not None
    assert one_minute.minutes == 1
    assert catalog.aggregation_interval("missing") is None

    assert [(item.name, item.label) for item in catalog.timezones] == [
        ("Asia/Kolkata", "Asia/Kolkata (IST)"),
        ("UTC", "UTC"),
    ]
    assert catalog.default_timezone == "Asia/Kolkata"
    assert catalog.timezone("UTC") is not None
    assert catalog.timezone("Europe/Not-A-Zone") is None


def test_catalog_defaults_reference_configured_eta_choices() -> None:
    """Verify defaults reference valid ETA CO and reliability choices."""

    catalog = get_scheduled_job_catalog()
    eta_defaults = catalog.defaults["eta_co"]
    reliability_defaults = catalog.defaults["reliability"]

    assert catalog.eta_co_signal(eta_defaults["signal"]) is not None
    assert (
        catalog.aggregation_interval(eta_defaults["aggregation_interval"]) is not None
    )
    assert eta_defaults["critical_threshold"] < eta_defaults["warning_threshold"]
    assert reliability_defaults == {
        "maximum_attempts": 3,
        "retry_interval_seconds": 60,
        "timeout_seconds": 600,
        "notify_on_failure": True,
        "overlap_policy": "skip",
        "misfire_policy": "fire_once_latest",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["job_types"].append(deepcopy(raw["job_types"][0])),
            "duplicate identifiers in job_types",
        ),
        (
            lambda raw: raw["job_types"][0].update(
                default_analysis_level="unsupported"
            ),
            "unsupported default analysis level",
        ),
        (
            lambda raw: raw["job_types"][0].update(id="Invalid Job Type"),
            "invalid job type identifier",
        ),
        (
            lambda raw: raw["data_sources"][0].update(id=None),
            r"data_sources\.id is required",
        ),
        (
            lambda raw: raw["aggregation_intervals"][0].update(minutes=0),
            "aggregation minutes must be positive",
        ),
        (
            lambda raw: raw["timezones"].append(deepcopy(raw["timezones"][0])),
            "duplicate identifiers in timezones",
        ),
        (
            lambda raw: raw["timezones"][0].update(name="Mars/Olympus_Mons"),
            "invalid IANA timezone",
        ),
        (
            lambda raw: raw.update(timezones=[]),
            "must define every operator choice catalog",
        ),
        (
            lambda raw: raw["defaults"]["eta_co"].update(signal="missing"),
            "ETA default signal is not configured",
        ),
        (
            lambda raw: raw["defaults"].update(target_device_id="missing"),
            "defaults.target_device_id must reference a configured target device",
        ),
        (
            lambda raw: raw["defaults"]["reliability"].update(maximum_attempts=0),
            "default maximum attempts are invalid",
        ),
        (
            lambda raw: raw.update(target_devices=[]),
            "must define every operator choice catalog",
        ),
    ],
)
def test_catalog_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    message: str,
) -> None:
    """Verify malformed catalog configurations raise descriptive errors."""

    raw = _configured_catalog()
    mutate(raw)
    monkeypatch.setattr(catalog_module, "load_config", lambda _name: raw)

    with pytest.raises(ValueError, match=message):
        get_scheduled_job_catalog()


def test_catalog_requires_stable_target_device_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify target devices require nonblank stable identifiers."""

    raw = _configured_catalog()
    raw["target_devices"][0]["device_id"] = "  "
    monkeypatch.setattr(catalog_module, "load_config", lambda _name: raw)

    with pytest.raises(ValueError, match=r"target_devices\.device_id is required"):
        get_scheduled_job_catalog()


def test_catalog_default_timezone_must_be_one_of_the_configured_zones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the default timezone references a configured zone."""

    raw = _configured_catalog()
    raw["timezones"] = [
        item for item in raw["timezones"] if item["name"] != "Asia/Kolkata"
    ]
    monkeypatch.setattr(catalog_module, "load_config", lambda _name: raw)

    with pytest.raises(ValueError, match="must reference a configured timezone"):
        get_scheduled_job_catalog()
