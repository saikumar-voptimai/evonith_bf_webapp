"""Configuration and architecture contracts for Furnace Status."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from data import furnace_status as fs
from data.furnace_status.catalogue import CatalogueError, load_catalogue
from furnace_data.influx.query import influx_fields
from ui.furnace_status.layout import validate_parameter_references

EXPECTED_KEYS = [
    "production_theoretical",
    "production_rate",
    "hot_blast_volume",
    "blast_pressure",
    "top_pressure",
    "co_utilization",
    "top_gas_h2",
    "hbt",
    "raft",
    "uptake_t1",
    "uptake_t2",
    "uptake_t3",
    "uptake_t4",
    "uptake_avg",
    "steam_injection",
    "steam_bypass_flow",
    "o2_injection",
    "oxygen_flow",
    "fuel_rate",
    "pci_rate",
    "slag_rate",
    "permeability",
    "tuyere_velocity",
    "furnace_level",
    "heat_load_total",
    "heat_load_q1",
    "heat_load_q2",
    "heat_load_q3",
    "heat_load_q4",
    "hearth_temp_a",
    "hearth_temp_b",
    "hearth_temp_c",
    "hearth_temp_d",
    "hearth_temp_avg",
]


def test_real_yaml_is_the_ordered_single_source_of_truth() -> None:
    loaded = load_catalogue(fs.CONFIG_PATH)
    assert loaded == fs.CONFIG
    assert len(loaded.parameters) == 34
    assert [spec.key for spec in loaded.parameters] == EXPECTED_KEYS
    assert len({spec.key for spec in loaded.parameters}) == 34
    assert all(spec.section in loaded.sections for spec in loaded.parameters)
    assert all(isinstance(spec.components, tuple) for spec in loaded.parameters)


def test_real_yaml_sources_and_derived_definitions_are_valid() -> None:
    for spec in fs.PARAMETERS:
        if spec.has_source:
            allowed = set(influx_fields(spec.measurement))
            assert set(spec.source_fields) <= allowed, spec.key
        else:
            assert spec.unavailable_reason, spec.key

    for quadrant in range(1, 5):
        spec = fs.PARAMETERS_BY_KEY[f"heat_load_q{quadrant}"]
        assert spec.aggregate == "sum"
        assert spec.components == tuple(
            f"heat_load_r{row}_q{quadrant}" for row in range(6, 11)
        )
    total = fs.PARAMETERS_BY_KEY["heat_load_total"]
    assert total.components == tuple(
        f"heat_load_r{row}_q{quadrant}"
        for quadrant in range(1, 5)
        for row in range(6, 11)
    )
    hearth = fs.PARAMETERS_BY_KEY["hearth_temp_avg"]
    assert hearth.aggregate == "mean"
    assert hearth.components == (
        "temp_4373_a",
        "temp_5411_b",
        "temp_5757_c",
        "temp_6103_d",
    )


def test_yaml_settings_drive_runtime_policy() -> None:
    assert fs.SETTINGS.timezone_name == "Asia/Kolkata"
    assert fs.STATUS_LOOKBACK == "last 15 minutes"
    assert fs.CACHE_TTL_SECONDS == 60
    assert fs.DEFAULT_INTERVAL == "8h"
    assert fs.MAX_CUSTOM_RANGE.days == 90
    assert fs.choose_window(fs.FIXED_INTERVALS["24h"]) == "15 minutes"


def _raw_config() -> dict:
    return yaml.safe_load(fs.CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(tmp_path, config: dict):
    path = tmp_path / "furnace_status.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config["parameters"].append(deepcopy(config["parameters"][0])),
        lambda config: config["parameters"][0].update(section="Unknown"),
        lambda config: config["parameters"][0].update(aggregate="median"),
        lambda config: config["parameters"][0].pop("field"),
        lambda config: config["parameters"][0].update(
            components=["production_per_hour"]
        ),
        lambda config: config["parameters"][0].update(field="does_not_exist"),
        lambda config: config["parameters"][15].update(setpoint_field="bad"),
        lambda config: config["parameters"][15].pop("unavailable_reason"),
    ],
)
def test_loader_rejects_invalid_catalogues(tmp_path, mutation) -> None:
    config = _raw_config()
    mutation(config)
    with pytest.raises(CatalogueError):
        load_catalogue(_write_config(tmp_path, config))


def test_ui_parameter_references_are_checked_against_catalogue() -> None:
    validate_parameter_references()
    with pytest.raises(ValueError, match="does_not_exist"):
        validate_parameter_references(["does_not_exist"])
