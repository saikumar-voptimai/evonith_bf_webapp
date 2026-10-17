"""Configuration and architecture contracts for Furnace Status."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from data import furnace_status as fs
from data.furnace_status.catalogue import CatalogueError, load_catalogue
from furnace_data.influx.query import influx_fields
from ui.furnace_status import components
from ui.furnace_status.layout import validate_parameter_references

EXPECTED_KEYS = [
    "production_theoretical",
    "production_rate",
    "charges_per_hr",
    "hot_blast_volume",
    "blast_pressure",
    "top_pressure",
    "body_dp_total",
    "body_dp_bottom",
    "co_utilization",
    "top_gas_h2",
    "hbt",
    "raft",
    "uptake_t1",
    "uptake_t2",
    "uptake_t3",
    "uptake_t4",
    "uptake_avg",
    "uptake_std",
    "steam_injection",
    "o2_injection",
    "oxygen_flow",
    "flare_bypass_press",
    "flare_press",
    "fuel_rate",
    "coke_rate",
    "nut_coke_rate",
    "pci_rate",
    "slag_rate",
    "permeability",
    "tuyere_velocity",
    "furnace_level",
    "stock_rod1_pos",
    "stock_rod2_pos",
    "heat_load_total",
    "heat_load_q1",
    "heat_load_q2",
    "heat_load_q3",
    "heat_load_q4",
    "hearth_temp_a",
    "hearth_temp_b",
    "hearth_temp_c",
    "hearth_temp_d",
]


def test_real_yaml_is_the_ordered_single_source_of_truth() -> None:
    loaded = load_catalogue(fs.CONFIG_PATH)
    assert loaded == fs.CONFIG
    assert len(loaded.parameters) == 42
    assert [spec.key for spec in loaded.parameters] == EXPECTED_KEYS
    assert len({spec.key for spec in loaded.parameters}) == 42
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
    uptake = fs.PARAMETERS_BY_KEY["uptake_std"]
    assert uptake.aggregate == "std"
    assert uptake.components == (
        "top_temp_1",
        "top_temp_2",
        "top_temp_3",
        "top_temp_4",
    )
    uptake_avg = fs.PARAMETERS_BY_KEY["uptake_avg"]
    assert (uptake_avg.measurement, uptake_avg.field) == (
        "process_params",
        "top_temp_avg",
    )
    fuel = fs.PARAMETERS_BY_KEY["fuel_rate"]
    assert fuel.components == (
        "coal_rate_actual_value",
        "coke_rate",
        "nut_coke_rate",
    )
    assert fuel.setpoint_field == "fuel_rate"


def test_yaml_settings_drive_runtime_policy() -> None:
    assert fs.SETTINGS.timezone_name == "Asia/Kolkata"
    assert fs.STATUS_LOOKBACK == "last 15 minutes"
    assert fs.CACHE_TTL_SECONDS == 5
    assert dict(fs.SETTINGS.refresh_choices) == {
        "10 s": 10,
        "30 s": 30,
        "1 min": 60,
        "5 min": 300,
    }
    assert fs.SETTINGS.refresh_default == "10 s"
    assert fs.DEFAULT_INTERVAL == "8h"
    assert fs.MAX_CUSTOM_RANGE.days == 7
    assert fs.SETTINGS.raw_gap_break_seconds == 120
    assert fs.SETTINGS.operating_state.parameter == "hot_blast_volume"
    assert fs.SETTINGS.operating_state.normal_min == 25000


def test_yaml_owns_the_palette_and_zone_mapping() -> None:
    colours = dict(fs.UI.colours)
    assert colours["accent"] == "#1F6FB2"
    assert colours["body_bottom"] == "#F0BF98"
    assert colours["abnormal"] == "#DC2626"
    assert "--fs-accent:#1F6FB2" in components._palette_css()
    assert "--fs-abnormal:#DC2626" in components._palette_css()
    assert [zone.name for zone in fs.UI.zones] == [
        "Stack",
        "Belly",
        "Bosh",
        "Tuyere",
        "Hearth",
    ]
    assert fs.UI.zones[-1].lower_m == 0.0
    assert "body_dp_bottom" in next(
        zone.readings for zone in fs.UI.zones if zone.name == "Tuyere"
    )


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
        lambda config: config["ui"]["colours"].pop("accent"),
        lambda config: config["ui"]["colours"].update(accent="orange"),
        lambda config: config["settings"]["trend"].update(
            raw_gap_break_seconds=0
        ),
        lambda config: config["settings"]["operating_state"].update(
            parameter="does_not_exist"
        ),
        lambda config: config["settings"]["operating_state"].update(
            parameter="slag_rate"
        ),
        lambda config: config["settings"]["operating_state"].update(
            normal_min="twenty-five thousand"
        ),
        lambda config: config["ui"]["zones"][0]["readings"].append(
            "does_not_exist"
        ),
        lambda config: next(
            item
            for item in config["parameters"]

        ).update(setpoint_field="bad"),
        lambda config: next(
            item
            for item in config["parameters"]
            if "unavailable_reason" in item
        ).pop("unavailable_reason"),
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
