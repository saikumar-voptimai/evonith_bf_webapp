"""The operator-input registry for everything the plant has not measured.

The point of this registry is that a guess can be replaced by a measurement
without touching code. That only works if overrides actually reach the balance,
survive a restart, and cannot be used to enter something physically impossible.
"""

from __future__ import annotations

import json

import pytest

from utils.energy_balance import EnergyBalanceInputs, run_energy_balance
from utils.energy_balance.assumptions import (
    ASSUMPTIONS,
    BY_KEY,
    apply_overrides,
    clamp,
    current_values,
    load_overrides,
    save_overrides,
)
from utils.energy_balance.constants import load_config


def _day() -> EnergyBalanceInputs:
    """The worked day, 2025-12-31, now carrying its dust."""

    return EnergyBalanceInputs(
        hot_metal_mt=2275.34,
        slag_mt=756.29,
        coke_mt=767.22,
        nut_coke_mt=171.48,
        pci_mt=342.63,
        blast_volume_nm3_per_hr=111392.03,
        blast_temperature_c=1200.0,
        oxygen_enrichment_pct=3.81,
        top_gas_co_pct=24.95,
        top_gas_co2_pct=18.75,
        top_gas_h2_pct=2.87,
        top_gas_temperature_c=137.26,
        hm_carbon_pct=4.28,
        hm_iron_pct=94.96,
        hm_silicon_pct=0.41,
        slag_feo_pct=0.36,
        flue_dust_mt=41.4,
        gcp_dust_mt=24.8,
        flux_mt=9.5,
        ore_mt=1400.0,
        sinter_mt=2100.0,
        pellet_mt=380.0,
        flux_loi_pct=40.0,
        fuel_vm_pct={"coke": 0.9, "nut_coke": 1.0, "pci": 19.9},
        moisture_pct={"ore": 4.0, "pellet": 2.0, "coke": 0.4, "nut_coke": 8.5},
        shell_loss_gj_per_hr=28.29,  # stave rows 6-10 only - see the closure test
    )


# --- the registry itself --------------------------------------------------------


def test_no_physical_constant_is_exposed_as_an_operator_input():
    """Overriding physics would break the balance, not calibrate it.

    Iron oxide reduction enthalpy, calorific values and molar volume are
    constants of nature. If one ever appears in this table, someone has
    confused a plant-specific assumption with a law.
    """

    forbidden = (
        "fe_reduction", "fe_to_feo", "co_lhv", "h2_lhv", "carbon_full",
        "hydrogen_lhv", "n2_in_air", "silicon_mj", "manganese_mj",
    )
    for spec in ASSUMPTIONS:
        assert not any(token in spec.key for token in forbidden), spec.key


def test_every_assumption_states_its_basis_and_confidence():
    """A literature value must never be mistaken for a measurement."""

    for spec in ASSUMPTIONS:
        assert spec.basis.strip()
        assert spec.impact.strip()
        assert spec.confidence in {"measured", "literature", "assumed"}
        assert spec.minimum < spec.default < spec.maximum, spec.key


def test_every_key_resolves_against_the_real_config():
    """A typo in a dotted path would silently create a dead config branch.

    apply_overrides happily creates missing nodes, so a mistyped key would
    write somewhere nothing reads and the operator's value would vanish with
    no error at all.
    """

    cfg = load_config()
    for spec in ASSUMPTIONS:
        node = cfg
        for part in spec.key.split("."):
            assert isinstance(node, dict) and part in node, spec.key
            node = node[part]
        assert isinstance(node, (int, float)), spec.key


# --- overrides reaching the balance ---------------------------------------------


def test_an_override_changes_the_config_it_targets():
    cfg = load_config()
    out = apply_overrides(cfg, {"fuels.dust_carbon_pct.flue": 42.0})

    assert out["fuels"]["dust_carbon_pct"]["flue"] == pytest.approx(42.0)


def test_apply_overrides_does_not_mutate_the_cached_config():
    """load_config returns a cached singleton; mutating it would leak."""

    cfg = load_config()
    before = cfg["fuels"]["dust_carbon_pct"]["flue"]

    apply_overrides(cfg, {"fuels.dust_carbon_pct.flue": 55.0})

    assert cfg["fuels"]["dust_carbon_pct"]["flue"] == before


def test_dust_carbon_actually_moves_the_balance():
    """If the input does nothing, it is decoration.

    Dust carbon is charged but never burnt, so raising it must REDUCE the
    credited input - the same mechanism as carbon dissolved in hot metal.
    """

    day = _day()
    base = run_energy_balance(day, apply_overrides(load_config(), {}))
    more = run_energy_balance(
        day, apply_overrides(load_config(), {"fuels.dust_carbon_pct.flue": 45.0})
    )

    assert more.diagnostics["carbon_to_dust_kg_per_thm"] > (
        base.diagnostics["carbon_to_dust_kg_per_thm"]
    )
    assert more.total_input_mj_per_thm < base.total_input_mj_per_thm


def test_zero_dust_reproduces_the_pre_dust_balance():
    """Setting both dust carbons to zero must restore the old behaviour exactly.

    Not a closure assertion - see test_balance_closes_on_the_reference_day for
    why the reference day's closure is contested. This only checks that the new
    dust term is inert when zeroed, so adding it cannot have changed any
    existing result.
    """

    day = _day()
    zero = apply_overrides(
        load_config(),
        {"fuels.dust_carbon_pct.flue": 0.0, "fuels.dust_carbon_pct.gcp": 0.0},
    )

    result = run_energy_balance(day, zero)
    no_dust_day = EnergyBalanceInputs(
        **{**day.__dict__, "flue_dust_mt": 0.0, "gcp_dust_mt": 0.0}
    )
    unchanged = run_energy_balance(no_dust_day, load_config())

    assert result.diagnostics["carbon_to_dust_kg_per_thm"] == pytest.approx(0.0)
    assert result.closure == pytest.approx(unchanged.closure, rel=1e-9)


# --- bounds and persistence -----------------------------------------------------


def test_values_are_clamped_to_something_physical():
    """A slipped decimal point must not silently reshape the balance."""

    assert clamp("fuels.dust_carbon_pct.flue", 900.0) == BY_KEY[
        "fuels.dust_carbon_pct.flue"
    ].maximum
    assert clamp("fuels.carbon_fraction.pci", -3.0) == BY_KEY[
        "fuels.carbon_fraction.pci"
    ].minimum


def test_out_of_range_overrides_are_clamped_on_the_way_into_config():
    out = apply_overrides(load_config(), {"fuels.carbon_fraction.pci": 5.0})

    assert out["fuels"]["carbon_fraction"]["pci"] == pytest.approx(0.90)


def test_a_round_trip_survives_a_restart(tmp_path):
    path = tmp_path / "assumptions.json"
    save_overrides({"fuels.dust_carbon_pct.gcp": 26.5}, path)

    assert json.loads(path.read_text())["fuels.dust_carbon_pct.gcp"] == 26.5
    assert load_overrides(path) == {"fuels.dust_carbon_pct.gcp": 26.5}


def test_saving_a_default_does_not_freeze_it(tmp_path):
    """Writing the shipped value would pin it against future revision.

    An operator who opens the table, changes nothing and hits save must not
    thereby override every default with a stale copy of today's.
    """

    path = tmp_path / "assumptions.json"
    defaults = {spec.key: spec.default for spec in ASSUMPTIONS}

    save_overrides(defaults, path)

    assert json.loads(path.read_text()) == {}


def test_a_retired_key_is_ignored_rather_than_crashing(tmp_path):
    path = tmp_path / "assumptions.json"
    path.write_text(json.dumps({"fuels.something_removed": 1.0}))

    assert load_overrides(path) == {}


def test_a_corrupt_file_falls_back_to_defaults(tmp_path):
    """A half-written file must not take the optimiser down."""

    path = tmp_path / "assumptions.json"
    path.write_text("{not json")

    assert load_overrides(path) == {}


# --- the table the operator sees ------------------------------------------------


def test_table_rows_label_operator_values_distinctly():
    rows = current_values({"fuels.dust_carbon_pct.flue": 33.0})
    by_key = {r["key"]: r for r in rows}

    assert by_key["fuels.dust_carbon_pct.flue"]["Source"] == "operator"
    assert by_key["fuels.dust_carbon_pct.flue"]["Value"] == pytest.approx(33.0)
    assert by_key["fuels.dust_carbon_pct.gcp"]["Source"] == "assumed"


def test_the_weakest_numbers_are_listed_first():
    """Whoever fills this in should spend their effort where it matters."""

    keys = [spec.key for spec in ASSUMPTIONS[:2]]

    assert keys == ["fuels.dust_carbon_pct.flue", "fuels.dust_carbon_pct.gcp"]
