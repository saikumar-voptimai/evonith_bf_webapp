"""The slag-side HM chemistry is held, not taken from the latest cast.

Four values - HM C / Si / S / Others - set the pig-iron closure and the SiO2
consumed by Si reduction inside the slag balance. They are deliberately pinned
at the operating point instead of tracking each HM_SLAG row, because a single
cast's Si swings far more than the real operating point does, and feeding that
swing into the balance moves calculated slag and basicity for reasons that have
nothing to do with the blend under comparison.

Scope matters here: this hold applies to the slag side ONLY. Mn/Ti partitioning
still comes from the live snapshot, and nothing here touches the Si prediction
model or the coke-correction Si term.
"""

from __future__ import annotations

import pytest
import yaml

from data.bmo.ore_editor_preferences import (
    apply_hm_chemistry_preferences,
    build_hm_chemistry_preferences,
)
from ui.bmo.editor_inputs import slag_balance_settings_from_editor
from utils.bmo.calculations import evaluate_blend
from utils.bmo.types import OreChemistry, OreInput, SlagBalanceSettings

CONFIG_PATH = "src/config/setting_bmo.yml"

# What the operator asked to hold.
HELD = {
    "carbon_pct": 4.2,
    "silicon_pct": 0.6,
    "sulphur_pct": 0.03,
    "other_pct": 0.0,
}


def _slag_balance_cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["bmo"]["slag_balance"]


def _ore() -> OreInput:
    return OreInput(
        ore_id="burden",
        display_name="BURDEN",
        stock_mt=99_999.0,
        price_rs_per_mt=7000.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(
            fe_t_pct=56.0,
            sio2_pct=5.6,
            al2o3_pct=2.71,
            cao_pct=7.27,
            mgo_pct=2.6,
            moisture_pct=4.0,
        ),
    )


def _settings(**overrides) -> SlagBalanceSettings:
    values = dict(HELD)
    values.update(overrides)
    return SlagBalanceSettings(enabled=True, **values)


def _slag(settings: SlagBalanceSettings):
    ore = _ore()
    return evaluate_blend(
        ores=[ore],
        quantities_mt={ore.ore_id: 3900.0},
        feo_in_slag_pct=0.4,
        fuel_cost_per_thm_rs=0.0,
        slag_balance_settings=settings,
        hot_metal_target_mt=2350.0,
    )


def test_shipped_config_holds_the_operator_specified_values() -> None:
    """Pinned so nobody quietly reverts them to the old live-tracking defaults."""

    cfg = _slag_balance_cfg()

    assert cfg["carbon_pct"] == pytest.approx(4.2)
    assert cfg["silicon_pct"] == pytest.approx(0.6)
    assert cfg["sulphur_pct"] == pytest.approx(0.03)
    assert cfg["other_pct"] == pytest.approx(0.0)


def test_held_values_are_not_overwritten_by_the_live_cast() -> None:
    """The live snapshot must not reach the four PI-chemistry fields.

    ``slag_balance_settings_from_editor`` takes PI chemistry from the edited
    values and Mn/Ti from the snapshot. A live cast well away from the held
    point must therefore change Mn/Ti and nothing else.
    """

    live_cast = {
        "chem_pct_c": 4.36,
        "chem_pct_si": 0.81,
        "chem_pct_s": 0.020,
        "others_pct": 0.29,
        "chem_pct_mn": 0.25,
        "chem_pct_ti": 0.06,
    }

    settings = slag_balance_settings_from_editor({"enabled": True}, HELD, live_cast)

    assert settings.carbon_pct == pytest.approx(4.2)
    assert settings.silicon_pct == pytest.approx(0.6)
    assert settings.sulphur_pct == pytest.approx(0.03)
    assert settings.other_pct == pytest.approx(0.0)
    # Mn/Ti partitioning is explicitly still live.
    assert settings.mn_pct == pytest.approx(0.25)
    assert settings.ti_pct == pytest.approx(0.06)


def test_holding_si_is_what_makes_two_blends_comparable() -> None:
    """Si is the value with real leverage, which is why it must not drift.

    SiO2 consumed by Si reduction is ``PI x Si% x 2.14``, so a cast-to-cast Si
    swing silently moves calculated slag. Same burden, two different Si values,
    must give different slag - which is exactly the artefact the hold removes.
    """

    at_held = _slag(_settings())
    at_live = _slag(_settings(silicon_pct=0.81))

    # Higher Si consumes more SiO2, so less reports to slag.
    assert at_live.slag_mt < at_held.slag_mt
    # And the shift is large enough to matter against a ~4% closure tolerance.
    shift_pct = abs(at_live.slag_mt - at_held.slag_mt) / at_held.slag_mt * 100.0
    assert shift_pct > 1.0

    # Basicity moves too, because SiO2 is its denominator.
    assert at_live.slag_basicity > at_held.slag_basicity


def test_holding_the_values_makes_the_balance_reproducible() -> None:
    """Same burden evaluated twice gives the same slag, whatever the cast did."""

    first = _slag(_settings())
    second = _slag(_settings())

    assert first.slag_mt == pytest.approx(second.slag_mt)
    assert first.slag_basicity == pytest.approx(second.slag_basicity)
    assert first.slag_al2o3_pct == pytest.approx(second.slag_al2o3_pct)


def test_hm_chemistry_preferences_round_trip() -> None:
    """Save writes its own block; apply overlays only the four PI keys."""

    payload = build_hm_chemistry_preferences(
        {**HELD, "pi_loss_pct": 0.2, "mn_recovery_pct": 60.0}
    )

    assert payload == {"hot_metal_chemistry": HELD}
    # Unrelated slag_balance settings must not be swept into the block.
    assert "pi_loss_pct" not in payload["hot_metal_chemistry"]
    assert "mn_recovery_pct" not in payload["hot_metal_chemistry"]

    defaults = {**HELD, "pi_loss_pct": 0.2, "alkali_to_slag_fraction": 0.8}
    applied = apply_hm_chemistry_preferences(
        defaults, {"hot_metal_chemistry": {"silicon_pct": 0.55}}
    )

    assert applied["silicon_pct"] == pytest.approx(0.55)
    assert applied["carbon_pct"] == pytest.approx(4.2)
    # Everything else in slag_balance stays config-driven.
    assert applied["pi_loss_pct"] == pytest.approx(0.2)
    assert applied["alkali_to_slag_fraction"] == pytest.approx(0.8)


def test_apply_is_a_no_op_without_saved_preferences() -> None:
    defaults = dict(HELD)

    assert apply_hm_chemistry_preferences(defaults, {}) == defaults
    assert apply_hm_chemistry_preferences(defaults, {"model_inputs": {}}) == defaults
