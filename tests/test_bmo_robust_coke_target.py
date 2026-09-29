from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.bmo.robust_coke_target import (
    NUT_COKE_FEATURE,
    PCI_FEATURE,
    PELLET_FE_PER_FE2O3,
    TARGET_COLUMN,
    RobustCokeSettings,
    fe_charged_mt,
    monotone_constraint_string,
    robust_coke_target,
    training_columns,
    trusted_coke_hours,
    window_coverage,
    window_features,
)

# Per charge: 6 t ore (62% Fe dry, 5% moisture), 16 t sinter (55% Fe),
# 2 t pellet (94% Fe2O3, 2% moisture), 4.4 t coke, 1.1 t nut coke.
FE_PER_CHARGE = 6 * 0.95 * 0.62 + 16 * 0.55 + 2 * 0.98 * 0.94 * PELLET_FE_PER_FE2O3
COKE_PER_CHARGE = 4.4


def _furnace(days: int = 12, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charges per hour vary at random; production runs on its own smooth clock.

    Like the plant, production agrees with the Fe charged over a day (here at
    0.935 t Fe per t HM) but not hour by hour.
    """

    index = pd.date_range("2026-09-01 00:30:00Z", periods=24 * days, freq="h")
    rng = np.random.default_rng(seed)
    charges = rng.integers(5, 9, size=len(index)).astype(float)
    daily_charges = (
        pd.Series(charges).rolling(24, center=True, min_periods=1).mean().to_numpy()
    )
    production = daily_charges * FE_PER_CHARGE / 0.935
    frame = pd.DataFrame(
        {
            "COKE_CALC_MT": COKE_PER_CHARGE * charges,
            "NUTCOKE_CALC_MT": 1.1 * charges,
            "PCI_CALC_MT": np.full(len(index), 16.0),
            "FLUX_CALC_MT": np.full(len(index), 0.5),
            "ORE_1_CALC_MT": 4.0 * charges,
            "ORE_2_CALC_MT": 2.0 * charges,
            "ORE_CALC_MT": 6.0 * charges,
            "ORE_FE(T)%": 62.0,
            "ORE_TM%": 5.0,
            "SINTER_CALC_MT": 16.0 * charges,
            "SINTER_FE(T)%": 55.0,
            "TOTAL_PELLET_CALC_MT": 2.0 * charges,
            "PELLET_PCT_FE2O3": 94.0,
            "PELLET_PCT_TM": 2.0,
            "PRODUCTIONTONNESPERHR": production,
            "HOT BLAST VOLUMENM3/HR.": 108_000.0,
            "COKE RATE KG/THM": 300.0,
        },
        index=index,
    )
    audit = pd.DataFrame({"normal_eligible": True}, index=index)
    return frame, audit


def test_fe_charged_converts_wet_ore_and_pellet_fe2o3():
    frame = pd.DataFrame(
        {
            "ORE_CALC_MT": [10.0],
            "ORE_FE(T)%": [60.0],
            "ORE_TM%": [10.0],
            "SINTER_CALC_MT": [20.0],
            "SINTER_FE(T)%": [50.0],
            "TOTAL_PELLET_CALC_MT": [5.0],
            "PELLET_PCT_FE2O3": [90.0],
            "PELLET_PCT_TM": [0.0],
        }
    )

    expected = 10 * 0.9 * 0.60 + 20 * 0.50 + 5 * 0.90 * PELLET_FE_PER_FE2O3
    assert fe_charged_mt(frame).iloc[0] == pytest.approx(expected)
    assert PELLET_FE_PER_FE2O3 == pytest.approx(0.6994, abs=1e-4)


def test_uncharged_material_with_missing_assay_contributes_no_fe():
    frame = pd.DataFrame(
        {
            "ORE_CALC_MT": [0.0],
            "ORE_FE(T)%": [np.nan],
            "SINTER_CALC_MT": [20.0],
            "SINTER_FE(T)%": [50.0],
            "TOTAL_PELLET_CALC_MT": [np.nan],
        }
    )

    assert fe_charged_mt(frame).iloc[0] == pytest.approx(10.0)


def test_charge_counting_cancels_in_the_robust_target():
    frame, audit = _furnace()
    hourly = 1000 * frame["COKE_CALC_MT"] / frame["PRODUCTIONTONNESPERHR"]

    target = robust_coke_target(frame, audit, RobustCokeSettings())[TARGET_COLUMN]

    # Coke and Fe share the charge count, so their ratio is steady even though
    # the hourly mass ratio swings by tens of kg/THM.
    assert hourly.std() > 20.0
    assert target.dropna().std() < 1.0
    assert target.dropna().median() == pytest.approx(
        1000 * COKE_PER_CHARGE / FE_PER_CHARGE * 0.935, rel=0.01
    )


def test_undercount_spell_and_weighing_spike_leave_the_target():
    frame, audit = _furnace(days=16)
    spell = (frame.index >= "2026-09-06 00:00Z") & (frame.index < "2026-09-09 00:00Z")
    frame.loc[spell, "COKE_CALC_MT"] *= 0.5
    spike = frame.index[24 * 12 + 5]
    frame.loc[spike, "COKE_CALC_MT"] *= 5.0
    settings = RobustCokeSettings()

    trusted = trusted_coke_hours(frame, audit, settings)
    target = robust_coke_target(frame, audit, settings)[TARGET_COLUMN]

    assert not trusted.loc[spell].any()
    assert not trusted.loc[spike]
    assert target.loc[spell].isna().all()
    clean = target.loc["2026-09-10 12:00Z":"2026-09-12 00:00Z"].dropna()
    assert not clean.empty
    assert clean.std() < 1.0


def test_missing_setpoint_column_skips_only_that_check():
    frame, audit = _furnace()
    frame = frame.drop(columns=["COKE RATE KG/THM"])

    target = robust_coke_target(frame, audit, RobustCokeSettings())[TARGET_COLUMN]

    assert target.notna().sum() > 100


def test_window_features_are_window_mass_ratios_without_coke():
    frame, audit = _furnace()
    settings = RobustCokeSettings(window_hours=24)
    slag = pd.DataFrame({"slag_kg_thm_repo": 330.0}, index=frame.index)

    features = window_features(frame, slag, audit, settings)

    last = features.index[-1]
    window = frame.loc[:last].tail(24)
    assert features.loc[last, PCI_FEATURE] == pytest.approx(
        1000 * window["PCI_CALC_MT"].sum() / window["PRODUCTIONTONNESPERHR"].sum()
    )
    assert features.loc[last, NUT_COKE_FEATURE] == pytest.approx(
        1000 * window["NUTCOKE_CALC_MT"].sum() / window["PRODUCTIONTONNESPERHR"].sum()
    )
    shares = features.loc[last, [c for c in features if c.endswith("_SHARE__win")]]
    assert shares.sum() == pytest.approx(1.0)
    assert features.loc[last, "BURDEN_PER_FE__win"] == pytest.approx(24.0 / FE_PER_CHARGE)
    assert features.loc[last, "slag_kg_thm_repo__win"] == pytest.approx(330.0)
    assert not any("COKE_CALC" in c or c.startswith("COKE RATE") for c in features)


def test_window_coverage_counts_running_hours():
    frame, audit = _furnace(days=2)
    audit.loc[audit.index[-6:], "normal_eligible"] = False

    coverage = window_coverage(frame, audit, RobustCokeSettings(window_hours=24))

    assert coverage.iloc[-1] == pytest.approx(18 / 24)


def test_training_columns_put_drivers_first_and_drop_constants():
    frame = pd.DataFrame(
        {
            "HOT BLAST TEMP.OC__win": [1100.0, 1110.0, 1120.0],
            "constant__win": [1.0, 1.0, 1.0],
            NUT_COKE_FEATURE: [70.0, 72.0, 74.0],
            PCI_FEATURE: [170.0, 180.0, 190.0],
        }
    )

    columns = training_columns(frame)

    assert columns == [PCI_FEATURE, NUT_COKE_FEATURE, "HOT BLAST TEMP.OC__win"]
    assert monotone_constraint_string(columns) == "(-1,-1,0)"


def test_settings_read_bundle_config():
    settings = RobustCokeSettings.from_config(
        {"robust_target": {"window_hours": 12, "assay_columns": ["COKE_ASH%"], "x": 1}}
    )

    assert settings.window_hours == 12
    assert settings.assay_columns == ("COKE_ASH%",)
    assert RobustCokeSettings.from_config({}).window_hours == 24
