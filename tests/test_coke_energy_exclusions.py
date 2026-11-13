"""Shifts with a data error (14 Nov 2026): the bad-shift list, an ore moisture that cannot be real and a charge record
whose dry fuel rate lies outside the plausible band (``coke_energy.yml -> fuel_rate_valid_kg_thm``) are left out of
the balance, and the burnt-vs-charged view lists them with the reason.

The band catches the two record errors found in the burnt-against-charged analysis: PCI recorded at about 40 t/h
after the stop of 15 Oct 2026 (790-920 kg/tHM) and the charged coke of 31 Oct - 5 Dec 2025 (340-460 kg/tHM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import burnt_vs_charged as bvc
from utils.coke_energy.pipeline import (
    PipelineResult,
    PipelineSettings,
    excluded_reasons,
    excluded_shifts,
    fuel_rate_kg_thm,
)

KEYS = ["2026-10-15_C", "2026-10-16_A", "2026-10-16_B", "2026-10-16_C", "2026-10-17_A"]
BAND = PipelineSettings(fuel_rate_valid_kg_thm=(470.0, 640.0))


def frame() -> pd.DataFrame:
    """Five shifts with what the balance needs: a normal one, an ore-moisture error, two PCI record errors (the
    16 Oct 2026 pattern) and one without production."""
    n = len(KEYS)
    return pd.DataFrame(
        {
            "COKE_DRY": [220.0, 220.0, 225.0, 230.0, 210.0],  # t in the shift
            "NUT_DRY": [50.0] * n,
            "PCI_DRY": [150.0, 150.0, 330.0, 320.0, 150.0],
            "PRODUCTIONTONNESPERHR": [740.0, 740.0, 669.0, 740.0, 0.0],  # t of hot metal in the shift
            "ORE_TM%": [5.0, 129.7, 5.0, 5.0, 5.0],
            "n_hours": [8] * n,
            "n_ok10": [48] * n,
            "n_taps": [4] * n,
            "chem_pct_fe": [94.5] * n,
            "slag_pct_al2o3": [17.5] * n,
            "dpr_dc_dust_kgthm": [9.0] * n,
            "COKE_CALC_MT": [230.0] * n,
        },
        index=KEYS,
    )


def test_the_fuel_rate_is_the_dry_fuel_of_the_record_per_tonne() -> None:
    fuel = fuel_rate_kg_thm(frame())
    assert fuel.round(1).tolist()[:4] == [567.6, 567.6, 904.3, 810.8]
    assert np.isnan(fuel.iloc[4])  # no production: no rate
    assert fuel_rate_kg_thm(frame().drop(columns="PCI_DRY")).isna().all()


def test_without_a_band_only_the_list_and_the_moisture_rule_hold() -> None:
    settings = PipelineSettings()
    assert settings.fuel_rate_valid_kg_thm is None
    assert excluded_shifts(frame(), settings) == {"2026-10-07_C", "2026-10-16_A"}


def test_the_band_leaves_out_the_fuel_record_errors_with_the_reason() -> None:
    reasons = excluded_reasons(frame(), BAND)
    assert set(reasons) == {"2026-10-07_C", "2026-10-16_A", "2026-10-16_B", "2026-10-16_C"}
    assert reasons["2026-10-16_A"] == "ore moisture 129.7 % above 15 %"  # the first reason holds
    assert reasons["2026-10-16_B"] == "fuel record 904 kg/tHM dry, outside 470-640 kg/tHM"
    assert reasons["2026-10-07_C"] == "listed as a data error (bad_shifts)"
    low = frame().assign(COKE_DRY=[100.0] * len(KEYS))  # the Nov 2025 pattern: coke under-recorded
    assert "2026-10-15_C" in excluded_shifts(low, BAND)


def test_the_band_comes_from_the_config() -> None:
    from services import coke_energy as ce

    assert PipelineSettings.from_config(ce.config()).fuel_rate_valid_kg_thm == (470.0, 640.0)
    assert PipelineSettings.from_config({}).fuel_rate_valid_kg_thm is None
    assert PipelineSettings.from_config({"fuel_rate_valid_kg_thm": None}).fuel_rate_valid_kg_thm is None
    with pytest.raises(ValueError, match="fuel_rate_valid_kg_thm"):
        PipelineSettings(fuel_rate_valid_kg_thm=(640.0, 470.0))


def test_the_view_lists_the_excluded_shifts_with_data_for_a_balance() -> None:
    empty = pd.DataFrame()
    result = PipelineResult(frame(), empty, empty, empty, empty, empty, empty, BAND)
    rows = bvc.excluded(result)
    assert [r["shift"] for r in rows] == ["2026-10-16_A", "2026-10-16_B", "2026-10-16_C"]  # 17 A has no production
    assert rows[1] == {
        "shift": "2026-10-16_B",
        "day": "2026-10-16",
        "reason": "fuel record 904 kg/tHM dry, outside 470-640 kg/tHM",
        "fuel_kg_thm": pytest.approx(904.33, abs=0.01),
    }
    assert bvc.excluded(PipelineResult(empty, empty, empty, empty, empty, empty, empty, BAND)) == []
    partial = frame()[["ORE_TM%"]]  # a frame without the balance's columns (the service tests' fixtures)
    assert bvc.excluded(PipelineResult(partial, empty, empty, empty, empty, empty, empty, BAND)) == []
