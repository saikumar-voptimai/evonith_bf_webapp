"""Charge-record check of the Coke Energy pipeline (20 Nov 2026): a shift whose charge record disagrees with the
plant's fuel-rate tag beyond ``PipelineSettings.fuel_rate_band_pct`` is a data error. Since 25 Nov 2026 the record is
on the tag's basis: dry coke + dry nut coke (wet tonnes less the total moisture) + PCI as injected.

Acceptance: the deviation is record / tag - 1; shifts outside the band are left out of the balances with the error in
words (``PipelineResult.excluded``), every other shift keeps its values; a twin window over such a shift fails with
the same words; without the tag, or without a band, nothing is left out; the admin analysis lists the shifts.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from services import burnt_vs_charged as bvc
from tests.twin_data import synthetic_frames
from utils.coke_energy.pipeline import (
    FUEL_RATE_TAG,
    PipelineSettings,
    charge_record_deviation,
    charge_record_fuel,
    exclusion_reasons,
    run_pipeline,
)
from utils.coke_energy.twin import Window, evaluate_window

BAND = (-9.0, 9.0)
BASE = dict(history_start="2026-04-01", predict_from="2026-04-01")
CORRUPT = "2026-05-12_B"  # PCI read 2.2 x, as on 16-21 Oct 2026


def inputs(**rows: dict) -> pd.DataFrame:
    # wet coke 230 t at 2.2 % and wet nut coke 56 t at 6 % total moisture
    # the coal rate (PCI) is dry as recorded
    cols = {"COKE_CALC_MT": 230.0, "NUTCOKE_CALC_MT": 56.0, "COKE_DRY": 224.94, "NUT_DRY": 52.64, "PCI_DRY": 135.0,
            "PRODUCTIONTONNESPERHR": 745.0, FUEL_RATE_TAG: 548.0, "ORE_TM%": 8.0}  # fmt: skip
    return pd.DataFrame({k: {**cols, **v} for k, v in rows.items()}).T


def test_the_deviation_is_record_over_tag() -> None:
    S = inputs(a={}, b={"PCI_DRY": 300.0}, c={FUEL_RATE_TAG: np.nan}, d={"PRODUCTIONTONNESPERHR": 0.0})
    fuel = charge_record_fuel(S)
    assert fuel["a"] == pytest.approx((224.94 + 52.64 + 135) * 1000 / 745)  # dry coke and nut coke, PCI as injected
    dev = charge_record_deviation(S)
    assert dev["a"] == pytest.approx(100 * (fuel["a"] / 548 - 1))
    assert abs(dev["a"]) < 1.5  # as charged (wet) the same record reads +3.1 %
    assert dev["b"] > 9 and np.isnan(dev["c"]) and np.isnan(dev["d"])
    assert charge_record_deviation(S.drop(columns=[FUEL_RATE_TAG])).isna().all()
    assert charge_record_fuel(S.drop(columns=["COKE_DRY"])).isna().all()  # never the wet tonnes against a dry tag


def test_exclusion_reasons_name_the_error() -> None:
    S = inputs(
        ok={},
        pci_high={"PCI_DRY": 300.0},  # +44 %
        coke_half={"COKE_CALC_MT": 115.0, "COKE_DRY": 112.47},  # -20 %
        no_tag={"PCI_DRY": 300.0, FUEL_RATE_TAG: np.nan},
        wet={"ORE_TM%": 129.7, "PCI_DRY": 300.0},
    )
    out = exclusion_reasons(S, PipelineSettings(bad_shifts=("listed",), fuel_rate_band_pct=BAND))
    assert set(out) == {"listed", "pci_high", "coke_half", "wet"}
    assert out["listed"] == "data error: on the list of known bad shifts"
    assert out["wet"].startswith("data error: ore moisture 129.7 %")  # the first error found is given
    assert out["pci_high"].startswith("data error: charge record fuel 775 kg/tHM is +41 % against the plant fuel")
    assert out["pci_high"].endswith("rate 548 kg/tHM (accepted -9 to +9 %)")
    assert "is -26 %" in out["coke_half"]
    assert set(exclusion_reasons(S, PipelineSettings(bad_shifts=("listed",)))) == {"listed", "wet"}  # no band: off


def test_settings_read_the_band_from_the_config() -> None:
    cfg = {"charge_record_check": {"fuel_rate_band_pct": [-9, 9]}}
    assert PipelineSettings.from_config(cfg).fuel_rate_band_pct == BAND
    assert PipelineSettings.from_config({"charge_record_check": {"fuel_rate_band_pct": None}}).fuel_rate_band_pct is None
    assert PipelineSettings.from_config({}).fuel_rate_band_pct is None  # the web model
    with pytest.raises(ValueError):
        PipelineSettings(fuel_rate_band_pct=(9.0, -9.0))


def test_the_app_config_turns_the_check_on() -> None:
    from services import coke_energy as ce

    band = PipelineSettings.from_config(ce.config()).fuel_rate_band_pct
    assert band is not None and band[0] < 0 < 3.0 < band[1]  # the record's usual +3 % lies inside


# ---------------------------------------------------------------------------------------------- pipeline and twin
@pytest.fixture(scope="module")
def world():
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=18)
    w = Window.of_shift(CORRUPT)
    bad = fd.copy()
    bad.loc[(bad.time >= w.start) & (bad.time < w.end), "PCI_CALC_MT"] *= 2.2  # the hourly record
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checked = run_pipeline(bad, m10, hm, dpr, PipelineSettings(**BASE, fuel_rate_band_pct=BAND))
        clean = run_pipeline(fd, m10, hm, dpr, PipelineSettings(**BASE, fuel_rate_band_pct=BAND))
        plain = run_pipeline(fd, m10, hm, dpr, PipelineSettings(**BASE))
    return (bad, m10, hm, dpr), checked, clean, plain


def test_the_corrupt_shift_is_left_out_with_its_error(world) -> None:
    _, checked, _, _ = world
    assert list(checked.excluded) == [CORRUPT]
    assert checked.excluded[CORRUPT].startswith("data error: charge record fuel")
    assert CORRUPT not in checked.R.index and CORRUPT not in checked.shifts.index
    assert not checked.errors


def test_the_check_changes_nothing_on_a_sound_record(world) -> None:
    _, _, clean, plain = world
    assert clean.excluded == {} and plain.excluded == {}
    pd.testing.assert_frame_equal(clean.shifts, plain.shifts)


def test_the_other_shifts_keep_their_balances(world) -> None:
    _, checked, clean, _ = world
    keep = checked.R.index
    assert len(keep) == len(clean.R.index) - 1
    pd.testing.assert_frame_equal(checked.R.loc[keep], clean.R.loc[keep])


def test_a_twin_window_on_the_corrupt_shift_fails_with_the_error(world) -> None:
    (fd, m10, hm, dpr), checked, _, _ = world
    settings = PipelineSettings(**BASE, fuel_rate_band_pct=BAND)
    ev = evaluate_window(fd, m10, hm, dpr, Window.of_shift(CORRUPT), checked, settings, strata=False)
    assert ev.status == "failed"
    assert ev.notes == [checked.excluded[CORRUPT]]
    ok = evaluate_window(fd, m10, hm, dpr, Window.of_shift("2026-05-13_B"), checked, settings, strata=False)
    assert ok.status in ("complete", "partial") and ok.k_trace is not None


def test_the_admin_analysis_lists_the_shifts_left_out(world) -> None:
    _, checked, clean, _ = world
    built = bvc.build(checked, {}, now=pd.Timestamp("2026-05-19 07:20", tz="Asia/Kolkata"))
    assert built["excluded"] == [
        {
            "skey": CORRUPT,
            "start": "2026-05-12T08:30:00+00:00",  # 14:00 IST
            "end": "2026-05-12T16:30:00+00:00",
            "reason": checked.excluded[CORRUPT],
        }
    ]
    assert all(CORRUPT not in str(p) for p in built["bases"]["all"]["points"]["Shift (8 h)"])
    assert bvc.build(clean, {}, now=pd.Timestamp("2026-05-19 07:20", tz="Asia/Kolkata"))["excluded"] == []
