"""Dry basis of the coke and nut coke charged (25 Nov 2026).

The charge record weighs coke and nut coke wet and the plant's rate tags are dry. The balance takes the laboratory's
TOTAL moisture (``fuel_chemistry.tm``) of each coke type, weighted by the tonnes of that type charged in the hour; the
hourly dataset's ``COKE_MOIST%`` has held the inherent moisture (about 0.45 %) since 7 Dec 2025.

Acceptance: an hour's moisture is the tonnes-weighted latest sample taken before the hour ends; a coke type without a
sample takes the others' value; a reading that is not a moisture is ignored; the dataset gains COKE_TM% / NUTCOKE_TM%
and keeps its own columns; the shift builder's dry tonnes use the total moisture where known and the dataset's column
elsewhere; without a moisture frame nothing changes; the page and the twin read the same frame.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from services import coke_energy as ce
from tests.twin_data import synthetic_frames
from utils.coke_energy.build_shifts import build_shift_inputs
from utils.coke_energy.fuel_moisture import hourly_total_moisture, on_total_moisture, with_total_moisture

H = pd.Timestamp("2026-05-10 06:00")


def samples(*rows: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"material": m, "t": pd.Timestamp(t), "tm": tm} for m, t, tm in rows],
                        columns=["material", "t", "tm"])  # fmt: skip


def charged(**columns: list[float]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.date_range(H, periods=n, freq="h"))


def test_an_hour_takes_the_tonnes_weighted_latest_sample_before_its_end() -> None:
    tm = hourly_total_moisture(
        charged(coke_1=[30.0, 30.0, 30.0], coke_2=[10.0, 0.0, 10.0], nut_coke_1=[8.0, 8.0, 0.0]),
        samples(
            ("coke_1", "2026-05-10 05:00", 2.0),
            ("coke_1", "2026-05-10 07:30", 3.0),  # inside the second hour
            ("coke_2", "2026-05-09 23:00", 6.0),
            ("nut_coke_1", "2026-05-10 07:00", 7.0),
        ),
    )
    assert tm["COKE_TM%"].tolist() == pytest.approx([(30 * 2 + 10 * 6) / 40, 3.0, (30 * 3 + 10 * 6) / 40])
    # the nut-coke sample is taken exactly when the first hour ends: it counts from the second; the third charges none
    assert np.isnan(tm["NUTCOKE_TM%"].iloc[0]) and tm["NUTCOKE_TM%"].iloc[1] == pytest.approx(7.0)
    assert np.isnan(tm["NUTCOKE_TM%"].iloc[2])


def test_a_coke_type_without_a_sample_takes_the_others_value() -> None:
    tm = hourly_total_moisture(
        charged(nut_coke_1=[6.0], nut_coke_2=[2.0], coke_2=[5.0]), samples(("nut_coke_1", "2026-05-10 01:00", 6.5))
    )
    assert tm["NUTCOKE_TM%"].iloc[0] == pytest.approx(6.5)  # nut_coke_2 is never sampled
    assert np.isnan(tm["COKE_TM%"].iloc[0])  # no coke sample at all


def test_a_reading_that_is_not_a_moisture_is_ignored() -> None:
    tm = hourly_total_moisture(
        charged(coke_1=[30.0]), samples(("coke_1", "2026-05-10 02:00", 2.4), ("coke_1", "2026-05-10 05:00", 87.3))
    )
    assert tm["COKE_TM%"].iloc[0] == pytest.approx(2.4)
    assert hourly_total_moisture(charged(coke_1=[30.0]), samples()).isna().all().all()


def test_the_dataset_gains_the_total_moisture_and_keeps_its_own_columns() -> None:
    fd = pd.DataFrame({"time": pd.date_range(H, periods=4, freq="h"), "COKE_MOIST%": 0.45, "NUTCOKE_MOIST%": 0.46})
    tm = pd.DataFrame(
        {"COKE_TM%": [2.0, np.nan], "NUTCOKE_TM%": [np.nan, 6.0]}, index=pd.date_range(H, periods=2, freq="h")
    )
    out = with_total_moisture(fd, tm)
    assert out["COKE_TM%"].tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])  # hours without a charge: the last value
    assert np.isnan(out["NUTCOKE_TM%"].iloc[0]) and out["NUTCOKE_TM%"].iloc[1:].tolist() == pytest.approx([6.0] * 3)
    assert out["COKE_MOIST%"].tolist() == [0.45] * 4 and "COKE_TM%" not in fd.columns  # a copy; the feature is kept
    assert with_total_moisture(fd, None) is fd and with_total_moisture(fd, pd.DataFrame()) is fd
    late = with_total_moisture(fd.assign(time=fd.time + pd.Timedelta(days=3)), tm)
    assert late["COKE_TM%"].isna().all()  # carried at most a day


def test_the_balance_reads_the_total_moisture_where_it_is_known() -> None:
    f = pd.DataFrame({"COKE_MOIST%": [0.45, 0.45], "COKE_TM%": [2.5, np.nan], "NUTCOKE_MOIST%": [0.46, 0.46]})
    out = on_total_moisture(f)
    assert out["COKE_MOIST%"].tolist() == pytest.approx([2.5, 0.45])
    assert out["NUTCOKE_MOIST%"].tolist() == pytest.approx([0.46, 0.46])  # no NUTCOKE_TM% column: unchanged


def test_the_shift_builders_dry_tonnes_use_the_total_moisture() -> None:
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=4)
    plain = fd.drop(columns=["COKE_TM%", "NUTCOKE_TM%"])  # the dataset as published: inherent moisture only
    wet = plain.assign(**{"COKE_TM%": 2.5, "NUTCOKE_TM%": 6.0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = build_shift_inputs(plain, m10, hm, dpr, start="2026-05-01")
        S = build_shift_inputs(wet, m10, hm, dpr, start="2026-05-01")
    k = "2026-05-02_A"
    assert S.at[k, "COKE_DRY"] == pytest.approx(S.at[k, "COKE_CALC_MT"] * 0.975)
    assert S.at[k, "NUT_DRY"] == pytest.approx(S.at[k, "NUTCOKE_CALC_MT"] * 0.94)
    assert S.at[k, "COKE_MOIST%"] == pytest.approx(2.5) and S.at[k, "NUTCOKE_MOIST%"] == pytest.approx(6.0)
    assert S.at[k, "COKE_DRY"] < base.at[k, "COKE_DRY"]  # the dataset's own column is the smaller moisture
    assert S.at[k, "PCI_DRY"] == pytest.approx(base.at[k, "PCI_DRY"])  # PCI is left as it is
    pd.testing.assert_series_equal(S["ORE_DRY"], base["ORE_DRY"])


def test_the_page_and_the_twin_build_from_the_same_frame() -> None:
    fd = pd.DataFrame({"time": pd.date_range(H, periods=3, freq="h"), "COKE_MOIST%": 0.45, "NUTCOKE_MOIST%": 0.46})
    tm = pd.DataFrame({"COKE_TM%": [2.2] * 3, "NUTCOKE_TM%": [5.6] * 3}, index=pd.date_range(H, periods=3, freq="h"))
    asked = []

    def moisture(start: str, clock: str, nonce: int) -> pd.DataFrame:
        asked.append((start, clock, nonce))
        return tm

    keys = ce.DataKeys("f.csv", 1, "2026-05-10T09:00", 3, "2026-04-01", "{}", "cache", 30.0)
    loaders = ce.Loaders(lambda path, mtime: fd, None, None, None, None, moisture=moisture)
    out = ce.hourly_frame(keys, loaders)
    assert out["COKE_TM%"].tolist() == pytest.approx([2.2] * 3) and asked == [("2026-04-01", "2026-05-10T09:00", 3)]
    pd.testing.assert_frame_equal(ce.hourly_frame(keys, loaders, fd), out)  # the twin passes the frame it observed
    plain = ce.Loaders(lambda path, mtime: fd, None, None, None, None)  # a loader set without the laboratory
    assert ce.hourly_frame(keys, plain) is fd
    assert ce.CACHED_LOADERS.moisture is not ce.no_fuel_moisture
