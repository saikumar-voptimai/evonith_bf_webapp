"""Charged coke of the last charges (``utils.coke_energy.charge_record``): the charge table, the ratio, the factor
from the stored windows and the figure on the pipeline's basis."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.coke_energy import charge_record as cr

S = cr.Settings(charges=6, min_window_charges=40, min_windows=3, max_windows=5)


def _raw(n: int, coke: float = 4.5) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    times = pd.date_range("2026-11-10 00:05", periods=n, freq="10min")
    raw = pd.DataFrame(
        {
            "charge_no": range(n),
            "coke_1_mt": coke,
            "coke_2_mt": 0.0,
            "nut_coke_1_mt": 1.1,
            "sinter_1_mt": 15.0,
            "ore_3_mt": 6.0,
            "pellet_1_mt": 4.0,
            "flux_1_mt": 0.4,
            "pci_mt": 0.0,
        },
        index=times,
    )
    return raw, times


def _table(n: int = 200, coke: float = 4.5) -> pd.DataFrame:
    raw, times = _raw(n, coke)
    return cr.charge_table(raw, times, times + pd.Timedelta(minutes=30))


def test_the_charge_table_sums_the_groups_per_charge() -> None:
    t = _table(3)
    assert list(t.columns) == ["coke_t", "nut_coke_t", "ferrous_t", "flux_t", "received"]
    assert t["coke_t"].tolist() == [4.5] * 3 and t["nut_coke_t"].tolist() == [1.1] * 3  # nut coke is not coke
    assert t["ferrous_t"].tolist() == [25.0] * 3 and t["flux_t"].tolist() == [0.4] * 3
    assert t["received"].iloc[0] == pd.Timestamp("2026-11-10 00:35")
    raw, times = _raw(3)
    raw.loc[times[1], ["sinter_1_mt", "ore_3_mt", "pellet_1_mt"]] = 0.0  # a row without burden is no charge
    assert len(cr.charge_table(raw, times)) == 2 and cr.charge_table(raw, times)["received"].isna().all()
    assert cr.charge_table(pd.DataFrame(), pd.DatetimeIndex([])).empty


def test_the_ratio_is_coke_per_tonne_of_ferrous_burden() -> None:
    t = _table()
    r, n = cr.ratio(t, "2026-11-10 00:00", "2026-11-10 08:00")
    assert n == 48 and r == pytest.approx(180.0)
    assert cr.ratio(t, "2026-11-09 00:00", "2026-11-09 08:00") == (None, 0)


def test_the_factor_is_the_median_over_the_newest_windows() -> None:
    t = _table()
    ends = pd.date_range("2026-11-10 10:00", periods=6, freq="1h")
    windows = [(e, k) for e, k in zip(ends, [300.0, 306.0, 309.0, 303.0, 312.0, None], strict=True)]
    f, n = cr.factor(t, windows, S)
    assert n == 5 and f == pytest.approx(306.0 / 180.0)  # the None window is left out, the median of the other five
    assert cr.factor(t, windows[:2], S) == (S.default_factor, 2)  # fewer than min_windows
    # a window with too few charges does not count
    assert cr.factor(t, [(pd.Timestamp("2026-11-10 03:00"), 300.0)] * 5, S) == (S.default_factor, 0)


def test_charged_now_is_on_the_basis_of_the_windows() -> None:
    t = _table()
    richer = t.copy()
    richer.loc[richer.index[-6:], "coke_t"] = 4.75  # the last six charges carry more coke
    ends = pd.date_range("2026-11-10 10:00", periods=5, freq="1h")
    windows = [(e, 306.0) for e in ends]
    now = cr.charged_now(richer, windows, S)
    assert now["charges"] == 6 and now["basis"] == "windows" and now["factor"] == pytest.approx(1.7)
    assert now["value"] == pytest.approx(1.7 * 190.0)  # 4.75 / 25 = 190 kg/t, against 306 for 180 kg/t
    assert now["first"] == richer.index[-6] and now["last"] == richer.index[-1]
    assert now["received"] == richer.index[-1] + pd.Timedelta(minutes=30)
    # only the charges up to ``at``; the default factor without windows; no value with fewer charges than asked
    earlier = cr.charged_now(richer, windows, S, at=richer.index[-7])
    assert earlier["value"] == pytest.approx(306.0) and earlier["last"] == richer.index[-7]
    default = cr.charged_now(t, [], S)
    assert default["basis"] == "default" and default["value"] == pytest.approx(S.default_factor * 180.0)
    few = cr.charged_now(t.iloc[:4], windows, S)
    assert few["value"] is None and few["charges"] == 4
    assert cr.charged_now(t.iloc[:0], windows, S)["value"] is None


def test_the_settings_come_from_the_config() -> None:
    s = cr.Settings.from_config({"charges": 12, "default_factor": 1.66, "read_days": 4})
    assert (s.charges, s.default_factor, s.window_h) == (12, 1.66, 8.0)
