"""Burden column of the live twin (SV-1.3, #283): descent times, the ledger of the charges in the column, coke by zone
and the implied lag between charged and burnt coke (``utils.coke_energy.burden_column``, ``services.transit``)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import transit
from utils.coke_energy import burden_column as bc

REF = bc.TransitReference()
AT = pd.Timestamp("2026-05-02 00:00")


def record(
    hours: int = 24, charges: float = 6.0, per_charge: dict | None = None, end: pd.Timestamp = AT
) -> pd.DataFrame:
    """Hourly charge record ending at ``end``: ``charges`` per hour, ``per_charge`` tonnes of each material."""
    per = per_charge or {
        "COKE_CALC_MT": 4.55,
        "SINTER_CALC_MT": 16.0,
        "TOTAL_PELLET_CALC_MT": 2.5,
        "ORE_CALC_MT": 7.5,
        "ORE_2_CALC_MT": 5.0,
        "ORE_3_CALC_MT": 2.5,
        "NUTCOKE_CALC_MT": 0.46,
        "FLUX_CALC_MT": 0.0,
    }
    t = pd.date_range(end - pd.Timedelta(hours=hours), periods=hours, freq="h")
    df = pd.DataFrame({m: v * charges for m, v in per.items()}, index=t)
    df["CHARGES/HRS."] = charges
    df["PRODUCTIONTONNESPERHR"] = 15.0 * charges
    df.index.name = "time"
    return df


# ---------------------------------------------------------------------------------------------- descent times
def test_descent_times_at_the_reference_rate() -> None:
    d = bc.descent_times(5.9)
    assert d["hours_to_cz"] == pytest.approx(2.2, abs=0.1) and d["hours_to_raceway"] == pytest.approx(6.2, abs=0.1)
    assert (d["hours_above_cz"], d["hours_in_cz"], d["hours_below_cz"]) == pytest.approx((2.2, 0.9, 3.1))
    faster = bc.descent_times(6.5)
    assert faster["hours_to_raceway"] == pytest.approx(6.2 * 5.9 / 6.5)
    with pytest.raises(ValueError):
        bc.descent_times(0.0)
    assert REF.zone_charges() == pytest.approx({"above_cz": 12.98, "in_cz": 5.31, "below_cz": 18.29})


# ---------------------------------------------------------------------------------------------- ledger
def test_ledger_at_a_constant_rate() -> None:
    st = bc.column(record(), AT)
    above, inside, below = st.bands
    assert [b.zone for b in st.bands] == ["above_cz", "in_cz", "below_cz"] and st.complete and st.missing_hours == 0
    assert (above.charges, inside.charges, below.charges) == pytest.approx((12.98, 5.31, 18.29))
    assert above.newest_h == pytest.approx(0.0) and above.oldest_h == pytest.approx(12.98 / 6)
    assert inside.oldest_h == pytest.approx((12.98 + 5.31) / 6) and below.oldest_h == pytest.approx(36.58 / 6)
    assert st.tau["to_cz"] == pytest.approx(12.98 / 6) and st.tau["to_raceway"] == pytest.approx(36.58 / 6)
    assert st.tau_rate["hours_to_raceway"] == pytest.approx(6.2 * 5.9 / 6.0)  # rate formula: the same at 6/h
    assert above.tonnes["COKE_CALC_MT"] == pytest.approx(12.98 * 4.55)
    assert above.coke_t_per_charge == pytest.approx(4.55) and above.coke_kg_thm == pytest.approx(4550 / 15)
    assert above.coke_kg_per_t_ferrous == pytest.approx(4550 / 26.0)
    assert above.blend_wet_pct == pytest.approx(
        {
            "SINTER_CALC_MT": 1600 / 26,
            "TOTAL_PELLET_CALC_MT": 250 / 26,
            "ORE_2_CALC_MT": 500 / 26,
            "ORE_3_CALC_MT": 250 / 26,
        }
    )


def test_coke_by_zone_follows_the_coke_share_of_the_charge_volume() -> None:
    reference = {"COKE_CALC_MT": 4.55, "SINTER_CALC_MT": 26.46}  # the reference charge: 4.55 t coke, 26.46 t ore layer
    st = bc.column(record(per_charge=reference), AT)
    assert [b.coke_inventory_t for b in st.bands] == pytest.approx([61.0, 14.0, 59.0])
    assert st.coke_in_transit_t == pytest.approx(134.0)
    assert st.bands[0].coke_volume_fraction == pytest.approx(0.3872, abs=1e-4)
    richer = bc.column(record(per_charge={"COKE_CALC_MT": 5.0, "SINTER_CALC_MT": 26.46}), AT)
    phi = REF.coke_volume_fraction(5.0, 26.46)
    assert richer.bands[0].coke_inventory_t == pytest.approx(61.0 * phi / REF.coke_volume_fraction(4.55, 26.46))
    assert richer.bands[2].coke_inventory_t == pytest.approx(59.0)  # coke bed of fixed volume below the zone
    assert richer.bands[0].coke_inventory_t > 61.0


def test_a_varying_rate_moves_the_zone_boundaries_in_the_record() -> None:
    slow = record(hours=20, charges=3.0, end=AT - pd.Timedelta(hours=4))
    fast = record(hours=4, charges=6.0)
    st = bc.column(pd.concat([slow, fast]), AT)
    # the last 4 h hold 24 charges: the cohesive-zone top (12.98 charges back) lies in them
    assert st.tau["to_cz"] == pytest.approx(12.98 / 6)
    # the raceway boundary (36.58 charges back) lies 12.58 charges into the slow hours: 4 h + 12.58 / 3 h
    assert st.tau["to_raceway"] == pytest.approx(4 + (36.58 - 24) / 3)
    assert st.tau_rate["hours_to_raceway"] == pytest.approx(6.2 * 5.9 / 6.0)  # the rate formula sees only 6/h
    assert st.bands[2].oldest_h > st.tau_rate["hours_to_raceway"]


def test_stoppages_missing_hours_and_a_short_record() -> None:
    rec = record()
    rec.loc[AT - pd.Timedelta(hours=2), "CHARGES/HRS."] = 0.0  # no charging in one hour
    st = bc.column(rec, AT)
    assert st.tau["to_cz"] == pytest.approx(1 + 12.98 / 6)  # the stoppage adds its hour to the age
    gap = record().drop(AT - pd.Timedelta(hours=3))
    st_gap = bc.column(gap, AT)
    assert st_gap.missing_hours == 1 and st_gap.tau["to_cz"] == pytest.approx(1 + 12.98 / 6)
    short = bc.column(record(hours=3), AT)
    assert not short.complete and np.isnan(short.tau["to_raceway"])
    assert short.bands[0].charges == pytest.approx(12.98) and short.bands[1].charges == pytest.approx(18 - 12.98)
    late = bc.column(record(), AT + pd.Timedelta(minutes=30))  # an hour that has not ended is not counted
    assert late.bands[0].newest_h == pytest.approx(0.5)


def test_charged_coke_series() -> None:
    rec = record(hours=5)
    assert bc.charged_coke(rec).to_numpy() == pytest.approx(np.full(5, 4550 / 26.0))
    assert bc.charged_coke(rec.reset_index(), "charge").to_numpy() == pytest.approx(np.full(5, 4.55))
    with pytest.raises(ValueError):
        bc.charged_coke(rec, "tonnes")


# ---------------------------------------------------------------------------------------------- implied lag
def synthetic_lag(seed: int, lag: int = 6, days: int = 14) -> tuple[pd.Series, pd.Series]:
    """Hourly coke content of the charges (AR(1) around 180 kg/t) and K_TRACE of 8-h windows burning it ``lag`` h
    later, with measurement noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-09-01", periods=24 * days + 24, freq="h")
    x = np.zeros(len(idx))
    for i in range(1, len(idx)):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 4)
    charged = pd.Series(180 + x, idx)
    burnt = charged.shift(lag).rolling(8).mean().shift(1) * 1.7 + rng.normal(0, 4, len(idx))
    return charged, burnt.dropna()


def test_the_implied_lag_recovers_a_known_transit_time() -> None:
    charged, burnt = synthetic_lag(seed=4)
    res = bc.implied_lag(charged, burnt, n_boot=300, seed=1)
    assert res.lag_h == pytest.approx(6.0, abs=0.5) and res.ci[0] <= 6.0 <= res.ci[1]
    assert res.scan.set_index("lag_h").r.idxmax() == 6.0 and res.r_peak > 0.5
    assert res.ci[1] - res.ci[0] < 2.0 and res.n_hours > 300


def test_the_implied_lag_is_unbiased_over_repetitions() -> None:
    est, covered = [], 0
    for seed in range(20):
        charged, burnt = synthetic_lag(seed=100 + seed)
        res = bc.implied_lag(charged, burnt, n_boot=150, seed=seed)
        est.append(res.lag_h)
        covered += res.ci[0] <= 6.0 <= res.ci[1]
    assert abs(np.mean(est) - 6.0) < 0.15 and covered >= 16  # simulated coverage of the 95 % interval: 0.90-0.92


# ---------------------------------------------------------------------------------------------- service
def test_service_settings_quantities_and_bands() -> None:
    cfg = {"twin": {"transit": {"reference": {"charges_per_hour": 6.0}, "implied_lag": {"days": 7, "n_boot": 50}}}}
    ref, lag = transit.settings(cfg)
    assert ref.charges_per_hour == 6.0 and ref.hours["above_cz"] == 2.2 and lag.days == 7 and lag.n_boot == 50
    st = transit.state(record(), AT, cfg)
    q = transit.quantities(st)
    assert q["tau_cz_h"] == pytest.approx(2.2) and q["tau_cz_rate_h"] == pytest.approx(2.2)  # 13.2 charges at 6/h
    assert {"coke_in_transit_t", "coke_above_cz_t", "coke_below_cz_t", "tau_raceway_rate_h"} <= set(q)
    rows = transit.bands(st)
    assert [r["zone"] for r in rows] == ["above_cz", "in_cz", "below_cz"] and rows[0]["tonnes"]["COKE_CALC_MT"] > 0


def test_service_implied_lag_from_stored_windows(tmp_path) -> None:
    charged, burnt = synthetic_lag(seed=4)
    fer = 26.0
    fd = pd.DataFrame(
        {"COKE_CALC_MT": charged.to_numpy() * fer / 1000 * 6, "SINTER_CALC_MT": fer * 6, "CHARGES/HRS.": 6.0},
        index=charged.index,
    )
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    for t, k in burnt.items():  # window ends in IST, stored in UTC
        end = t.tz_localize("Asia/Kolkata").tz_convert("UTC").to_pydatetime()
        store.snapshots.upsert(
            {
                "window_start": end - timedelta(hours=8),
                "window_end": end,
                "evaluated_at": end,
                "status": "complete",
                "k_trace": float(k),
            }
        )
    rows = store.snapshots.series(datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 10, 1, tzinfo=UTC))
    assert transit.burnt_series(rows).index[0] == burnt.index[0]  # back on the IST time base
    res = transit.implied_lag(fd, rows, burnt.index[-1], {"twin": {"transit": {"implied_lag": {"n_boot": 200}}}})
    assert res.lag_h == pytest.approx(6.0, abs=0.5) and res.ci[0] <= 6.0 <= res.ci[1]
    engine.dispose()


def test_the_report_script_prints_the_column(tmp_path, capsys) -> None:
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("transit_report", root / "scripts" / "transit_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    path = tmp_path / "hourly.csv"
    record().reset_index().to_csv(path, index=False)
    assert report.main(["--dataset", str(path)]) == 0
    text = capsys.readouterr().out
    assert "burden column at 2026-05-02 00:00 IST" in text and "charging rate 6.00 charges/h" in text
    assert "cohesive-zone top 2.16 h, raceways 6.10 h" in text and "above_cz  12.98 charges" in text
