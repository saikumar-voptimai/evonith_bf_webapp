"""Live Twin page support (SV-1.9, #289): blend change in the burden column, the window's ferrous shares, data age of
each source, model gaps against TRACE, the Live Twin alarms and the routes the page reads (/api/twin/now sources and
gaps, /api/twin/window)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from services import store as store_service
from services import twin as twin_service
from services import twin_alarms, twin_views
from services.events import MemoryEventBus
from tests.test_api_twin import Frozen, client, get, ist, snapshot, twin  # noqa: F401 - fixtures
from utils.coke_energy import burden_column as bc

AT = pd.Timestamp("2026-05-02 00:00")
OLD = {"COKE_CALC_MT": 4.55, "SINTER_CALC_MT": 16.0, "TOTAL_PELLET_CALC_MT": 2.5, "ORE_CALC_MT": 7.5}
NEW = {"COKE_CALC_MT": 4.55, "SINTER_CALC_MT": 12.0, "TOTAL_PELLET_CALC_MT": 6.5, "ORE_CALC_MT": 7.5}


def record(change_h: int | None = 4, hours: int = 24, charges: float = 6.0) -> pd.DataFrame:
    """Hourly charge record ending at AT; the last ``change_h`` hours charged with the NEW blend."""
    t = pd.date_range(AT - pd.Timedelta(hours=hours), periods=hours, freq="h")
    rows = [NEW if change_h and i >= hours - change_h else OLD for i in range(hours)]
    df = pd.DataFrame([{m: v * charges for m, v in r.items()} for r in rows], index=t)
    df["CHARGES/HRS."] = charges
    df["PRODUCTIONTONNESPERHR"] = 15.0 * charges
    df.index.name = "time"
    return df


# ------------------------------------------------------------------------------------------- burden column
def test_ferrous_shares_of_the_three_groups() -> None:
    assert bc.ferrous_shares(OLD) == pytest.approx({"sinter": 1600 / 26, "pellet": 250 / 26, "ore": 750 / 26})
    assert bc.ferrous_shares({"COKE_CALC_MT": 4.0}) == {}


def test_blend_change_found_at_its_hour_with_its_arrival_times() -> None:
    ch = bc.blend_change(record(4), AT)
    assert ch is not None and ch.changed_at == AT - pd.Timedelta(hours=4)
    assert ch.step_pt == pytest.approx(100 * (6.5 / 26 - 2.5 / 26))
    assert ch.before_pct["pellet"] == pytest.approx(250 / 26) and ch.after_pct["pellet"] == pytest.approx(650 / 26)
    assert ch.charges_since == pytest.approx(24.0)
    need = bc.TransitReference().zone_charges()
    # 24 charges since the change: past the cohesive zone (12.98 + 5.31), not yet at the raceways (36.58)
    assert ch.zone == "below_cz"
    ms = pd.Timedelta(milliseconds=1)
    assert abs(ch.cz_at - (ch.changed_at + pd.Timedelta(hours=need["above_cz"] / 6.0))) < ms
    assert abs(ch.raceway_at - (AT + pd.Timedelta(hours=(sum(need.values()) - 24.0) / 6.0))) < ms


def test_blend_change_still_above_the_zone_and_none_without_a_change() -> None:
    ch = bc.blend_change(record(1), AT)
    assert ch is not None and ch.zone == "above_cz" and ch.charges_since == pytest.approx(6.0)
    assert abs(ch.cz_at - (AT + pd.Timedelta(hours=(12.98 - 6.0) / 6.0))) < pd.Timedelta(milliseconds=1)
    assert bc.blend_change(record(None), AT) is None
    assert bc.blend_change(record(4), AT, threshold_pt=20.0) is None


# ------------------------------------------------------------------------------------------- views
def test_source_freshness_from_the_recorded_ends() -> None:
    now = datetime(2026, 10, 27, 6, 0, tzinfo=UTC)
    latest = {
        "evaluated_at": now - timedelta(minutes=20),
        "flags": {
            "provisional_dpr": True,
            "sources": {
                "influxdb": (now - timedelta(minutes=50)).isoformat(),
                "hourly": (now - timedelta(hours=1.5)).isoformat(),
                "hm_slag": (now - timedelta(hours=2)).isoformat(),
                "dpr": (now - timedelta(hours=30)).isoformat(),
                "open_meteo": None,
            },
        },
    }
    rows = {r["key"]: r for r in twin_views.source_freshness(latest, now)}
    assert list(rows) == ["influxdb", "hourly", "hm_slag", "dpr", "open_meteo"]
    assert rows["influxdb"]["stale"] and rows["influxdb"]["age_h"] == pytest.approx(50 / 60)
    assert not rows["hourly"]["stale"] and not rows["hm_slag"]["stale"]
    assert rows["dpr"]["provisional"] and not rows["dpr"]["stale"]
    assert rows["open_meteo"]["stale"] and rows["open_meteo"]["data_end"] is None
    assert all(r["stale"] for r in twin_views.source_freshness(None, now))
    tight = twin_views.source_freshness(latest, now, {"hourly": 1.0})
    assert next(r for r in tight if r["key"] == "hourly")["stale"]


def series(n: int, end: datetime, gap: float = 0.0, last: int = 0, last_gap: float = 0.0) -> list[dict]:
    """``n`` hourly rows ending at ``end``: STRATA - TRACE alternates +-gap; the last ``last`` rows have last_gap."""
    rows = []
    for i in range(n):
        g = last_gap if i >= n - last else (gap if i % 2 else -gap)
        rows.append(dict(window_end=end - timedelta(hours=n - 1 - i), k_trace=300.0, k_forge=301.0, k_strata=300.0 + g))
    return rows


def test_model_gaps_now_rmse_and_the_excess_run() -> None:
    end = datetime(2026, 10, 27, 5, 30, tzinfo=UTC)
    g = twin_views.model_gaps(series(100, end, gap=2.0, last=4, last_gap=6.0), end)
    assert g["forge_trace_kg_thm"] == pytest.approx(1.0) and g["forge_trace_rmse_kg_thm"] == pytest.approx(1.0)
    assert g["strata_trace_kg_thm"] == pytest.approx(6.0) and g["n_strata"] == 100
    rmse = ((96 * 4 + 4 * 36) / 100) ** 0.5
    assert g["strata_trace_rmse_kg_thm"] == pytest.approx(rmse)
    assert g["strata_excess_h"] == 4 and g["strata_excess_since"] == end - timedelta(hours=3)
    # rows older than 7 days are left out; a missing hour ends the run
    rows = series(200, end, gap=2.0, last=4, last_gap=6.0)
    del rows[-3]
    g = twin_views.model_gaps(rows, end)
    assert g["n_strata"] == 7 * 24 - 1 and g["strata_excess_h"] == 2


# ------------------------------------------------------------------------------------------- alarms
def test_alarm_conditions() -> None:
    s = twin_alarms.AlarmSettings()
    end = datetime(2026, 10, 27, 5, 30, tzinfo=UTC)
    gaps = twin_views.model_gaps(series(100, end, gap=2.0, last=4, last_gap=6.0), end)
    c = twin_alarms.gap_condition(gaps, s)
    assert c.active and "4 h" in c.message and c.value == pytest.approx(6.0)
    assert twin_alarms.gap_condition({**gaps, "strata_excess_h": 2}, s).active is False
    assert twin_alarms.gap_condition({**gaps, "n_strata": 10}, s).active is None
    assert twin_alarms.lag_condition(6.2, (5.5, 7.0), s).active is False
    assert twin_alarms.lag_condition(9.1, (8.0, 10.0), s).active is True
    assert twin_alarms.lag_condition(None, (None, None), s).active is None
    assert twin_alarms.job_condition("twin.evaluate", "failed", "boom").active is True
    assert twin_alarms.job_condition("twin.evaluate", "done", None).active is False
    assert twin_alarms.job_condition("twin.evaluate", "cancelled", None).active is None
    cfg = {"twin": {"alarms": {"strata_trace_gap": {"min_hours": 5}}, "freshness": {"stale_after_h": {"dpr": 30}}}}
    assert twin_alarms.AlarmSettings.from_config(cfg).min_hours == 5
    assert twin_alarms.AlarmSettings.from_config(cfg).stale_after_h["dpr"] == 30


def test_alarms_are_raised_cleared_and_published(twin) -> None:  # noqa: F811 - fixture
    now = datetime(2026, 10, 27, 5, 30, tzinfo=UTC)
    for row in series(100, now - timedelta(minutes=30), gap=2.0, last=4, last_gap=6.0):
        twin.snapshots.upsert(snapshot(row["window_end"], k_strata=row["k_strata"], k_trace=300.0))
    bus = MemoryEventBus()
    changes = asyncio.run(twin_alarms.run(twin, bus, {}, now, lag=False))
    kinds = {(k, sub) for e, k, sub in changes if e == "alarm.raised"}
    assert ("strata_trace_gap", "") in kinds
    assert {sub for k, sub in kinds if k == "source_stale"} == set(twin_views.SOURCE_LABELS)  # no ends recorded
    events = asyncio.run(bus.recent())
    assert {e.data.code for e in events} == {"STRATA_TRACE_GAP", "SOURCE_STALE"}
    # The next run with the gap back inside clears it; a second run with nothing changed publishes nothing new.
    twin.snapshots.upsert(snapshot(now, k_strata=300.5, k_trace=300.0))
    later = asyncio.run(twin_alarms.run(twin, bus, {}, now + timedelta(minutes=5), lag=False))
    assert ("alarm.cleared", "strata_trace_gap", "") in later
    again = asyncio.run(twin_alarms.run(twin, bus, {}, now + timedelta(minutes=10), lag=False))
    assert again == []


def test_a_failed_scheduled_job_raises_and_its_next_run_clears(twin) -> None:  # noqa: F811 - fixture
    bus = MemoryEventBus()
    assert asyncio.run(twin_alarms.job_outcome(bus, "twin.evaluate", "failed", "RuntimeError: x")) == [
        ("alarm.raised", "job_failed", "twin.evaluate")
    ]
    assert asyncio.run(twin_alarms.job_outcome(bus, "system.selftest", "failed", "x")) == []  # not scheduled
    opened = twin.alarms.open_alarms()
    assert [(a.kind, a.subject) for a in opened] == [("job_failed", "twin.evaluate")]
    assert asyncio.run(twin_alarms.job_outcome(bus, "twin.evaluate", "done", None)) == [
        ("alarm.cleared", "job_failed", "twin.evaluate")
    ]
    store_service.use(None)
    assert asyncio.run(twin_alarms.job_outcome(bus, "twin.evaluate", "failed", "x")) == []


def test_source_ends_recorded_by_the_evaluation() -> None:
    fd = pd.DataFrame({"time": pd.to_datetime(["2026-10-27 09:00", "2026-10-27 10:00"])})
    m10 = pd.DataFrame({"ts": pd.to_datetime(["2026-10-27 10:50"])})
    hm = pd.DataFrame({"t": pd.to_datetime(["2026-10-27 08:40"])})
    dpr = pd.DataFrame({"hm_t": [2400.0, 2400.0]}, index=pd.to_datetime(["2026-10-26", "2026-10-27"]))
    frames = twin_service.TwinFrames(fd, m10, hm, dpr, (pd.Timestamp("2026-10-27").date(),))
    ends = twin_service.source_ends(frames)
    assert ends["hourly"] == "2026-10-27T04:30:00+00:00" and ends["influxdb"] == "2026-10-27T05:20:00+00:00"
    assert ends["hm_slag"] == "2026-10-27T03:10:00+00:00"
    assert ends["dpr"] == "2026-10-25T18:30:00+00:00"  # the provisional 27 Oct is not a report
    assert set(ends) == set(twin_views.SOURCE_LABELS)


# ------------------------------------------------------------------------------------------- routes
def test_now_carries_the_source_ages_and_gaps_and_window_returns_one_row(twin, client, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(twin_views, "datetime", Frozen)
    end = ist("2026-10-27 11:00")
    for row in series(30, end, gap=2.0):
        twin.snapshots.upsert(snapshot(row["window_end"], k_strata=row["k_strata"]))
    flags = {
        "notes": [],
        "sources": {"hourly": (end - timedelta(hours=1)).isoformat()},
        "strata": {"network_solves": 3},
    }
    twin.snapshots.upsert(snapshot(end, flags=flags))
    body = get(client, "op", "/api/twin/now").json()
    sources = {s["key"]: s for s in body["sources"]}
    assert sources["hourly"]["age_h"] == pytest.approx(1.5) and not sources["hourly"]["stale"]
    assert sources["dpr"]["stale"] and sources["dpr"]["data_end"] is None
    assert body["gaps"]["forge_trace_kg_thm"] == pytest.approx(-2.0) and body["gaps"]["n_forge"] == 30
    one = get(client, "op", "/api/twin/window", params={"end": end.isoformat()})
    assert one.status_code == 200 and one.json()["flags"]["strata"]["network_solves"] == 3
    missing = get(client, "op", "/api/twin/window", params={"end": (end + timedelta(hours=5)).isoformat()})
    assert missing.status_code == 404 and missing.headers["content-type"].startswith("application/problem+json")


def test_hot_metal_returns_each_tap_and_si_sample(client, monkeypatch) -> None:  # noqa: F811 - fixture
    import numpy as np

    from data.coke_energy import sources
    from services import status

    now = datetime(2026, 10, 27, 9, 30, 15, tzinfo=UTC)  # 15:00:15 IST
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        calls.append((start, end))
        t = pd.to_datetime(["2026-10-27 11:05", "2026-10-27 12:40", "2026-10-27 13:55", "2026-10-27 14:30"])
        return pd.DataFrame(
            {"t": t, "hmt_gt_1480c": [1488.0, 1495.0, 1502.0, 1210.0], "chem_pct_si": [0.41, 0.52, np.nan, 0.47]}
        )

    monkeypatch.setattr(status, "clock", lambda: now)
    monkeypatch.setattr(sources, "fetch_hm_slag", fetch)
    body = get(client, "op", "/api/twin/hot-metal", params={"hours": 24}).json()
    assert [(r["at"], r["value"]) for r in body["hmt"]] == [
        ("2026-10-27T05:35:00Z", 1488.0),
        ("2026-10-27T07:10:00Z", 1495.0),
        ("2026-10-27T08:25:00Z", 1502.0),
    ]  # the 1,210 °C sample is not a tap
    assert [r["value"] for r in body["si"]] == [0.41, 0.52, 0.47]
    assert calls[0][0] == pd.Timestamp("2026-10-26 15:00:15")

    def down(start, end):
        raise ConnectionError("no route to Neon")

    monkeypatch.setattr(sources, "fetch_hm_slag", down)
    failed = get(client, "op", "/api/twin/hot-metal")
    assert failed.status_code == 503 and failed.headers["content-type"].startswith("application/problem+json")
    assert get(client, "op", "/api/twin/hot-metal", params={"hours": 24 * 40}).status_code == 422


def test_critical_assumptions_are_stored_for_strata_plans_only() -> None:
    from services import optimiser

    rows = [{"assumption": "size exponent of the reduction rate", "value": "1", "alternative": "1.5", "d_coke_kg": 0.2}]

    class Eng:
        def critical_assumptions(self, plan, s):
            assert s.omega_mode == "blend"
            return rows

    class Broken:
        def critical_assumptions(self, plan, s):
            raise RuntimeError("no anchor")

    plan = {"omega": {"mode": "blend"}, "settings": {"omega_mode": "blend", "not_a_setting": 1}}
    assert optimiser.critical_assumptions(Eng(), plan) == rows
    assert optimiser.critical_assumptions(Eng(), {"omega": {"mode": "fixed"}, "settings": {}}) == []
    assert optimiser.critical_assumptions(Eng(), None) == []
    assert optimiser.critical_assumptions(Broken(), plan) == []
