"""Live model view of the Blend & Fuel Optimiser (user decisions 12 Nov 2026): the data clock, fuel and blast now, the
coke balance with the charged coke of the last charges, the HMT filter with its band, score and check, the twin's
hourly windows with the PCI guard, page 11's model check on the FORGE view, and the route."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import forge, live_model
from services import store as store_service
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from utils.coke_energy.charge_record import charge_table

NOW = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)  # 14:30 IST


@pytest.fixture()
def twin(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store_service.use(store)
    yield store
    store_service.reset()
    store.dispose()


def _windows(store: TwinStore, hours: int = 60, stuck: tuple[int, ...] = ()) -> None:
    for h in range(hours, 0, -1):
        end = NOW - timedelta(hours=h)
        store.snapshots.upsert(
            {
                "window_start": end - timedelta(hours=8),
                "window_end": end,
                "evaluated_at": end,
                "status": "complete",
                "k_trace": 300.0 + h % 5,
                "k_forge": 302.0 + h % 3,
                "k_strata": 301.0,
                "k_charged": 320.0,
                "quantities": {
                    "hmt_c": 1480.0,
                    "pci_kg_thm": 30.0 if h in stuck else 195.0,
                    "tau_cz_h": 2.1,
                    "tau_raceway_h": 6.0,
                    "eta_co_pct": 46.0,
                    "eta_co_forge_pct": 46.5,
                    "top_gas_c": 150.0,
                    "top_gas_forge_c": 155.0,
                },
                "flags": {"hmt_forecast_gate": "pending"},
            }
        )


def _taps() -> pd.DataFrame:
    """Taps every 2 h (naive IST) around 1,480 °C, published 40 min after the tap."""
    idx = pd.date_range("2026-10-18 06:00", "2026-11-12 14:00", freq="2h")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"hmt": 1480.0 + rng.normal(0.0, 12.0, len(idx)), "received": idx + pd.Timedelta(minutes=40)}, index=idx
    )


def _charges() -> pd.DataFrame:
    """A charge every 10 min, 4.5 t coke on 25 t ferrous burden (180 kg/t), published 30 min later."""
    times = pd.date_range("2026-11-08 14:00", "2026-11-12 14:20", freq="10min")  # naive IST
    raw = pd.DataFrame({"coke_1_mt": 4.5, "nut_coke_1_mt": 1.1, "sinter_1_mt": 19.0, "ore_1_mt": 6.0}, index=times)
    return charge_table(raw, times, times + pd.Timedelta(minutes=30))


def _status() -> SimpleNamespace:
    at = NOW - timedelta(seconds=20)

    def tag(key: str, value: float, setpoint: float | None = None) -> SimpleNamespace:
        return SimpleNamespace(key=key, value=value, at=at, setpoint=setpoint)

    return SimpleNamespace(
        parameters=[
            tag("pci_rate", 200.0, 204.0), tag("production_rate", 90.0), tag("hbt", 1193.0),
            tag("hot_blast_volume", 109800.0), tag("oxygen_flow", 5690.0), tag("raft", 2217.0),
            tag("co_utilization", 42.3), tag("uptake_avg", 212.0),
        ],
        freshness=SimpleNamespace(data_at=at, telemetry="live"),
    )  # fmt: skip


def _view(store: TwinStore, **overrides) -> live_model.LiveModelView:
    args = {
        "now": NOW, "taps": lambda a, b: _taps(), "charges": lambda a, b: _charges(), "process": _status,
        "shifts": _shifts(), "built_at": NOW, "cfg": {"model_check": {}}, "settings": live_model.Settings(),
    } | overrides  # fmt: skip
    return live_model.LiveModelView.model_validate(live_model.view(store, **args))


def _shifts() -> pd.DataFrame:
    keys = [f"2026-11-1{d}_{s}" for d, s in zip("0001112", "ABCABCA", strict=True)]
    return pd.DataFrame(
        {
            "sdate": pd.to_datetime(["2026-11-10"] * 3 + ["2026-11-11"] * 3 + ["2026-11-12"]),
            "shift": list("ABCABCA"),
            "t_mid": pd.to_datetime(
                [
                    "2026-11-10 10:00",
                    "2026-11-10 18:00",
                    "2026-11-11 02:00",
                    "2026-11-11 10:00",
                    "2026-11-11 18:00",
                    "2026-11-12 02:00",
                    "2026-11-12 10:00",
                ]
            ),  # fmt: skip
            "stable": [True, True, True, True, True, False, True],
            "K_burnt": [300.0, 305.0, 298.0, 301.0, 303.0, 299.0, 302.0],
            "K_energy": [302.0, 303.0, 300.0, 299.0, 305.0, 330.0, 300.0],
            "K_obal": [301.0, 304.0, 299.0, 300.0, 304.0, 298.0, 301.0],
            "K_charged": [320.0] * 7,
        },
        index=keys,
    )


def test_the_view_composes_clock_process_coke_hmt_windows_and_the_shift_check(twin) -> None:
    _windows(twin)
    v = _view(twin)
    assert v.hours == 48 and len(v.hourly) == 48 and v.latest is not None and v.notes == []
    assert v.latest.end == NOW - timedelta(hours=1) and v.latest.k_burnt == 301.0 and v.latest.k_forge == 303.0
    assert v.latest.eta_co_pct == 46.0 and v.latest.top_gas_forge_c == 155.0 and v.latest.tau_cz_h == 2.1
    # the data clock: process tags 20 s old, the newest charge 14:20 IST published 14:50, the newest tap 14:00
    clock = {s.key: s for s in v.clock.sources}
    assert list(clock) == ["process", "charges", "taps"] and clock["process"].age_min == pytest.approx(1 / 3)
    assert clock["charges"].newest == NOW - timedelta(minutes=10) and clock["charges"].delay_min == 30
    assert clock["charges"].received == NOW + timedelta(minutes=20) and clock["charges"].age_min == pytest.approx(10)
    assert clock["taps"].newest == NOW - timedelta(minutes=30) and clock["taps"].delay_min == 40
    assert "published once an hour (median age of a charge at publication 30 min)" in v.clock.notice
    # fuel and blast now: the status tags and PCI in t/h = rate x production; a tag the snapshot lacks has no value
    tags = {r.key: r for r in v.process.readings}
    assert v.process.telemetry == "live" and v.process.at == NOW - timedelta(seconds=20)
    assert tags["pci_rate"].value == 200.0 and tags["pci_rate"].setpoint == 204.0 and tags["pci_rate"].unit == "kg/tHM"
    assert tags["pci_t_h"].value == pytest.approx(18.0) and tags["o2_injection"].value is None
    # the coke balance: K_charged 320 for 180 kg coke per t ferrous -> factor 320 / 180; the last 6 charges the same
    c = v.coke
    assert c.charged_basis == "windows" and c.charged_factor == pytest.approx(320.0 / 180.0) and c.charged_charges == 6
    assert c.charged_now == pytest.approx(320.0) and c.charged_last == NOW - timedelta(minutes=10)
    assert c.charged_first == NOW - timedelta(minutes=60) and c.charged_received == NOW + timedelta(minutes=20)
    assert c.window_end == v.latest.end and (c.burnt, c.model, c.charged_window) == (301.0, 303.0, 320.0)
    assert c.gap == pytest.approx(2.0) and c.limit == 8.0 and c.state == "ok" and "+2.0 kg/tHM" in c.text
    # the taps shown are those of the hours; the HMT filter ran over all of them
    assert len(v.taps) == 24 and all(NOW - timedelta(hours=48) <= t.at <= NOW for t in v.taps)
    assert v.taps[-1].received == v.taps[-1].at + timedelta(minutes=40)
    h = v.hmt
    assert h.available and not h.stale and h.taps_used == len(_taps()) and h.last_tap == v.taps[-1]
    assert h.tap_interval_h == pytest.approx(2.0) and h.tap_age_h == pytest.approx(0.5)
    assert h.now is not None and h.now.at == NOW and 1465 < h.now.value < 1495
    assert h.next_tap is not None and h.next_tap.at == NOW + timedelta(minutes=90) and h.next_tap.sd > h.now.sd
    assert h.ahead is not None and h.ahead.at == NOW + timedelta(hours=6) and h.ahead.sd > h.next_tap.sd
    # a cold tap is one below the lower edge of BAND-02 unless the config names a temperature
    assert h.p_cold_next is not None and 0 < h.p_cold_next < 0.1 and h.cold_tap_c == 1440.0 and h.horizon_h == 6.0
    set_c = _view(twin, settings=live_model.Settings(cold_tap_c=1470.0)).hmt
    assert set_c.cold_tap_c == 1470.0 and set_c.p_cold_next is not None and set_c.p_cold_next > h.p_cold_next
    # the band runs from the newest tap to the horizon and widens with the lead; the estimate has a point per tap shown
    assert h.band[0].at == h.last_tap.at and h.band[-1].at == NOW + timedelta(hours=6)
    widths = [b.hi - b.lo for b in h.band]
    assert widths == sorted(widths) and any(b.at == NOW for b in h.band) and len(h.estimate) == 24
    assert h.plant_band is not None and (h.plant_band.lo, h.plant_band.hi, h.plant_band.fact) == (1440, 1520, "BAND-02")
    # the score: every tap of the last 14 days against the forecast made 2 h and 6 h before it; white-noise taps are
    # forecast better by the level than by the last tap held
    assert [s.horizon_h for s in h.scores] == [2.0, 6.0] and all(s.days == 14 and s.n > 150 for s in h.scores)
    assert all(s.verdict == "better" and s.ratio is not None and s.ratio < 0.9 for s in h.scores)
    assert "for the last tap held" in h.scores[0].text and h.check.state in ("ok", "watch") and h.check.text
    # page 11's model check: stable shifts only (the unstable 330 is not scored), on track, its thresholds
    k = v.shifts
    assert k is not None and k.days == 14 and k.state == "ok" and k.energy.n == 6 and k.oxygen.n == 6
    assert k.energy.bias is not None and abs(k.energy.bias) < 8 and k.latest_shift == "2026-11-12_A"
    assert k.age_days == 0 and k.thresholds["bias_kgthm"] == 8.0 and len(k.rows) == 7
    assert [r.stable for r in k.rows].count(False) == 1 and k.built_at == NOW


def test_a_window_with_a_stuck_pci_tag_shows_no_coke_rates(twin) -> None:
    _windows(twin, hours=30, stuck=(1, 2, 3))
    v = _view(twin, hours=24)
    assert [w.valid for w in v.hourly[-4:]] == [True, False, False, False]
    assert v.latest is not None and not v.latest.valid and v.latest.k_burnt is None and v.latest.k_forge is None
    assert v.latest.pci_kg_thm == 30.0 and v.latest.eta_co_pct == 46.0  # the measured quantities stay
    assert v.coke.state == "unknown" and "PCI is outside the valid range" in v.coke.text and v.coke.gap is None
    assert v.coke.charged_now == pytest.approx(320.0)  # from the valid windows' factor
    assert any("3 of the 24 windows have a PCI outside 80–320 kg/tHM" in n for n in v.notes)
    # model − burnt beyond the limit is a watch
    biased = live_model.coke_balance({"end": NOW, "k_burnt": 300.0, "k_forge": 290.5, "valid": True}, {}, 8.0)
    assert biased["state"] == "watch" and "−9.5 kg/tHM" in biased["text"] and biased["charged_basis"] == "none"


def test_the_hmt_view_checks_the_newest_tap_and_flags_stale_taps(twin) -> None:
    _windows(twin, hours=10)
    taps = _taps()
    hot = taps.copy()
    hot.iloc[-1, hot.columns.get_loc("hmt")] = 1560.0  # far above the band
    v = _view(twin, taps=lambda a, b: hot)
    assert v.hmt.check.state == "watch" and v.hmt.check.rule == "single" and "σ above the expected" in v.hmt.check.text
    assert v.hmt.check.value == 1560.0 and v.hmt.check.z is not None and v.hmt.check.z > 2
    # no tap for a day: the estimate is the level with the full band, and the view says so
    old = taps[taps.index <= pd.Timestamp("2026-11-11 12:00")]
    v = _view(twin, taps=lambda a, b: old)
    assert v.hmt.stale and v.hmt.tap_age_h == pytest.approx(26.5) and v.hmt.now is not None and v.hmt.now.sd > 14
    assert any("No tap published for 26 h" in n for n in v.notes)
    # a series without publication times (the status service's form) and too few taps
    few = _view(twin, taps=lambda a, b: taps["hmt"].iloc[-5:])
    assert not few.hmt.available and few.hmt.now is None and few.hmt.band == [] and few.hmt.check.state == "unknown"
    assert len(few.taps) == 5 and few.taps[0].received is None and any("needs 8" in n for n in few.notes)


def test_the_check_watches_a_biased_model_and_says_when_shifts_are_too_few() -> None:
    from utils.burden_engine.model_check import Thresholds

    shifts = _shifts()
    shifts["K_energy"] = shifts["K_burnt"] + 15.0
    today = pd.Timestamp("2026-11-12")
    c = live_model.shift_check(shifts, 14, thresholds=Thresholds(), today=today)
    assert c["state"] == "watch" and "bias +15.0" in c["reason"]
    few = live_model.shift_check(shifts.iloc[:3], 14, thresholds=Thresholds(), today=today)
    assert few["state"] == "insufficient" and few["energy"]["n"] == 3 and few["age_days"] == 2
    nothing = live_model.shift_check(None, 7, thresholds=Thresholds(), today=today)
    assert nothing["state"] == "insufficient" and nothing["rows"] == [] and nothing["latest_shift"] is None


def test_without_its_sources_the_view_stands_with_notes(twin, monkeypatch) -> None:
    _windows(twin, hours=10)

    def down(*args):
        raise ConnectionError("unreachable")

    def no_view():
        raise forge.ViewUnavailable("The FORGE view has not been built yet: the twin.forge job writes it hourly.")

    monkeypatch.setattr(forge, "load", no_view)
    v = _view(twin, taps=down, charges=down, process=down, shifts="load")
    assert len(v.hourly) == 10 and v.taps == [] and v.shifts is None and not v.hmt.available
    assert v.process.readings == [] and v.process.telemetry == "offline" and v.coke.charged_now is None
    assert v.coke.burnt == v.latest.k_burnt and v.coke.state == "ok"  # the stored window still stands
    assert all(s.newest is None for s in v.clock.sources) and "published once an hour." in v.clock.notice
    for part in ("tap temperatures", "charge record", "process tags", "FORGE view"):
        assert any(part in n for n in v.notes), part
    empty = _view(twin, now=NOW - timedelta(days=30), shifts=None)
    assert empty.hourly == [] and empty.latest is None and empty.coke.state == "unknown"
    assert any("No twin window" in n for n in empty.notes)


def test_the_route_serves_the_view_to_users(twin, monkeypatch) -> None:
    _windows(twin, hours=20)
    monkeypatch.setattr(live_model, "_taps", lambda a, b: _taps())
    monkeypatch.setattr(live_model, "_charges", lambda a, b: _charges())
    monkeypatch.setattr(live_model, "_status_view", _status)
    monkeypatch.setattr(forge, "load", lambda: forge.StoredView(meta={"built_at": NOW.isoformat()}, shifts=_shifts()))
    app = build()
    with TestClient(app, base_url="https://testserver") as client:
        cookies = {u: session_cookie(app, u) for u in PASSWORDS}
        assert client.get("/api/optimiser/live-model").status_code == 401
        r = client.get("/api/optimiser/live-model?hours=12&days=7", headers={"Cookie": cookies["op"]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["hours"] == 12 and body["shifts"]["days"] == 7 and body["process"]["telemetry"] == "live"
        assert [s["key"] for s in body["clock"]["sources"]] == ["process", "charges", "taps"]
        assert body["hmt"]["cold_tap_c"] == 1440.0 and body["coke"]["limit"] == 8.0
        assert client.get("/api/optimiser/live-model?hours=4", headers={"Cookie": cookies["op"]}).status_code == 422
        schemas = client.get("/api/openapi.json").json()["components"]["schemas"]
        for name, keys in {
            "LiveModelWindow": ("k_burnt", "valid", "top_gas_forge_c"),
            "CokeBalance": ("charged_now", "gap", "limit"),
            "HmtView": ("now", "next_tap", "p_cold_next", "band"),
        }.items():
            assert all(schemas[name]["properties"][k].get("description") for k in keys), name
