"""Implied-omega drift alarm (SV-1.2, #282): ``utils.coke_energy.drift`` and ``services.drift``.

CUSUM arithmetic on constructed series, the AR(1) estimate on simulated series, the replay of the STRATA evidence over
1 Jan - 7 Oct 2026 (acceptance: alarm dates and in-control run length) and the alarm lifecycle through the twin store
and the event bus.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import APIRouter, Depends, Request

from api.auth import require_role
from api.live import live_state
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import drift as service
from services.events import MemoryEventBus
from tests.test_api_auth import build
from tests.test_api_live import frames, live_server, next_frames, session_cookie
from utils.coke_energy import drift

ROOT = Path(__file__).resolve().parents[1]
DAY = pd.Timedelta(days=1)


def _series(values, start="2026-01-01", freq="D", tz=None) -> pd.Series:
    return pd.Series(np.asarray(values, float), index=pd.date_range(start, periods=len(values), freq=freq, tz=tz))


def _ar1(n: int, rho: float, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) * sigma
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - rho**2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


# ---------------------------------------------------------------------------------------------- CUSUM arithmetic
def test_a_constant_offset_signals_every_eleven_steps() -> None:
    s = drift.DriftSettings(lag=DAY, rho=0.0, sigma_e=1.0)
    rep = drift.evaluate(_series(np.ones(30)), s)
    assert rep.points.z.tolist() == [1.0] * 30
    assert rep.points.s_pos.iloc[:3].tolist() == [0.5, 1.0, 1.5]
    signals = list(np.flatnonzero(rep.points.signal.notna()))
    assert signals == [10, 21] and set(rep.points.signal.dropna()) == {"high"}  # S+ = 5.5 > 5, then restart
    first, second = rep.episodes  # 11 days between signals > clear_after (3 lags): two episodes
    assert (first.direction, first.raised_at, first.cleared_at) == (
        "high",
        pd.Timestamp("2026-01-11"),
        pd.Timestamp("2026-01-14"),
    )
    assert second.raised_at == pd.Timestamp("2026-01-22") and second.cleared_at == pd.Timestamp("2026-01-25")


def test_gaps_are_bridged_with_the_lagged_variance_and_restart_beyond_max_gap() -> None:
    rho, sig = 0.5, 0.01
    s = drift.DriftSettings(lag=DAY, rho=rho, sigma_e=sig, max_gap=2)
    r = pd.Series([0.01, 0.02, 0.03], index=pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-07"]))
    pts = drift.evaluate(r, s).points
    assert pts.z.iloc[0] == pytest.approx(0.01 * np.sqrt(1 - rho**2) / sig)  # a new chain: stationary
    assert pts.z.iloc[1] == pytest.approx((0.02 - rho**2 * 0.01) / (sig * np.sqrt(1 + rho**2)))  # two-day gap
    assert pts.z.iloc[2] == pytest.approx(0.03 * np.sqrt(1 - rho**2) / sig)  # four days > max_gap: restart
    assert pts.s_pos.iloc[2] == pytest.approx(max(0.0, pts.z.iloc[2] - 0.5))  # from 0, not from the earlier sum


def test_hourly_windows_form_eight_independent_chains() -> None:
    s = drift.DriftSettings(lag=pd.Timedelta(hours=8), rho=0.0, sigma_e=1.0)
    t = pd.date_range("2026-05-01", periods=8 * 20, freq="h", tz="UTC")
    r = pd.Series(np.where(np.arange(len(t)) % 8 == 3, 1.0, 0.0), index=t)  # one chain carries an offset
    pts = drift.evaluate(r, s).points
    chain = pts.iloc[3::8]
    assert (chain.s_pos > 0).all() and (pts.drop(chain.index).s_pos == 0).all()
    assert chain.signal.notna().sum() == 1 and chain.signal.dropna().index[0] == t[3 + 8 * 10]


def test_episodes_clear_after_the_last_signal() -> None:
    s = drift.DriftSettings(lag=DAY, rho=0.0, sigma_e=1.0, clear_after=pd.Timedelta(days=7))
    values = np.r_[np.full(15, 1.5), np.zeros(10)]  # S+ +1 per day: signals on days 6 and 12, then in control
    rep = drift.evaluate(_series(values), s)
    (ep,) = rep.episodes
    assert ep.raised_at == pd.Timestamp("2026-01-06") and ep.last_signal == pd.Timestamp("2026-01-12")
    assert ep.cleared_at == pd.Timestamp("2026-01-19") and ep.signals == 2
    still = drift.evaluate(_series(values[:13]), s)
    assert still.open_episodes()["high"].cleared_at is None  # last evaluation before the clear time


def test_too_few_pairs_give_no_evaluation() -> None:
    rep = drift.evaluate(_series(np.zeros(10)), drift.DriftSettings(lag=DAY, min_pairs=30))
    assert rep.ar1 is None and rep.points.empty and "need 30" in rep.note


# ---------------------------------------------------------------------------------------------- estimate and ARL
def test_the_ar1_estimate_recovers_rho_and_sigma() -> None:
    r = _series(_ar1(3000, 0.7, 0.02, seed=3))
    ar = drift.fit_ar1(r, drift.DriftSettings(lag=DAY))
    assert ar.rho == pytest.approx(0.7, abs=0.03) and ar.sigma_e == pytest.approx(0.02, rel=0.05)
    assert ar.source == "estimated" and ar.n_pairs == 2999


def test_siegmund_nominal_run_length() -> None:
    assert drift.siegmund_arl(0.5, 5.0) == pytest.approx(469, abs=1)  # two-sided, in control
    assert drift.siegmund_arl(0.5, 5.0, 1.0) < 15


def test_a_large_level_shift_is_detected_within_two_weeks_in_the_median() -> None:
    s = drift.DriftSettings(lag=DAY, max_gap=7, min_pairs=30, rho=0.693, sigma_e=0.02)
    delays = []
    for seed in range(25):
        x = _ar1(300, 0.693, 0.02, seed=seed)
        x[150:] += 0.05  # about 12 kg/tHM of coke
        pts = drift.evaluate(_series(x), s).points
        first = pts.signal.iloc[150:].first_valid_index()
        delays.append((first - pts.index[150]).days + 1 if first is not None else 999)
        assert first is None or pts.signal.loc[first] == "high"
    assert np.median(delays) <= 14 and max(delays) < 150  # simulation: median 8 days (a 0.02 shift: 70 days)


# ---------------------------------------------------------------------------------------------- replay (acceptance)
def test_replay_of_the_strata_evidence_over_2026() -> None:
    cs = json.loads((ROOT / "src/assets/burden_engine/omega_model.json").read_text(encoding="utf-8"))["channel_scales"]
    seed = pd.read_csv(ROOT / "src/assets/burden_engine/strata_evidence_seed.csv", parse_dates=["date"])
    r = drift.evidence_residuals(seed, cs)
    assert len(r) == 191
    rep = drift.evaluate(r, drift.DriftSettings(lag=DAY, max_gap=7, min_pairs=30))
    assert rep.ar1.rho == pytest.approx(0.693, abs=0.001) and rep.ar1.sigma_e == pytest.approx(0.0200, abs=0.0001)
    year = rep.points.loc["2026-01-01":"2026-10-07"]
    assert len(year) == 175 and year.signal.isna().all() and not rep.episodes  # no alarm: in control
    assert year.s_pos.idxmax() == pd.Timestamp("2026-06-23") and year.s_pos.max() == pytest.approx(2.53, abs=0.01)
    assert year.s_neg.idxmax() == pd.Timestamp("2026-07-19") and year.s_neg.max() == pytest.approx(3.26, abs=0.01)
    done, censored = drift.run_lengths(year, DAY)
    assert done == [] and censored == [175] and drift.arl_estimate(done, censored) is None


# ---------------------------------------------------------------------------------------------- service and alarms
T0 = datetime(2026, 5, 1, tzinfo=UTC)
CFG = {"twin": {"drift": {"lag_hours": 8, "rho": 0.0, "sigma_e": 0.01, "clear_after_hours": 24, "history_days": 30}}}


def _snapshots(store: TwinStore, hours: int, drift_from: int, drift_to: int, offset: float = 0.02) -> None:
    for i in range(hours):
        end = T0 + timedelta(hours=i + 1)
        r = offset if drift_from <= i < drift_to else 0.0
        store.snapshots.upsert(
            {
                "window_start": end - timedelta(hours=8),
                "window_end": end,
                "evaluated_at": end + timedelta(minutes=5),
                "status": "complete",
                "omega_imp": 0.80 + r,
                "omega_strata": 0.80,
            }
        )


@pytest.fixture()
def store(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    yield TwinStore(session_factory(engine), engine)
    engine.dispose()


def test_the_alarm_is_raised_kept_and_cleared_with_events(store) -> None:
    config = service.DriftConfig.from_config(CFG)
    _snapshots(store, 24 * 12, drift_from=24 * 5, drift_to=24 * 7)  # 2 days of r = +2 sigma_e from day 5
    bus = MemoryEventBus()

    async def at(hours: int):
        return await service.run(store, bus, config, now=T0 + timedelta(hours=hours))

    before = asyncio.run(at(24 * 5))
    assert not before.episodes and store.alarms.open_alarms() == []

    during = asyncio.run(at(24 * 5 + 30))
    (alarm,) = store.alarms.open_alarms()
    assert alarm.kind == service.KIND and alarm.subject == "high" and alarm.severity == "warning"
    assert alarm.message.startswith("Implied omega above STRATA since")
    # chain of the first drifting window (end T0 + 121 h): fourth step at T0 + 145 h, S+ = 4 x 1.5 = 6 > 5
    assert during.open_episodes()["high"].raised_at == pd.Timestamp(T0 + timedelta(hours=145))
    asyncio.run(at(24 * 5 + 31))  # still raised: updated, no second event

    asyncio.run(at(24 * 10))
    assert store.alarms.open_alarms() == []
    events = asyncio.run(bus.read("0-0", 0))
    assert [e.type for e in events] == ["alarm.raised", "alarm.cleared"]
    assert {e.data.code for e in events} == {"OMEGA_DRIFT"} and events[0].data.alarm_id == events[1].data.alarm_id
    (history,) = store.alarms.history()
    assert history.cleared_at is not None


def test_without_enough_history_the_alarms_are_left_alone(store) -> None:
    config = service.DriftConfig.from_config({"twin": {"drift": {"min_pairs": 500}}})
    _snapshots(store, 48, drift_from=0, drift_to=48, offset=0.1)
    report = asyncio.run(service.run(store, MemoryEventBus(), config, now=T0 + timedelta(hours=48)))
    assert report.ar1 is None and store.alarms.open_alarms() == []


# ---------------------------------------------------------------------------------------------- over server-sent events
def test_drift_alarms_reach_the_event_stream(store) -> None:
    """The drift service, run in the API process on its bus, raises and clears the alarm; a client of
    ``/api/stream`` receives ``alarm.raised`` and ``alarm.cleared`` (acceptance criterion of #282)."""
    config = service.DriftConfig.from_config(CFG)
    _snapshots(store, 24 * 12, drift_from=24 * 5, drift_to=24 * 7)
    app = build(stream_heartbeat_s=0.3)
    router = APIRouter(prefix="/api/_probe", dependencies=[Depends(require_role("admin"))])

    @router.post("/drift")
    async def run_drift(body: dict, request: Request) -> dict:
        report = await service.run(store, live_state(request).bus, config, now=T0 + timedelta(hours=body["hours"]))
        return {"episodes": len(report.episodes)}

    app.include_router(router)
    cookie = session_cookie(app, "adm")
    with live_server(app) as base, httpx.Client(base_url=base, headers={"Cookie": cookie}, timeout=5) as http:
        with http.stream("GET", "/api/stream") as response:
            stream = frames(response)
            assert next(stream) == {"retry": "3000"}
            assert http.post("/api/_probe/drift", json={"hours": 24 * 5 + 30}).status_code == 200
            assert http.post("/api/_probe/drift", json={"hours": 24 * 10}).status_code == 200
            raised, cleared = next_frames(stream, 2)
    assert (raised["event"], cleared["event"]) == ("alarm.raised", "alarm.cleared")
    data = [json.loads(f["data"])["data"] for f in (raised, cleared)]
    assert {d["code"] for d in data} == {"OMEGA_DRIFT"} and data[0]["alarm_id"] == data[1]["alarm_id"]
    assert data[0]["severity"] == "warning" and data[0]["message"].startswith("Implied omega above STRATA")


def test_the_replay_script_reports_alarms_and_the_run_length(tmp_path, capsys) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("drift_replay", ROOT / "scripts" / "drift_replay.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    out = tmp_path / "replay.json"
    assert replay.main(["--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "175 evaluations 2026-01-01 to 2026-10-07, lag 1 day" in text
    assert "largest S+ 2.53 on 2026-06-23" in text and "alarm episodes: 0" in text
    assert "in-control ARL > 175 (no signal); nominal ARL for iid N(0, 1) innovations 469" in text
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["episodes"] == [] and report["run_lengths"]["censored"] == [175] and len(report["points"]) == 175
