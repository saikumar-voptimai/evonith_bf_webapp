"""Local buffering of the worker's twin-store writes (SV-6.5, #319): the SQLite outbox, the buffered store on a
failing Neon, the ordered and idempotent replay, and ``twin.evaluate`` through a 1-h outage without a lost snapshot."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.outbox import Outbox, decode, encode
from data.twin.repositories import TwinStore
from services import outbox as outbox_service
from services import source_health, twin
from services.jobs import TwinEvaluateParams
from services.outbox import BufferedTwinStore, is_outage
from tests.test_scheduled_jobs import context, record, utc_of
from utils.coke_energy.twin import Window, WindowEvaluation
from worker.kinds import twin as twin_kind

T0 = datetime(2026, 10, 27, 0, 0, tzinfo=UTC)
H = timedelta(hours=1)


def refused(message: str = "connection refused") -> OperationalError:
    return OperationalError("SELECT 1", {}, ConnectionRefusedError(message))


class _Repo:
    """A repository that raises an outage while ``owner.down``."""

    def __init__(self, owner: OutageStore, inner: Any) -> None:
        self._owner, self._inner = owner, inner

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            if self._owner.down:
                self._owner.refusals += 1
                raise refused()
            self._owner.calls.append(name)
            return attr(*args, **kwargs)

        return call


class OutageStore:
    """A SQLite ``TwinStore`` whose queries fail like a Neon outage while ``down`` is set."""

    def __init__(self, inner: TwinStore) -> None:
        self.inner, self.down, self.refusals, self.calls = inner, False, 0, []
        self.snapshots = _Repo(self, inner.snapshots)
        self.jobs = _Repo(self, inner.jobs)
        self.alarms = _Repo(self, inner.alarms)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@pytest.fixture()
def neon(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'neon.db'}")
    TwinBase.metadata.create_all(engine)
    store = OutageStore(TwinStore(session_factory(engine), engine))
    yield store
    engine.dispose()


@pytest.fixture()
def buffered(neon, tmp_path):
    clock = {"t": 1000.0}
    store = BufferedTwinStore(neon, Outbox(tmp_path / "outbox.sqlite"), retry_after_s=30.0, clock=lambda: clock["t"])
    store.tick = lambda s: clock.__setitem__("t", clock["t"] + s)  # type: ignore[attr-defined]
    return store


def snapshot(k: int, **extra: Any) -> dict[str, Any]:
    end = T0 + k * H
    return {
        "window_start": end - 8 * H,
        "window_end": end,
        "evaluated_at": end + timedelta(minutes=12),
        "status": "complete",
        "k_trace": 500.0 + k,
        "quantities": {"hm_t": 780.5, "nan_value": float("nan")},
        "flags": {"sources": {"influxdb": end.isoformat()}, "n_hours": 8},
        **extra,
    }


# ------------------------------------------------------------------------------------------------- the outbox
def test_outbox_keeps_order_types_and_survives_reopening(tmp_path) -> None:
    path = tmp_path / "outbox.sqlite"
    box = Outbox(path)
    payload = {"window_end": T0, "run_id": uuid.UUID(int=7), "ratio": float("nan"), "day": T0.date(), "n": 3}
    first = box.append("snapshot.upsert", str(T0), payload)
    second = box.append("job.upsert", "j1", {"id": "j1", "status": "done"})
    assert (first, second) == (1, 2) and box.pending() == 2
    del box  # no close: the file is the store
    again = Outbox(path)
    entries = again.oldest()
    assert [e.op for e in entries] == ["snapshot.upsert", "job.upsert"]
    assert entries[0].payload == {"window_end": T0, "run_id": uuid.UUID(int=7), "ratio": None, "day": T0.date(), "n": 3}
    assert entries[0].queued_at.tzinfo is not None
    again.mark_failed(1, "x" * 1000)
    assert again.oldest(1)[0].tries == 1 and len(again.oldest(1)[0].last_error) == 400
    again.delete(1)
    assert again.pending() == 1 and again.oldest()[0].seq == 2
    assert again.oldest_queued_at() == entries[1].queued_at


def test_encoding_round_trips_pandas_and_numpy_values() -> None:
    import numpy as np

    value = {"ts": pd.Timestamp("2026-10-27 05:30", tz="UTC"), "x": np.float64(1.5), "s": {1, 2}, "inf": float("inf")}
    out = decode(encode(value))
    assert out == {"ts": datetime(2026, 10, 27, 5, 30, tzinfo=UTC), "x": 1.5, "s": [1, 2], "inf": None}
    with pytest.raises(TypeError):
        encode({"bad": object()})


def test_is_outage_classifies_connection_errors_only() -> None:
    assert is_outage(refused()) and is_outage(ConnectionResetError()) and is_outage(TimeoutError())
    assert not is_outage(IntegrityError("x", {}, Exception())) and not is_outage(ValueError("bad payload"))
    assert not is_outage(ProgrammingError("x", {}, Exception("relation does not exist")))


# ---------------------------------------------------------------------------------------- the buffered store
def test_writes_go_direct_while_neon_answers(buffered, neon) -> None:
    row = buffered.snapshots.upsert(snapshot(1))
    assert row is not None and row.k_trace == 501.0 and buffered.pending == 0
    assert buffered.snapshots.latest().window_end.replace(tzinfo=UTC) == T0 + H  # reads are the store's
    assert buffered.alarms.open_alarms() == [] and buffered.revision() is None


def test_an_outage_buffers_in_order_and_the_replay_restores_everything(buffered, neon) -> None:
    buffered.snapshots.upsert(snapshot(1))
    neon.down = True
    assert buffered.snapshots.upsert(snapshot(2)) is None  # the first failed write
    assert neon.refusals == 1 and buffered.holding and buffered.down_since is not None
    for k in range(3, 6):  # the hour's later writes go straight to the outbox: no connect timeout each
        buffered.snapshots.upsert(snapshot(k))
    buffered.jobs.upsert({"id": "job-1", "kind": "twin.evaluate", "status": "done", "submitted_by": "scheduler",
                          "submitted_at": T0, "input_hash": "h"})  # fmt: skip
    assert neon.refusals == 1 and buffered.pending == 5
    buffered.tick(31)  # the hold ends; Neon still down: one probe, then the outbox again
    buffered.snapshots.upsert(snapshot(6))
    assert neon.refusals == 2 and buffered.pending == 6
    assert buffered.replay().error and buffered.pending == 6  # no replay while down (nothing was lost)

    neon.down = False
    buffered.tick(31)
    report = buffered.replay()
    assert report.as_dict() == {"replayed": 6, "dropped": 0, "pending": 0, "error": None}
    rows = buffered.snapshots.series(T0, T0 + 10 * H)
    assert [r.k_trace for r in rows] == [501.0, 502.0, 503.0, 504.0, 505.0, 506.0]
    assert rows[2].evaluated_at.replace(tzinfo=UTC) == T0 + 3 * H + timedelta(minutes=12)  # the original time
    assert rows[2].quantities == {"hm_t": 780.5, "nan_value": None} and rows[2].flags["n_hours"] == 8
    assert buffered.jobs.get("job-1").status == "done"
    assert buffered.down_since is None and not buffered.holding and buffered.replayed == 6
    assert buffered.snapshots.upsert(snapshot(7)) is not None  # direct again


def test_a_write_after_recovery_replays_the_queue_first(buffered, neon) -> None:
    neon.down = True
    buffered.snapshots.upsert(snapshot(1, k_trace=1.0))
    buffered.snapshots.upsert(snapshot(1, k_trace=2.0))  # the same window again, later
    neon.down = False
    buffered.tick(31)
    assert buffered.snapshots.upsert(snapshot(2)) is not None  # replays 1.0, 2.0, then writes 2 directly
    assert buffered.pending == 0
    assert [r.k_trace for r in buffered.snapshots.series(T0, T0 + 3 * H)] == [2.0, 502.0]


def test_replay_is_idempotent_and_keeps_order_at_a_failure(buffered, neon) -> None:
    neon.down = True
    for k in (1, 2, 3):
        buffered.snapshots.upsert(snapshot(k))
    neon.down = False
    buffered.tick(31)
    entries = buffered.outbox.oldest()
    neon.inner.snapshots.upsert(entries[0].payload)  # as if the process died after the commit, before the delete
    assert buffered.replay().replayed == 3 and buffered.pending == 0
    assert [r.k_trace for r in buffered.snapshots.series(T0, T0 + 4 * H)] == [501.0, 502.0, 503.0]
    # a refused entry (not an outage) stops the replay so later writes never overtake it, until it is dropped
    buffered.outbox.append("snapshot.upsert", "bad", {"window_end": T0 + 9 * H})  # lacks window_start: ValueError
    buffered.outbox.append("snapshot.upsert", str(T0 + 5 * H), snapshot(5))
    for _ in range(outbox_service.POISON_TRIES - 1):
        report = buffered.replay()
        assert report.replayed == 0 and report.pending == 2 and "snapshot needs" in report.error
    report = buffered.replay()
    assert report.as_dict() == {"replayed": 1, "dropped": 1, "pending": 0, "error": None}


def test_the_process_store_and_the_heartbeat_replay(buffered, neon) -> None:
    outbox_service.use(buffered)
    try:
        assert outbox_service.replay_pending() is None  # nothing queued
        neon.down = True
        buffered.snapshots.upsert(snapshot(1))
        assert outbox_service.replay_pending() is None  # within the hold: no probe
        buffered.tick(31)
        assert outbox_service.replay_pending().error  # probed, still down
        neon.down = False
        buffered.tick(31)
        assert outbox_service.replay_pending().replayed == 1
        health = outbox_service.outbox_health()
        assert health["pending"] == 0 and health["replayed"] == 1 and health["down_since"] is None
    finally:
        outbox_service.reset()


# ------------------------------------------------------------------------------------- twin.evaluate in an outage
def stub_evaluation(t: Any, strata: bool = True) -> WindowEvaluation:
    window = Window.ending(pd.Timestamp(t), 8)
    sources = {"influxdb": utc_of(str(window.end)).isoformat()}
    return WindowEvaluation(window, "complete", k_trace=500.0, k_forge=498.0, omega_imp=0.9,
                            flags={"n_hours": 8, "sources": sources})  # fmt: skip


def test_a_one_hour_neon_outage_loses_no_hourly_snapshot(buffered, neon, monkeypatch) -> None:
    monkeypatch.setattr(twin, "evaluate", stub_evaluation)
    monkeypatch.setattr(twin, "ensure_dataset", lambda end: (True, twin.DatasetState(end - twin.HOUR, end)))
    source_health.HEALTH.reset()
    outbox_service.use(buffered)
    try:
        params = TwinEvaluateParams(backfill_hours=2, strata=False)
        slots = [utc_of(f"2026-05-15 {h:02d}:10") for h in (9, 10, 11, 12)]
        job, _ = context("twin.evaluate", record("twin.evaluate", slot=slots[0]))
        out = asyncio.run(twin_kind.evaluate(job, params))
        assert out["stored"] and out["buffered"] == 0 and out["backfilled"] == 2

        neon.down = True  # 10:10 and 11:10 run during the outage
        for slot in slots[1:3]:
            buffered.tick(31)
            job, bus = context("twin.evaluate", record("twin.evaluate", slot=slot))
            out = asyncio.run(twin_kind.evaluate(job, params))
            assert out["stored"] and out["backfilled"] == 0  # the backfill lookup is skipped, noted
            assert any("backfill lookup skipped" in n for n in out["notes"])
            assert any("drift and alarms skipped" in n for n in out["notes"])
            events = [e.type for e in asyncio.run(bus.recent(20))]
            assert "twin.snapshot" in events and "alarm.raised" not in events
        assert out["buffered"] == 4 and buffered.pending == 4  # 2 windows and 2 job records, in order
        assert [e.op for e in buffered.outbox.oldest()] == ["job.upsert", "snapshot.upsert"] * 2

        neon.down = False  # Neon is back for the 12:10 run; the heartbeat has not replayed yet
        buffered.tick(31)
        job, _ = context("twin.evaluate", record("twin.evaluate", slot=slots[3]))
        out = asyncio.run(twin_kind.evaluate(job, params))
        assert out["buffered"] == 0 and buffered.pending == 0 and out["backfilled"] == 0
        rows = neon.inner.snapshots.series(utc_of("2026-05-15 07:00"), utc_of("2026-05-15 12:00"))
        ends = [r.window_end.replace(tzinfo=UTC) for r in rows]
        assert ends == [utc_of(f"2026-05-15 {h:02d}:00") for h in (7, 8, 9, 10, 11, 12)]
        assert neon.inner.jobs.get(job.job_id).kind == "twin.evaluate"
    finally:
        outbox_service.reset()
        source_health.HEALTH.reset()
