"""Shift handover records (SV-2.5, #297): ``shift.close`` freezes the closed shift's twin snapshots, its latest plan
decision and its report link into ``twin.shift_record`` once, run as the worker runs it on a SQLite twin store (and,
for the trigger that refuses edits, on a migrated PostgreSQL database)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import Actor, TwinStore
from services import shift_records
from services import store as store_service
from services.events import MemoryEventBus
from services.jobs import CATALOG, JobRecord, ShiftCloseParams, input_hash
from tests.pg_server import fresh_database, needs_postgres, postgres_server
from worker.kinds import shift as shift_kind
from worker.runtime import JobContext, SkipJob, TransientJobError

IST = "Asia/Kolkata"
H = timedelta(hours=1)
SUPERVISOR = Actor("sv1", "supervisor")
KIND = CATALOG["shift.close"]


def ist(value: str) -> datetime:
    return pd.Timestamp(value, tz=IST).tz_convert("UTC").to_pydatetime()


class FakeJobs:
    """The parts of ``JobStore`` the handler uses."""

    def __init__(self, rec: JobRecord | None) -> None:
        self.rec = rec

    async def get(self, job_id: str) -> JobRecord | None:
        return self.rec

    async def update(self, job_id: str, **changes: Any) -> None:
        return None


def run_close(slot: datetime | None, *, attempt: int = 1, by: str = "scheduler", **params: Any) -> dict[str, Any]:
    rec = JobRecord(
        id=uuid.uuid4().hex,
        kind="shift.close",
        status="running",
        params=params,
        input_hash=input_hash("shift.close", params),
        submitted_by=by,
        submitted_at=slot or datetime.now(UTC),
        scheduled_for=slot,
    )
    job = JobContext(job_id=rec.id, kind=KIND, attempt=attempt, store=FakeJobs(rec), bus=MemoryEventBus(), pool=None)
    return asyncio.run(shift_kind.close(job, ShiftCloseParams(**params)))


def snapshot(end: datetime, k: float, **extra: Any) -> dict[str, Any]:
    return {
        "window_start": end - 8 * H,
        "window_end": end,
        "evaluated_at": end + timedelta(minutes=12),
        "status": "complete",
        "k_trace": k,
        "k_forge": k - 4.0,
        "k_strata": k - 2.5,
        "k_strata_sd": 5.1,
        "k_charged": k + 6.0,
        "omega_imp": 0.912,
        "omega_strata": 0.905,
        "quantities": {"production_t_h": 262.5, "hmt_c": 1478.0},
        "flags": {"notes": []},
        "versions": {"strata": "1.5"},
        **extra,
    }


def run_row(at: datetime) -> dict[str, Any]:
    return {
        "trigger": "schedule",
        "requested_by": "scheduler",
        "mode": "step2",
        "omega_mode": "strata",
        "started_at": at,
        "finished_at": at + timedelta(seconds=74),
        "status": "ok",
        "production_t_d": 6150.0,
        "cost_rs_thm": 24810.5,
        "coke_kg_thm": 371.2,
        "certainty": "Medium",
    }


@pytest.fixture()
def store(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    twin = TwinStore(session_factory(engine), engine)
    store_service.use(twin)
    yield twin
    store_service.reset()
    twin.dispose()


# Shift C of 26 Oct: 22:00 IST on the 26th to 06:00 IST on the 27th; the job's slot is 06:10 IST on the 27th.
C_START, C_END = ist("2026-10-26 22:00"), ist("2026-10-27 06:00")
SLOT = ist("2026-10-27 06:10")


def fill_shift_c(store: TwinStore) -> None:
    for i in range(0, 9):  # windows ending 22:00 (shift B's, not C's) to 06:00
        store.snapshots.upsert(snapshot(C_START + i * H, 400.0 + i))


def test_the_slot_freezes_the_closed_shift_with_its_snapshots_decision_and_report_link(store) -> None:
    fill_shift_c(store)
    early, late = store.optimiser.add(run_row(C_START - 3 * H)), store.optimiser.add(run_row(C_START + H))
    store.optimiser.decide(early.id, "accepted", SUPERVISOR, at=C_START - 2 * H)  # shift B's decision
    store.optimiser.decide(late.id, "accepted", SUPERVISOR, at=C_START + 2 * H)
    store.optimiser.decide(late.id, "rejected", SUPERVISOR, at=C_START + 5 * H, comment="pellet short")
    store.optimiser.decide(early.id, "accepted", SUPERVISOR, at=C_END + timedelta(minutes=5))  # next shift's

    out = run_close(SLOT, backfill_shifts=0)
    assert out["shift_key"] == "2026-10-26_C" and out["frozen"] == ["2026-10-26_C"] and out["shift_window"]
    row = store.shifts.get("2026-10-26_C")
    assert row.letter == "C" and shift_records.summary_data(row)["shift_start"] == C_START
    assert (row.k_trace, row.k_forge, row.k_strata, row.k_charged, row.omega_imp) == (408.0, 404.0, 405.5, 414.0, 0.912)
    hourly = row.twin["hourly"]
    assert [pd.Timestamp(h["window_end"]) for h in hourly] == [pd.Timestamp(C_START + i * H) for i in range(1, 9)]
    assert row.twin["missing_windows"] == [] and hourly[-1]["quantities"]["hmt_c"] == 1478.0
    assert (str(row.plan_run_id), row.plan_decision) == (str(late.id), "rejected")  # the latest decision in the shift
    assert row.twin["plan"]["decision"]["comment"] == "pellet short" and row.twin["plan"]["run"]["mode"] == "step2"
    assert row.report_ref == "/reports?date=2026-10-26&shift=C"
    assert row.twin["frozen_by"]["submitted_by"] == "scheduler"


def test_the_job_is_idempotent_per_shift(store) -> None:
    fill_shift_c(store)
    first = run_close(SLOT, backfill_shifts=0)
    frozen = store.shifts.get("2026-10-26_C")
    store.snapshots.upsert(snapshot(C_END, 999.0))  # a later re-evaluation of the window changes nothing frozen
    again = run_close(SLOT, backfill_shifts=0)
    by_hand = run_close(None, by="sv1", shift_key="2026-10-26_C", backfill_shifts=0)
    assert first["frozen"] == ["2026-10-26_C"] and again["frozen"] == [] and by_hand["frozen"] == []
    assert again["existing"] == ["2026-10-26_C"] == by_hand["existing"]
    row = store.shifts.get("2026-10-26_C")
    assert row.k_trace == 408.0 and row.closed_at == frozen.closed_at
    assert len(store.shifts.recent(limit=10)) == 1


def test_the_job_waits_for_the_shift_window_and_its_last_try_freezes_what_there_is(store) -> None:
    for i in range(1, 7):  # windows ending 23:00 to 04:00; 05:00 and the shift's own window (06:00) are missing
        store.snapshots.upsert(snapshot(C_START + i * H, 400.0 + i))
    for attempt in range(1, KIND.max_tries):
        with pytest.raises(TransientJobError, match="2026-10-26_C"):
            run_close(SLOT, attempt=attempt, backfill_shifts=0)
    assert store.shifts.get("2026-10-26_C") is None
    out = run_close(SLOT, attempt=KIND.max_tries, backfill_shifts=0)
    row = store.shifts.get("2026-10-26_C")
    assert out["frozen"] == ["2026-10-26_C"] and not out["shift_window"]
    assert row.k_trace is None and len(row.twin["hourly"]) == 6 and row.twin["shift_window_end"] is None
    assert [pd.Timestamp(t) for t in row.twin["missing_windows"]] == [pd.Timestamp(C_END - H), pd.Timestamp(C_END)]
    assert row.plan_run_id is None and row.plan_decision is None
    # The schedule reaches HH:30 before the last try: 5 tries 5 min apart from HH:10.
    assert KIND.schedule == ({6, 14, 22}, {10}) and (KIND.max_tries - 1) * KIND.retry_delay_s == 20 * 60


def test_shifts_before_without_a_record_are_frozen_first(store) -> None:
    fill_shift_c(store)
    store.shifts.freeze(shift_records.record_values(store, shift_records.Shift.parse("2026-10-26_A")))
    out = run_close(SLOT)  # backfill of 3: 25 Oct C, 26 Oct A (recorded), 26 Oct B
    assert out["frozen"] == ["2026-10-25_C", "2026-10-26_B", "2026-10-26_C"]
    keys = [r.shift_key for r in store.shifts.recent(limit=10)]
    assert keys == ["2026-10-26_C", "2026-10-26_B", "2026-10-26_A", "2026-10-25_C"]
    assert store.shifts.get("2026-10-26_B").k_trace == 400.0  # its window (ending 22:00) is stored


def test_one_record_per_shift_over_14_days_with_missed_and_repeated_slots(store) -> None:
    start = ist("2026-10-01 06:00")
    for i in range(14 * 24 + 17):  # windows ending 01 Oct 06:00 to 15 Oct 22:00
        store.snapshots.upsert(snapshot(start + i * H, 380.0 + i % 24))
    slots = [ist(f"2026-10-{d:02d} {h:02d}:10") for d in range(2, 16) for h in (6, 14, 22)]
    for n, slot in enumerate(slots):
        if n % 7 == 3:  # the worker was down at this slot: the next one freezes the shift
            continue
        run_close(slot)
        if n % 5 == 0:
            run_close(slot)  # a catch-up or a second worker
    expected = ["2026-10-01_C", *[f"2026-10-{d:02d}_{s}" for d in range(2, 15) for s in "ABC"], "2026-10-15_A",
                "2026-10-15_B"]  # fmt: skip
    rows = store.shifts.between(ist("2026-10-01 22:00"), ist("2026-10-15 22:00"))
    assert [r.shift_key for r in rows] == expected and len(rows) == 42
    k_of_window_end = {"A": 388.0, "B": 396.0, "C": 380.0}  # 14:00, 22:00, 06:00 IST: 8, 16, 0 h after 06:00
    assert all(r.twin["shift_window_end"] and r.k_trace == k_of_window_end[r.letter] for r in rows)
    assert all(len(r.twin["hourly"]) == 8 and r.report_ref.endswith(f"shift={r.letter}") for r in rows)


def test_a_shift_that_has_not_ended_and_a_missing_store_are_refused(store) -> None:
    with pytest.raises(ValueError, match="has not ended"):
        run_close(None, by="sv1", shift_key="2030-01-01_A")
    store_service.use(None)
    with pytest.raises(SkipJob, match="BF_TWIN_DB_URL"):
        run_close(SLOT)


def test_the_closed_shift_of_each_slot() -> None:
    assert shift_records.closed_shift(ist("2026-10-27 06:10")).key == "2026-10-26_C"
    assert shift_records.closed_shift(ist("2026-10-27 14:10")).key == "2026-10-27_A"
    assert shift_records.closed_shift(ist("2026-10-27 22:10")).key == "2026-10-27_B"
    start, end = shift_records.bounds(shift_records.Shift.parse("2026-10-26_C"))
    assert (start, end) == (C_START, C_END) and end - start == 8 * H
    assert shift_records.day_bounds(pd.Timestamp("2026-10-27").date()) == (
        ist("2026-10-27 00:00"),
        ist("2026-10-28 00:00"),
    )


# ------------------------------------------------------------------------------------------------- PostgreSQL
@pytest.fixture(scope="module")
def pg_url():
    from alembic import command

    from tests.test_twin_store import alembic_config

    with postgres_server() as server, fresh_database(server) as url:
        command.upgrade(alembic_config(url), "head")
        yield url


@needs_postgres
def test_the_frozen_record_cannot_be_changed_in_the_database(pg_url) -> None:
    twin = TwinStore.from_url(pg_url)
    store_service.use(twin)
    try:
        fill_shift_c(twin)
        run = twin.optimiser.add(run_row(C_START + H))
        twin.optimiser.decide(run.id, "accepted", SUPERVISOR, at=C_START + 2 * H)
        assert run_close(SLOT, backfill_shifts=0)["frozen"] == ["2026-10-26_C"]
        row = twin.shifts.get("2026-10-26_C")
        assert row.k_trace == 408.0 and len(row.twin["hourly"]) == 8 and row.plan_decision == "accepted"
        for sql in ("UPDATE twin.shift_record SET k_trace = 1", "DELETE FROM twin.shift_record"):
            with pytest.raises(IntegrityError, match="written once"), twin.engine.begin() as connection:
                connection.execute(text(sql))
        assert twin.shifts.get("2026-10-26_C").k_trace == 408.0
        assert run_close(SLOT, backfill_shifts=0)["existing"] == ["2026-10-26_C"]
    finally:
        store_service.reset()
        twin.dispose()
