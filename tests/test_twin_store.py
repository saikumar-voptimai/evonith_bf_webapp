"""The ``twin`` schema (#278): migration up, down and up again; models equal to the migration; time indexes;
append-only tables; the repositories; the job archive of the worker; the models on SQLite (outage buffer, SV-6.5).

Needs a PostgreSQL server (``tests/pg_server.py``: a throw-away local cluster, or ``BF_TEST_DATABASE_URL``); the archive
test also needs redis-server. Each module works in a database of its own.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import CheckConstraint, UniqueConstraint, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool

from data.twin import Actor, AlreadyRecorded, NotFound, ReviewRefused, TwinStore
from data.twin.engine import build_engine, session_factory
from data.twin.models import APPEND_ONLY, SCHEMA, TIME_INDEXED, TwinBase
from tests.pg_server import fresh_database, needs_postgres, postgres_server
from tests.redis_server import needs_redis, redis_server

REPO = Path(__file__).resolve().parents[1]
TABLES = (
    "jobs",
    "snapshot",
    "optimiser_run",
    "plan_decision",
    "shift_record",
    "alarm_event",
    "facts_version",
    "audit_log",
)
T0 = datetime(2026, 10, 27, 0, 30, tzinfo=UTC)  # 06:00 IST, start of shift A
H = timedelta(hours=1)
ALICE = Actor("alice", "admin", "10.0.0.5")
BOB = Actor("bob", "admin")
SUPERVISOR = Actor("sv1", "supervisor", "10.0.0.7")


def alembic_config(url: str) -> Config:
    config = Config(str(REPO / "alembic.ini"), ini_section="twin")
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def job_columns(url: str, table: str = "jobs") -> set[str]:
    engine = create_engine(url, poolclass=NullPool)
    try:
        return {c["name"] for c in inspect(engine).get_columns(table, schema=SCHEMA)}
    finally:
        engine.dispose()


def twin_tables(url: str) -> set[str]:
    engine = create_engine(url, poolclass=NullPool)
    try:
        return set(inspect(engine).get_table_names(schema=SCHEMA))
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def pg_server():
    with postgres_server() as url:
        yield url


@pytest.fixture(scope="module")
def db_url(pg_server):
    with fresh_database(pg_server) as url:
        command.upgrade(alembic_config(url), "head")
        yield url


@pytest.fixture()
def store(db_url):
    twin = TwinStore.from_url(db_url)
    assert twin is not None and twin.engine is not None
    yield twin
    with twin.engine.begin() as connection:  # TRUNCATE is not a row change: the append-only triggers allow it
        connection.execute(text(f"TRUNCATE {', '.join(f'{SCHEMA}.{t}' for t in TABLES)} RESTART IDENTITY CASCADE"))
    twin.dispose()


# ------------------------------------------------------------------------------------------------------- migration


@needs_postgres
def test_migration_upgrades_downgrades_and_upgrades_again(pg_server) -> None:
    with fresh_database(pg_server) as url:
        config = alembic_config(url)
        command.upgrade(config, "head")
        assert twin_tables(url) == {*TABLES, "alembic_version"}
        assert TwinStore.from_url(url).revision() == "twin_0003"
        assert {"scheduled_for", "run_s"} <= job_columns(url)
        assert {"before", "after"} <= job_columns(url, "audit_log")

        command.downgrade(config, "-1")  # twin_0003: the values before and after of the audit log
        assert TwinStore.from_url(url).revision() == "twin_0002"
        assert not {"before", "after"} & job_columns(url, "audit_log")

        command.downgrade(config, "-1")  # twin_0002: the slot and execution time of the jobs
        assert TwinStore.from_url(url).revision() == "twin_0001"
        assert twin_tables(url) == {*TABLES, "alembic_version"}
        assert not {"scheduled_for", "run_s"} & job_columns(url)

        command.downgrade(config, "base")
        assert twin_tables(url) == {"alembic_version"}  # the schema and Alembic's table stay
        engine = create_engine(url, poolclass=NullPool)
        with engine.connect() as connection:
            functions = connection.execute(
                text("SELECT proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace WHERE nspname = :s"),
                {"s": SCHEMA},
            ).all()
        engine.dispose()
        assert functions == []
        assert TwinStore.from_url(url).revision() is None

        command.upgrade(config, "head")
        assert twin_tables(url) == {*TABLES, "alembic_version"}


@needs_postgres
def test_models_equal_the_migration(db_url) -> None:
    engine = create_engine(db_url, poolclass=NullPool)
    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={
                "include_schemas": True,
                "include_name": lambda name, type_, _parents: (
                    name == SCHEMA if type_ == "schema" else name != "alembic_version"
                ),
                "compare_type": True,
                "compare_server_default": True,
            },
        )
        diff = compare_metadata(context, TwinBase.metadata)
    engine.dispose()
    assert diff == []


@needs_postgres
def test_constraint_and_index_names_equal_the_models(db_url) -> None:
    """``compare_metadata`` does not compare check constraints or names: the migration must give every constraint
    and index the name the models' naming convention gives it (``ck_jobs_status``, never ``ck_jobs_ck_jobs_status``)."""
    engine = create_engine(db_url, poolclass=NullPool)
    inspector = inspect(engine)
    for table in TwinBase.metadata.sorted_tables:
        name = table.name
        in_db = {
            "pk": inspector.get_pk_constraint(name, schema=SCHEMA)["name"],
            "ck": sorted(c["name"] for c in inspector.get_check_constraints(name, schema=SCHEMA)),
            "uq": sorted(c["name"] for c in inspector.get_unique_constraints(name, schema=SCHEMA)),
            "fk": sorted(c["name"] for c in inspector.get_foreign_keys(name, schema=SCHEMA)),
            "ix": sorted(
                i["name"] for i in inspector.get_indexes(name, schema=SCHEMA) if not i.get("duplicates_constraint")
            ),
        }
        in_models = {
            "pk": table.primary_key.name,
            "ck": sorted(c.name for c in table.constraints if isinstance(c, CheckConstraint)),
            "uq": sorted(c.name for c in table.constraints if isinstance(c, UniqueConstraint)),
            "fk": sorted(c.name for c in table.foreign_key_constraints),
            "ix": sorted(i.name for i in table.indexes),
        }
        assert in_db == in_models, name
    engine.dispose()


@needs_postgres
def test_time_columns_lead_an_index(db_url) -> None:
    engine = create_engine(db_url, poolclass=NullPool)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT t.relname, a.attname FROM pg_index i "
                "JOIN pg_class t ON t.oid = i.indrelid JOIN pg_namespace n ON n.oid = t.relnamespace "
                "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = i.indkey[0] WHERE n.nspname = :s"
            ),
            {"s": SCHEMA},
        ).all()
    engine.dispose()
    leading = {(table, column) for table, column in rows}
    missing = [(t, c) for t, columns in TIME_INDEXED.items() for c in columns if (t, c) not in leading]
    assert missing == []
    assert set(TIME_INDEXED) == set(TABLES)


@needs_postgres
def test_append_only_rows_cannot_be_changed_or_deleted(store) -> None:
    run = store.optimiser.add(_run(T0))
    store.optimiser.decide(run.id, "accepted", SUPERVISOR, at=T0 + H)
    store.shifts.freeze(_shift("2026-10-27_A", T0))
    statements = {
        "shift_record": ("UPDATE twin.shift_record SET k_trace = 1", "DELETE FROM twin.shift_record"),
        "plan_decision": ("UPDATE twin.plan_decision SET comment = 'x'", "DELETE FROM twin.plan_decision"),
        "audit_log": ("UPDATE twin.audit_log SET actor = 'x'", "DELETE FROM twin.audit_log"),
    }
    assert set(statements) == set(APPEND_ONLY)
    for table, sqls in statements.items():
        for sql in sqls:
            with pytest.raises(IntegrityError, match="written once"), store.engine.begin() as connection:
                connection.execute(text(sql))
        with store.engine.connect() as connection:
            assert connection.execute(text(f"SELECT count(*) FROM twin.{table}")).scalar_one() >= 1
    # Deleting the run would change the decision rows (cascade) and is refused as well.
    with pytest.raises(IntegrityError), store.engine.begin() as connection:
        connection.execute(text("DELETE FROM twin.optimiser_run"))


# ----------------------------------------------------------------------------------------------------- repositories


def _job(job_id: str, status: str, at: datetime, **extra) -> dict:
    return {
        "id": job_id,
        "kind": "twin.evaluate",
        "status": status,
        "params": {"window_end": at.isoformat()},
        "input_hash": "abc123",
        "submitted_by": "scheduler",
        "submitted_at": at,
        "progress": 0.5,  # not archived
        "cancel_requested": False,  # not archived
        **extra,
    }


def _run(at: datetime, **extra) -> dict:
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
        "saving_rs_thm": 182.4,
        "coke_kg_thm": 371.2,
        "coke_sd_kg_thm": 5.9,
        "certainty": "Medium",
        "result": {"blend": {"sinter": 0.68, "pellet": 0.175}},
        **extra,
    }


def _shift(key: str, start: datetime, **extra) -> dict:
    return {
        "shift_key": key,
        "shift_start": start,
        "shift_end": start + 8 * H,
        "closed_at": start + 8 * H + timedelta(minutes=10),
        "k_trace": 370.7,
        "twin": {"hours": 8},
        **extra,
    }


@needs_postgres
def test_job_records_are_upserted_and_listed(store) -> None:
    store.jobs.upsert(_job("j1", "queued", T0))
    store.jobs.upsert_many(
        [
            _job("j1", "running", T0, started_at=T0 + timedelta(seconds=2), tries=1),
            _job("j2", "queued", T0 + H),
        ]
    )
    store.jobs.upsert(
        _job(
            "j1",
            "done",
            T0,
            started_at=T0 + timedelta(seconds=2),
            finished_at=T0 + 20 * timedelta(seconds=1),
            tries=1,
            result={"k_trace": 370.7},
        ),
        at=T0 + H,
    )
    j1 = store.jobs.get("j1")
    assert j1 is not None
    assert (j1.status, j1.tries, j1.result, j1.archived_at) == ("done", 1, {"k_trace": 370.7}, T0 + H)
    assert j1.params == {"window_end": T0.isoformat()}
    assert [j.id for j in store.jobs.recent()] == ["j2", "j1"]  # newest first
    assert [j.id for j in store.jobs.recent(status="done")] == ["j1"]
    assert [j.id for j in store.jobs.recent(since=T0 + H)] == ["j2"]
    with pytest.raises(ValueError, match="timezone-aware"):
        store.jobs.upsert(_job("j3", "queued", datetime(2026, 10, 27, 6)))
    with pytest.raises(IntegrityError):
        store.jobs.upsert(_job("j4", "paused", T0))


@needs_postgres
def test_jobs_of_a_period_include_the_slots_served_in_it(store) -> None:
    """``jobs.window``: submitted in the period, or serving a slot in it (a slot recorded as skipped at a later
    worker start); with the slot and the execution time over the tries (twin_0002)."""
    store.jobs.upsert_many(
        [
            _job("in", "done", T0 + H, scheduled_for=T0 + H, run_s=12.5),
            _job("late", "cancelled", T0 + 30 * H, scheduled_for=T0 + 2 * H),  # recorded the next day
            _job("before", "done", T0 - H),
            _job("user", "done", T0 + 3 * H),
        ]
    )
    rows = store.jobs.window(T0, T0 + 24 * H)
    assert [j.id for j in rows] == ["in", "user", "late"]  # oldest first by submission
    assert rows[0].scheduled_for == T0 + H and rows[0].run_s == 12.5
    assert [j.id for j in store.jobs.window(T0, T0 + 24 * H, kinds=["other.kind"])] == []


@needs_postgres
def test_snapshots_are_one_per_window_and_form_a_series(store) -> None:
    for hour in range(3):
        end = T0 + (8 + hour) * H
        store.snapshots.upsert(
            {
                "window_start": end - 8 * H,
                "window_end": end,
                "evaluated_at": end + timedelta(minutes=5),
                "status": "complete",
                "k_trace": 370.0 + hour,
                "k_strata": 368.0,
                "k_strata_sd": 5.9,
                "quantities": {"hmt_c": 1486, "transit_rw_h": 6.2},
                "versions": {"strata": "1.5", "facts": "1.8"},
            }
        )
    # A re-evaluation of the last window replaces it.
    last = store.snapshots.upsert(
        {
            "window_start": T0 + 2 * H,
            "window_end": T0 + 10 * H,
            "evaluated_at": T0 + 10 * H + timedelta(minutes=40),
            "status": "partial",
            "k_trace": 380.0,
            "flags": {"dpr": "provisional"},
        }
    )
    assert (last.k_trace, last.status, last.flags) == (380.0, "partial", {"dpr": "provisional"})
    assert (last.k_strata, last.quantities, last.versions) == (None, {}, {})  # nothing left of the first evaluation
    assert store.snapshots.latest().window_end == T0 + 10 * H
    series = store.snapshots.series(T0 + 8 * H, T0 + 10 * H)
    assert [s.k_trace for s in series] == [370.0, 371.0, 380.0]
    assert store.snapshots.get(T0 + 9 * H).quantities == {"hmt_c": 1486, "transit_rw_h": 6.2}
    with pytest.raises(IntegrityError):  # window_start must precede window_end
        store.snapshots.upsert({"window_start": T0, "window_end": T0, "evaluated_at": T0, "status": "complete"})
    with pytest.raises(ValueError, match="no column"):
        store.snapshots.upsert({"window_end": T0, "k_unknown": 1})


@needs_postgres
def test_runs_and_decisions_with_their_audit_rows(store) -> None:
    first = store.optimiser.add(_run(T0))
    second = store.optimiser.add(_run(T0 + 8 * H, mode="step1", cost_rs_thm=24790.0))
    assert store.optimiser.latest().id == second.id
    assert store.optimiser.latest(mode="step2").id == first.id
    assert [r.id for r in store.optimiser.recent(since=T0 + H)] == [second.id]

    store.optimiser.decide(first.id, "accepted", SUPERVISOR, at=T0 + H, shift_key="2026-10-27_A")
    store.optimiser.decide(str(first.id), "rejected", SUPERVISOR, at=T0 + 2 * H, comment="Pellet not available")
    assert store.optimiser.decision(first.id).decision == "rejected"
    assert [d.decision for d in store.optimiser.decisions(first.id)] == ["accepted", "rejected"]
    assert store.optimiser.decision(second.id) is None

    audit = store.audit.query(action="plan.decide")
    assert [(a.actor, a.role, a.client_ip, a.details["decision"]) for a in audit] == [
        ("sv1", "supervisor", "10.0.0.7", "rejected"),
        ("sv1", "supervisor", "10.0.0.7", "accepted"),
    ]
    assert audit[0].object_id == str(first.id)
    with pytest.raises(NotFound):
        store.optimiser.decide(uuid.uuid4(), "accepted", SUPERVISOR)
    with pytest.raises(ValueError):
        store.optimiser.decide(first.id, "maybe", SUPERVISOR)
    with pytest.raises(IntegrityError):  # ω mode outside strata / fixed
        store.optimiser.add(_run(T0, omega_mode="learned"))


@needs_postgres
def test_shift_records_are_written_once(store) -> None:
    run = store.optimiser.add(_run(T0))
    record = store.shifts.freeze(_shift("2026-10-27_A", T0, plan_run_id=str(run.id), plan_decision="accepted"))
    assert (record.letter, record.plan_run_id) == ("A", run.id)
    store.shifts.freeze(_shift("2026-10-27_B", T0 + 8 * H))
    with pytest.raises(AlreadyRecorded):
        store.shifts.freeze(_shift("2026-10-27_A", T0, k_trace=999.0))
    assert store.shifts.get("2026-10-27_A").k_trace == 370.7
    assert [r.shift_key for r in store.shifts.between(T0, T0 + 24 * H)] == ["2026-10-27_A", "2026-10-27_B"]
    with pytest.raises(ValueError, match="shift_key"):
        store.shifts.freeze(_shift("2026-10-27_D", T0))
    with pytest.raises(ValueError, match="does not match"):
        store.shifts.freeze(_shift("2026-10-28_A", T0 + 24 * H, letter="B"))


@needs_postgres
def test_alarms_open_once_per_kind_and_subject(store) -> None:
    drift, new = store.alarms.raise_alarm(
        "omega_drift", severity="warning", message="CUSUM above h = 5", details={"cusum": 5.4}, at=T0
    )
    assert new
    again, new = store.alarms.raise_alarm("omega_drift", severity="critical", message="CUSUM 7.9", at=T0 + H)
    assert (again.id, new, again.severity, again.raised_at) == (drift.id, False, "critical", T0)
    stale, new = store.alarms.raise_alarm("source_stale", subject="influxdb", severity="warning", message="19 min")
    other, new2 = store.alarms.raise_alarm("source_stale", subject="dpr", severity="warning", message="2 days")
    assert new and new2 and stale.id != other.id
    assert {a.id for a in store.alarms.open_alarms()} == {drift.id, stale.id, other.id}

    cleared = store.alarms.clear_alarm("omega_drift", at=T0 - H)  # a clock behind the raise: clamped to raised_at
    assert cleared.cleared_at == T0
    assert store.alarms.clear_alarm("omega_drift") is None
    reopened, new = store.alarms.raise_alarm("omega_drift", severity="warning", message="CUSUM 5.1", at=T0 + 3 * H)
    assert new and reopened.id != drift.id
    assert [a.id for a in store.alarms.history(kind="omega_drift")] == [reopened.id, drift.id]

    acknowledged = store.alarms.acknowledge(stale.id, SUPERVISOR, at=T0 + H)
    assert (acknowledged.acknowledged_by, acknowledged.acknowledged_at) == ("sv1", T0 + H)
    assert store.audit.query(action="alarm.acknowledge")[0].object_id == str(stale.id)
    with pytest.raises(NotFound):
        store.alarms.acknowledge(10_000, SUPERVISOR)
    with pytest.raises(ValueError):
        store.alarms.raise_alarm("omega_drift", severity="info", message="x")


@needs_postgres
def test_fact_register_versions_need_a_second_reviewer_for_status_changes(store) -> None:
    v1 = store.facts.propose(
        base_version="1.8",
        overlay={"ASM-08": {"status": "retired"}},
        actor=ALICE,
        second_review=True,
        note="ASM-08 replaced by the regime medians",
        at=T0,
    )
    assert (v1.version, v1.status, len(v1.content_hash)) == (1, "pending", 64)
    with pytest.raises(ReviewRefused, match="second administrator"):
        store.facts.review(1, approve=True, actor=ALICE)
    approved = store.facts.review(1, approve=True, actor=BOB, at=T0 + H)
    assert (approved.status, approved.reviewed_by) == ("approved", "bob")
    with pytest.raises(ReviewRefused, match="not pending"):
        store.facts.review(1, approve=False, actor=BOB)

    v2 = store.facts.propose(base_version="1.8", overlay={"DATA-14": {"value": 457.85}}, actor=BOB, parent_version=1)
    store.facts.review(v2.version, approve=True, actor=BOB)  # a value change: the author may approve it
    store.facts.propose(base_version="1.8", overlay={"ASM-20": {"value": 460}}, actor=ALICE)
    store.facts.review(3, approve=False, actor=BOB)
    assert store.facts.latest_approved().version == 2
    assert [v.version for v in store.facts.recent()] == [3, 2, 1]
    assert [a.action for a in store.audit.query(object_type="facts_version")].count("facts.review") == 3
    with pytest.raises(NotFound):
        store.facts.review(99, approve=True, actor=BOB)
    # The rule holds in the database as well.
    with pytest.raises(IntegrityError), store.engine.begin() as connection:
        connection.execute(text("UPDATE twin.facts_version SET reviewed_by = created_by WHERE version = 1"))


@needs_postgres
def test_audit_log_filters_newest_first(store) -> None:
    store.audit.record(ALICE, "user.create", "user", "u-17", {"role": "user"}, at=T0)
    store.audit.record(BOB, "hopper.update", "hopper", "hopper_04", {"material": "ore_6"}, at=T0 + H)
    store.audit.record(ALICE, "user.disable", "user", "u-17", at=T0 + 2 * H)
    assert [a.action for a in store.audit.query()] == ["user.disable", "hopper.update", "user.create"]
    assert [a.action for a in store.audit.query(actor="alice", since=T0 + H)] == ["user.disable"]
    assert [a.action for a in store.audit.query(object_type="user", object_id="u-17", until=T0 + H)] == ["user.create"]


@needs_postgres
def test_revision_is_none_before_the_migration(pg_server) -> None:
    with fresh_database(pg_server) as url:
        assert TwinStore.from_url(url).revision() is None


# ------------------------------------------------------------------------------------------------------ job archive


@needs_postgres
@needs_redis
def test_worker_archives_every_change_of_state(db_url, tmp_path) -> None:
    from arq.connections import RedisSettings, create_pool

    from services.job_archive import TwinJobArchive
    from services.jobs import JobStore, NoParams
    from worker.kinds.system import housekeeping

    async def run(redis_url: str) -> None:
        twin = TwinStore.from_url(db_url)
        archive = TwinJobArchive(twin)
        archive.start()
        pool = await create_pool(RedisSettings.from_dsn(redis_url))
        jobs = JobStore(pool, archive=archive)
        try:
            record, _ = await jobs.submit("system.selftest", {"steps": 2}, "alice")
            await jobs.update(record.id, status="running", started_at=datetime.now(UTC), tries=1)
            await jobs.update(record.id, progress=0.5, message="Step 1 of 2")  # not a change of state
            await jobs.finish(record.id, "done", result={"steps": 2})
            await archive.flush()
            row = twin.jobs.get(record.id)
            assert row is not None
            assert (row.status, row.tries, row.submitted_by, row.result) == ("done", 1, "alice", {"steps": 2})
            assert row.started_at is not None and row.finished_at is not None
            assert archive.failed == 0 and 1 <= archive.written <= 3  # changes of one batch are coalesced

            # Housekeeping copies the records of the last 7 days again (repairs a write lost to an outage).
            with twin.engine.begin() as connection:
                connection.execute(text("DELETE FROM twin.jobs"))
            context = type("Ctx", (), {"store": jobs})()
            result = await housekeeping(context, NoParams())
            assert result["records_archived"] == 1
            assert twin.jobs.get(record.id).status == "done"
        finally:
            await archive.close()
            await pool.aclose()

    with redis_server(tmp_path) as redis_url:
        asyncio.run(run(redis_url))


def test_archive_failures_are_counted_not_raised() -> None:
    from services.job_archive import TwinJobArchive
    from services.jobs import JobRecord

    async def run() -> None:
        unreachable = TwinStore.from_url("postgresql+psycopg2://nobody@127.0.0.1:9/none")
        archive = TwinJobArchive(unreachable)
        archive.start()
        archive.record(
            JobRecord(
                id="j", kind="system.selftest", status="queued", input_hash="h", submitted_by="a", submitted_at=T0
            )
        )
        await archive.flush(timeout=30)
        assert (archive.written, archive.failed) == (0, 1)
        await archive.close()

    asyncio.run(run())


# ----------------------------------------------------------------------------------------------------------- SQLite


def test_models_work_on_sqlite_for_the_outage_buffer(tmp_path) -> None:
    engine = build_engine(f"sqlite:///{tmp_path / 'buffer.db'}")
    TwinBase.metadata.create_all(engine)
    twin = TwinStore(session_factory(engine), engine)
    twin.jobs.upsert(_job("j1", "queued", T0))
    twin.jobs.upsert(_job("j1", "done", T0, finished_at=T0 + H))
    assert twin.jobs.get("j1").status == "done"
    snap = twin.snapshots.upsert(
        {"window_start": T0, "window_end": T0 + 8 * H, "evaluated_at": T0 + 8 * H, "status": "complete", "k_trace": 1.0}
    )
    assert snap.id == 1
    alarm, new = twin.alarms.raise_alarm("source_stale", subject="influxdb", severity="warning", message="20 min")
    _, again = twin.alarms.raise_alarm("source_stale", subject="influxdb", severity="warning", message="25 min")
    assert new and not again
    assert twin.alarms.clear_alarm("source_stale", subject="influxdb").id == alarm.id
    run = twin.optimiser.add(_run(T0))
    twin.optimiser.decide(run.id, "accepted", SUPERVISOR)
    assert twin.optimiser.decision(run.id).decision == "accepted"
    assert twin.revision() is None
    engine.dispose()
