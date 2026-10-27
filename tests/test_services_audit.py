"""Admin writes and the audit log (SV-4.1, #304): one audit row with the values before and after in the transaction of
the write (``services.audit.admin_write``), JSON values, the changed fields, and the pages of the viewer.

SQLite store for the logic; the PostgreSQL tests (throw-away cluster, ``tests/pg_server.py``) check the transaction on
the production database: a write whose audit row the database refuses is rolled back with it.
"""

from __future__ import annotations

import enum
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from data.twin.engine import build_engine, session_factory
from data.twin.models import AlarmEvent, AuditLog, FactsVersion, TwinBase
from data.twin.repositories import Actor, TwinStore
from services import audit
from tests.pg_server import needs_postgres

T0 = datetime(2026, 10, 27, 0, 30, tzinfo=UTC)  # 06:00 IST
H = timedelta(hours=1)
ALICE = Actor("alice", "admin", "10.0.0.5")
BOB = Actor("bob", "admin")
FIELDS = ("severity", "message", "acknowledged_by", "acknowledged_at")


def sqlite_store(tmp_path) -> TwinStore:
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    return TwinStore(session_factory(engine), engine)


@pytest.fixture()
def store(tmp_path):
    twin = sqlite_store(tmp_path)
    yield twin
    twin.dispose()


def audit_rows(store: TwinStore) -> list[AuditLog]:
    with store.sessions() as session:
        return list(session.scalars(select(AuditLog).order_by(AuditLog.id)))


def acknowledge(store: TwinStore, alarm_id: int, actor: Actor = ALICE, *, at: datetime = T0) -> audit.AdminWrite:
    """The admin write of the tests: acknowledge an alarm and raise its severity, with its values around the write."""
    with audit.admin_write(store, actor, "alarm.acknowledge", "alarm_event", alarm_id, at=at) as write:
        row = write.session.get(AlarmEvent, alarm_id, with_for_update=True)
        write.before = audit.row_values(row, include=FIELDS)
        row.severity = "critical"
        row.acknowledged_by = actor.name
        row.acknowledged_at = at
        write.after = audit.row_values(row, include=FIELDS)
    return write


def raise_alarm(store: TwinStore, kind: str = "omega_drift") -> AlarmEvent:
    alarm, _ = store.alarms.raise_alarm(kind, subject="high", severity="warning", message="ω above STRATA", at=T0)
    return alarm


# ------------------------------------------------------------------------------------------------ admin writes


def test_an_admin_write_records_one_audit_row_with_the_values_before_and_after(store) -> None:
    alarm = raise_alarm(store)
    write = acknowledge(store, alarm.id, at=T0 + H)
    rows = audit_rows(store)
    assert len(rows) == 1 and rows[0].id == write.entry.id
    row = rows[0]
    assert (row.actor, row.role, row.client_ip) == ("alice", "admin", "10.0.0.5")
    assert (row.action, row.object_type, row.object_id) == ("alarm.acknowledge", "alarm_event", str(alarm.id))
    assert row.before == {
        "severity": "warning",
        "message": "ω above STRATA",
        "acknowledged_by": None,
        "acknowledged_at": None,
    }
    assert row.after == {
        "severity": "critical",
        "message": "ω above STRATA",
        "acknowledged_by": "alice",
        "acknowledged_at": "2026-10-27T01:30:00+00:00",
    }
    assert row.at.replace(tzinfo=UTC) == T0 + H
    stored = store.alarms.open_alarms()[0]
    assert (stored.severity, stored.acknowledged_by) == ("critical", "alice")  # the write itself is committed
    assert [c["path"] for c in audit.entry_data(row)["changes"]] == ["severity", "acknowledged_by", "acknowledged_at"]


def test_each_admin_write_adds_exactly_one_row(store) -> None:
    first, second = raise_alarm(store), raise_alarm(store, "source_stale")
    acknowledge(store, first.id, ALICE)
    acknowledge(store, second.id, BOB, at=T0 + H)
    acknowledge(store, first.id, BOB, at=T0 + 2 * H)  # a second write on the same object: its own row
    rows = audit_rows(store)
    assert [(r.actor, r.object_id) for r in rows] == [
        ("alice", str(first.id)),
        ("bob", str(second.id)),
        ("bob", str(first.id)),
    ]
    assert rows[2].before["acknowledged_by"] == "alice" and rows[2].after["acknowledged_by"] == "bob"


def test_a_failing_write_leaves_neither_the_change_nor_an_audit_row(store) -> None:
    alarm = raise_alarm(store)
    with pytest.raises(RuntimeError, match="limit check"):
        with audit.admin_write(store, ALICE, "alarm.acknowledge", "alarm_event", alarm.id) as write:
            row = write.session.get(AlarmEvent, alarm.id)
            write.before = audit.row_values(row, include=FIELDS)
            row.severity = "critical"
            write.session.flush()
            raise RuntimeError("limit check failed")
    assert audit_rows(store) == []
    assert store.alarms.open_alarms()[0].severity == "warning"


def test_a_refused_audit_row_rolls_back_the_write(store, monkeypatch) -> None:
    alarm = raise_alarm(store)

    def refuse(*args, **kwargs):
        raise RuntimeError("audit row refused")

    monkeypatch.setattr(audit, "audit_row", refuse)
    with pytest.raises(RuntimeError, match="audit row refused"):
        acknowledge(store, alarm.id)
    assert store.alarms.open_alarms()[0].severity == "warning"  # flushed before the row, then rolled back


@pytest.mark.parametrize("missing", ["before", "after", "both", "none"])
def test_a_write_that_does_not_state_its_values_is_refused(store, missing) -> None:
    alarm = raise_alarm(store)
    with pytest.raises(audit.AuditIncomplete):
        with audit.admin_write(store, ALICE, "alarm.acknowledge", "alarm_event", alarm.id) as write:
            row = write.session.get(AlarmEvent, alarm.id)
            row.severity = "critical"
            if missing == "before":
                write.after = {"severity": "critical"}
            elif missing == "after":
                write.before = {"severity": "warning"}
            elif missing == "none":  # both stated, both None: no object before or after
                write.before = write.after = None
    assert audit_rows(store) == [] and store.alarms.open_alarms()[0].severity == "warning"


def test_created_and_deleted_objects(store) -> None:
    with audit.admin_write(store, ALICE, "facts.propose", "facts_version", 1, details={"note": "PCI 185"}) as write:
        row = FactsVersion(
            version=1,
            base_version="1.8",
            status="pending",
            overlay={"ASM-08": {"value": 185}},
            content_hash="h",
            created_by="alice",
            created_at=T0,
        )
        write.session.add(row)
        write.before = None
        write.after = audit.row_values(row, include=("version", "status", "overlay"))
    alarm = raise_alarm(store)
    with audit.admin_write(store, BOB, "alarm.delete", "alarm_event", alarm.id) as write:
        row = write.session.get(AlarmEvent, alarm.id)
        write.before = audit.row_values(row, include=("kind", "subject"))
        write.session.delete(row)
        write.after = None
    created, deleted = (audit.entry_data(r) for r in audit_rows(store))
    assert created["before"] is None and created["details"] == {"note": "PCI 185"}
    assert created["changes"] == [
        {"path": "version", "kind": "added", "before": None, "after": 1},
        {"path": "status", "kind": "added", "before": None, "after": "pending"},
        {"path": "overlay.ASM-08.value", "kind": "added", "before": None, "after": 185},
    ]
    assert deleted["after"] is None and [c["kind"] for c in deleted["changes"]] == ["removed", "removed"]
    assert store.facts.get(1) is not None and store.alarms.open_alarms() == []


def test_inputs_are_checked_before_the_transaction(store) -> None:
    for kwargs, message in [
        (dict(actor=Actor(""), action="a.b", object_type="t"), "actor"),
        (dict(actor=ALICE, action="", object_type="t"), "action"),
        (dict(actor=ALICE, action="a" * 65, object_type="t"), "at most 64"),
        (dict(actor=ALICE, action="a.b", object_type="t", object_id="x" * 129), "at most 128"),
        (dict(actor=ALICE, action="a.b", object_type="t", at=datetime(2026, 10, 27)), "timezone-aware"),
    ]:
        with pytest.raises(ValueError, match=message), audit.admin_write(store, **kwargs):
            pytest.fail("the block must not run")
    with pytest.raises(TypeError, match="mapping"), audit.admin_write(store, ALICE, "a.b", "t") as write:
        write.before, write.after = ["not", "a", "mapping"], {}
    assert audit_rows(store) == []


# ------------------------------------------------------------------------------------------------------ values


class Colour(enum.Enum):
    TEAL = "teal"


@dataclass
class Point:
    x: float
    y: float


class Limits(BaseModel):
    lo_kg_thm: float
    hi_kg_thm: float


def test_json_safe_values() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    key = uuid.UUID("12345678-1234-5678-1234-567812345678")
    value = {
        "at": datetime(2026, 10, 27, 6, 0, tzinfo=ist),
        "naive": datetime(2026, 10, 27, 6, 0),
        "day": date(2026, 10, 27),
        "run": key,
        "price": Decimal("457.85"),
        "nan": math.nan,
        "inf": -math.inf,
        "colour": Colour.TEAL,
        "tags": {"b", "a"},
        "pair": (1, 2.5),
        "point": Point(1.0, 2.0),
        "limits": Limits(lo_kg_thm=520, hi_kg_thm=600),
        "np": np.float64(1.5),
        "arr": np.array([1, 2]),
        7: True,
    }
    assert audit.json_safe(value) == {
        "at": "2026-10-27T00:30:00+00:00",
        "naive": "2026-10-27T06:00:00",
        "day": "2026-10-27",
        "run": str(key),
        "price": 457.85,
        "nan": None,
        "inf": None,
        "colour": "teal",
        "tags": ["a", "b"],
        "pair": [1, 2.5],
        "point": {"x": 1.0, "y": 2.0},
        "limits": {"lo_kg_thm": 520.0, "hi_kg_thm": 600.0},
        "np": 1.5,
        "arr": [1, 2],
        "7": True,
    }
    with pytest.raises(TypeError, match="object"):
        audit.json_safe({"x": object()})


def test_row_values_of_rows_and_mappings(store) -> None:
    alarm = raise_alarm(store)
    assert audit.row_values(alarm, include=("severity", "kind")) == {"severity": "warning", "kind": "omega_drift"}
    everything = audit.row_values(alarm, exclude=("details",))
    assert everything["raised_at"] == "2026-10-27T00:30:00+00:00" and "details" not in everything
    with pytest.raises(ValueError, match="no field colour"):
        audit.row_values(alarm, include=("colour",))
    assert audit.row_values({"b": 2, "a": Decimal("1.5")}) == {"b": 2, "a": 1.5}
    assert audit.row_values(Point(1, 2), include=("y",)) == {"y": 2}
    with store.sessions() as session:
        core = session.execute(text("SELECT 1 AS one, 'x' AS name")).one()
    assert audit.row_values(core) == {"one": 1, "name": "x"}
    assert audit.row_values(None) is None


def test_changes_flatten_nested_values() -> None:
    before = {
        "overlay": {"ASM-08": {"value": 190, "band": [170, 200]}, "ASM-20": {"value": 457.85}},
        "coke_angles_deg": [28, 30, 32],
        "enabled": True,
        "count": 3,
        "note": None,
        "empty": {},
    }
    after = {
        "overlay": {"ASM-08": {"value": 185, "band": [170, 200]}, "ASM-20": {"value": 457.85}},
        "coke_angles_deg": [28, 31],
        "enabled": 1,
        "count": 3.0,
        "note": "trial",
        "empty": {},
        "added_t": 4.2,
    }
    assert audit.changes(before, after) == [
        {"path": "overlay.ASM-08.value", "kind": "changed", "before": 190, "after": 185},
        {"path": "coke_angles_deg[1]", "kind": "changed", "before": 30, "after": 31},
        {"path": "coke_angles_deg[2]", "kind": "removed", "before": 32, "after": None},
        {"path": "enabled", "kind": "changed", "before": True, "after": 1},  # a boolean is not the number 1
        {"path": "note", "kind": "changed", "before": None, "after": "trial"},
        {"path": "added_t", "kind": "added", "before": None, "after": 4.2},
    ]
    assert audit.changes(None, None) == [] and audit.changes(before, before) == []
    assert audit.flatten({"a": {"b": [{"c": 1}, []]}}) == {"a.b[0].c": 1, "a.b[1]": []}


# ------------------------------------------------------------------------------------------------------ viewer


def seed(store: TwinStore) -> None:
    """Seven entries over three days: four by alice, three by bob, on three object types."""
    for i in range(7):
        actor = ALICE if i % 2 == 0 else BOB
        kind = ("user", "hopper", "facts_version")[i % 3]
        store.audit.record(
            actor,
            f"{kind}.update",
            kind,
            f"{kind}-{i % 2}",
            {"i": i},
            at=T0 + i * 10 * H,
            before={"value": i},
            after={"value": i + 1},
        )


def test_pages_newest_first_with_the_total(store) -> None:
    seed(store)
    page = audit.read_page(store, limit=3, offset=0)
    assert page["total"] == 7 and (page["limit"], page["offset"]) == (3, 0)
    assert [e["details"]["i"] for e in page["entries"]] == [6, 5, 4]
    last = audit.read_page(store, limit=3, offset=6)
    assert [e["details"]["i"] for e in last["entries"]] == [0] and last["total"] == 7
    entry = page["entries"][0]
    assert entry["at"] == T0 + 60 * H and entry["at"].tzinfo is not None
    assert entry["changes"] == [{"path": "value", "kind": "changed", "before": 6, "after": 7}]
    assert page["facets"] == {
        "actors": ["alice", "bob"],
        "actions": ["facts_version.update", "hopper.update", "user.update"],
        "object_types": ["facts_version", "hopper", "user"],
    }


def test_filters_by_user_object_and_time(store) -> None:
    seed(store)

    def ids(**filters) -> list[int]:
        page = audit.read_page(store, audit.AuditFilter(**filters))
        assert page["total"] == len(page["entries"])
        return [e["details"]["i"] for e in page["entries"]]

    assert ids(actor="alice") == [6, 4, 2, 0]
    assert ids(object_type="hopper") == [4, 1]
    assert ids(object_type="hopper", object_id="hopper-0") == [4]
    assert ids(action="user.update") == [6, 3, 0]
    assert ids(since=T0 + 20 * H, until=T0 + 50 * H) == [4, 3, 2]  # [since, until)
    assert ids(actor="carol") == []
    with pytest.raises(ValueError, match="before until"):
        audit.read_page(store, audit.AuditFilter(since=T0, until=T0))


# -------------------------------------------------------------------------------------------------- PostgreSQL


@pytest.fixture(scope="module")
def pg_store():
    from alembic import command

    from tests.pg_server import fresh_database, postgres_server
    from tests.test_twin_store import alembic_config

    with postgres_server() as server, fresh_database(server) as url:
        command.upgrade(alembic_config(url), "head")
        twin = TwinStore.from_url(url)
        yield twin
        twin.dispose()


@needs_postgres
def test_on_postgresql_the_write_and_its_audit_row_commit_together(pg_store) -> None:
    alarm = raise_alarm(pg_store, "pg_write")
    before = len(pg_store.audit.query(object_id=str(alarm.id)))
    write = acknowledge(pg_store, alarm.id, at=T0 + H)
    rows = pg_store.audit.query(object_type="alarm_event", object_id=str(alarm.id))
    assert len(rows) == before + 1 and rows[0].id == write.entry.id
    assert rows[0].before["severity"] == "warning" and rows[0].after["acknowledged_at"] == "2026-10-27T01:30:00+00:00"
    page = audit.read_page(pg_store, audit.AuditFilter(object_type="alarm_event", object_id=str(alarm.id)))
    assert page["total"] == 1 and page["entries"][0]["changes"][0]["path"] == "severity"
    with pytest.raises(DBAPIError, match="written once"), pg_store.engine.begin() as connection:
        connection.execute(text("UPDATE twin.audit_log SET after = NULL"))  # the values cannot be rewritten


@needs_postgres
def test_on_postgresql_a_refused_audit_row_rolls_back_the_write(pg_store) -> None:
    """PostgreSQL refuses a NUL character in JSON text: the audit row fails after the write was flushed."""
    alarm = raise_alarm(pg_store, "pg_refused")
    with pytest.raises(DBAPIError):
        with audit.admin_write(pg_store, ALICE, "alarm.acknowledge", "alarm_event", alarm.id) as write:
            row = write.session.get(AlarmEvent, alarm.id)
            write.before = audit.row_values(row, include=FIELDS)
            row.severity = "critical"
            write.after = audit.row_values(row, include=FIELDS)
            write.details["note"] = "nul \x00 in the note"
    with pg_store.sessions() as session:
        assert session.get(AlarmEvent, alarm.id).severity == "warning"
        assert session.scalar(select(func.count()).where(AuditLog.object_id == str(alarm.id))) == 0
