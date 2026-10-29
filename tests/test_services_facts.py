"""Fact register editor (SV-4.2, #305): versions as overlays in ``twin.facts_version``, the second review of status
changes, the register the engine loads (latest approved version, else the repository file), retired facts refused,
the version diff and the schema-valid export. SQLite store for the logic, PostgreSQL for the constraints and the lock.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from jsonschema import Draft202012Validator
from sqlalchemy import func, select

from data.twin.engine import build_engine, session_factory
from data.twin.models import AuditLog, FactsVersion, TwinBase
from data.twin.repositories import Actor, ReviewRefused, TwinStore
from services import facts
from services import store as store_service
from tests.pg_server import needs_postgres
from utils.burden_engine import facts as register
from utils.burden_engine.engine import Engine, Limits
from utils.burden_engine.facts import REGISTER_PATH, SCHEMA_PATH, FactRegister, RetiredFactError, load_register

T0 = datetime(2026, 10, 28, 4, 30, tzinfo=UTC)  # 10:00 IST
H = timedelta(hours=1)
ALICE = Actor("alice", "admin", "10.0.0.5")
BOB = Actor("bob", "admin")
FILE = json.loads(REGISTER_PATH.read_text(encoding="utf-8"))
FILE_FACTS = {f["id"]: f for f in FILE["facts"]}


@pytest.fixture()
def store(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    twin = TwinStore(session_factory(engine), engine)
    store_service.use(twin)
    facts.approved_overlay.clear()
    yield twin
    store_service.reset()
    facts.approved_overlay.clear()
    twin.dispose()


def propose(store, changes, actor=ALICE, parent=None, note="Monthly refit", at=T0):
    if parent is None:
        parent = facts.register_view(store)["in_force_version"]
    return facts.propose(store, actor, parent_version=parent, changes=changes, note=note, at=at)


def audit_rows(store) -> list[AuditLog]:
    with store.sessions() as session:
        return list(session.scalars(select(AuditLog).order_by(AuditLog.id)))


# ------------------------------------------------------------------------------------------------- engine loading


def test_without_a_store_the_engine_loads_the_repository_file() -> None:
    store_service.use(None)
    facts.approved_overlay.clear()
    try:
        reg = load_register()
        assert reg is load_register(overlay=False) and reg.overlay_version is None and reg.label == FILE["version"]
        assert len(reg.facts()) == len(FILE["facts"]) == 116
        view = facts.register_view()
        assert view["in_force_version"] == 0 and view["version_status"] == "file" and "No twin store" in view["reason"]
    finally:
        store_service.reset()


def test_the_engine_loads_the_latest_approved_version(store) -> None:
    assert load_register().label == "1.8"  # empty store: the file
    propose(store, {"BAND-01": {"band": [510, 600]}, "COKE-01": {"value": -9.0}})
    assert load_register().overlay_version is None  # pending: not in force
    facts.review(store, BOB, 1, approve=True, at=T0 + H)
    reg = load_register()
    assert reg.label == "1.8 + overlay 1" and reg.band("BAND-01") == (510.0, 600.0) and reg.value("COKE-01") == -9.0
    assert Limits.from_register().fuel_min == 510.0  # the engine's default limits read the overlay
    propose(store, {"BAND-02": {"band": [1430, 1520]}}, at=T0 + 2 * H)
    facts.review(store, ALICE, 2, approve=False, at=T0 + 3 * H)  # rejected: version 1 stays in force
    assert load_register().overlay_version == 1
    propose(store, {"BAND-02": {"band": [1445, 1515]}}, at=T0 + 4 * H)
    facts.review(store, ALICE, 3, approve=True, at=T0 + 5 * H)  # values only: the author may approve
    eng = Engine()
    assert eng.facts.label == "1.8 + overlay 3" and eng.band_hmt == (1445.0, 1515.0)
    assert eng.facts.band("BAND-01") == (510.0, 600.0)  # cumulative: version 3 keeps version 1's change
    assert load_register(overlay=False).band("BAND-01") == (520.0, 600.0)  # the file is unchanged


def test_an_overlay_on_another_file_version_or_an_unreadable_store_leaves_the_file(store) -> None:
    store.facts.propose(base_version="1.7", overlay={"COKE-01": {"value": -1.0}}, actor=ALICE, at=T0)
    store.facts.review(1, approve=True, actor=BOB, at=T0 + H)
    facts.approved_overlay.clear()
    assert load_register().overlay_version is None  # made against v1.7, the file is v1.8
    assert facts.register_view(store)["in_force_version"] == 0
    with store.engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE facts_version")
    facts.approved_overlay.clear()
    assert load_register().overlay_version is None  # the store fails: the file is in force
    assert "not reachable" in facts.register_view(store)["reason"]


def test_retired_facts_are_refused_to_the_engine(store) -> None:
    propose(store, {"COKE-01": {"status": "Retired"}})
    with pytest.raises(ReviewRefused, match="second administrator"):
        facts.review(store, ALICE, 1, approve=True, at=T0 + H)  # a status change needs another administrator
    facts.review(store, BOB, 1, approve=True, at=T0 + H)
    reg = load_register()
    assert reg.overlay_version == 1 and reg["COKE-01"].retired
    with pytest.raises(RetiredFactError):
        reg.active("COKE-01")
    with pytest.raises(RetiredFactError):
        reg.value("COKE-01")
    assert "COKE-01" in [f.id for f in reg.facts()]  # shown, never used
    with pytest.raises(RetiredFactError):
        load_register(overlay=False).active("GAS-19")  # retired in the file itself
    with pytest.raises(facts.InvalidOverlay, match="the engine reads it for its hydraulics rules"):
        propose(store, {"FLOW-02": {"status": "Retired"}})  # retiring a fact the engine is built on: refused


# ------------------------------------------------------------------------------------------------- versions


def test_a_proposal_is_validated_against_the_schema_and_the_register(store) -> None:
    cases = [
        ({"COKE-01": {"status": "Obsolete"}}, "unknown status"),
        ({"COKE-01": {"band": [-8.0, -9.0]}}, "lies above"),
        ({"COKE-99": {"value": 1.0}}, "no such fact"),
        ({"COKE-01": {"statement": "x"}}, "cannot be changed"),
        ({"BAND-01": {"band": None}}, "cannot be removed"),
        ({"COKE-01": {"se": -1.0}}, "cannot be changed"),
        ({"COKE-01": {"value": FILE_FACTS["COKE-01"]["value"]}}, "leave every fact as it is"),
    ]
    for changes, message in cases:
        with pytest.raises(facts.InvalidOverlay, match=message):
            propose(store, changes)
    assert store.facts.recent() == [] and audit_rows(store) == []  # a refused write leaves neither


def test_the_overlay_is_cumulative_and_a_field_set_back_to_the_file_leaves_it(store) -> None:
    propose(store, {"COKE-01": {"value": -9.0, "status": "Disputed"}, "BAND-03": {"band": [0.25, 0.8]}})
    facts.review(store, BOB, 1, approve=True, at=T0 + H)
    v2 = propose(store, {"COKE-01": {"value": -8.7}}, at=T0 + 2 * H)  # back to the file's value
    assert v2["parent_version"] == 1 and v2["facts"] == ["BAND-03", "COKE-01"] and not v2["second_review"]
    with store.sessions() as session:
        overlay = session.scalars(select(FactsVersion.overlay).where(FactsVersion.version == 2)).one()
    assert overlay == {"COKE-01": {"status": "Disputed"}, "BAND-03": {"band": [0.25, 0.8]}}
    view = facts.register_view(store, 2)
    coke = next(f for f in view["facts"] if f["id"] == "COKE-01")
    assert view["version_status"] == "pending" and coke["changed"] == ["status"] and coke["file"] == {"status": "Model"}
    assert coke["value"] == -8.7 and coke["band"] == [-9.9, -8.7] and coke["band_kind"] == "range"


def test_a_version_must_be_made_from_the_version_in_force(store) -> None:
    propose(store, {"COKE-01": {"value": -9.0}}, parent=0)
    propose(store, {"COKE-02": {"value": 1.0}}, parent=0, at=T0 + H)
    facts.review(store, BOB, 1, approve=True, at=T0 + 2 * H)
    with pytest.raises(facts.FactsConflict, match="version 1 is in force now"):
        facts.review(store, BOB, 2, approve=True, at=T0 + 3 * H)
    with pytest.raises(facts.FactsConflict, match="the edit started from the repository file"):
        propose(store, {"COKE-03": {"value": 1.0}}, parent=0)
    with pytest.raises(ReviewRefused, match="is approved, not pending"):
        facts.review(store, BOB, 1, approve=False)
    listed = facts.versions(store)
    assert [(v["version"], v["status"], v["in_force"]) for v in listed] == [
        (2, "pending", False),
        (1, "approved", True),
    ]


def test_every_write_is_audited_with_the_version_before_and_after(store) -> None:
    propose(store, {"COKE-01": {"status": "Disputed"}}, note="T3 result disputes the replacement ratio")
    with pytest.raises(ReviewRefused):
        facts.review(store, ALICE, 1, approve=True, at=T0 + H)
    facts.review(store, BOB, 1, approve=True, note="Agreed in the review", at=T0 + H)
    rows = audit_rows(store)
    assert [(r.action, r.object_type, r.object_id, r.actor) for r in rows] == [
        ("facts.propose", "facts_version", "1", "alice"),
        ("facts.review", "facts_version", "1", "bob"),
    ]  # the refused review left no row
    propose_row, review_row = rows
    assert propose_row.before is None and propose_row.after["status"] == "pending"
    assert propose_row.after["overlay"] == {"COKE-01": {"status": "Disputed"}}
    assert propose_row.details["status_changes"] == ["COKE-01"] and propose_row.details["second_review"] is True
    assert review_row.before["status"] == "pending" and review_row.after["status"] == "approved"
    assert review_row.after["reviewed_by"] == "bob" and review_row.details == {
        "decision": "approved",
        "note": "Agreed in the review",
    }
    assert review_row.client_ip is None and propose_row.client_ip == "10.0.0.5"


def test_the_diff_lists_values_bands_and_statuses(store) -> None:
    propose(store, {"COKE-01": {"value": -9.0}, "BAND-01": {"band": [515, 600]}})
    facts.review(store, ALICE, 1, approve=True, at=T0 + H)
    propose(store, {"COKE-01": {"status": "Disputed"}, "BAND-01": {"band": [520, 600]}}, at=T0 + 2 * H)
    rows = facts.diff(store, 1, 2)["changes"]
    assert [(r["id"], r["field"], r["before"], r["after"]) for r in rows] == [
        ("COKE-01", "status", "Model", "Disputed"),
        ("BAND-01", "band", [515.0, 600.0], [520.0, 600.0]),
    ]
    assert [r["id"] for r in facts.diff(store, 0, 1)["changes"]] == ["COKE-01", "BAND-01"]
    assert facts.diff(store, 2, 2)["changes"] == []


# ------------------------------------------------------------------------------------------------- export


def test_the_export_is_schema_valid_and_loads_as_the_next_register_version(store) -> None:
    assert facts.export_text(FILE) == REGISTER_PATH.read_text(encoding="utf-8")  # the file's format: a minimal PR
    propose(store, {"COKE-01": {"value": -9.0}, "BAND-02": {"band": [1445, 1515]}, "COKE-02": {"status": "Disputed"}})
    with pytest.raises(ReviewRefused, match="pending: only an approved version"):
        facts.export(store, 1)
    with pytest.raises(ReviewRefused, match="no approved version"):
        facts.export(store)
    facts.review(store, BOB, 1, approve=True, at=datetime(2026, 10, 28, 20, 0, tzinfo=UTC))  # 29 Oct 01:30 IST
    doc = json.loads(facts.export_text(facts.export(store)))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(doc)) == []
    assert register.register_problems(doc) == []
    reg = FactRegister(doc)
    assert reg.version == "1.9" and doc["date"] == "2026-10-29" and reg.revision_log[0]["version"] == "1.9"
    assert (
        "COKE-01 value -8.7 -> -9" in reg.revision_log[0]["change"] and "Monthly refit" in reg.revision_log[0]["change"]
    )
    assert (
        reg.value("COKE-01") == -9.0 and reg.band("BAND-02") == (1445.0, 1515.0) and reg["COKE-02"].status == "Disputed"
    )
    stamped = {f["id"] for f in doc["facts"] if f["version"] == "1.9"}
    assert stamped == {"COKE-01", "BAND-02", "COKE-02"} and reg["COKE-01"].date == "2026-10-29"
    assert [f for f in doc["facts"] if f["id"] not in stamped] == [f for f in FILE["facts"] if f["id"] not in stamped]
    assert facts.next_version("1.9") == "1.10"


# ------------------------------------------------------------------------------------------------- PostgreSQL


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
def test_on_postgresql_versions_and_their_audit_rows_commit_together(pg_store) -> None:
    store_service.use(pg_store)
    facts.approved_overlay.clear()
    try:
        first = propose(pg_store, {"COKE-01": {"status": "Disputed"}, "BAND-01": {"band": [515, 600]}})
        with pytest.raises(ReviewRefused):
            facts.review(pg_store, ALICE, first["version"], approve=True)
        approved = facts.review(pg_store, BOB, first["version"], approve=True, at=T0 + H)
        assert approved["in_force"] and load_register().band("BAND-01") == (515.0, 600.0)
        rows = pg_store.audit.query(object_type="facts_version", object_id=str(first["version"]))
        assert sorted(r.action for r in rows) == ["facts.propose", "facts.review"]
        assert facts.export(pg_store)["version"] == "1.9"
        with pg_store.sessions() as session:
            assert session.scalar(select(func.count()).select_from(FactsVersion)) == 1
    finally:
        store_service.reset()
        facts.approved_overlay.clear()
