"""User, hopper and burden-distribution admin (SV-4.7, #310): the twin-api's writes are those of the Streamlit tools.

The Welcome page's tools (``ui.user_management``, ``ui.hopper_admin_page``, ``ui.burden_admin_page``, Streamlit
stubbed) and ``services.plant_admin`` run the same forms on two databases seeded alike; the rows of
``identity.users`` / ``identity.user_roles``, ``ops_config.hopper_raw_material_history`` and
``ops_config.burden_history`` must be equal. The history stays SCD Type-2 (earlier snapshots untouched, each
snapshot a copy of the one in force at its time), and every write commits with its audit row. Shared domain
functions (``domain.plant_config``) are tested without a database. Needs PostgreSQL (``tests/pg_server.py``).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from alembic import command
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from data.db import password_digest
from data.material_mapping import MaterialNameMapper
from data.twin.repositories import Actor, TwinStore
from domain import plant_config
from furnace_data.relational import (
    BURDEN_VALUE_COLUMNS,
    HOPPER_COLUMNS,
    BurdenHistory,
    Hopper,
    HopperRawMaterialHistory,
    Material,
    MaterialCategory,
    Unit,
    User,
    UserRepository,
    UserRoleAssignment,
    build_relational_session_factory,
)
from furnace_data.relational import Base as RelationalBase
from services import audit, plant_admin
from tests.module_stubs import import_with_stubs
from tests.pg_server import fresh_database, needs_postgres, postgres_server
from tests.test_twin_store import alembic_config

IST = timezone(timedelta(hours=5, minutes=30))
ACTOR = Actor("adm", "admin", "10.0.0.5")
#: Times of the scenario: the imported history, the assignment, a back-dated assignment (UTC).
T_IMPORT = datetime(2025, 6, 28, 0, 30, tzinfo=UTC)
T_ASSIGN = datetime(2025, 7, 5, 0, 30, tzinfo=UTC)  # 06:00 IST
T_BACK = datetime(2025, 7, 1, 8, 30, tzinfo=UTC)  # 14:00 IST
TABLES = [
    User.__table__,
    UserRoleAssignment.__table__,
    MaterialCategory.__table__,
    Unit.__table__,
    Material.__table__,
    Hopper.__table__,
    HopperRawMaterialHistory.__table__,
    BurdenHistory.__table__,
]


# ---------------------------------------------------------------------------------------------------- domain


def test_registration_strips_and_requires_both_fields() -> None:
    assert plant_config.registration("  qa ", " pw ") == ("qa", "pw")
    for name, secret in [("", "pw"), ("qa", "  "), (" ", " ")]:
        with pytest.raises(plant_config.EmptyField):
            plant_config.registration(name, secret)


def test_hopper_changes_are_the_hoppers_that_differ_from_the_mapping_in_force() -> None:
    current = {"hopper_01": "Sinter", "hopper_02": "UNASSIGNED"}
    form = {"hopper_01": "Sinter", "hopper_02": "Quartz", "hopper_03": "UNASSIGNED", "hopper_04": "Dolomite"}
    assert plant_config.hopper_changes(form, current) == {"hopper_02": "Quartz", "hopper_04": "Dolomite"}


def test_burden_changes_compare_the_values_as_the_snapshot_stores_them() -> None:
    current = {
        "coke_p01_rings": 3.0,
        "coke_p01_angles": None,
        "coke_charge_pattern": None,
        "burden_changing_purpose": "x",
    }
    form = {
        "coke_p01_rings": "3",  # stored as 3.0: unchanged
        "coke_p01_angles": 0.0,  # NULL -> 0.0 (the Streamlit number input shows 0.0): changed
        "coke_charge_pattern": "",  # stored as NULL: unchanged
        "burden_changing_purpose": "y",
    }
    assert plant_config.burden_changes(form, current) == {"coke_p01_angles": 0.0, "burden_changing_purpose": "y"}
    nan = float("nan")
    assert plant_config.burden_changes({"coke_p02_rings": nan}, {"coke_p02_rings": nan}) == {}
    with pytest.raises(ValueError, match="Invalid burden field"):
        plant_config.burden_changes({"coke_p12_rings": 1.0}, current)
    with pytest.raises(ValueError, match="is not a number"):
        plant_config.burden_changes({"coke_p01_rings": "three"}, current)
    with pytest.raises(ValueError, match="is not a number"):
        plant_config.burden_changes({"coke_p01_rings": True}, current)
    with pytest.raises(ValueError, match="is not text"):
        plant_config.burden_changes({"coke_charge_pattern": 4.0}, current)


def test_burden_fields_carry_their_units_and_place_in_the_charge_matrix() -> None:
    fields = {f["name"]: f for f in plant_admin.burden_fields()}
    assert list(fields) == list(BURDEN_VALUE_COLUMNS)
    assert fields["coke_p03_angles"] | {"label": None} == {
        "name": "coke_p03_angles",
        "label": None,
        "kind": "number",
        "unit": "°",
        "charge": "coke",
        "position": 3,
        "quantity": "angles",
    }
    assert (fields["noncoke_p11_rings"]["unit"], fields["noncoke_p11_rings"]["charge"]) == ("rotations", "non_coke")
    assert (fields["non_coke_discharge_time"]["unit"], fields["non_coke_discharge_time"]["charge"]) == ("s", "non_coke")
    assert {name for name, f in fields.items() if f["kind"] == "text"} == set(plant_config.BURDEN_TEXT_FIELDS)
    assert fields["burden_changing_purpose"]["charge"] is None


# ---------------------------------------------------------------------------------------------------- database


@pytest.fixture(scope="module")
def pg_server() -> Iterator[str]:
    with postgres_server() as server:
        yield server


def seed(url: str) -> None:
    """The plant tables of the tools on a fresh database (with the twin schema), seeded with the plant master, the
    accounts ``admin`` and ``adm`` and one imported snapshot of each history."""
    command.upgrade(alembic_config(url), "head")
    engine = create_engine(url, poolclass=NullPool)
    try:
        with engine.begin() as connection:
            for schema in ("identity", "plant_master", "ops_config"):
                connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        RelationalBase.metadata.create_all(engine, tables=TABLES)
        sessions = build_relational_session_factory(engine)
        names = sorted({entry.material_name for entry in MaterialNameMapper.from_file().entries})
        with sessions.begin() as session:
            session.add(MaterialCategory(category_code="ORE", category_name="Ore"))
            session.add(Unit(unit_code="MT", unit_name="Tonne"))
            session.flush()
            for i, name in enumerate(names, start=1):
                session.add(Material(material_code=f"M{i:02d}", material_name=name, category_code="ORE"))
            for i in range(1, 20):
                session.add(
                    Hopper(hopper_code=f"hopper_{i:02d}", display_name=f"Bin {i}", sort_order=i, is_active=i < 19)
                )
        users = UserRepository(sessions)
        users.seed_admin_user(password_hash=password_digest("admin123"))
        users.add_user("adm", password_digest("pw-adm"), "admin")
        codes = {name: f"M{i:02d}" for i, name in enumerate(names, start=1)}
        with sessions.begin() as session:
            session.add(
                HopperRawMaterialHistory(
                    date_time=T_IMPORT,
                    source_type="import",
                    hopper_01=codes["sinter_sp02_online"],
                    hopper_02=codes["coke_online"],
                    hopper_03=codes["lloyds_ore"],
                    hopper_05=codes["dolomite"],
                    hopper_07=codes["quartz"],
                    hopper_19=codes["limestone"],
                )
            )
            values: dict[str, Any] = {}
            for i in range(1, 11):  # P11 is not used on the plant: NULL
                values[f"coke_p{i:02d}_rings"] = float(1 + i % 3)
                values[f"coke_p{i:02d}_angles"] = 40.0 - 2 * i
                values[f"noncoke_p{i:02d}_rings"] = float(2 + i % 2)
                values[f"noncoke_p{i:02d}_angles"] = 41.5 - 2 * i
            values |= {"coke_discharge_time": 92.0, "non_coke_discharge_time": 68.0, "coke_charge_pattern": "C 3 2 2"}
            session.add(BurdenHistory(date_time=T_IMPORT, source_type="import", **values))
    finally:
        engine.dispose()


@pytest.fixture()
def database(pg_server) -> Iterator[str]:
    with fresh_database(pg_server) as url:
        seed(url)
        yield url


@pytest.fixture()
def store(database) -> Iterator[TwinStore]:
    twin = TwinStore.from_url(database)
    assert twin is not None
    yield twin
    twin.dispose()


def rows(url: str, sql: str) -> list[dict[str, Any]]:
    engine = create_engine(url, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            return [dict(r._mapping) for r in connection.execute(text(sql))]
    finally:
        engine.dispose()


def dump(url: str) -> dict[str, list[dict[str, Any]]]:
    """The tools' tables without generated keys and times; account ids as user names."""
    names = {r["id"]: r["username"] for r in rows(url, "SELECT id, username FROM identity.users")}

    def history(table: str) -> list[dict[str, Any]]:
        out = []
        for r in rows(url, f"SELECT * FROM ops_config.{table} ORDER BY date_time, id"):
            del r["id"], r["created_at"]
            r["user_modified"] = names.get(r["user_modified"])
            out.append(r)
        return out

    return {
        "users": rows(url, "SELECT username, password_hash, role, is_active FROM identity.users ORDER BY username"),
        "user_roles": [
            {**r, "assigned_by_user_id": names.get(r["assigned_by_user_id"])}
            for r in rows(
                url,
                "SELECT u.username, r.role, r.assigned_by_user_id FROM identity.user_roles r "
                "JOIN identity.users u ON u.id = r.user_id ORDER BY 1, 2",
            )
        ],
        "hoppers": history("hopper_raw_material_history"),
        "burden": history("burden_history"),
    }


def audit_rows(url: str) -> list[dict[str, Any]]:
    return rows(url, "SELECT * FROM twin.audit_log ORDER BY id")


# ------------------------------------------------------------------------------------------- the two write paths


class Rerun(BaseException):
    """``st.rerun()`` (Streamlit's own is a BaseException too, so the tools' ``except Exception`` lets it pass)."""


def streamlit_stub(messages: list[tuple[str, str]]) -> dict[str, Any]:
    def say(kind: str):
        return lambda text, *a, **k: messages.append((kind, str(text)))

    def rerun() -> None:
        raise Rerun

    return {
        "session_state": {},
        "success": say("success"),
        "info": say("info"),
        "warning": say("warning"),
        "error": say("error"),
        "rerun": rerun,
    }


def streamlit_form(current: dict[str, Any]) -> dict[str, Any]:
    """The burden form as the Streamlit tool fills it: text inputs show the text or "", number inputs the number or
    0.0."""
    return {
        name: (
            ("" if value is None else str(value))
            if name in plant_config.BURDEN_TEXT_FIELDS
            else (float(value) if isinstance(value, int | float) else 0.0)
        )
        for name, value in current.items()
    }


def hopper_form(current: dict[str, str], **changes: str) -> dict[str, str]:
    return {**current, **changes}


ASSIGN = {"hopper_02": "Sinter", "hopper_05": "UNASSIGNED", "hopper_07": "LLOYDS FINES OVER SIZE"}  # an alias
BACK = {"hopper_03": "Quartz"}
BURDEN_EDIT = {"coke_p03_angles": 31.5, "coke_charge_pattern": "C 4 3 2", "burden_changing_purpose": "Open the centre"}


def run_streamlit(url: str, monkeypatch) -> list[tuple[str, str]]:
    """The scenario through the Streamlit tools' own submission code, on ``DATABASE_URL`` = ``url``."""
    monkeypatch.setenv("DATABASE_URL", url)
    messages: list[tuple[str, str]] = []
    stubs = {"streamlit": streamlit_stub(messages)}
    um = import_with_stubs("ui.user_management", stubs)
    hp = import_with_stubs("ui.hopper_admin_page", stubs)
    bp = import_with_stubs("ui.burden_admin_page", stubs)

    page = um.RegisterPage()
    with pytest.raises(Rerun):
        page.handle_registration(" qa ", " s3cret ", "supervisor")
    page.db.dispose()

    hopper = hp.HopperAdminPage()
    with pytest.raises(Rerun):  # the tool's time is naive and stored as UTC
        hopper.handle_submission(
            hopper_form(hopper.hopper_materials, **ASSIGN), T_ASSIGN.replace(tzinfo=None), "adm", "10.0.0.5"
        )
    hopper.db.dispose()
    hopper = hp.HopperAdminPage()
    with pytest.raises(Rerun):
        hopper.handle_submission(
            hopper_form(hopper.hopper_materials, **BACK), T_BACK.replace(tzinfo=None), "adm", "10.0.0.5"
        )

    burden = bp.BurdenAdminPage()
    current = burden.get_current_burden_values()
    with pytest.raises(Rerun):
        burden.handle_submission(
            {**streamlit_form(current), **BURDEN_EDIT}, current, T_ASSIGN.replace(tzinfo=None), "adm", "10.0.0.5"
        )

    back_id = next(r["id"] for r in hopper.db.get_hopper_material_history() if r["date_time"] == T_BACK)
    hopper.db.delete_hopper_material_history([back_id])
    import_id = next(r["id"] for r in burden.db.get_burden_history() if r["date_time"] == T_IMPORT)
    burden.db.delete_burden_history([import_id])
    hopper.db.dispose()
    burden.db.dispose()
    return messages


def run_service(url: str) -> list[dict[str, Any]]:
    """The same scenario through ``services.plant_admin`` (times entered in IST)."""
    twin = TwinStore.from_url(url)
    assert twin is not None
    results: list[dict[str, Any]] = []
    try:
        results.append(plant_admin.register_user(twin, ACTOR, " qa ", " s3cret ", "supervisor"))
        current = plant_admin.hopper_view(twin)["current"]["materials"]
        results.append(
            plant_admin.assign_hoppers(twin, ACTOR, hopper_form(current, **ASSIGN), T_ASSIGN.astimezone(IST))
        )
        current = plant_admin.hopper_view(twin)["current"]["materials"]
        results.append(plant_admin.assign_hoppers(twin, ACTOR, hopper_form(current, **BACK), T_BACK.astimezone(IST)))
        view = plant_admin.burden_view(twin)
        form = {**streamlit_form(view["current"]["values"]), **BURDEN_EDIT}
        results.append(plant_admin.update_burden(twin, ACTOR, form, T_ASSIGN.astimezone(IST)))
        back_id = next(r["id"] for r in plant_admin.hopper_view(twin)["history"] if r["date_time"] == T_BACK)
        results.append(plant_admin.delete_hopper_history(twin, ACTOR, [back_id]))
        import_id = next(r["id"] for r in plant_admin.burden_view(twin)["history"] if r["date_time"] == T_IMPORT)
        results.append(plant_admin.delete_burden_history(twin, ACTOR, [import_id]))
    finally:
        twin.dispose()
    return results


@needs_postgres
def test_the_twin_api_writes_the_rows_of_the_streamlit_tools(pg_server, monkeypatch) -> None:
    with fresh_database(pg_server) as a, fresh_database(pg_server) as b:
        seed(a)
        seed(b)
        assert dump(a) == dump(b)
        messages = run_streamlit(a, monkeypatch)
        assert not [m for m in messages if m[0] in ("error", "warning")], messages
        results = run_service(b)
        assert [r.get("written", True) for r in results] == [True] * 6
        streamlit, service = dump(a), dump(b)
        assert streamlit == service

        # The writes themselves: the account, the snapshots (copies of the one in force at their time), the deletes.
        assert [u["username"] for u in service["users"]] == ["adm", "admin", "qa"]
        qa = next(u for u in service["users"] if u["username"] == "qa")
        assert qa == {
            "username": "qa",
            "password_hash": password_digest("s3cret"),
            "role": "supervisor",
            "is_active": True,
        }
        assert {"username": "qa", "role": "supervisor", "assigned_by_user_id": None} in service["user_roles"]
        imported, assigned = service["hoppers"]  # the back-dated snapshot was added, then deleted
        assert assigned["date_time"] == T_ASSIGN and assigned["user_modified"] == "adm"
        assert (assigned["ip_address"], assigned["source_type"]) == ("10.0.0.5", "webapp")
        sinter, oversize = (
            r["material_code"]
            for name in ("sinter_sp02_online", "lloyds_oversize")
            for r in rows(b, f"SELECT material_code FROM plant_master.materials WHERE material_name = '{name}'")
        )
        expected = {h: imported[h] for h in HOPPER_COLUMNS} | {
            "hopper_02": sinter,
            "hopper_05": None,
            "hopper_07": oversize,
        }
        assert {h: assigned[h] for h in HOPPER_COLUMNS} == expected
        (burden,) = service["burden"]  # the imported snapshot was deleted
        assert burden["date_time"] == T_ASSIGN
        assert (burden["coke_p03_angles"], burden["coke_charge_pattern"]) == (31.5, "C 4 3 2")
        assert burden["coke_p11_rings"] == 0.0  # the Streamlit number input sends 0.0 for an empty field
        assert burden["burden_changing_purpose"] == "Open the centre" and burden["coke_discharge_time"] == 92.0
        assert audit_rows(a) == []
        assert [r["action"] for r in audit_rows(b)] == [
            "user.create",
            "hopper.assign",
            "hopper.assign",
            "burden.update",
            "hopper.delete",
            "burden.delete",
        ]


@needs_postgres
def test_a_back_dated_assignment_copies_the_snapshot_then_in_force_and_keeps_the_history(database, store) -> None:
    before = dump(database)["hoppers"]
    current = plant_admin.hopper_view(store)["current"]["materials"]
    plant_admin.assign_hoppers(store, ACTOR, hopper_form(current, **ASSIGN), T_ASSIGN)
    later = plant_admin.hopper_view(store)["current"]["materials"]
    result = plant_admin.assign_hoppers(store, ACTOR, hopper_form(later, **BACK), T_BACK)
    assert result["written"] and result["changed"] == ["hopper_03"]
    after = dump(database)["hoppers"]
    assert after[0] == before[0]  # the imported snapshot is untouched
    imported, back, assigned = after
    assert back["date_time"] == T_BACK and assigned["date_time"] == T_ASSIGN
    # The back-dated snapshot is the imported one with hopper_03 set; the later snapshot keeps its own values.
    assert {h: back[h] for h in HOPPER_COLUMNS if h != "hopper_03"} == {
        h: imported[h] for h in HOPPER_COLUMNS if h != "hopper_03"
    }
    assert assigned["hopper_03"] == imported["hopper_03"] != back["hopper_03"]
    # The mapping in force is still the latest snapshot.
    view = plant_admin.hopper_view(store)
    assert view["current"]["id"] == view["history"][0]["id"] and view["current"]["date_time"] == T_ASSIGN
    assert [r["date_time"] for r in view["history"]] == [T_ASSIGN, T_BACK, T_IMPORT]


@needs_postgres
def test_every_write_has_its_audit_row_with_the_stored_values(database, store) -> None:
    created = plant_admin.register_user(store, ACTOR, "qa", "pw", "user")
    current = plant_admin.hopper_view(store)["current"]
    assigned = plant_admin.assign_hoppers(store, ACTOR, hopper_form(current["materials"], **ASSIGN), T_ASSIGN)
    view = plant_admin.burden_view(store)
    changed = plant_admin.update_burden(store, ACTOR, {"coke_p03_angles": 31.5, "coke_p11_rings": None}, T_ASSIGN)
    assert changed["changed"] == ["coke_p03_angles"]  # p11 is NULL in force: unchanged
    deleted = plant_admin.delete_burden_history(store, ACTOR, [view["current"]["id"]])

    user, hopper, burden, removed = audit_rows(database)
    assert [r["id"] for r in (user, hopper, burden, removed)] == [
        created["audit_id"],
        assigned["audit_id"],
        changed["audit_id"],
        deleted["audit_id"],
    ]
    assert (user["actor"], user["role"], user["client_ip"]) == ("adm", "admin", "10.0.0.5")
    assert (user["object_type"], user["object_id"], user["before"]) == ("user", "qa", None)
    assert user["after"] == {"username": "qa", "role": "user", "is_active": True, "roles": ["user"]}
    assert user["details"] == {"user_id": created["user"]["id"]}
    assert "pw" not in str(user) and password_digest("pw") not in str(user)

    assert (hopper["object_type"], hopper["object_id"]) == ("hopper_mapping", str(assigned["id"]))
    assert hopper["details"]["based_on"] == current["id"] and hopper["details"]["changed"] == ASSIGN
    changes = {c["path"]: (c["before"], c["after"]) for c in audit.changes(hopper["before"], hopper["after"])}
    assert set(changes) == {"date_time", "hopper_02", "hopper_05", "hopper_07"}
    assert changes["hopper_05"][1] is None and changes["date_time"][1] == T_ASSIGN.isoformat()

    assert (burden["object_type"], burden["object_id"]) == ("burden_distribution", str(changed["id"]))
    assert {c["path"] for c in audit.changes(burden["before"], burden["after"])} == {"date_time", "coke_p03_angles"}
    assert (removed["object_id"], removed["after"]) == (str(view["current"]["id"]), None)
    assert removed["before"][str(view["current"]["id"])]["coke_charge_pattern"] == "C 3 2 2"


@needs_postgres
def test_a_write_whose_audit_row_fails_is_rolled_back(database, store, monkeypatch) -> None:
    before = dump(database)
    current = plant_admin.hopper_view(store)["current"]["materials"]
    burden_id = plant_admin.burden_view(store)["history"][0]["id"]

    def refuse(*args, **kwargs):
        raise RuntimeError("audit log refused")

    monkeypatch.setattr(audit, "audit_row", refuse)
    with pytest.raises(RuntimeError, match="audit log refused"):
        plant_admin.assign_hoppers(store, ACTOR, hopper_form(current, **ASSIGN), T_ASSIGN)
    with pytest.raises(RuntimeError, match="audit log refused"):
        plant_admin.register_user(store, ACTOR, "qa", "pw", "user")
    with pytest.raises(RuntimeError, match="audit log refused"):
        plant_admin.delete_burden_history(store, ACTOR, [burden_id])
    assert dump(database) == before and audit_rows(database) == []


@needs_postgres
def test_refused_and_empty_writes_change_nothing(database, store) -> None:
    before = dump(database)
    view = plant_admin.hopper_view(store)
    current = view["current"]["materials"]
    assert plant_admin.assign_hoppers(store, ACTOR, current, T_ASSIGN) == {
        "written": False,
        "id": None,
        "changed": [],
        "audit_id": None,
    }
    burden = plant_admin.burden_view(store)["current"]["values"]
    assert not plant_admin.update_burden(store, ACTOR, burden, T_ASSIGN)["written"]
    refused = [
        (plant_admin.Invalid, lambda: plant_admin.assign_hoppers(store, ACTOR, {"hopper_19": "Quartz"}, T_ASSIGN)),
        (plant_admin.Invalid, lambda: plant_admin.assign_hoppers(store, ACTOR, {"hopper_01": "Gold"}, T_ASSIGN)),
        (plant_admin.Invalid, lambda: plant_admin.assign_hoppers(store, ACTOR, current, T_ASSIGN.replace(tzinfo=None))),
        (plant_admin.Invalid, lambda: plant_admin.update_burden(store, ACTOR, {"coke_p01_rings": "x"}, T_ASSIGN)),
        (plant_admin.Invalid, lambda: plant_admin.update_burden(store, ACTOR, {"coke_p12_rings": 1.0}, T_ASSIGN)),
        (plant_admin.Invalid, lambda: plant_admin.register_user(store, ACTOR, " ", "pw", "user")),
        (plant_admin.Invalid, lambda: plant_admin.register_user(store, ACTOR, "qa", "pw", "root")),
        (plant_admin.Conflict, lambda: plant_admin.register_user(store, ACTOR, "adm", "pw", "user")),
        (plant_admin.Invalid, lambda: plant_admin.delete_hopper_history(store, ACTOR, [])),
        (plant_admin.NotFound, lambda: plant_admin.delete_hopper_history(store, ACTOR, [view["current"]["id"], 999])),
    ]
    for error, call in refused:
        with pytest.raises(error):
            call()
    assert dump(database) == before and audit_rows(database) == []


@needs_postgres
def test_the_views_read_the_accounts_hoppers_and_burden(database, store) -> None:
    users = plant_admin.users_view(store)
    assert users["roles"] == ["user", "supervisor", "admin"]
    assert [(u["username"], u["role"], u["roles"], u["is_active"]) for u in users["users"]] == [
        ("adm", "admin", ["admin"], True),
        ("admin", "admin", ["admin"], True),
    ]
    assert all(u["created_at"].tzinfo is not None for u in users["users"])

    hoppers = plant_admin.hopper_view(store)
    assert [h["code"] for h in hoppers["hoppers"]] == [f"hopper_{i:02d}" for i in range(1, 19)]  # 19 is inactive
    assert hoppers["hoppers"][0]["display_name"] == "Bin 1"
    assert hoppers["materials"][0] == "UNASSIGNED" and "LLOYDS FINES OVER SIZE" in hoppers["materials"]
    current = hoppers["current"]
    assert current["date_time"] == T_IMPORT and current["user_name"] is None and current["source_type"] == "import"
    assert current["materials"]["hopper_01"] == "Sinter" and current["materials"]["hopper_04"] == "UNASSIGNED"
    assert "hopper_19" not in current["materials"]

    plant_admin.update_burden(store, ACTOR, {"coke_p01_rings": 4.0}, T_ASSIGN + timedelta(days=36500))  # future
    burden = plant_admin.burden_view(store, now=T_ASSIGN)
    assert burden["at"] == T_ASSIGN and burden["current"]["date_time"] == T_IMPORT  # in force now, not the future
    assert burden["current"]["values"]["coke_discharge_time"] == 92.0
    assert burden["history"][0]["user_name"] == "adm" and len(burden["history"]) == 2


@needs_postgres
def test_without_the_plant_tables_the_views_are_unavailable(pg_server) -> None:
    with fresh_database(pg_server) as url:
        command.upgrade(alembic_config(url), "head")  # the twin schema only
        twin = TwinStore.from_url(url)
        assert twin is not None
        try:
            for view in (plant_admin.users_view, plant_admin.hopper_view, plant_admin.burden_view):
                with pytest.raises(plant_admin.Unavailable, match="identity, plant_master and ops_config"):
                    view(twin)
        finally:
            twin.dispose()
