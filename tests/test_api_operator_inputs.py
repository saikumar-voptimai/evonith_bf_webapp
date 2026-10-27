"""Operator inputs API (SV-3.1, #298): GET the version in force or at a date (role user), PUT a new version (role
supervisor, audited), 409 / 422 / 503 problems, units in the schema, and an edit in either UI (page 9 in Streamlit,
the SPA through the API) read by page 9's loader, page 11's yard loader and the API, on SQLite and on PostgreSQL."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from alembic import command
from fastapi.testclient import TestClient

from data.bmo import context_provider
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import Actor, TwinStore
from services import operator_inputs as svc
from services import store as store_service
from tests.pg_server import fresh_database, needs_postgres, postgres_server
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from tests.test_burden_engine_inputs import REF, _ore
from tests.test_twin_store import alembic_config

URL = "/api/operator-inputs"
T0 = datetime(2026, 10, 20, 4, 30, tzinfo=UTC)  # Tue 20 Oct 2026 10:00 IST
DAY = timedelta(days=1)
SINTER = "sinter_sp_02"


def _sqlite(tmp_path) -> TwinStore:
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    return TwinStore(session_factory(engine), engine)


def _use(store: TwinStore | None) -> None:
    store_service.use(store)
    svc.invalidate()


@pytest.fixture()
def twin(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "_modified", lambda _path: T0 - 3 * DAY)
    store = _sqlite(tmp_path)
    _use(store)
    yield store
    store_service.reset()
    svc.invalidate()
    store.dispose()


@pytest.fixture()
def client(twin):
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def call(client: TestClient, method: str, user: str, url: str = URL, **kw: Any):
    return client.request(method, url, headers={"Cookie": client.cookies_by_user[user]}, **kw)  # type: ignore[attr-defined]


def sinter_price(body: dict) -> float:
    return body["inputs"]["ore_editor"]["rows"][SINTER]["price_rs_per_mt"]


def ore_rows(price: float) -> dict:
    """The ore section as the SPA sends it: the YAML's rows with a new sinter price."""
    section = json.loads(json.dumps(svc.read_yaml()["ore_editor"]))
    section["rows"][SINTER]["price_rs_per_mt"] = price
    return section


# --------------------------------------------------------------------------------------------------------- reading
def test_get_answers_from_the_yaml_file_until_the_first_version(client) -> None:
    assert client.get(URL).status_code == 401
    body = call(client, "GET", "op").json()
    assert (body["source"], body.get("version"), body["reason"]) == ("yaml", None, svc.EMPTY)
    assert body["inputs"] == svc.normalise(svc.read_yaml())


def test_without_a_store_get_reads_the_file_and_the_rest_answers_503(client) -> None:
    _use(None)
    body = call(client, "GET", "op").json()
    assert (body["source"], body["reason"]) == ("yaml", svc.NO_STORE)
    at = call(client, "GET", "op", params={"at": "2026-10-20"})
    assert at.status_code == 503 and "needs the twin store" in at.json()["detail"]
    put = call(client, "PUT", "sup", json={"inputs": {"model_inputs": {"charge_mass_mt": 26.0}}})
    assert put.status_code == 503 and put.headers["content-type"] == "application/problem+json"


def test_every_past_version_is_retrievable_by_date(twin, client) -> None:
    sv1 = Actor("sv1", "supervisor")
    for value, when in [(9000.0, T0), (9100.0, T0 + DAY), (9200.0, T0 + DAY + timedelta(hours=5))]:
        svc.write(svc.replace_sections({"ore_editor": ore_rows(value)}), actor=sv1, origin="api", at=when)
    expected = {
        "2026-10-16": None,  # before the YAML file's modification time (version 1 starts there)
        "2026-10-17": (1, 8784.0),
        "2026-10-20": (2, 9000.0),
        "2026-10-21": (4, 9200.0),  # the last version of the day
        "2026-11-30": (4, 9200.0),
    }
    for day, want in expected.items():
        r = call(client, "GET", "op", params={"at": day})
        if want is None:
            assert r.status_code == 404 and r.headers["content-type"] == "application/problem+json"
            continue
        body = r.json()
        assert (body["version"], sinter_price(body), body["source"]) == (*want, "neon"), day
        assert pd.Timestamp(body["at"]) == pd.Timestamp(svc.as_of(date.fromisoformat(day)))
    morning = call(client, "GET", "op", params={"at": "2026-10-21T12:00:00+05:30"}).json()  # between v3 and v4
    assert (morning["version"], sinter_price(morning), morning["author"], morning["origin"]) == (
        3,
        9100.0,
        "sv1",
        "api",
    )
    assert pd.Timestamp(morning["valid_upto"]) == pd.Timestamp(T0 + DAY + timedelta(hours=5))
    assert call(client, "GET", "op", params={"at": "21 Oct"}).status_code == 422


# --------------------------------------------------------------------------------------------------------- writing
def test_put_needs_a_supervisor_and_is_audited(twin, client) -> None:
    edit = {"inputs": {"ore_editor": ore_rows(9050.0)}, "base_version": 0, "note": " cost sheet 27 Oct "}
    assert call(client, "PUT", "op", json=edit).status_code == 403
    r = call(client, "PUT", "sup", json=edit)
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["version"], body["changed"], body["author"], body["origin"], body["note"]) == (
        2, True, "sup", "api", "cost sheet 27 Oct"
    )  # fmt: skip
    assert sinter_price(call(client, "GET", "op").json()) == 9050.0
    audit = list(reversed(twin.audit.query(object_type="operator_inputs")))
    assert [(a.action, a.object_id, a.actor) for a in audit] == [
        ("operator_inputs.seed", "1", "sup"),
        ("operator_inputs.update", "2", "sup"),
    ]
    assert audit[1].role == "supervisor" and audit[1].client_ip
    assert audit[1].details["changes"] == [
        {"path": f"ore_editor.rows.{SINTER}.price_rs_per_mt", "before": 8784.0, "after": 9050.0}
    ]
    again = call(client, "PUT", "adm", json={**edit, "base_version": 2}).json()
    assert (again["version"], again["changed"]) == (2, False)  # nothing changed: no version, no audit row
    assert len(twin.audit.query(object_type="operator_inputs")) == 2


def test_put_conflicts_invalid_values_and_section_removal(twin, client) -> None:
    hm = {"hot_metal_chemistry": {"carbon_pct": 4.7, "silicon_pct": 0.45}}
    assert call(client, "PUT", "sup", json={"inputs": hm, "base_version": 0}).json()["version"] == 2
    stale = call(client, "PUT", "sup", json={"inputs": {"model_inputs": {"charge_mass_mt": 26.0}}, "base_version": 1})
    assert stale.status_code == 409 and "version 2 is current" in stale.json()["detail"]
    bad = {"model_inputs": {"target_slag_basicity_min": 1.2, "target_slag_basicity_max": 1.0}}
    invalid = call(client, "PUT", "sup", json={"inputs": bad})
    assert invalid.status_code == 422 and "lower bound 1.2 above upper bound 1" in invalid.json()["detail"]
    unknown = call(client, "PUT", "sup", json={"inputs": {"model_inputs": {"charge_mass_t": 26.0}}})
    assert unknown.status_code == 422 and unknown.json()["errors"]
    assert call(client, "GET", "op").json()["version"] == 2  # nothing of the refused requests was stored

    removed = call(client, "PUT", "sup", json={"inputs": {"hot_metal_chemistry": None}, "base_version": 2}).json()
    assert removed["version"] == 3 and "hot_metal_chemistry" not in removed["inputs"]
    assert removed["inputs"]["ore_editor"] == svc.normalise(svc.read_yaml())["ore_editor"]  # sections left out: kept


# ------------------------------------------------------------------------------------------------------------ units
UNITS = ("Rs/t", "t", "t/d", "%", "kg/tHM", "kg/charge", "charges/h", "dimensionless", "count")
SCHEMAS = ("OperatorInputsView", "OperatorInputsSaved", "OperatorInputsUpdate", "OreInputRow", "FluxInputRow",
           "SlagChargingInputs", "FuelInputRow", "HotMetalChemistryInputs", "DustInputRow")  # fmt: skip


@pytest.mark.parametrize("schema", SCHEMAS)
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    missing = []
    for name, prop in spec.get("properties", {}).items():
        types = {prop.get("type")} | {p.get("type") for p in prop.get("anyOf", [])}
        if not types & {"number", "integer"}:
            continue
        unit = (prop.get("description") or "").rsplit(",", 1)[-1].strip()
        if unit not in UNITS:
            missing.append(f"{name}: {prop.get('description')!r}")
    assert not missing, missing


# --------------------------------------------------------------------------------------- one source for every UI
class YardProvider(context_provider.EvonithBmoContextProvider):
    """Page 11's provider with the reference yard's chemistry instead of Neon; the operator inputs are its own."""

    def __init__(self) -> None:  # no database
        pass

    def build_ore_inputs(self, mode: str = "latest", window_days: int | None = None):
        return [_ore(m) for m in REF["materials"]], {"warnings": []}


class StreamlitPages:
    """The Streamlit read paths: page 9's cached operator inputs and page 11's cached yard."""

    def __init__(self) -> None:
        from ui.bmo import operator_inputs as page9
        from ui.burden_optimiser import loaders as page11

        self.page9, self.page11 = page9, page11
        page9.cached_inputs.clear()
        page11.yard.clear()

    def page9_price(self) -> float:
        inputs = self.page9.current_inputs(svc.YAML_PATH)
        return inputs["ore_editor"]["rows"][SINTER]["price_rs_per_mt"]

    def page11_price(self) -> tuple[float, dict]:
        yard, source, message = self.page11.yard(json.dumps({"yard": {}}), 0, svc.token())
        assert source == "live", message
        return {m["id"]: m["price"] for m in yard["materials"]}[SINTER], yard["operator_inputs"]


def edit_in_either_ui(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(context_provider, "EvonithBmoContextProvider", YardProvider)
    pages = StreamlitPages()
    assert pages.page9_price() == pages.page11_price()[0] == 8784.0  # the YAML file: no version yet

    # 1. A write through the service (what both UIs call) is read by page 9, page 11 and the API.
    svc.write(svc.replace_sections({"ore_editor": ore_rows(9000.0)}), actor=Actor("sv1", "supervisor"), origin="api")
    price11, provenance = pages.page11_price()
    assert pages.page9_price() == price11 == sinter_price(call(client, "GET", "op").json()) == 9000.0
    assert (provenance["source"], provenance["version"], provenance["author"]) == ("neon", 2, "sv1")

    # 2. Page 9's save (Streamlit) appears in the API and on page 11.
    editor = pd.DataFrame([{"ore_id": SINTER, "selected": True, "price_rs_per_mt": 9100.0,
                            "min_share_pct": 58.0, "max_share_pct": 70.0}])  # fmt: skip
    saved = pages.page9.save_ore_editor_preferences(svc.YAML_PATH, editor)
    assert saved.changed and str(saved) == "twin.operator_inputs version 3"
    api = call(client, "GET", "op").json()
    assert (api["version"], api["origin"], sinter_price(api)) == (3, "streamlit", 9100.0)
    assert pages.page11_price()[0] == pages.page9_price() == 9100.0

    # 3. An edit through the API (the SPA) appears on page 9 and page 11.
    put = call(client, "PUT", "sup", json={"inputs": {"ore_editor": ore_rows(9200.0)}, "base_version": 3})
    assert put.status_code == 200 and put.json()["version"] == 4
    assert pages.page9_price() == pages.page11_price()[0] == 9200.0
    assert svc.load_at(datetime.now(UTC) - timedelta(microseconds=1)).version == 4


def test_an_edit_in_either_ui_appears_in_both(client, monkeypatch) -> None:
    edit_in_either_ui(client, monkeypatch)


@pytest.fixture(scope="module")
def pg_url():
    with postgres_server() as server, fresh_database(server) as url:
        command.upgrade(alembic_config(url), "head")
        yield url


@needs_postgres
def test_an_edit_in_either_ui_appears_in_both_on_postgresql(pg_url, monkeypatch) -> None:
    store = TwinStore.from_url(pg_url)
    assert store is not None
    _use(store)
    try:
        app = build()
        with TestClient(app, base_url="https://testserver") as c:
            c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
            edit_in_either_ui(c, monkeypatch)
        versions = store.operator_inputs.history()
        assert [(v.version, v.origin) for v in versions] == [(4, "api"), (3, "streamlit"), (2, "api"), (1, "seed")]
        assert all(v.valid_upto is not None for v in versions[1:]) and versions[0].valid_upto is None
    finally:
        store_service.reset()
        svc.invalidate()
        store.dispose()
