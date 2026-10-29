"""Validation and STRATA-Learn console (SV-4.5, #308): the console's numbers equal ``learn.run.shadow_run`` on the same
evidence (relative difference <= 1e-6, through the stored files), the asset's out-of-sample results are passed as
stated, and decisions on the register proposal write through the fact-register editor (SV-4.2) with their audit rows
in the same transaction. SQLite twin store."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select

from data.twin.engine import build_engine, session_factory
from data.twin.models import AuditLog, FactsVersion, TwinBase
from data.twin.repositories import Actor, ReviewRefused, TwinStore
from services import facts, learn, strata
from services import learn_console as console
from services import store as store_service
from tests.pg_server import needs_postgres
from tests.test_services_facts import pg_store  # noqa: F401 (the PostgreSQL fixture)
from utils.burden_engine.facts import REGISTER_PATH
from utils.burden_engine.learn.evidence import PARAMS, EvidenceStore
from utils.burden_engine.learn.run import shadow_run
from utils.burden_engine.omega_model import MODEL_PATH

SPEC = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
REGISTER = json.loads(REGISTER_PATH.read_text(encoding="utf-8"))
#: The asset deployed on 1 Jun 2026 instead of 25 Oct: the gate evaluates and the CUSUM runs on the champion's errors.
EARLY = dict(SPEC, version="STRATA 1.5 (2026-06-01): test deployment")
T0 = datetime(2026, 10, 29, 4, 30, tzinfo=UTC)
H = timedelta(hours=1)
#: Console field -> field of the run's family table.
FAMILY_FIELDS = (
    ("realised_kg_thm", "realised"), ("fixed_omega_kg_thm", "fixed_omega"), ("strata_kg_thm", "strata"),
    ("rmse_fixed_kg_thm", "rmse_fixed"), ("rmse_strata_kg_thm", "rmse_strata"), ("skill", "skill"),
    ("bias_kg_thm", "bias"), ("se_bias_kg_thm", "se_bias"),
)  # fmt: skip
ALICE = Actor("alice", "admin", "10.0.0.5")
BOB = Actor("bob", "admin")


def close(a: Any, b: Any, rel: float = 1e-6) -> bool:
    if b is None or (isinstance(b, float) and not math.isfinite(b)):
        return a is None
    return a is not None and abs(float(a) - float(b)) <= rel * max(abs(float(b)), 1e-12)


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """Shadow runs on the seed: the deployed asset and the early deployment, each with the in-memory reference."""
    out = {}
    for name, spec in (("deployed", SPEC), ("early", EARLY)):
        folder = tmp_path_factory.mktemp(name)
        learn.shadow(folder, spec=spec)
        out[name] = (folder, shadow_run(EvidenceStore(), dict(spec), dict(REGISTER)))
    return out


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


def rows(store: TwinStore) -> list[AuditLog]:
    with store.sessions() as session:
        return list(session.scalars(select(AuditLog).order_by(AuditLog.id)))


def versions(store: TwinStore) -> int:
    with store.sessions() as session:
        return int(session.scalar(select(func.count()).select_from(FactsVersion)) or 0)


# ------------------------------------------------------------------------------------------------------ parity


@pytest.mark.parametrize("case", ["deployed", "early"])
def test_the_console_equals_the_shadow_run(stored, case) -> None:
    """Acceptance: the numbers equal learn.run output (relative difference <= 1e-6)."""
    folder, ref = stored[case]
    v = console.view(None, directory=folder)
    g, rg = v["gate"], ref["gate"]
    assert (g["status"], g["n"], g["window"], g["promote"], g["reason"]) == (
        rg["status"], rg["n"], rg["window"], rg["promote"], rg["reason"],
    )  # fmt: skip
    for name in ("mse_champion", "mse_challenger", "gain", "dm_t"):
        assert close(g[name], rg.get(name)), name
    assert (g["t_min"], g["window"], g["lags"]) == (2.0, 60, 5)
    d, rd = v["cusum"], ref["drift"]
    assert close(d["s_plus"], rd["s_plus"]) and close(d["s_minus"], rd["s_minus"])
    assert (d["alarm"], d["ever_alarm"], d["basis"], d["first_alarm"]) == (
        rd["alarm"], rd["ever_alarm"], rd["basis"], rd["first_alarm"],
    )  # fmt: skip
    assert str(d["first_alarm_date"]) == str(rd.get("first_alarm_date"))
    assert [(s["label"], s["predictor"]) for s in v["skill"]] == [(k, k.split(" ")[0]) for k in ref["skill"]]
    for s in v["skill"]:
        assert close(s["omega_day"], ref["skill"][s["label"]]["omega_day"])
        assert close(s["coke_day"], ref["skill"][s["label"]]["coke_day"])
    assert [f["family"] for f in v["families"]] == [r["family"] for r in ref["families"]]
    for f, r in zip(v["families"], ref["families"], strict=True):
        assert f["n"] == r["n"] and f["verdict"] == r["verdict"]
        for a, b in FAMILY_FIELDS:
            assert close(f[a], r[b]), (f["family"], a)
    t, hist = v["trajectory"], ref["history"]
    assert t["dates"] == [str(x.date()) for x in hist["date"]] and t["n"] == ref["state"]["n"]
    for p in t["params"]:
        assert all(close(a, b) for a, b in zip(p["mean"], hist[p["name"]], strict=True)), p["name"]
        assert all(close(a, b) for a, b in zip(p["sd"], hist[p["name"] + "_sd"], strict=True)), p["name"]
        assert close(p["latest"], ref["state"]["mean"][p["name"]]) and close(p["champion"], ref["champion"][p["name"]])
    assert {p["name"] for p in t["params"] if p["fixed"]} == {"lam_production", "a_moisture"}
    assert [p["name"] for p in t["params"]] == list(PARAMS)
    for a, b in (
        ("observed", "y"),
        ("challenger", "pred"),
        ("challenger_sd", "pred_sd"),
        ("champion", "pred_champion"),
    ):
        assert all(close(x, y) for x, y in zip(t[a], hist[b], strict=True)), a
    p, rp = v["proposal"], ref["proposal"]
    assert (p["n_auto"], p["n_approval"], str(p["day"])) == (rp["n_auto"], rp["n_approval"], rp["date"])
    assert [i["key"] for i in p["items"]] == [it["id"] or f"new:{it['change']['family']}" for it in rp["items"]]
    for item, it in zip(p["items"], rp["items"], strict=True):
        assert (item["kind"], item["automatic"], item["reason"]) == (it["kind"], it["automatic"], it["reason"])
        for c in item["register_changes"]:
            assert (
                c["proposed"] == it["change"][c["field"]]
                if c["field"] == "status"
                else close(c["proposed"], it["change"]["value"])
            )
    assert v["run"]["n_days"] == ref["n_days"] == 191 and str(v["asset"]["deployed"]) == ref["deployed"]


def test_the_early_deployment_evaluates_the_gate(stored) -> None:
    v = console.view(None, directory=stored["early"][0])
    assert v["gate"]["status"] == "evaluated" and v["gate"]["n"] >= 60 and v["gate"]["dm_t"] is not None
    assert v["cusum"]["basis"] == "champion errors after deployment"


def test_the_asset_results_are_those_of_the_strata_page(stored) -> None:
    v = console.view(None, directory=stored["deployed"][0])["validation"]
    assert v == strata.validation(SPEC)
    assert (v["expanding"]["rmse_strata"], v["expanding"]["rmse_fixed"]) == (5.99, 6.39)
    assert (v["strict"]["rmse_fixed"], v["strict"]["rmse_strata"], v["strict"]["skill_coke_day"]) == (8.23, 8.0, 0.055)


def test_the_families_carry_the_val_fact_in_force(stored) -> None:
    fams = {f["family"]: f for f in console.view(None, directory=stored["deployed"][0])["families"]}
    n1 = fams["N1 No pellet, sinter + CLO"]
    assert n1["fact"]["id"] == "VAL-01" and n1["fact"]["status"] == "Neutral" and n1["fact"]["value_kg_thm"] == 296.0
    assert all(f["fact"] is not None for f in fams.values())


def test_without_a_run_the_console_shows_the_asset_only(tmp_path) -> None:
    v = console.view(None, directory=tmp_path)
    assert v["run"] is None and v["proposal"] is None and v["trajectory"] is None and v["runs"] == []
    assert v["validation"]["expanding"]["rmse_strata"] == 5.99 and v["job"]["schedule"] == "03:00"
    with pytest.raises(console.NoRun):
        console.decide(None, ALICE, proposal="x", keys=["VAL-02"], approve=False, directory=tmp_path)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------------------ items


def test_items_that_an_overlay_cannot_carry_are_not_approvable() -> None:
    fact = dict(id="GAS-20", statement="s", unit="", value=0.12, status="Validated", retired=False)
    drift = console.item_view(dict(id="GAS-20", kind="drift", automatic=False, change={}, reason="r"), fact, None)
    assert not drift["approvable"] and "recalibration" in drift["why"] and drift["decidable"]
    promo = console.item_view(dict(id="ASSET", kind="promotion", automatic=False, change={}, reason="r"), None, None)
    assert not promo["approvable"] and promo["fact_id"] is None and "release of the asset" in promo["why"]
    new = console.item_view(dict(id=None, kind="new", automatic=False, change={"family": "X"}, reason="r"), None, None)
    assert new["key"] == "new:X" and not new["approvable"]
    same = console.item_view(
        dict(id="GAS-20", kind="value", automatic=True, change={"value": 0.12, "se": 0.1}, reason="r"), fact, None
    )
    assert not same["approvable"] and same["not_carried"] == ["se"] and same["register_changes"] == []


# ------------------------------------------------------------------------------------------------------ decisions


def _hash(folder) -> str:
    return console.view(None, directory=folder)["proposal"]["hash"]


def test_an_approval_writes_a_pending_version_through_the_editor_with_its_audit_row(stored, store) -> None:
    """Acceptance: approvals write through SV-4.2 with audit."""
    folder, ref = stored["deployed"]
    h = _hash(folder)
    out = console.decide(
        store, ALICE, proposal=h, keys=["VAL-02", "GAS-14"], approve=True, note="Monthly", directory=folder, at=T0
    )
    ver = out["facts_version"]
    assert out["decision"] == "approved" and ver["version"] == 1 and ver["status"] == "pending"
    assert not ver["second_review"] and ver["facts"] == ["GAS-14", "VAL-02"]
    with store.sessions() as session:
        row = session.scalars(select(FactsVersion)).one()
    items = {it["id"]: it for it in ref["proposal"]["items"]}
    assert row.overlay == {"VAL-02": {"value": items["VAL-02"]["change"]["value"]},
                           "GAS-14": {"value": items["GAS-14"]["change"]["value"]}}  # fmt: skip
    assert row.note.startswith(f"STRATA-Learn proposal {ref['proposal']['date']} ({h}): VAL-02 n 9 -> 9")
    assert row.note.endswith(". Monthly")
    decision, propose = rows(store)
    obj = f"{ref['proposal']['date']}/{h}"
    assert (decision.action, decision.object_type, decision.object_id) == ("learn.approve", "learn_proposal", obj)
    assert decision.id == out["audit_id"] and decision.at == propose.at
    assert decision.before == {"items": {"VAL-02": "open", "GAS-14": "open"}}
    assert decision.after == {"items": {"VAL-02": "proposed", "GAS-14": "proposed"}, "facts_version": 1}
    assert decision.details["changes"]["GAS-14"] == {"value": items["GAS-14"]["change"]["value"]}
    assert (propose.action, propose.object_id) == ("facts.propose", "1")
    assert propose.details["learn_proposal"] == obj and propose.details["learn_items"] == ["VAL-02", "GAS-14"]
    # the console shows the decision, and a second decision on the items is refused
    shown = {i["key"]: i for i in console.view(store, directory=folder)["proposal"]["items"]}
    assert shown["VAL-02"]["state"] == "proposed" and not shown["VAL-02"]["decidable"]
    assert shown["VAL-02"]["decision"]["by"] == "alice" and shown["VAL-02"]["decision"]["version_status"] == "pending"
    for approve in (True, False):
        with pytest.raises(console.AlreadyDecided, match="VAL-02 was approved by alice, facts version 1 pending"):
            console.decide(store, BOB, proposal=h, keys=["VAL-02"], approve=approve, directory=folder)
    assert len(rows(store)) == 2


def test_a_version_the_editor_rejects_opens_the_items_again(stored, store) -> None:
    folder = stored["deployed"][0]
    h = _hash(folder)
    console.decide(store, ALICE, proposal=h, keys=["VAL-02"], approve=True, directory=folder, at=T0)
    facts.review(store, BOB, 1, approve=False, at=T0 + H)
    item = {i["key"]: i for i in console.view(store, directory=folder)["proposal"]["items"]}["VAL-02"]
    assert item["state"] == "not_taken" and item["decidable"]
    out = console.decide(store, ALICE, proposal=h, keys=["VAL-02"], approve=True, directory=folder, at=T0 + 2 * H)
    assert out["facts_version"]["version"] == 2 and out["facts_version"]["parent_version"] == 0
    facts.review(store, BOB, 2, approve=True, at=T0 + 3 * H)
    v = console.view(store, directory=folder)
    item = {i["key"]: i for i in v["proposal"]["items"]}["VAL-02"]
    assert (
        item["state"] == "approved" and item["register_changes"] == [] and v["facts_register"]["in_force_version"] == 2
    )
    assert {f["family"]: f for f in v["families"]}["N2 No pellet, with Gomti"]["fact"]["value_kg_thm"] == item[
        "change"
    ]["value"]


def test_a_status_change_needs_a_second_administrator(stored, store) -> None:
    folder = stored["deployed"][0]
    view = {i["key"]: i for i in console.view(store, directory=folder)["proposal"]["items"]}
    status_items = [k for k, i in view.items() if any(c["field"] == "status" for c in i["register_changes"])]
    assert status_items, "the seed proposal has a status change"
    out = console.decide(store, ALICE, proposal=_hash(folder), keys=status_items[:1], approve=True, directory=folder)
    assert out["facts_version"]["second_review"]
    with pytest.raises(ReviewRefused):
        facts.review(store, ALICE, 1, approve=True)


def test_a_rejection_is_audited_and_final(stored, store) -> None:
    folder, ref = stored["deployed"]
    h = _hash(folder)
    with pytest.raises(console.DecisionRefused, match="GAS-23: Nothing an overlay carries"):
        console.decide(store, ALICE, proposal=h, keys=["GAS-23"], approve=True, directory=folder)
    out = console.decide(
        store, ALICE, proposal=h, keys=["GAS-23", "VAL-03"], approve=False, note="noise", directory=folder, at=T0
    )
    assert out["decision"] == "rejected" and out["facts_version"] is None and versions(store) == 0
    (entry,) = rows(store)
    assert (entry.action, entry.object_id, entry.id) == (
        "learn.reject",
        f"{ref['proposal']['date']}/{h}",
        out["audit_id"],
    )
    assert entry.before == {"items": {"GAS-23": "open", "VAL-03": "open"}}
    assert entry.after == {"items": {"GAS-23": "rejected", "VAL-03": "rejected"}}
    assert entry.details["note"] == "noise" and entry.details["items"] == ["GAS-23", "VAL-03"]
    item = {i["key"]: i for i in console.view(store, directory=folder)["proposal"]["items"]}["VAL-03"]
    assert item["state"] == "rejected" and item["decision"]["note"] == "noise"
    with pytest.raises(console.AlreadyDecided):
        console.decide(store, BOB, proposal=h, keys=["VAL-03"], approve=True, directory=folder)


def test_a_decision_on_another_proposal_or_item_is_refused(stored, store) -> None:
    folder = stored["deployed"][0]
    with pytest.raises(console.ProposalChanged, match="reload the console"):
        console.decide(store, ALICE, proposal="0" * 16, keys=["VAL-02"], approve=True, directory=folder)
    with pytest.raises(console.DecisionRefused, match="no item VAL-99"):
        console.decide(store, ALICE, proposal=_hash(folder), keys=["VAL-99"], approve=False, directory=folder)
    assert rows(store) == [] and versions(store) == 0


def test_a_decision_made_meanwhile_rolls_the_approval_back(stored, store, monkeypatch) -> None:
    """The items are checked again inside the transaction: a decision recorded between the check and the write
    refuses the approval, and neither the version nor an audit row is kept."""
    folder = stored["deployed"][0]
    h = _hash(folder)
    console.decide(store, BOB, proposal=h, keys=["VAL-04"], approve=False, directory=folder, at=T0)
    monkeypatch.setattr(console, "decisions", lambda *_: {})  # the check before the transaction misses it
    with pytest.raises(console.AlreadyDecided, match="VAL-04 was rejected by bob"):
        console.decide(store, ALICE, proposal=h, keys=["VAL-04"], approve=True, directory=folder, at=T0 + H)
    assert versions(store) == 0 and [r.action for r in rows(store)] == ["learn.reject"]


def test_a_new_run_with_other_items_is_a_new_proposal(stored, store) -> None:
    folder = stored["deployed"][0]
    state = console.read_state(folder)
    other = dict(state["proposal"], items=state["proposal"]["items"][:1])
    assert console.proposal_hash(other) != console.proposal_hash(state["proposal"])
    assert console.proposal_object(state["proposal"]).startswith(state["proposal"]["date"] + "/")


# ------------------------------------------------------------------------------------------------- PostgreSQL


@needs_postgres
def test_on_postgresql_decisions_commit_with_their_versions(stored, pg_store) -> None:  # noqa: F811
    """Advisory lock, JSONB details and the append-only audit log: the approval's version and both audit rows commit
    together; the decision is read back and a second decision is refused."""
    folder = stored["deployed"][0]
    store_service.use(pg_store)
    facts.approved_overlay.clear()
    try:
        h = _hash(folder)
        out = console.decide(pg_store, ALICE, proposal=h, keys=["VAL-02"], approve=True, directory=folder, at=T0)
        console.decide(pg_store, BOB, proposal=h, keys=["VAL-03"], approve=False, directory=folder, at=T0 + H)
        obj = console.view(pg_store, directory=folder)["proposal"]["object_id"]
        actions = sorted(r.action for r in pg_store.audit.query(object_type="learn_proposal", object_id=obj))
        assert actions == ["learn.approve", "learn.reject"]
        assert [r.action for r in pg_store.audit.query(object_type="facts_version")] == ["facts.propose"]
        states = {i["key"]: i["state"] for i in console.view(pg_store, directory=folder)["proposal"]["items"]}
        assert (states["VAL-02"], states["VAL-03"], out["facts_version"]["version"]) == ("proposed", "rejected", 1)
        with pytest.raises(console.AlreadyDecided):
            console.decide(pg_store, BOB, proposal=h, keys=["VAL-02"], approve=False, directory=folder)
    finally:
        store_service.reset()
        facts.approved_overlay.clear()
