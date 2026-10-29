"""What-if analysis (SV-4.3, #306): the deck's what-ifs reproduced, STRATA forced blends against the case report,
what-if days, the register version of every result, and the saved cases with their audit rows (SQLite twin store).

The deck is "BF-02 coke model: what the latest results mean" (15 Oct 2026), page 10: the what-if table on the
Jul-Oct 2026 operation (coke 305.5 kg/tHM, PCI 184, nut coke 73, ηCO 42.2 %, top gas 208 °C) with the old
fixed-ηCO model and the two-zone model; its two-zone values are the register's COKE-01..08 and COKE-14. They were
computed on the Jul-Oct 2026 calibration at 15 g/Nm³ (``tests/fixtures/cal_2z_jul_oct_2026_case_report_15g.json``).
The STRATA plans reproduce the 16 Oct 2026 case report (``tests/test_burden_engine.py``, case 2).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from datetime import UTC, date, datetime

import pytest

from data.burden_engine.calibration import CalibrationInfo
from data.twin.engine import build_engine, session_factory
from data.twin.models import AuditLog, TwinBase
from data.twin.repositories import Actor, NotFound, TwinStore
from services import day_calibration, whatif
from services import store as store_service
from services.events import MemoryEventBus
from services.jobs import CATALOG, WhatIfForgeParams, WhatIfStrataParams
from utils.burden_engine import facts as register
from utils.burden_engine.engine import REFERENCE_INPUTS_PATH, Engine, Limits
from utils.burden_engine.facts import ApprovedOverlay, load_register

CASE_CAL = os.path.join(os.path.dirname(__file__), "fixtures", "cal_2z_jul_oct_2026_case_report_15g.json")
slow = pytest.mark.skipif(os.environ.get("BURDEN_ENGINE_FAST") == "1", reason="STRATA-ω network solves are slow")
ALICE = Actor("alice", "admin", "10.0.0.5")
BOB = Actor("bob", "admin")
T0 = datetime(2026, 10, 29, 6, 0, tzinfo=UTC)
#: The case report's Step 2 optimum (tests/test_burden_engine.py::test_case2_fuel_added).
CASE2_BLEND = {"sinter_sp_02": 58.0, "geomin_clo": 13.378254849223312, "lloyds_clo": 18.621745150776697,
               "lloyds_pellet": 10.0}  # fmt: skip


@pytest.fixture(scope="module")
def cal():
    with open(CASE_CAL, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def base(cal):
    return whatif.forge_compute(cal, [])["base"]


def lever(base, name, value):
    return {"name": name, "levers": {name: value}}


@pytest.fixture(scope="module")
def deck(cal, base):
    """The deck's levers (whatif_2z.py of the web model): PCI and nut coke +10 kg/tHM, HBT +50 °C, HM Si +0.1 %,
    HMT +10 °C, O2 +10 % (relative), blast moisture +5 g/Nm³, production +5 % (losses fixed in GJ/h), ω +0.01."""
    b = base["inputs"]
    cases = [
        lever(b, "pci_kg_thm", b["pci_kg_thm"] + 10),
        lever(b, "nut_coke_kg_thm", b["nut_coke_kg_thm"] + 10),
        lever(b, "hbt_c", b["hbt_c"] + 50),
        lever(b, "hm_si_pct", b["hm_si_pct"] + 0.1),
        lever(b, "hmt_c", b["hmt_c"] + 10),
        lever(b, "o2_per_air", b["o2_per_air"] * 1.1),
        lever(b, "blast_moisture_g_nm3", b["blast_moisture_g_nm3"] + 5),
        lever(b, "production_t_h", b["production_t_h"] * 1.05),
        lever(b, "shaft_efficiency", b["shaft_efficiency"] + 0.01),
    ]
    out = whatif.forge_compute(cal, cases)
    return {c["name"]: c for c in out["cases"]} | {"_run": out}


# ------------------------------------------------------------------------------------------------- FORGE


def test_the_base_is_the_calibration_point(base) -> None:
    """The deck's base: Jul-Oct 2026 operation, coke 305.5 kg/tHM, PCI 184, nut coke 73, ηCO 42.2 %, top gas 208 °C."""
    tz, fx, b = base["two_zone"], base["fixed_eta"], base["inputs"]
    assert tz["coke_kg_thm"] == pytest.approx(305.5, abs=0.05) and fx["coke_kg_thm"] == pytest.approx(305.5, abs=0.05)
    assert tz["eta_co_pct"] == pytest.approx(42.2, abs=0.05) and tz["top_gas_c"] == pytest.approx(208, abs=0.5)
    assert b["pci_kg_thm"] == pytest.approx(186.8, abs=0.1) and b["nut_coke_kg_thm"] == pytest.approx(73.6, abs=0.1)
    assert round(b["shaft_efficiency"], 3) == 0.8 and tz["d_coke_kg_thm"] == 0.0 == fx["d_coke_kg_thm"]


@pytest.mark.parametrize(
    "name,two_zone,fixed_eta",
    # page 10 of the deck: coke, two-zone (new) and fixed ηCO (old), kg/tHM, one decimal; HM Si, fixed ηCO: the deck's
    # +1.5 holds the basicity with quartzite, run_scenario with limestone only (+1.35), so it is not compared
    [
        ("pci_kg_thm", -8.7, -9.9),
        ("nut_coke_kg_thm", -9.9, -9.9),
        ("hbt_c", -6.7, -4.1),
        ("hm_si_pct", 2.2, None),
        ("hmt_c", 1.0, 0.6),
        ("o2_per_air", 0.3, 1.0),
    ],
)
def test_the_decks_what_ifs_are_reproduced(deck, name, two_zone, fixed_eta) -> None:
    case = deck[name]
    assert case["status"] == "ok" and not case["warnings"]
    assert case["two_zone"]["d_coke_kg_thm"] == pytest.approx(two_zone, abs=0.06)  # the deck's one decimal
    if fixed_eta is not None:
        assert case["fixed_eta"]["d_coke_kg_thm"] == pytest.approx(fixed_eta, abs=0.06)


def test_the_decks_other_effects_are_reproduced(deck, base) -> None:
    """The deck's "other effects in the two-zone model": PCI ηCO −0.06 points, top gas +7 °C; nut coke none; hot
    blast ηCO +0.23 points, top gas −11 °C; O2 top gas −6 °C."""
    tz0 = base["two_zone"]

    def d(name, key):
        return deck[name]["two_zone"][key] - tz0[key]

    assert d("pci_kg_thm", "eta_co_pct") == pytest.approx(-0.06, abs=0.01) and round(d("pci_kg_thm", "top_gas_c")) == 7
    assert abs(d("nut_coke_kg_thm", "eta_co_pct")) < 0.01 and abs(d("nut_coke_kg_thm", "top_gas_c")) < 0.1
    assert d("hbt_c", "eta_co_pct") == pytest.approx(0.23, abs=0.01) and round(d("hbt_c", "top_gas_c")) == -11
    assert round(d("o2_per_air", "top_gas_c")) == -6


@pytest.mark.parametrize(
    "name,fact",
    [
        ("pci_kg_thm", "COKE-01"),
        ("nut_coke_kg_thm", "COKE-02"),
        ("hm_si_pct", "COKE-03"),
        ("hmt_c", "COKE-04"),
        ("hbt_c", "COKE-05"),
        ("blast_moisture_g_nm3", "COKE-06"),
        ("production_t_h", "COKE-08"),
        ("shaft_efficiency", "COKE-14"),
    ],
)
def test_the_registers_coke_responses_are_reproduced(deck, name, fact) -> None:
    """The register's COKE facts are the deck's two-zone what-ifs on the Jul-Oct 2026 calibration."""
    value = load_register()[fact].value
    assert deck[name]["two_zone"]["d_coke_kg_thm"] == pytest.approx(value, abs=0.06)


def test_each_lever_reaches_its_model_only(deck) -> None:
    omega = deck["shaft_efficiency"]
    assert omega["fixed_eta"]["d_coke_kg_thm"] == pytest.approx(0.0, abs=1e-9)
    assert omega["two_zone"]["eta_co_pct"] - deck["_run"]["base"]["two_zone"]["eta_co_pct"] == pytest.approx(
        0.40, abs=0.01
    )
    assert omega["notes"] and "two-zone" in omega["notes"][0]


def test_the_register_version_and_its_bands_are_those_in_force(cal, monkeypatch) -> None:
    monkeypatch.setattr(register, "_source", None)  # the file alone
    out = whatif.forge_compute(cal, [{"name": "hot", "levers": {"hmt_c": 1530.0}}])
    assert out["facts_version"] == load_register().label == "1.8"
    assert out["cases"][0]["warnings"] == ["HMT 1,530 °C is outside BAND-02 (1,440–1,520 °C)."]
    approved = ApprovedOverlay(3, "1.8", {"BAND-02": {"band": [1440, 1540]}}, "h3")
    monkeypatch.setattr(register, "_source", lambda: approved)  # an approved overlay in force (SV-4.2)
    out = whatif.forge_compute(cal, [{"name": "hot", "levers": {"hmt_c": 1530.0}}])
    assert out["facts_version"] == "1.8 + overlay 3" and out["cases"][0]["warnings"] == []


def test_eta_and_coke_chemistry_levers(cal) -> None:
    out = whatif.forge_compute(
        cal,
        [
            {"name": "eta", "levers": {"eta_co_pct": 43.2}},
            {"name": "ash", "levers": {"coke_ash_pct": 13.45, "coke_fc_pct": 85.62}},
            {"name": "sum", "levers": {"coke_ash_pct": 14.0}},
        ],
    )["cases"]
    eta, ash, bad = out
    assert eta["two_zone"]["d_coke_kg_thm"] == pytest.approx(0.0, abs=1e-9) and eta["fixed_eta"]["d_coke_kg_thm"] < -5
    assert ash["two_zone"]["d_coke_kg_thm"] > 3 and ash["two_zone"]["slag_kg_thm"] > out[0]["two_zone"]["slag_kg_thm"]
    assert not ash["warnings"] and "does not add up to 100 %" in bad["warnings"][0]


# ------------------------------------------------------------------------------------------------ STRATA


@pytest.fixture(scope="module")
def strata_inputs(cal):
    with open(REFERENCE_INPUTS_PATH, encoding="utf-8") as fh:
        inputs = json.load(fh)
    settings = asdict(Engine(inputs, cal).cal_settings())  # PCI 190, nut coke 70, HMT 1,470 °C, 15 g/Nm³
    names = {m["id"]: m["name"] for m in inputs["materials"]}
    return inputs, settings, names


def test_strata_forced_blends_reproduce_the_case_report(cal, strata_inputs) -> None:
    inputs, settings, names = strata_inputs
    cases = [
        {"name": "Cheapest", "blend": None},
        {"name": "Case 2 blend, forced", "blend": CASE2_BLEND},
        {"name": "Jul-Oct blend", "blend": {"sinter_sp_02": 64, "lloyds_clo": 25, "lloyds_pellet": 11}},
        {"name": "PCI 200", "blend": None, "settings": {"pci_kg_thm": 200}},
        {"name": "Unknown", "blend": {"foo": 100}},
    ]
    cheapest, forced, jul_oct, pci, unknown = whatif.strata_compute(
        json.dumps(inputs), json.dumps(cal), settings, asdict(Limits()), 2270.0, cases, names
    )
    assert cheapest["status"] == "ok" and not cheapest["forced"]
    assert cheapest["coke_kg_thm"] == pytest.approx(305.7, abs=0.3)
    assert cheapest["cost_rs_thm"] == pytest.approx(30139, abs=5)
    assert cheapest["fuel_rate_kg_thm"] == pytest.approx(567.0, abs=0.3)
    assert forced["forced"] and forced["status"] == "ok"
    for key in ("coke_kg_thm", "cost_rs_thm", "fuel_rate_kg_thm", "slag_kg_thm"):
        assert forced[key] == pytest.approx(cheapest[key], abs=0.01), key
    assert {b["id"]: round(b["share_pct"], 6) for b in jul_oct["blend"]} == {
        "sinter_sp_02": 64.0, "lloyds_clo": 25.0, "lloyds_pellet": 11.0}  # fmt: skip
    assert jul_oct["status"] == "ok" and jul_oct["settings"]["pci_kg_thm"] == 190.0
    assert pci["settings"]["pci_kg_thm"] == 200.0 and pci["coke_kg_thm"] < cheapest["coke_kg_thm"] - 7
    assert unknown["status"] == "infeasible" and "foo" in unknown["error"]
    assert {r.get("facts_version") for r in (cheapest, forced, jul_oct, pci)} == {load_register().label}
    assert cheapest["certainty"] == "Held" and cheapest["coke_sd_kg_thm"] is None


@slow
def test_strata_omega_forced_blend_carries_its_certainty(cal, strata_inputs) -> None:
    inputs, settings, names = strata_inputs
    s = dict(settings, omega_mode="blend", certainty="any")
    (r,) = whatif.strata_compute(
        json.dumps(inputs), json.dumps(cal), s, asdict(Limits()), 2270.0, [{"name": "B", "blend": CASE2_BLEND}], names
    )
    assert r["status"] == "ok" and r["certainty"] in ("High", "Medium", "Low", "Investigate")
    assert r["coke_sd_kg_thm"] is not None and r["coke_sd_kg_thm"] > 0


# ------------------------------------------------------------------------------------------------ what-if days


def test_what_if_days_use_the_stored_day_calibration(tmp_path, monkeypatch, cal) -> None:
    monkeypatch.setattr(day_calibration, "STORE_DIR", tmp_path)
    with pytest.raises(whatif.WhatIfInputError, match="No day calibration is stored yet"):
        whatif.resolve_calibration("day")
    for d, omega in ((date(2026, 10, 27), 0.79), (date(2026, 10, 28), 0.81)):
        info = CalibrationInfo(dict(cal, omega=omega), None, "live", ("2026-10-13", "2026-10-26"), 40)
        day_calibration.save(info, d, made_at=T0)
    assert whatif.stored_days() == [date(2026, 10, 28), date(2026, 10, 27)]
    latest, fields = whatif.resolve_calibration("day")
    assert latest["omega"] == 0.81 and fields["day"] == date(2026, 10, 28) and fields["n_shifts"] == 40
    older, fields = whatif.resolve_calibration("day", date(2026, 10, 27))
    assert older["omega"] == 0.79 and fields["window"] == "2026-10-13 to 2026-10-26"
    with pytest.raises(whatif.WhatIfInputError, match="No calibration is stored for 2026-10-20"):
        whatif.resolve_calibration("day", date(2026, 10, 20))
    shipped, fields = whatif.resolve_calibration("reference")
    assert fields["calibration"] == "reference" and fields["day"] is None and shipped["period"]
    base = whatif.forge_compute(older, [])["base"]
    assert base["inputs"]["shaft_efficiency"] == 0.79


# ------------------------------------------------------------------------------------------------ saved cases


@pytest.fixture()
def twin(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store_service.use(store)
    yield store
    store_service.reset()
    store.dispose()


def audit_rows(store: TwinStore) -> list[AuditLog]:
    with store.sessions() as session:
        return list(session.query(AuditLog).order_by(AuditLog.id))


def test_cases_are_saved_per_user_with_their_audit_rows(twin) -> None:
    forge = whatif.case_inputs("forge", forge={"pci_kg_thm": 196.8})
    view = whatif.save_case(twin, ALICE, kind="forge", name=" PCI +10 ", inputs=forge, at=T0)
    assert view["name"] == "PCI +10" and view["forge"] == {"pci_kg_thm": 196.8} and view["owner"] == "alice"
    strata = whatif.case_inputs("strata", strata={"blend": {"sinter_sp_02": 64, "lloyds_clo": 36, "gomti_clo": 0}})
    assert strata == {"blend": {"sinter_sp_02": 64.0, "lloyds_clo": 36.0}, "settings": None}
    whatif.save_case(twin, BOB, kind="strata", name="Bob's", inputs=strata, at=T0)
    assert [c["name"] for c in whatif.list_cases(twin, "alice")] == ["PCI +10"]
    assert whatif.list_cases(twin, "alice", "strata") == []
    with pytest.raises(NotFound):
        whatif.save_case(twin, BOB, kind="forge", name="x", inputs=forge, case_id=view["id"])
    with pytest.raises(NotFound):
        whatif.delete_case(twin, BOB, view["id"])
    rows = audit_rows(twin)
    assert [(r.action, r.actor, r.object_type) for r in rows] == [
        ("whatif.save", "alice", "whatif_case"),
        ("whatif.save", "bob", "whatif_case"),
    ]
    assert rows[0].before is None and rows[0].after["inputs"] == {"levers": {"pci_kg_thm": 196.8}}
    whatif.delete_case(twin, ALICE, view["id"])
    assert whatif.list_cases(twin, "alice") == []
    last = audit_rows(twin)[-1]
    assert last.action == "whatif.delete" and last.after is None and last.before["name"] == "PCI +10"


def test_a_result_is_stored_with_an_unchanged_case_of_its_owner(twin) -> None:
    inputs = whatif.case_inputs("forge", forge={"hbt_c": 1238.4})
    view = whatif.save_case(twin, ALICE, kind="forge", name="HBT +50", inputs=inputs, at=T0)
    spec = {"case_id": view["id"], "name": "HBT +50", "levers": {"hbt_c": 1238.4}}
    run = {"job_id": "j1", "facts_version": "1.8", "computed_at": T0.isoformat()}
    assert whatif.store_results(twin, "forge", "bob", run, [{"x": 1}], [spec]) == []  # not Bob's
    assert whatif.store_results(twin, "forge", "alice", run, [{"x": 1}], [spec]) == [view["id"]]
    row = twin.whatif.get(view["id"])
    assert row.result == {**run, "result": {"x": 1}} and row.facts_version == "1.8" and row.result_job_id == "j1"
    renamed = whatif.save_case(twin, ALICE, kind="forge", name="HBT +50 °C", inputs=inputs, case_id=view["id"])
    assert renamed["forge_run"] == {**run, "result": {"x": 1}}  # a new name keeps the result
    changed = whatif.case_inputs("forge", forge={"hbt_c": 1250.0})
    whatif.save_case(twin, ALICE, kind="forge", name="HBT +62", inputs=changed, case_id=view["id"])
    assert twin.whatif.get(view["id"]).result is None  # new inputs: the result no longer belongs to the case
    assert whatif.store_results(twin, "forge", "alice", run, [{"x": 2}], [spec]) == []  # the run solved the old case


def test_run_parameters_are_built_from_the_owners_cases(twin) -> None:
    ids = [
        whatif.save_case(twin, ALICE, kind="forge", name=f"c{i}", inputs=whatif.case_inputs("forge", forge={}))["id"]
        for i in range(5)
    ]
    params = whatif.run_params(twin, "alice", "forge", ids[:4], calibration="reference", day=None)
    assert isinstance(params, WhatIfForgeParams) and [c.name for c in params.cases] == ["c0", "c1", "c2", "c3"]
    with pytest.raises(whatif.WhatIfInputError, match="1-4 cases"):
        whatif.run_params(twin, "alice", "forge", ids)
    with pytest.raises(whatif.WhatIfInputError, match="not strata cases"):
        whatif.run_params(twin, "alice", "strata", ids[:1])
    with pytest.raises(NotFound):
        whatif.run_params(twin, "bob", "forge", ids[:1])
    blend = whatif.case_inputs("strata", strata={"blend": CASE2_BLEND, "settings": {"pci_kg_thm": 200}})
    sid = whatif.save_case(twin, ALICE, kind="strata", name="s", inputs=blend)["id"]
    sp = whatif.run_params(twin, "alice", "strata", [sid], calibration="day", day=date(2026, 10, 28),
                           omega_mode="fixed", yard="reference", production_t_d=2300.0)  # fmt: skip
    assert isinstance(sp, WhatIfStrataParams) and sp.cases[0].settings.pci_kg_thm == 200
    assert whatif.spec_inputs("strata", sp.cases[0].model_dump(mode="json")) == blend


def test_the_forge_job_stores_each_result_with_its_case(twin, monkeypatch, cal) -> None:
    from worker.kinds import whatif as whatif_kind
    from worker.runtime import HANDLERS, JobContext

    assert HANDLERS["whatif.forge"] is whatif_kind.forge and HANDLERS["whatif.strata"] is whatif_kind.strata
    monkeypatch.setattr(whatif, "resolve_calibration", lambda basis, day=None: (cal, whatif._basis_fields(
        cal, "reference", None, "2026-07-01 to 2026-10-15", 80)))  # fmt: skip
    a = whatif.save_case(twin, ALICE, kind="forge", name="PCI +10", inputs=whatif.case_inputs(
        "forge", forge={"pci_kg_thm": 196.8}))  # fmt: skip
    params = whatif.run_params(twin, "alice", "forge", [a["id"]], calibration="reference", day=None)

    class Jobs:
        async def get(self, job_id):
            return type("R", (), {"submitted_by": "alice"})()

        async def update(self, job_id, **changes):
            return None

    job = JobContext(
        job_id="j7", kind=CATALOG["whatif.forge"], attempt=1, store=Jobs(), bus=MemoryEventBus(), pool=None
    )
    out = asyncio.run(HANDLERS["whatif.forge"](job, params))
    assert out["stored"] == [a["id"]] and out["facts_version"] == "1.8" and out["kind"] == "forge"
    (case,) = whatif.list_cases(twin, "alice")
    run = whatif.ForgeCaseRun.model_validate(case["forge_run"])
    assert run.job_id == "j7" and run.facts_version == "1.8" and run.basis.calibration == "reference"
    assert run.result.two_zone.d_coke_kg_thm == pytest.approx(-8.7, abs=0.05)
    assert run.base.two_zone.coke_kg_thm == pytest.approx(305.5, abs=0.05)
