"""Scheduled twin, optimiser, calibration and STRATA-Learn jobs (SV-1.7, #287), run as the worker runs them, on
synthetic plant data, a SQLite twin store and stubbed optimiser solves; plus the job statistics of housekeeping and
the routes ``GET /api/jobs/stats`` and ``GET /api/jobs/schedule``."""

from __future__ import annotations

import asyncio
import functools
import json
import uuid
import warnings
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from data.burden_engine.calibration import CalibrationInfo, fallback_calibration
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import coke_energy as ce
from services import day_calibration, learn, optimiser, twin
from services import schedule as sch
from services import store as store_service
from services.events import MemoryEventBus
from services.jobs import CATALOG, JobRecord, NoParams, OptimiserRunParams, TwinEvaluateParams, input_hash
from tests.redis_server import needs_redis, redis_server
from tests.twin_data import synthetic_frames
from worker.kinds import calibration as calibration_kind
from worker.kinds import learn as learn_kind
from worker.kinds import optimiser as optimiser_kind
from worker.kinds import system as system_kind
from worker.kinds import twin as twin_kind
from worker.runtime import JobContext, TransientJobError

IST = "Asia/Kolkata"
CFG = {"history_start": "2026-04-01", "predict_from": "2026-04-01", "rolling_window_days": 14, "rolling_min_shifts": 10}
KEYS = ce.DataKeys("synthetic.csv", 1, "2026-05-15T10:00:00+05:30", 0, "2026-04-01", "{}", "", 30.0)


def utc_of(text: str) -> datetime:
    return pd.Timestamp(text, tz=IST).tz_convert("UTC").to_pydatetime()


def record(kind: str, *, slot: datetime | None = None, by: str = "scheduler", **kw: Any) -> JobRecord:
    return JobRecord(
        id=uuid.uuid4().hex,
        kind=kind,
        status="running",
        input_hash=input_hash(kind, {}),
        submitted_by=by,
        submitted_at=slot or datetime.now(UTC),
        scheduled_for=slot,
        **kw,
    )


class FakeJobs:
    """The parts of ``JobStore`` a handler uses."""

    def __init__(self, rec: JobRecord | None = None, records: list[JobRecord] | None = None) -> None:
        self.rec, self.records, self.archive, self.trimmed = rec, list(records or []), None, 0

    async def get(self, job_id: str) -> JobRecord | None:
        return self.rec

    async def list(self, limit: int = 50) -> list[JobRecord]:
        return self.records[:limit]

    async def trim(self) -> int:
        self.trimmed += 1
        return 0

    async def update(self, job_id: str, **changes: Any) -> None:
        return None


def context(kind: str, rec: JobRecord | None, attempt: int = 1, jobs: FakeJobs | None = None):
    bus = MemoryEventBus()
    jobs = jobs or FakeJobs(rec)
    return JobContext(job_id=rec.id if rec else "j1", kind=CATALOG[kind], attempt=attempt, store=jobs, bus=bus,
                      pool=None), bus  # fmt: skip


@pytest.fixture()
def store(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    twin_store = TwinStore(session_factory(engine), engine)
    store_service.use(twin_store)
    yield twin_store
    store_service.reset()
    twin_store.dispose()


# ----------------------------------------------------------------------------------------------- twin.evaluate
@pytest.fixture(scope="module")
def synthetic_evaluate():
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=18)
    loaders = ce.Loaders(
        hourly=lambda path, mtime: fd,
        online=lambda start, clock, nonce, fields, cache, minutes: m10,
        neon=lambda start, clock, nonce: (hm, dpr, []),
        lab=lambda start, clock, nonce: (pd.DataFrame(columns=["source", "material", "t"]), pd.DataFrame()),
        names=dict,
    )

    @functools.cache
    def page_data_for(keys, settings):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ce.build_page_data(keys, settings, loaders)

    original = twin.evaluate  # the tests replace twin.evaluate by this function

    def evaluate(t, strata=True):
        return original(t, cfg=CFG, keys=KEYS, loaders=loaders, page_data_for=page_data_for, strata=strata)

    return evaluate


def ready(holds: bool, data_end: str = "2026-05-15 09:00"):
    state = twin.DatasetState(pd.Timestamp(data_end), pd.Timestamp("2026-05-15 10:06"))
    return lambda end: (holds, state)


def test_the_hourly_job_evaluates_the_window_of_its_slot_with_the_missing_ones(
    store, synthetic_evaluate, monkeypatch
) -> None:
    monkeypatch.setattr(twin, "evaluate", synthetic_evaluate)
    monkeypatch.setattr(twin, "ensure_dataset", ready(True))
    rec = record("twin.evaluate", slot=utc_of("2026-05-15 10:10"))
    store.snapshots.upsert(  # 08:00 is stored already; 07:00 and 09:00 are missing
        twin.snapshot_values(
            synthetic_evaluate(pd.Timestamp("2026-05-15 08:00"), strata=False), evaluated_at=utc_of("2026-05-15 08:12")
        )  # fmt: skip
    )
    job, bus = context("twin.evaluate", rec)
    out = asyncio.run(twin_kind.evaluate(job, TwinEvaluateParams(backfill_hours=3, strata=False)))
    assert out["window_end"] == "2026-05-15T10:00:00" and out["backfilled"] == 2 and out["stored"]
    assert [w["window_end"] for w in out["windows"]] == ["2026-05-15T07:00:00", "2026-05-15T09:00:00",
                                                          "2026-05-15T10:00:00"]  # fmt: skip
    assert {w["status"] for w in out["windows"]} == {"complete"} and not out["last_row_pending"]
    rows = store.snapshots.series(utc_of("2026-05-15 07:00"), utc_of("2026-05-15 10:00"))
    assert len(rows) == 4 and {r.job_id for r in rows} == {rec.id, None}
    assert store.jobs.get(rec.id).scheduled_for is not None  # the job record the snapshots refer to
    published = asyncio.run(bus.recent(50))
    events = [e for e in published if e.type not in ("job.progress", "alarm.raised", "alarm.cleared")]
    assert [e.type for e in events] == ["twin.snapshot"]  # the window of the slot only, not the backfilled ones
    # the Live Twin alarms ran after the windows (SV-1.9): the synthetic evaluation records no source times
    assert {e.data.code for e in published if e.type == "alarm.raised"} == {"SOURCE_STALE"}
    assert pd.Timestamp(events[0].data.window_end) == pd.Timestamp(utc_of("2026-05-15 10:00"))


def test_a_late_dataset_is_retried_and_its_window_evaluated_again_later(store, synthetic_evaluate, monkeypatch) -> None:
    monkeypatch.setattr(twin, "evaluate", synthetic_evaluate)
    monkeypatch.setattr(twin, "ensure_dataset", ready(False, "2026-05-15 08:00"))
    rec = record("twin.evaluate", slot=utc_of("2026-05-15 10:10"))
    params = TwinEvaluateParams(backfill_hours=0, strata=False)
    job, _ = context("twin.evaluate", rec, attempt=1)
    with pytest.raises(TransientJobError, match="needs the row of 09:00"):
        asyncio.run(twin_kind.evaluate(job, params))
    job, _ = context("twin.evaluate", rec, attempt=CATALOG["twin.evaluate"].max_tries)  # the last try evaluates
    out = asyncio.run(twin_kind.evaluate(job, params))
    assert out["last_row_pending"] and "did not hold the row of 09:00" in out["notes"][-1]
    pending = store.snapshots.get(utc_of("2026-05-15 10:00"))
    assert pending.flags[twin.PENDING_FLAG] is True

    stuck = record("twin.evaluate", slot=utc_of("2026-05-15 11:10"))  # the next hour, the dataset still ends 08:00
    job, _ = context("twin.evaluate", stuck, attempt=CATALOG["twin.evaluate"].max_tries)
    out = asyncio.run(twin_kind.evaluate(job, TwinEvaluateParams(backfill_hours=1, strata=False)))
    assert [w["window_end"] for w in out["windows"]] == ["2026-05-15T10:00:00", "2026-05-15T11:00:00"]
    assert store.snapshots.get(utc_of("2026-05-15 10:00")).flags[twin.PENDING_FLAG] is True  # still lacks 09:00

    monkeypatch.setattr(twin, "ensure_dataset", ready(True, "2026-05-15 10:00"))
    nxt = record("twin.evaluate", slot=utc_of("2026-05-15 11:10"))
    job, _ = context("twin.evaluate", nxt)
    out = asyncio.run(twin_kind.evaluate(job, TwinEvaluateParams(backfill_hours=2, strata=False)))
    assert [w["window_end"] for w in out["windows"]] == ["2026-05-15T09:00:00", "2026-05-15T10:00:00",
                                                          "2026-05-15T11:00:00"]  # fmt: skip
    again = store.snapshots.get(utc_of("2026-05-15 10:00"))
    assert twin.PENDING_FLAG not in again.flags and again.job_id == nxt.id
    assert (
        not out["last_row_pending"] and twin.PENDING_FLAG not in store.snapshots.get(utc_of("2026-05-15 11:00")).flags
    )


def test_the_window_end_of_a_request_a_slot_or_now() -> None:
    assert twin.target_window_end("2026-05-15T10:00:00+05:30") == pd.Timestamp("2026-05-15 10:00")
    assert twin.target_window_end(datetime(2026, 5, 15, 4, 30, tzinfo=UTC)) == pd.Timestamp("2026-05-15 10:00")
    assert twin.target_window_end(None, utc_of("2026-05-15 10:10")) == pd.Timestamp("2026-05-15 10:00")
    assert twin.target_window_end(None, None, now=utc_of("2026-05-15 10:59")) == pd.Timestamp("2026-05-15 10:00")


def test_the_dataset_is_fetched_only_when_it_lacks_the_row_and_predates_this_hours_build() -> None:
    end, calls = pd.Timestamp("2026-05-15 10:00"), []
    states = iter(
        [
            twin.DatasetState(pd.Timestamp("2026-05-15 08:00"), pd.Timestamp("2026-05-15 09:06")),
            twin.DatasetState(pd.Timestamp("2026-05-15 09:00"), pd.Timestamp("2026-05-15 10:06")),
        ]
    )

    def refresh(cfg):
        calls.append(cfg)
        return {"refreshed": True}

    ok, state = twin.ensure_dataset(end, cfg={}, now=utc_of("2026-05-15 10:10"), refresh_fn=refresh,
                                    state_fn=lambda cfg: next(states))  # fmt: skip
    assert ok and len(calls) == 1 and state.refresh == {"refreshed": "True"}
    late = twin.DatasetState(pd.Timestamp("2026-05-15 08:00"), pd.Timestamp("2026-05-15 10:06"))
    ok, _ = twin.ensure_dataset(end, cfg={}, now=utc_of("2026-05-15 10:15"), refresh_fn=refresh,
                                state_fn=lambda cfg: late)  # fmt: skip
    assert not ok and len(calls) == 1  # this hour's build is local already: nothing newer to fetch


# ------------------------------------------------------------------------------------------- optimiser schedule
def fake_compute(mode, inputs_json, cal_json, settings, limits, prod, *rest, **kw):
    plan = {
        "P_tday": prod if mode == "fixed" else 2310.0,
        "feasible": True,
        "out": {"K": 301.0 if settings["omega_mode"] == "blend" else 303.0, "fuel_rate": 545.0},
        "cost": {"total": 29_500.0},
        "omega": {"mode": settings["omega_mode"], "sigma_K": 3.1, "certainty": "High"},
    }
    return {"mode": mode, "with_fuel": plan} if mode == "fixed" else {"mode": mode, "best": plan}


@pytest.fixture()
def day_dir(tmp_path, monkeypatch):
    folder = tmp_path / "calibration"
    monkeypatch.setattr(day_calibration, "STORE_DIR", folder)
    return folder


def shipped_info() -> CalibrationInfo:
    return CalibrationInfo(fallback_calibration(), None, "live", ("2026-10-13", "2026-10-26"), 40, "40 shifts", [])


def test_step2_runs_strata_and_fixed_at_the_current_production_on_the_days_calibration(
    store, day_dir, monkeypatch
) -> None:
    monkeypatch.setattr(optimiser, "compute_mode", fake_compute)
    day_calibration.save(shipped_info(), day_calibration.today(), made_at=datetime.now(UTC), job_id="cal-1")
    end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    store.snapshots.upsert({"window_start": (end - pd.Timedelta(hours=8)).to_pydatetime(), "window_end":
                            end.to_pydatetime(), "evaluated_at": end.to_pydatetime(), "status": "complete",
                            "quantities": {"production_t_h": 95.04}})  # fmt: skip
    rec = record("optimiser.step2", slot=datetime.now(UTC))
    job, _ = context("optimiser.step2", rec, attempt=1)
    with pytest.raises(TransientJobError, match="reference"):  # no Neon here: the yard is the reference yard
        asyncio.run(optimiser_kind.step2(job, NoParams()))
    job, _ = context("optimiser.step2", rec, attempt=CATALOG["optimiser.step2"].max_tries)
    out = asyncio.run(optimiser_kind.step2(job, NoParams()))
    assert out["production_t_d"] == 2280.0 and out["production_source"].startswith("twin window")
    assert [r["omega_mode"] for r in out["runs"]] == ["strata", "fixed"]
    assert [r["coke_kg_thm"] for r in out["runs"]] == [301.0, 303.0]
    runs = store.optimiser.recent()
    assert len(runs) == 2 and {r.trigger for r in runs} == {"schedule"} and {r.job_id for r in runs} == {rec.id}
    inputs = runs[0].inputs
    assert inputs["calibration"]["basis"] == "day" and inputs["calibration"]["made_at"]
    assert inputs["production_target_t_d"] == 2280.0 and runs[0].versions["calibration_basis"] == "day"
    assert inputs["yard_source"] == "reference"  # the last try runs on the reference, labelled


def test_step2_without_a_current_window_takes_page_11s_production(store, day_dir, monkeypatch) -> None:
    monkeypatch.setattr(optimiser, "compute_mode", fake_compute)
    rec = record("optimiser.step2", slot=datetime.now(UTC))
    job, _ = context("optimiser.step2", rec, attempt=CATALOG["optimiser.step2"].max_tries)
    out = asyncio.run(optimiser_kind.step2(job, NoParams()))
    assert "no current production: no twin window stored" in out["production_source"]
    assert store.optimiser.recent()[0].inputs["calibration"]["basis"] == "live"  # no calibration of today yet
    assert "calibration.nightly" in store.optimiser.recent()[0].inputs["calibration"]["note"]


def test_step1_runs_strata_once(store, day_dir, monkeypatch) -> None:
    monkeypatch.setattr(optimiser, "compute_mode", fake_compute)
    rec = record("optimiser.step1", slot=datetime.now(UTC))
    job, _ = context("optimiser.step1", rec, attempt=CATALOG["optimiser.step1"].max_tries)
    out = asyncio.run(optimiser_kind.step1(job, NoParams()))
    assert [(r["mode"], r["omega_mode"], r["production_t_d"]) for r in out["runs"]] == [("step1", "strata", 2310.0)]


def test_a_requested_run_takes_the_days_calibration_unless_asked_for_the_live_one(store, day_dir, monkeypatch) -> None:
    monkeypatch.setattr(optimiser, "compute_mode", fake_compute)
    day_calibration.save(shipped_info(), day_calibration.today(), made_at=datetime.now(UTC))
    for basis in ("day", "live"):
        rec = record("optimiser.run", by="sv1")
        job, _ = context("optimiser.run", rec)
        params = OptimiserRunParams(mode="step2", omega_mode="fixed", production_t_d=2200, calibration=basis)
        out = asyncio.run(optimiser_kind.run(job, params))
        assert out["production_t_d"] == 2200 and out["omega_mode"] == "fixed"
        run = store.optimiser.get(out["run_id"])
        assert run.inputs["calibration"]["basis"] == basis and run.trigger == "user"
        assert run.inputs["production_source"] == "requested"


def test_current_production_rules(store) -> None:
    now = datetime(2026, 10, 27, 6, 0, tzinfo=UTC)  # 11:30 IST
    assert optimiser.current_production(store, now) == (None, "no twin window stored")

    def window(end, **q):
        store.snapshots.upsert({"window_start": end - timedelta(hours=8), "window_end": end, "evaluated_at": end,
                                "status": "complete", "quantities": q})  # fmt: skip

    window(datetime(2026, 10, 27, 2, 30, tzinfo=UTC), production_t_h=94.0)  # 08:00 IST: 3.5 h old
    assert "3.5 h ago" in optimiser.current_production(store, now)[1]
    window(datetime(2026, 10, 27, 5, 30, tzinfo=UTC), production_t_h=40.0)  # 11:00 IST
    assert "outside" in optimiser.current_production(store, now)[1]
    window(datetime(2026, 10, 27, 5, 30, tzinfo=UTC), production_t_h=94.04)
    value, label = optimiser.current_production(store, now)
    assert value == 2260.0 and label == "twin window 27 Oct 03:00-11:00 IST (94.0 t/h)"


# ------------------------------------------------------------------------------------------ calibration.nightly
def test_the_nightly_calibration_saves_the_day_and_builds_the_jacobian(day_dir, monkeypatch) -> None:
    day = day_calibration.today()
    monkeypatch.setattr(day_calibration, "compute", lambda d, cfg: shipped_info())
    monkeypatch.setattr(day_calibration, "network_jacobian", lambda cal_json: {"jacobian": "computed (17 solves)"})
    prev = shipped_info()
    prev.cal = {**prev.cal, "omega": prev.cal["omega"] - 0.01}
    day_calibration.save(prev, day - timedelta(days=1), made_at=datetime.now(UTC))
    rec = record("calibration.nightly", slot=datetime.now(UTC))
    job, _ = context("calibration.nightly", rec)
    out = asyncio.run(calibration_kind.nightly(job, NoParams()))
    assert out["day"] == day.isoformat() and out["n_shifts"] == 40 and out["network"]["jacobian"].startswith("computed")
    assert out["change"]["omega"] == pytest.approx(0.01, abs=1e-4)
    doc = day_calibration.read(day)
    assert doc["job_id"] == rec.id and doc["network"] == out["network"]
    loaded = day_calibration.load(day)
    assert loaded.cal == json.loads(json.dumps(optimiser.jsonable(shipped_info().cal)))
    assert loaded.window == ("2026-10-13", "2026-10-26")


def test_the_nightly_calibration_retries_without_plant_data_and_fails_on_a_fallback(day_dir, monkeypatch) -> None:
    def unavailable(d, cfg):
        raise day_calibration.PlantDataUnavailable("InfluxDB timed out")

    monkeypatch.setattr(day_calibration, "compute", unavailable)
    job, _ = context("calibration.nightly", record("calibration.nightly"))
    with pytest.raises(TransientJobError, match="InfluxDB timed out"):
        asyncio.run(calibration_kind.nightly(job, NoParams()))
    fallback = CalibrationInfo(fallback_calibration(), None, "fallback", ("", ""), 3, "3 stable shifts (need 10)")
    monkeypatch.setattr(day_calibration, "compute", lambda d, cfg: fallback)
    with pytest.raises(TransientJobError, match="need 10"):
        asyncio.run(calibration_kind.nightly(job, NoParams()))
    last, _ = context("calibration.nightly", record("calibration.nightly"), attempt=3)
    with pytest.raises(RuntimeError, match="calibrate live"):
        asyncio.run(calibration_kind.nightly(last, NoParams()))
    assert day_calibration.read(day_calibration.today()) is None


def test_day_calibrations_are_kept_for_30_days(tmp_path) -> None:
    start = datetime(2026, 9, 1).date()
    for i in range(33):
        day_calibration.save(shipped_info(), start + timedelta(days=i), made_at=datetime.now(UTC), directory=tmp_path)
    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert len(files) == day_calibration.KEEP and files[0] == "2026-09-04.json"


# --------------------------------------------------------------------------------------------------- learn.shadow
def test_the_shadow_run_writes_its_outputs_and_a_daily_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(learn, "STORE_DIR", tmp_path)
    job, _ = context("learn.shadow", record("learn.shadow"))
    out = asyncio.run(learn_kind.shadow(job, NoParams()))
    assert out["evidence_days"] == 191 and out["evidence_to"] == "2026-10-07" and out["appended_days"] == 0
    assert out["gate"]["status"] in ("collecting", "evaluated") and isinstance(out["drift"]["alarm"], bool)
    for name in ("evidence.csv", "strata_learn_state.json", "strata_learn_history.csv", "register_proposal.json",
                 "strata_learn_report.md"):  # fmt: skip
        assert (tmp_path / name).is_file(), name
    stamp = (tmp_path / "evidence.csv").stat().st_mtime_ns
    again = learn.shadow(tmp_path)
    assert again["evidence_days"] == 191 and (tmp_path / "evidence.csv").stat().st_mtime_ns == stamp
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1


# ---------------------------------------------------------------------------------------- housekeeping statistics
def test_housekeeping_stores_the_runtime_statistics_of_14_days() -> None:
    now = datetime.now(UTC)
    records = []
    for i in range(30):
        t = now - timedelta(hours=i + 1)
        records.append(
            record("forecast.si", slot=t, started_at=t, finished_at=t + timedelta(seconds=4 + i % 5), run_s=4.0 + i % 5)
        )
        records[-1].status = "done"
    jobs = FakeJobs(records=records)
    job, _ = context("system.housekeeping", record("system.housekeeping"), jobs=jobs)
    out = asyncio.run(system_kind.housekeeping(job, NoParams()))
    stats = {k["kind"]: k for k in out["job_stats"]["kinds"]}
    assert out["job_stats"]["source"] == "redis" and jobs.trimmed == 1
    assert stats["forecast.si"]["runs"] == 30 and stats["forecast.si"]["runtime_max_s"] == 8.0
    assert stats["forecast.si"]["runtime_p95_s"] == pytest.approx(8.0)
    assert set(stats) == set(CATALOG)


# --------------------------------------------------------------------------------------------------------- routes
@pytest.mark.usefixtures("store")
@needs_redis
def test_the_stats_and_schedule_routes_read_the_archive(store, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from tests.test_api_auth import build
    from tests.test_api_live import Users

    now = datetime.now(UTC)
    slots = sch.slots(CATALOG["twin.evaluate"].schedule, now - timedelta(days=2), now - sch.SETTLE)
    rows = []
    for i, slot in enumerate(slots):
        if i == 5:
            continue  # a missed slot
        rows.append(
            dict(id=uuid.uuid4().hex, kind="twin.evaluate", status="failed" if i == 9 else "done", params={},
                 input_hash="h", submitted_by="scheduler", submitted_at=slot + timedelta(seconds=1),
                 scheduled_for=slot, started_at=slot + timedelta(seconds=2),
                 finished_at=slot + timedelta(seconds=40), run_s=30.0 + i % 10, tries=1)
        )  # fmt: skip
    store.jobs.upsert_many(rows)
    with redis_server(tmp_path) as url:
        app = build(redis_url=url)
        with TestClient(app, base_url="https://testserver") as client:
            users = Users(app, client)
            stats = users.get("op", "/api/jobs/stats", params={"days": 3}).json()
            audit = users.get("op", "/api/jobs/schedule", params={"days": 2}).json()
            kinds = {k["name"]: k for k in users.get("op", "/api/jobs/kinds").json()}
            assert users.get("op", "/api/jobs/stats", params={"days": 0}).status_code == 422
    evaluate = {k["kind"]: k for k in stats["kinds"]}["twin.evaluate"]
    assert stats["source"] == "twin.jobs" and evaluate["runs"] == len(slots) - 1 and evaluate["failed"] == 1
    done_runs = [30.0 + i % 10 for i in range(len(slots)) if i not in (5, 9)]
    assert evaluate["runtime_p95_s"] == pytest.approx(float(pd.Series(done_runs).quantile(0.95)))
    assert evaluate["start_delay_p95_s"] == pytest.approx(2.0)
    a = {k["kind"]: k for k in audit["kinds"]}["twin.evaluate"]
    assert audit["source"] == "twin.jobs" and not audit["clean"]
    assert a["expected"] == len(slots) and (a["missed"], a["failed"], a["done"]) == (1, 1, len(slots) - 2)
    assert [p["state"] for p in a["problems"]] == ["missed", "failed"]
    assert pd.Timestamp(a["problems"][0]["slot"]) == pd.Timestamp(slots[5])
    assert kinds["twin.evaluate"]["schedule"] == "every hour at :10"
    assert pd.Timestamp(kinds["twin.evaluate"]["next_slot"]) > pd.Timestamp(now)
    assert kinds["optimiser.run"]["next_slot"] is None


def test_the_stats_routes_need_the_archive_or_redis(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from tests.test_api_auth import build
    from tests.test_api_live import Users

    app = build()  # no Redis
    with TestClient(app, base_url="https://testserver") as client:
        users = Users(app, client)
        assert users.get("op", "/api/jobs/stats").status_code == 503  # no twin store either (BF_TWIN_DB_URL=off)
        engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
        TwinBase.metadata.create_all(engine)
        store_service.use(TwinStore(session_factory(engine), engine))
        try:
            body = users.get("op", "/api/jobs/schedule", params={"days": 1}).json()
            assert body["source"] == "twin.jobs" and not body["clean"]  # nothing ran: every slot missed
            assert {k["kind"]: k["missed"] for k in body["kinds"]}["twin.evaluate"] in (23, 24)
        finally:
            store_service.reset()
            engine.dispose()
