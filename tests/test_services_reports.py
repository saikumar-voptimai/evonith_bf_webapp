"""Shift report service and jobs (SV-2.3, #295): parity with the FurnaceMind Reports tab on a fixture, one retry of the
LLM, the report stored without narrative when the LLM fails, the shifts of the schedule slots, and the worker kinds
``report.shift`` / ``report.live`` on a SQLite twin store."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import math
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from reports.furnace_report import service as report_service
from reports.furnace_report.data import ReportTimeframe, ShiftRawData
from services import reports
from services import store as store_service
from services.events import MemoryEventBus
from services.jobs import CATALOG, JobRecord, LiveReportParams, ShiftReportParams, input_hash
from worker.kinds import reports as report_kinds
from worker.runtime import JobContext, TransientJobError

IST = "Asia/Kolkata"
SHIFT = reports.Shift(date(2026, 10, 26), "B")  # 14:00-22:00 IST
ONLINE = (
    "production_per_hour",
    "coke_rate",
    "nut_coke_rate",
    "coal_rate_actual_value",
    "body_raft",
    "body_perm",
    "body_etaco",
    "hot_blast_vol_nm3h",
    "hot_blast_temp",
    "hot_blast_press",
    "body_dp_top",
    "body_dp_bottom",
    "body_dp_total",
    "oxygen_flow",
    "oxygen_enrichment_pct",
    *(f"top_temp_{i}" for i in range(1, 5)),
    *(f"skin_flow_temp_{q}" for q in "abcd"),
    *(f"temp_{level}_{q}" for level in (18660, 15162, 12975) for q in "abcd"),
    "temp_4373_a",
    "temp_5411_c",
    "temp_5757_c",
    "temp_6103_b",
)
LEVELS = {  # mean and spread of each online field
    "production_per_hour": (112.0, 4.0),
    "coke_rate": (352.0, 6.0),
    "nut_coke_rate": (38.0, 2.0),
    "coal_rate_actual_value": (151.0, 5.0),
    "body_raft": (2210.0, 25.0),
    "body_perm": (1320.0, 60.0),
    "body_etaco": (46.3, 0.8),
    "hot_blast_vol_nm3h": (231000.0, 2500.0),
    "hot_blast_temp": (1105.0, 8.0),
    "hot_blast_press": (3.41, 0.04),
    "body_dp_top": (0.62, 0.03),
    "body_dp_bottom": (0.91, 0.04),
    "body_dp_total": (1.53, 0.05),
    "oxygen_flow": (14200.0, 300.0),
    "oxygen_enrichment_pct": (4.2, 0.2),
}


def raw_for(timeframe: ReportTimeframe) -> ShiftRawData:
    """Synthetic source data of a shift window, different for every window (seeded by its id)."""
    seed = int(hashlib.sha256(timeframe.window_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    index = pd.date_range(timeframe.start_ist, timeframe.end_ist, freq="15min", inclusive="left")
    online = {}
    for name in ONLINE:
        mean, sd = LEVELS.get(name, (180.0 if name.startswith(("top", "skin")) else 420.0, 12.0))
        online[name] = mean + sd * rng.standard_normal(len(index))
    start_utc = timeframe.start_ist.astimezone(UTC)
    taps = pd.date_range(start_utc + timedelta(minutes=40), periods=6, freq="80min")
    hm = pd.DataFrame(
        {
            "chem_pct_si": 0.45 + 0.08 * rng.standard_normal(6),
            "chem_pct_s": 0.031 + 0.004 * rng.standard_normal(6),
            "hmt_gt_1480c": 1488.0 + 9.0 * rng.standard_normal(6),
            "slag_pct_cao": 35.2 + 0.6 * rng.standard_normal(6),
            "slag_pct_sio2": 33.1 + 0.5 * rng.standard_normal(6),
        },
        index=taps,
    )
    n = 46
    charge = pd.DataFrame(
        {
            "coke_1_mt": 9.0 + 0.3 * rng.standard_normal(n),
            "nut_coke_1_mt": 1.0 + 0.05 * rng.standard_normal(n),
            "sinter_1_mt": 20.0 + 0.5 * rng.standard_normal(n),
            "ore_1_mt": 6.0 + 0.2 * rng.standard_normal(n),
            "ore_2_mt": 2.0 + 0.1 * rng.standard_normal(n),
            "pellet_1_mt": 3.0 + 0.1 * rng.standard_normal(n),
            "flux_1_mt": 0.8 + 0.05 * rng.standard_normal(n),
        },
        index=pd.date_range(start_utc, periods=n, freq="10min"),
    )
    lab = start_utc - timedelta(hours=5)
    ore = pd.DataFrame(
        [
            {"date_time": lab, "material_code": "ore_1", "tm": 3.7},
            {"date_time": lab, "material_code": "ore_2", "tm": 4.4},
            {"date_time": lab, "material_code": "pellet_1", "tm": 1.9},
        ]
    )
    fuel = pd.DataFrame(
        [
            {"date_time": lab, "material_code": "coke_1", "tm": 5.2},
            {"date_time": lab, "material_code": "nut_coke_1", "moisture": 7.5},
        ]
    )
    flux = pd.DataFrame([{"date_time": lab, "material_code": "flux_1", "tm": 1.1}])
    fines = pd.DataFrame(
        [
            {"date_time": lab, "material_code": "coke_1", "plus_20mm": 1.3, "plus_10mm": 2.2, "minus_10mm": 3.1},
            {"date_time": lab, "material_code": "ore_1", "plus_6mm": 6.4, "minus_6mm": 2.6},
            {"date_time": lab, "material_code": "sinter_1", "plus_5mm": 4.1, "minus_5mm": 3.3},
        ]
    )
    materials = pd.DataFrame(
        {
            "material_code": ["ore_1", "ore_2", "flux_1"],
            "material_name": ["Ore A", "Ore B", "Dolomite"],
            "is_active": [True, True, True],
        }  # fmt: skip
    )
    return ShiftRawData(
        shift_date=timeframe.selected_date,
        shift_label=timeframe.shift_label,
        shift_start_ist=timeframe.start_ist,
        shift_end_ist=timeframe.end_ist,
        online_df=pd.DataFrame(online, index=index),
        hm_slag_df=hm,
        charge_df=charge,
        ore_chemistry_df=ore,
        fuel_chemistry_df=fuel,
        flux_chemistry_df=flux,
        material_fines_df=fines,
        materials_df=materials,
        report_type=timeframe.report_type,
    )


class FakeFetcher:
    """``ShiftFetcher`` on the synthetic data; records the windows fetched."""

    windows: list[str] = []
    fail = False

    def fetch(self, *, timeframe: ReportTimeframe | None = None, **_: Any) -> ShiftRawData:
        assert timeframe is not None
        if FakeFetcher.fail:
            raise ConnectionError("InfluxDB is not reachable")
        FakeFetcher.windows.append(timeframe.window_id)
        return raw_for(timeframe)


class FakeLLM:
    """Page 7's client as the tests see it: the narrative names a digest of the prompt (current and previous shift)."""

    calls = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        self.last_actual_model = "fake/model-1"

    def generate(self, system_prompt: str, user_prompt: str, stop: Any = None) -> str:
        FakeLLM.calls += 1
        digest = hashlib.sha256((system_prompt + user_prompt).encode()).hexdigest()[:12]
        return f"  Shift B ran stable; fuel rate held. Prompt {digest}.\n"


class FlakyLLM:
    """Fails ``failures`` times (provider busy), then answers."""

    def __init__(self, failures: int, answer: str = "Watch the belly spread.") -> None:
        self.failures, self.answer, self.calls = failures, answer, 0

    def __call__(self) -> FlakyLLM:  # used as the client factory
        return self

    def generate(self, system_prompt: str, user_prompt: str, stop: Any = None) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("503 provider busy")
        return self.answer


@pytest.fixture(autouse=True)
def fake_sources(monkeypatch):
    FakeFetcher.windows, FakeFetcher.fail, FakeLLM.calls = [], False, 0
    monkeypatch.setattr(report_service, "ShiftFetcher", FakeFetcher)
    monkeypatch.setattr(reports, "LLM_RETRY_DELAY_S", 0.0)
    yield


@pytest.fixture()
def store(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    twin_store = TwinStore(session_factory(engine), engine)
    store_service.use(twin_store)
    yield twin_store
    store_service.reset()
    twin_store.dispose()


def assert_same(service_value: Any, tab_value: Any, path: str = "data") -> int:
    """Equal values, numbers within a relative difference of 1e-6; returns the numbers compared."""
    if isinstance(tab_value, dict):
        assert isinstance(service_value, dict) and set(service_value) >= set(tab_value), path
        return sum(assert_same(service_value[k], v, f"{path}.{k}") for k, v in tab_value.items())
    if isinstance(tab_value, list | tuple):
        assert len(service_value) == len(tab_value), path
        return sum(
            assert_same(s, t, f"{path}[{i}]") for i, (s, t) in enumerate(zip(service_value, tab_value, strict=True))
        )
    if isinstance(tab_value, float) and not isinstance(tab_value, bool):
        assert service_value is not None, path
        assert math.isclose(service_value, tab_value, rel_tol=1e-6, abs_tol=1e-12), (path, service_value, tab_value)
        return 1
    if isinstance(tab_value, datetime | date):
        assert pd.Timestamp(service_value) == pd.Timestamp(tab_value), path
        return 0
    assert service_value == tab_value, (path, service_value, tab_value)
    return 0


# ---------------------------------------------------------------------------------------------------------- parity
def test_the_report_data_equal_the_reports_tab(store, monkeypatch) -> None:
    """The Reports tab's own function (``ui.furnacemind.reports._run_live_report``, agentic analysis on) and the
    service on the same fixture: data within 1e-6, the same tables and the same narrative."""
    import agents.llm.llm_client as llm_client
    from ui.furnacemind import reports as tab

    monkeypatch.setattr(llm_client, "OpenRouterClient", FakeLLM)
    seen: list[Any] = []
    original = report_service.ShiftReportService.generate_document

    def spy(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        seen.append(out)
        return out

    monkeypatch.setattr(report_service.ShiftReportService, "generate_document", spy)

    class StatusBox:
        def status(self, *_: Any, **__: Any) -> None:
            return None

    tab_document = tab._run_live_report(
        "Shift", SHIFT.date, SHIFT.letter, include_analysis=True, status_box=StatusBox()
    )
    (tab_data, _), = seen  # fmt: skip
    assert FakeFetcher.windows == ["2026-10-26_SHIFT_B", "2026-10-26_SHIFT_A"]  # the shift and the one before

    generated = reports.generate(SHIFT, kind="closed", client_factory=FakeLLM)
    values = reports.store_values(generated, job_id=None, trigger="schedule", requested_by="scheduler")
    row = store.reports.add(values)
    detail = reports.detail_data(store.reports.get(row.id))  # through the store, as the API serves it

    expected = dataclasses.asdict(tab_data)
    for name in reports.TEMP_ROWS:
        expected[name]["spread_std"] = getattr(tab_data, name).spread_std
    compared = assert_same(detail["data"], expected)
    assert compared >= 70  # every number of the report: rates, tonnages, statistics, temperatures
    assert detail["data"]["fuel_rate"] == tab_data.fuel_rate and tab_data.fuel_rate is not None
    assert detail["fuel_rate_kg_thm"] == tab_data.fuel_rate and detail["etaco_pct"] == tab_data.etaco.mean
    assert detail["furnace_status"] == tab_data.status

    document = detail["document"]
    assert document["title"] == tab_document.title == "Live Report (2026-10-26_SHIFT_B)"
    assert document["pre_blocks"] == list(tab_document.pre_blocks)
    assert [(s["title"], s["headers"], s["rows"], s["placement"]) for s in document["sections"]] == [
        (s.title, list(s.headers), [list(r) for r in s.rows], s.placement) for s in tab_document.sections
    ]
    assert [(n["text"], n["kind"]) for n in document["notes"]] == [(n.text, n.kind) for n in tab_document.notes]
    analysis = [n.text for n in tab_document.notes if n.kind == "analysis"]
    assert analysis and detail["narrative"] in analysis[0] and detail["narrative"].startswith("Shift B ran stable")
    assert (
        detail["status"] == "complete"
        and detail["narrative_tries"] == 1
        and detail["narrative_model"] == "fake/model-1"
    )


def test_the_tab_still_shows_its_report_through_the_shared_building_blocks(monkeypatch) -> None:
    """``generate_document`` without analysis fetches the shift only and stamps the document."""
    data, document = report_service.ShiftReportService().generate_document(SHIFT.date, "B", include_analysis=False)
    assert FakeFetcher.windows == ["2026-10-26_SHIFT_B"]
    assert data.shift_label == "B" and document.generated_at_ist is not None
    assert not [n for n in document.notes if n.kind == "analysis"]


# ------------------------------------------------------------------------------------------------------- narrative
def test_one_retry_after_an_llm_failure() -> None:
    sleeps: list[float] = []
    flaky = FlakyLLM(failures=1)
    data = reports.generate(SHIFT, narrative=False).data
    note = reports.write_narrative(data, None, client_factory=flaky, retry_delay_s=15, sleep=sleeps.append)
    assert (note.status, note.tries, note.text) == ("ok", 2, "Watch the belly spread.")
    assert flaky.calls == 2 and sleeps == [15]


def test_the_report_is_stored_without_narrative_when_the_llm_fails_twice() -> None:
    failing = FlakyLLM(failures=99)
    generated = reports.generate(SHIFT, client_factory=failing)
    assert failing.calls == reports.LLM_TRIES == 2  # the first call and one retry, no more
    note = generated.narrative
    assert note.status == "failed" and note.text is None and note.tries == 2
    assert note.error == "try 1: RuntimeError: 503 provider busy; try 2: RuntimeError: 503 provider busy"
    assert generated.data.fuel_rate is not None and generated.document.sections  # the data are kept
    assert not [n for n in generated.document.notes if n.kind == "analysis"]
    values = reports.store_values(generated, job_id=None, trigger="schedule", requested_by="scheduler")
    assert values["narrative_status"] == "failed" and values["narrative"] is None and values["data"]["status"]
    assert reports.report_status(values["narrative_status"]) == "no_narrative"


def test_an_empty_answer_counts_as_a_failure_and_a_missing_client_is_not_called() -> None:
    data = reports.generate(SHIFT, narrative=False).data
    empty = FlakyLLM(failures=0, answer="  \n")
    note = reports.write_narrative(data, None, client_factory=empty, retry_delay_s=0)
    assert note.status == "failed" and empty.calls == 2 and "empty narrative" in note.error

    def no_key() -> Any:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    missing = reports.write_narrative(data, None, client_factory=no_key)
    assert missing.status == "failed" and missing.tries == 0 and "OPENROUTER_API_KEY" in missing.error


def test_without_narrative_the_previous_shift_is_not_fetched() -> None:
    generated = reports.generate(SHIFT, narrative=False)
    assert FakeFetcher.windows == ["2026-10-26_SHIFT_B"] and generated.narrative.status == "skipped"


def test_a_data_source_failure_is_a_report_data_error() -> None:
    FakeFetcher.fail = True
    with pytest.raises(reports.ReportDataError, match="2026-10-26_B could not be fetched"):
        reports.generate(SHIFT)


def test_the_llm_client_is_page_7s_with_a_time_limit_and_no_sdk_retries() -> None:
    from agents.llm.llm_client import OpenRouterClient

    client = reports.narrative_client()
    assert isinstance(client, OpenRouterClient)
    assert client.client.timeout == reports.LLM_TIMEOUT_S and client.client.max_retries == 0


# ---------------------------------------------------------------------------------------------------------- shifts
@pytest.mark.parametrize(
    ("slot", "closed", "running"),
    [
        ("2026-10-27 06:20", "2026-10-26_C", "2026-10-27_A"),
        ("2026-10-27 14:20", "2026-10-27_A", "2026-10-27_B"),
        ("2026-10-27 22:20", "2026-10-27_B", "2026-10-27_C"),
        ("2026-10-28 02:00", "2026-10-27_B", "2026-10-27_C"),
    ],
)
def test_the_shift_of_each_slot(slot: str, closed: str, running: str) -> None:
    at = pd.Timestamp(slot, tz=IST).tz_convert("UTC").to_pydatetime()
    assert reports.closed_shift(at).key == closed and reports.running_shift(at).key == running


def test_the_report_kinds_are_catalogued() -> None:
    shift, live = CATALOG["report.shift"], CATALOG["report.live"]
    assert shift.schedule == ({6, 14, 22}, {20}) and shift.max_tries == 2 and shift.params is ShiftReportParams
    assert live.schedule is None and live.role == "user" and live.params is LiveReportParams
    budget = reports.LLM_TRIES * reports.LLM_TIMEOUT_S + reports.LLM_RETRY_DELAY_S
    assert shift.timeout_s >= budget + 120 and live.timeout_s >= budget + 120  # LLM calls and the data fetch fit
    assert reports.Shift.parse("2026-10-27_C").timeframe().duration_hours == 8.0
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        reports.Shift.parse("2026-10-27_D")


# ---------------------------------------------------------------------------------------------------------- worker
class FakeJobs:
    def __init__(self, rec: JobRecord | None) -> None:
        self.rec = rec

    async def get(self, job_id: str) -> JobRecord | None:
        return self.rec

    async def update(self, job_id: str, **changes: Any) -> None:
        return None


def job_record(kind: str, *, slot: datetime | None = None, by: str = "scheduler", **params: Any) -> JobRecord:
    return JobRecord(
        id=uuid.uuid4().hex,
        kind=kind,
        status="running",
        params=params,
        input_hash=input_hash(kind, params),
        submitted_by=by,
        submitted_at=slot or datetime.now(UTC),
        scheduled_for=slot,
    )


def context(kind: str, rec: JobRecord | None, attempt: int = 1) -> JobContext:
    return JobContext(job_id=rec.id if rec else "j1", kind=CATALOG[kind], attempt=attempt, store=FakeJobs(rec),
                      bus=MemoryEventBus(), pool=None)  # fmt: skip


def test_the_scheduled_job_stores_the_closed_shift_without_narrative_when_the_llm_fails(store, monkeypatch) -> None:
    failing = FlakyLLM(failures=99)
    monkeypatch.setattr(reports, "narrative_client", failing)
    slot = pd.Timestamp("2026-10-26 22:20", tz=IST).tz_convert("UTC").to_pydatetime()
    rec = job_record("report.shift", slot=slot)
    out = asyncio.run(report_kinds.shift(context("report.shift", rec), ShiftReportParams()))
    assert out["shift_key"] == "2026-10-26_B" and out["kind"] == "closed" and out["stored"]
    assert out["status"] == "no_narrative" and out["narrative_tries"] == 2 and failing.calls == 2
    row = store.reports.get(out["report_id"])
    assert (row.trigger, row.requested_by, row.job_id, row.narrative_status) == ("schedule", "scheduler", rec.id,
                                                                                  "failed")  # fmt: skip
    assert row.narrative is None and row.data["fuel_rate"] == out["fuel_rate_kg_thm"] is not None
    assert store.jobs.get(rec.id) is not None  # the job record the report refers to


def test_a_rerun_of_a_named_shift_with_the_narrative(store, monkeypatch) -> None:
    monkeypatch.setattr(reports, "narrative_client", FlakyLLM(failures=1))
    rec = job_record("report.shift", by="sup", shift_key="2026-05-08_B")
    out = asyncio.run(report_kinds.shift(context("report.shift", rec), ShiftReportParams(shift_key="2026-05-08_B")))
    row = store.reports.get(out["report_id"])
    assert (row.kind, row.trigger, row.narrative_status, row.narrative_tries) == ("closed", "user", "ok", 2)
    assert row.narrative == "Watch the belly spread."
    future = ShiftReportParams(shift_key="2099-01-01_A")
    with pytest.raises(ValueError, match="has not started"):
        asyncio.run(report_kinds.shift(context("report.shift", job_record("report.shift", by="sup")), future))


def test_unavailable_data_are_tried_again_then_fail(store) -> None:
    FakeFetcher.fail = True
    rec = job_record("report.shift", slot=datetime(2026, 10, 27, 0, 50, tzinfo=UTC))
    with pytest.raises(TransientJobError, match="could not be fetched"):
        asyncio.run(report_kinds.shift(context("report.shift", rec), ShiftReportParams(narrative=False)))
    with pytest.raises(reports.ReportDataError):
        asyncio.run(report_kinds.shift(context("report.shift", rec, attempt=2), ShiftReportParams(narrative=False)))
    assert store.reports.recent() == []


def test_the_live_job_reports_the_running_shift(store, monkeypatch) -> None:
    monkeypatch.setattr(reports, "narrative_client", FakeLLM)
    rec = job_record("report.live", by="op")
    out = asyncio.run(report_kinds.live(context("report.live", rec), LiveReportParams()))
    assert out["kind"] == "live" and out["shift_key"] == reports.running_shift().key
    row = store.reports.get(out["report_id"])
    assert (row.kind, row.trigger, row.requested_by, row.narrative_status) == ("live", "user", "op", "ok")


def test_without_a_store_the_report_is_the_job_result(monkeypatch) -> None:
    store_service.use(None)
    try:
        rec = job_record("report.live", by="op", narrative=False)
        out = asyncio.run(report_kinds.live(context("report.live", rec), LiveReportParams(narrative=False)))
    finally:
        store_service.reset()
    assert out["report_id"] is None and not out["stored"] and out["narrative_status"] == "skipped"
    assert out["data"]["production_rate"] is not None and out["document"]["sections"]
