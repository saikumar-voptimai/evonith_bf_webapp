"""Live forecasts in the worker (SV-1.5, #285): ``services.forecasts`` and ``worker.kinds.forecasts``.

The Si path and the charged-coke block are issued by the worker through the functions page 9 used, into the same
ledgers (the lines are compared with those of a direct call), and page 9 reads them back.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from data.bmo.si_forecast_context import SiliconForecastSources
from services import forecasts
from services.events import MemoryEventBus, make_event
from services.jobs import CATALOG, NoParams
from tests.redis_server import needs_redis, redis_server
from utils.bmo.charged_coke.bmo import _built_at, charged_coke_prediction
from utils.bmo.charged_coke.ledger import ForecastLedger, PlannedChangeRegister
from utils.bmo.charged_coke.service import ChargedCokeEngine, issue_latest
from utils.bmo.si_forecast.features import PLANT_TZ, matched_outcomes
from utils.bmo.si_forecast.ledger import SiliconForecastLedger
from utils.bmo.si_forecast.service import (
    PREDICTABLE_STATUSES,
    SiliconForecastService,
    _aware,
    forecast_horizons,
)
from worker.kinds import forecasts as forecast_kinds
from worker.runtime import JobContext, TransientJobError
from worker.settings import WorkerSettings

ROOT = Path(__file__).resolve().parents[1]
SI_BUNDLE = ROOT / "src/assets/models/bmo_si_forecast/20260924_initial"
SI_FIXTURE = ROOT / "tests/fixtures/bmo_si_forecast"
CHARGED_BUNDLE = ROOT / "src/assets/models/bmo_charged_coke/20261007_0400"
CHARGED_FIXTURE = ROOT / "tests/fixtures/charged_coke"


def _cfg(tmp_path: Path, **overrides) -> dict:
    cfg = {
        "forecast_issuer": "auto",
        "si_forecast": {
            "mode": "forecast",
            "default_bundle_dir": str(SI_BUNDLE),
            "storage_dir": str(tmp_path / "si"),
            "static_dataset_path": str(tmp_path / "data" / "furnace_dataset.csv"),
        },
        "data_driven_coke": {
            "dataset_path": str(tmp_path / "data" / "furnace_dataset.csv"),
            "charged_4h": {"storage_dir": str(tmp_path / "cc"), "bundle_dir": str(CHARGED_BUNDLE)},
        },
        "data_sources": {"static_dataset_url": ""},
    }
    cfg.update(overrides)
    return cfg


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ----------------------------------------------------------------------------------------------------------- Si
@pytest.fixture(scope="module")
def si_sources() -> SiliconForecastSources:
    kw = {"float_precision": "round_trip"}
    return SiliconForecastSources(
        online=pd.read_csv(SI_FIXTURE / "online_10min.csv.gz", **kw),
        labs=pd.read_csv(SI_FIXTURE / "raw_hm_slag.csv", **kw),
        charge=pd.read_csv(SI_FIXTURE / "raw_charge.csv.gz", **kw),
        hourly_gate=pd.read_csv(SI_FIXTURE / "hourly_gate_source.csv", **kw),
        fetched_at=pd.Timestamp("2026-10-07T00:00:00Z"),
    )


@pytest.fixture(scope="module")
def si_origin() -> pd.Timestamp:
    expected = pd.read_csv(SI_FIXTURE / "recent_predictions.csv", float_precision="round_trip")
    return pd.Timestamp(expected.iloc[-1]["origin"]).tz_convert("UTC")


def _factory(sources: SiliconForecastSources | Exception, calls: list | None = None):
    class Source:
        def __init__(self, *, bundle_dir, static_dataset_path) -> None:
            self.bundle_dir = bundle_dir

        def fetch(self, origin):
            if calls is not None:
                calls.append(pd.Timestamp(origin))
            if isinstance(sources, Exception):
                raise sources
            return sources

    return Source


def test_the_worker_writes_the_si_record_page_9_wrote(tmp_path, si_sources, si_origin) -> None:
    now = si_origin + pd.Timedelta(minutes=2)
    calls: list = []
    out = forecasts.issue_si(now=now, cfg=_cfg(tmp_path), source_factory=_factory(si_sources, calls))
    attempt, record = out["attempt"], out["record"]
    assert calls == [si_origin]  # the shipped bundle issues hourly: origin = the hour, floored in IST
    assert attempt["written"] and not attempt["existing"] and attempt["key"] == si_origin
    assert record["status"] in PREDICTABLE_STATUSES and attempt["issuer"] == "worker"

    direct = SiliconForecastService(bundle_dir=SI_BUNDLE, storage_dir=tmp_path / "direct")
    direct.issue(si_sources, origin=si_origin, issued_at=now, mode="forecast")
    assert _lines(tmp_path / "si" / "issued_forecasts.jsonl") == _lines(tmp_path / "direct" / "issued_forecasts.jsonl")

    later = forecasts.issue_si(
        now=now + pd.Timedelta(minutes=5), cfg=_cfg(tmp_path), source_factory=_factory(si_sources)
    )
    assert later["attempt"]["existing"] and not later["attempt"]["written"]
    assert later["record"] == record  # the first record at an origin stands
    issues = [r for r in _lines(tmp_path / "si" / "issued_forecasts.jsonl") if r["type"] == "issue"]
    assert len(issues) == 1
    log = forecasts.AttemptLog(tmp_path / "si")
    hour = pd.Timedelta(hours=1)
    assert [a["existing"] for a in log.between(now - hour, now + hour)] == [False, True]
    assert log.latest()["record"]["origin_at"] == record["origin_at"]

    view = forecasts.si_for_page(tmp_path / "si", SI_BUNDLE, now=now + pd.Timedelta(minutes=30))
    assert view.current and view.note is None and view.record["origin_at"] == record["origin_at"]


def test_unusable_inputs_write_no_record_and_page_9_shows_the_attempt(tmp_path, si_sources, si_origin) -> None:
    no_samples = SiliconForecastSources(
        online=si_sources.online,
        labs=si_sources.labs.iloc[:0],
        charge=si_sources.charge,
        hourly_gate=si_sources.hourly_gate,
        fetched_at=si_sources.fetched_at,
    )
    out = forecasts.issue_si(now=si_origin, cfg=_cfg(tmp_path), source_factory=_factory(no_samples))
    assert out["record"]["status"] == "stale_inputs" and not out["attempt"]["written"]
    assert not (tmp_path / "si" / "issued_forecasts.jsonl").exists()
    view = forecasts.si_for_page(tmp_path / "si", SI_BUNDLE, now=si_origin + pd.Timedelta(minutes=10))
    assert view.current and view.record["status"] == "stale_inputs"
    assert all(h["si_pct"] is None for h in forecast_horizons(view.record))


def test_a_source_failure_is_logged_and_shown(tmp_path, si_origin) -> None:
    with pytest.raises(RuntimeError, match="InfluxDB unavailable"):
        forecasts.issue_si(
            now=si_origin, cfg=_cfg(tmp_path), source_factory=_factory(RuntimeError("InfluxDB unavailable"))
        )
    latest = forecasts.AttemptLog(tmp_path / "si").latest()
    assert latest["status"] == "failed" and "InfluxDB unavailable" in latest["error"]
    view = forecasts.si_for_page(tmp_path / "si", SI_BUNDLE, now=si_origin + pd.Timedelta(minutes=5))
    assert view.record["status"] == "insufficient_data" and view.record["reason_codes"] == ["source_fetch_failed"]


def test_legacy_mode_issues_nothing(tmp_path, si_sources, si_origin) -> None:
    cfg = _cfg(tmp_path)
    cfg["si_forecast"]["mode"] = "legacy"
    out = forecasts.issue_si(now=si_origin, cfg=cfg, source_factory=_factory(si_sources))
    assert out["record"] is None and out["attempt"]["status"] == "skipped"
    assert not (tmp_path / "si").exists()


def _si_issue(ledger: SiliconForecastLedger, origin: pd.Timestamp, si: float = 0.42) -> None:
    ledger.record_issue(
        {
            "origin_at": origin.isoformat(),
            "issued_at": origin.isoformat(),
            "status": "ok",
            "si_pct": si,
            "horizons": [
                {
                    "horizon_minutes": 120,
                    "label": "+2 h",
                    "target_start": (origin + pd.Timedelta(hours=2)).isoformat(),
                    "target_end": (origin + pd.Timedelta(hours=3)).isoformat(),
                    "si_pct": si,
                    "lower_pct": None,
                    "upper_pct": None,
                    "status": "ok",
                }
            ],
        }
    )


def test_page_9_shows_only_current_si_records(tmp_path) -> None:
    origin = pd.Timestamp("2026-10-20T10:00:00+05:30").tz_convert("UTC")  # the shipped bundle: hourly origins
    now = origin + pd.Timedelta(minutes=10)
    storage = tmp_path / "si"
    assert forecasts.si_for_page(storage, SI_BUNDLE, now=now).record is None

    _si_issue(SiliconForecastLedger(storage), origin - pd.Timedelta(hours=2))
    stale = forecasts.si_for_page(storage, SI_BUNDLE, now=now)
    assert not stale.current and "No HM Si forecast has been issued since 20 Oct 08:00 IST" in stale.note
    assert all(h["si_pct"] is None for h in forecast_horizons(stale.record))  # nothing usable reaches the optimiser
    assert stale.record["reason_codes"] == ["forecast_out_of_date"]

    _si_issue(SiliconForecastLedger(storage), origin - pd.Timedelta(hours=1), si=0.45)
    previous = forecasts.si_for_page(storage, SI_BUNDLE, now=now)
    assert previous.current and previous.record["si_pct"] == 0.45 and "is being issued" in previous.note

    _si_issue(SiliconForecastLedger(storage), origin, si=0.47)
    current = forecasts.si_for_page(storage, SI_BUNDLE, now=now)
    assert current.current and current.record["si_pct"] == 0.47 and current.note is None


def test_who_issues(tmp_path) -> None:
    now = pd.Timestamp("2026-10-20T10:00:00Z")
    storage = tmp_path / "si"
    assert forecasts.page_issues(storage, "si", {"forecast_issuer": "page"}, now)
    assert not forecasts.page_issues(storage, "si", {"forecast_issuer": "worker"}, now)
    assert forecasts.page_issues(storage, "si", {"forecast_issuer": "auto"}, now)  # no worker attempt yet
    log = forecasts.AttemptLog(storage)
    log.record({"forecast": "si", "at": now - pd.Timedelta(minutes=20), "issuer": "worker", "status": "ok"}, None)
    assert forecasts.page_issues(storage, "si", {}, now)  # the worker's last attempt is 20 min old: silent
    log.record({"forecast": "si", "at": now - pd.Timedelta(minutes=10), "issuer": "worker", "status": "ok"}, None)
    assert not forecasts.page_issues(storage, "si", {}, now)
    log.record({"forecast": "si", "at": now - pd.Timedelta(minutes=1), "issuer": "page", "status": "ok"}, None)
    assert not forecasts.page_issues(storage, "si", {}, now)  # page attempts do not count as a worker
    assert forecasts.issuer({"forecast_issuer": "sometimes"}) == "auto"


def _reference_matches(service: SiliconForecastService, sources, available_by) -> list[tuple]:
    """The outcome matching before SV-1.5: every issue and horizon against ``matched_outcomes``."""
    known, out = set(), []
    for origin, issue in service.ledger.issues().items():
        if issue.get("status") not in PREDICTABLE_STATUSES:
            continue
        for h in forecast_horizons(issue):
            if h.get("si_pct") is None:
                continue
            end = _aware(h["target_end"]).tz_convert("UTC")
            if end > available_by:
                continue
            rows = matched_outcomes(
                sources.labs, target_start=_aware(h["target_start"]), target_end=end, available_by=available_by
            )
            for row in rows.itertuples():
                at = pd.Timestamp(row.sample).tz_localize(PLANT_TZ).tz_convert("UTC").isoformat()
                key = (origin.isoformat(), int(h["horizon_minutes"]), str(row.id), at)
                if key not in known:
                    known.add(key)
                    out.append(key)
    return out


def test_outcomes_are_matched_as_before(tmp_path, si_sources, si_origin) -> None:
    labs = si_sources.labs
    stamps = pd.to_datetime(labs["time (IST)"], format="mixed")
    recent = labs[stamps >= stamps.max() - pd.Timedelta(days=2)]  # two days of samples keep the reference fast
    sources = SiliconForecastSources(si_sources.online, recent, si_sources.charge, si_sources.hourly_gate,
                                     si_sources.fetched_at)  # fmt: skip
    ledger = SiliconForecastLedger(tmp_path)
    for k in range(80):  # 20-min origins over 26 h, four horizons each
        origin = si_origin - pd.Timedelta(minutes=20 * k)
        ledger.record_issue(
            {
                "origin_at": origin.isoformat(),
                "status": "ok" if k % 7 else "stale_inputs",
                "horizons": [
                    {
                        "horizon_minutes": h,
                        "target_start": (origin + pd.Timedelta(minutes=h)).isoformat(),
                        "target_end": (origin + pd.Timedelta(minutes=h + 20)).isoformat(),
                        "si_pct": 0.4,
                    }
                    for h in (0, 60, 120, 180)
                ],
            }
        )
    available_by = si_origin + pd.Timedelta(hours=1)
    service = SiliconForecastService(bundle_dir=SI_BUNDLE, storage_dir=tmp_path)
    reference = _reference_matches(service, sources, available_by)
    assert len(reference) > 20
    written = service.record_matured_outcomes(sources, available_by=available_by)
    got = [
        (pd.Timestamp(r["origin_at"]).isoformat(), r["horizon_minutes"], r["sample_id"], r["sample_at"])
        for r in ledger.outcomes()
    ]
    assert written == len(reference) and got == reference  # same matches, in the same order
    assert service.record_matured_outcomes(sources, available_by=available_by) == 0  # each recorded once


# ------------------------------------------------------------------------------------------------- charged coke
@pytest.fixture(scope="module")
def charged_source() -> pd.DataFrame:
    return pd.read_csv(CHARGED_FIXTURE / "source_window.csv.gz", parse_dates=["time"]).set_index("time")


@pytest.fixture(scope="module")
def charged_engine() -> ChargedCokeEngine:
    return forecasts.charged_engine(str(CHARGED_BUNDLE))


def _publish(tmp_path: Path, source: pd.DataFrame, row: pd.Timestamp) -> tuple[Path, pd.Timestamp]:
    """The build of ``row + 1 h`` at ``row + 1 h 06 min``, as the dataset manager saves it (versioned file, stable copy,
    cache_meta.json with the publisher's Last-Modified)."""
    built = row + pd.Timedelta(hours=1, minutes=6)
    folder = tmp_path / "data"
    folder.mkdir(parents=True, exist_ok=True)
    versioned = folder / f"furnace_dataset_{built:%Y%m%d_%H%M%S}.csv"
    source.loc[: row + pd.Timedelta(hours=1)].to_csv(versioned)
    (folder / "furnace_dataset.csv").write_bytes(versioned.read_bytes())
    gmt = built.tz_localize(PLANT_TZ).tz_convert("UTC").strftime("%a, %d %b %Y %H:%M:%S GMT")
    meta = {"csv_file": versioned.name, "source_last_modified": gmt, "source_url": ""}
    (folder / "cache_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return versioned, built


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is POSIX only")
def test_the_build_time_is_read_as_gmt_on_any_host(tmp_path, monkeypatch) -> None:

    meta = tmp_path / "cache_meta.json"
    meta.write_text(json.dumps({"source_last_modified": "Mon, 28 Sep 2026 07:36:00 GMT"}), encoding="utf-8")
    try:
        for zone in ("Europe/London", "Asia/Kolkata", "UTC"):  # London calls its winter time GMT
            monkeypatch.setenv("TZ", zone)
            time.tzset()
            assert _built_at(meta) == pd.Timestamp("2026-09-28 07:36:00", tz="UTC"), zone
    finally:
        monkeypatch.undo()
        time.tzset()
    meta.write_text(json.dumps({"source_last_modified": "2026-09-28T13:06:00+05:30"}), encoding="utf-8")
    assert _built_at(meta) == pd.Timestamp("2026-09-28 07:36:00", tz="UTC")


def test_the_worker_writes_the_charged_record_page_9_wrote(tmp_path, charged_source, charged_engine) -> None:
    row = pd.Timestamp("2026-09-28 12:00")
    csv, built = _publish(tmp_path, charged_source, row)
    now = built + pd.Timedelta(minutes=1)  # 13:07
    calls: list = []
    out = forecasts.issue_charged(now=now, cfg=_cfg(tmp_path), refresh_fn=lambda c: calls.append(c) or {})
    attempt, record = out["attempt"], out["record"]
    assert not calls  # the local build is this hour's
    assert attempt["written"] and not attempt["behind"] and pd.Timestamp(attempt["key"]) == row
    assert pd.Timestamp(record["data_row"]) == row

    direct = tmp_path / "direct"
    issue_latest(
        forecasts.read_charged_source(csv),
        charged_engine,
        ledger=ForecastLedger(direct),
        plans=PlannedChangeRegister(direct),
        now=now,
        built_at=_built_at(tmp_path / "data" / "cache_meta.json"),
        model_version=str(charged_engine.model.manifest.get("version", "")),
    )
    assert _lines(tmp_path / "cc" / "issued_forecasts.jsonl") == _lines(direct / "issued_forecasts.jsonl")

    again = forecasts.issue_charged(now=now + pd.Timedelta(minutes=20), cfg=_cfg(tmp_path), refresh=False)
    assert not again["attempt"]["written"] and again["record"] == record


def test_an_old_build_is_fetched_first_and_a_late_one_leaves_the_forecast_behind(
    tmp_path, charged_source, charged_engine
) -> None:
    row = pd.Timestamp("2026-09-28 12:00")
    _publish(tmp_path, charged_source, row - pd.Timedelta(hours=1))  # the 12:06 build, read at 13:08
    now = pd.Timestamp("2026-09-28 13:08")
    late = forecasts.issue_charged(now=now, cfg=_cfg(tmp_path), refresh_fn=lambda c: {"refreshed": False})
    assert late["attempt"]["behind"] and pd.Timestamp(late["record"]["data_row"]) == row - pd.Timedelta(hours=1)
    assert late["attempt"]["refresh"] == {"refreshed": "False"}

    seen: list = []

    def publish(data_cfg):
        seen.append(dict(data_cfg))
        _publish(tmp_path, charged_source, row)
        return {"refreshed": True}

    out = forecasts.issue_charged(now=now + pd.Timedelta(minutes=5), cfg=_cfg(tmp_path), refresh_fn=publish)
    assert seen and seen[0]["static_dataset_path"].endswith("furnace_dataset.csv")
    assert not out["attempt"]["behind"] and pd.Timestamp(out["record"]["data_row"]) == row


def test_page_9_reads_the_charged_record_of_the_latest_build(tmp_path, charged_source, charged_engine) -> None:
    row = pd.Timestamp("2026-09-28 12:00")
    csv, built = _publish(tmp_path, charged_source, row)
    issued = forecasts.issue_charged(now=built + pd.Timedelta(minutes=1), cfg=_cfg(tmp_path), refresh=False)["record"]
    storage = tmp_path / "cc"

    for now in ("2026-09-28 13:30", "2026-09-28 14:03"):  # the 12:00 row until the 14:06 build
        view = forecasts.charged_for_page(storage, now=pd.Timestamp(now))
        assert view.current and view.note is None and view.record == issued
    pending = forecasts.charged_for_page(storage, now=pd.Timestamp("2026-09-28 14:10"))
    assert pending.current and pending.record == issued and "being issued" in pending.note
    old = forecasts.charged_for_page(storage, now=pd.Timestamp("2026-09-28 15:10"))
    assert not old.current and old.record["state"] == "out_of_date" and not old.record["shown"]

    source = forecasts.read_charged_source(csv)
    kwargs = dict(now=built + pd.Timedelta(minutes=1), storage_dir=tmp_path / "unused", mode="fitted", built_at=built)
    read, _ = charged_coke_prediction(source, charged_engine, record=pending.record, **kwargs)
    issued_here, _ = charged_coke_prediction(source, charged_engine, **{**kwargs, "storage_dir": tmp_path / "page"})
    assert read == issued_here  # reading the ledger gives the level that issuing on the page gave
    unavailable, detail = charged_coke_prediction(source, charged_engine, record=old.record, **kwargs)
    assert not unavailable.usable and unavailable.value_kg_per_thm is None
    assert "No charged-coke forecast has been issued since the 28 Sep 12:00 row" in unavailable.reasons[0]


# ----------------------------------------------------------------------------------------------- events and jobs
def test_the_event_payloads_follow_their_schemas(tmp_path, si_sources, si_origin, charged_source) -> None:
    si = forecasts.issue_si(now=si_origin, cfg=_cfg(tmp_path), source_factory=_factory(si_sources))
    event = make_event(
        "1-0", "forecast.si", pd.Timestamp.now(tz="UTC"), forecasts.si_event(si["attempt"], si["record"])
    )
    assert event.data.recorded and event.data.origin_at == si_origin and event.data.horizons
    assert event.data.horizons[0].target_start.tzinfo is not None

    row = pd.Timestamp("2026-09-28 12:00")
    _, built = _publish(tmp_path, charged_source, row)
    cc = forecasts.issue_charged(now=built + pd.Timedelta(minutes=1), cfg=_cfg(tmp_path), refresh=False)
    payload = forecasts.charged_event(cc["record"])
    event = make_event("1-1", "forecast.charged_coke", pd.Timestamp.now(tz="UTC"), payload)
    assert event.data.data_row == row.tz_localize(PLANT_TZ) and len(event.data.path) == len(cc["record"]["path"])
    assert forecasts.charged_event({"data_row": None}) is None


def _context(kind: str, attempt: int = 1) -> tuple[JobContext, MemoryEventBus]:
    bus = MemoryEventBus()
    return JobContext(job_id="j1", kind=CATALOG[kind], attempt=attempt, store=None, bus=bus, pool=None), bus


def test_the_si_job_publishes_new_origins_only(monkeypatch) -> None:
    origin = pd.Timestamp("2026-10-20T04:30:00Z")
    record = {"origin_at": origin.isoformat(), "issued_at": origin.isoformat(), "status": "ok", "horizons": []}
    attempts = iter(
        [
            {"forecast": "si", "at": origin, "key": origin, "status": "ok", "written": True, "existing": False},
            {"forecast": "si", "at": origin, "key": origin, "status": "ok", "written": False, "existing": True},
        ]
    )
    monkeypatch.setattr(forecasts, "issue_si", lambda: {"attempt": next(attempts), "record": record})

    async def run():
        job, bus = _context("forecast.si")
        first = await forecast_kinds.si(job, NoParams())
        second = await forecast_kinds.si(job, NoParams())
        return first, second, await bus.recent(10)

    first, second, events = asyncio.run(run())
    assert first["written"] and second["existing"]
    assert [e.type for e in events] == ["forecast.si"] and events[0].data.recorded


def test_the_charged_job_retries_while_the_build_is_late(monkeypatch) -> None:
    attempt = {"forecast": "charged_coke", "key": "2026-10-20T09:00:00", "status": "cruise", "written": False,
               "behind": True, "built_at": "2026-10-20T09:06:53", "expected_row": "2026-10-20T10:00:00"}  # fmt: skip
    monkeypatch.setattr(forecasts, "issue_charged", lambda: {"attempt": attempt, "record": {"data_row": None}})
    kind = CATALOG["forecast.charged_coke"]

    async def run(n):
        job, bus = _context("forecast.charged_coke", attempt=n)
        return await forecast_kinds.charged_coke(job, NoParams()), await bus.recent(10)

    with pytest.raises(TransientJobError, match="not published yet"):
        asyncio.run(run(1))
    result, events = asyncio.run(run(kind.max_tries))  # the last try reports and ends
    assert result["behind"] and events == []


def test_the_forecast_jobs_are_scheduled_in_ist(monkeypatch) -> None:
    from worker import settings

    assert not any("forecast" in c.name for c in WorkerSettings.cron_jobs)  # BF_WORKER_SKIP_SCHEDULES of the tests
    monkeypatch.delenv("BF_WORKER_SKIP_SCHEDULES")
    names = {c.name: c for c in settings.cron_jobs()}
    si = names["cron:schedule_forecast_si"]
    assert si.hour is None and si.minute == set(range(0, 60, 5)) and si.second == 0
    cc = names["cron:schedule_forecast_charged_coke"]
    assert cc.hour is None and cc.minute == {7}
    kind = CATALOG["forecast.charged_coke"]
    assert kind.max_tries * kind.retry_delay_s <= 30 * 60  # retries end before the next hour's run
    assert CATALOG["forecast.si"].timeout_s < 300  # a run ends before the next one starts
    monkeypatch.setenv(
        "BF_WORKER_SKIP_SCHEDULES", "forecast.charged_coke, system., twin., optimiser., calibration., learn., report., strata."
    )
    assert sorted(c.name for c in settings.cron_jobs()) == ["cron:schedule_forecast_si"]

    from services.schedule import describe as _schedule_text  # as GET /api/jobs/kinds states them

    assert _schedule_text(CATALOG["forecast.si"].schedule) == "every 5 min"
    assert _schedule_text(CATALOG["forecast.charged_coke"].schedule) == "every hour at :07"
    assert _schedule_text((None, {10, 40})) == "every 30 min from :10" and _schedule_text(({4}, {0})) == "04:00"


def test_saving_the_dataset_does_not_load_streamlit(tmp_path) -> None:
    """The worker saves the refreshed dataset with ``StaticDatasetManager`` (atomic replace of the stable copy, then the
    metadata); its Streamlit cache is created only by a first call inside the app."""
    import os
    import subprocess
    import sys

    code = (
        "import sys, pandas as pd\n"
        "from data.ml.static_dataset_manager import StaticDatasetManager\n"
        f"m = StaticDatasetManager({str(tmp_path / 'furnace_dataset.csv')!r}, remote_url='http://plant/furnace.csv')\n"
        "df = pd.DataFrame({'x': [1.0, 2.0]}, index=pd.DatetimeIndex(['2026-01-01 00:00', '2026-01-01 01:00'], "
        "name='time'))\n"
        "df.attrs['source_last_modified'] = 'Mon, 28 Sep 2026 07:36:00 GMT'\n"
        "m.save(df)\n"
        "print('streamlit' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "furnace_data")])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "False"
    files = sorted(p.name for p in tmp_path.iterdir())
    assert (
        "furnace_dataset.csv" in files and "cache_meta.json" in files and not [f for f in files if f.endswith(".tmp")]
    )
    meta = json.loads((tmp_path / "cache_meta.json").read_text(encoding="utf-8"))
    assert meta["source_last_modified"] == "Mon, 28 Sep 2026 07:36:00 GMT" and meta["csv_file"] in files


@needs_redis
def test_the_jobs_run_in_the_worker(tmp_path, monkeypatch, si_sources, si_origin, charged_source) -> None:
    """Both kinds through arq: the Si path into the ledger with its event; the charged block after one retry for a
    late build."""
    from arq.connections import RedisSettings, create_pool
    from arq.worker import create_worker

    from services.events import RedisEventBus
    from services.jobs import ARQ_FUNCTION, QUEUE_NAME, JobStore

    cfg = _cfg(tmp_path)
    issue_si = forecasts.issue_si
    monkeypatch.setattr(
        forecasts,
        "issue_si",
        lambda: issue_si(now=si_origin + pd.Timedelta(minutes=1), cfg=cfg, source_factory=_factory(si_sources)),
    )
    row = pd.Timestamp("2026-09-28 12:00")
    _publish(tmp_path, charged_source, row - pd.Timedelta(hours=1))  # a late build: the 12:06 one at 13:08
    issue_charged, builds = forecasts.issue_charged, iter([None, row])

    def publish(data_cfg):
        build = next(builds)
        if build is not None:
            _publish(tmp_path, charged_source, build)
        return {"refreshed": build is not None}

    monkeypatch.setattr(
        forecasts,
        "issue_charged",
        lambda: issue_charged(now=pd.Timestamp("2026-09-28 13:08"), cfg=cfg, refresh_fn=publish),
    )
    monkeypatch.setitem(CATALOG, "forecast.charged_coke", replace(CATALOG["forecast.charged_coke"], retry_delay_s=0.2))

    async def run():
        with redis_server(tmp_path) as url:
            pool = await create_pool(RedisSettings.from_dsn(url), default_queue_name=QUEUE_NAME)

            async def enqueue(job_id: str) -> None:
                await pool.enqueue_job(ARQ_FUNCTION, job_id, _job_id=job_id, _queue_name=QUEUE_NAME)

            store = JobStore(pool, enqueue=enqueue)
            try:
                si_job, _ = await store.submit("forecast.si", {}, "test")
                cc_job, _ = await store.submit("forecast.charged_coke", {}, "test")
                worker = create_worker(
                    WorkerSettings, redis_settings=RedisSettings.from_dsn(url), poll_delay=0.05, handle_signals=False,
                    burst=True, cron_jobs=[],
                )  # fmt: skip
                await worker.main()
                await worker.close()
                return await store.get(si_job.id), await store.get(cc_job.id), await RedisEventBus(pool).recent(50)
            finally:
                await pool.aclose()

    si_done, cc_done, events = asyncio.run(run())
    assert si_done.status == "done" and si_done.result["written"] and si_done.result["key"] == si_origin.isoformat()
    assert cc_done.status == "done" and cc_done.tries == 2 and not cc_done.result["behind"]
    assert cc_done.result["key"] == row.isoformat()
    kinds = [e.type for e in events if e.type.startswith("forecast.")]
    assert sorted(kinds) == ["forecast.charged_coke", "forecast.charged_coke", "forecast.si"]  # 11:00 then 12:00 row
    assert len(_lines(tmp_path / "si" / "issued_forecasts.jsonl")) >= 1
    issued = [r["data_row"] for r in _lines(tmp_path / "cc" / "issued_forecasts.jsonl") if r["type"] == "issue"]
    assert issued == ["2026-09-28T11:00:00", "2026-09-28T12:00:00"]


# ----------------------------------------------------------------------------------------------------------- gaps
def test_gaps_list_the_intervals_without_a_record_and_the_attempts_in_them(tmp_path) -> None:
    storage = tmp_path / "si"
    ledger, log = SiliconForecastLedger(storage), forecasts.AttemptLog(storage)
    t0 = pd.Timestamp("2026-10-20T00:00:00Z")
    for k in range(13):  # 00:00 .. 01:00 every 5 min, without 00:25, 00:30, 00:35
        origin = t0 + pd.Timedelta(minutes=5 * k)
        if k in (5, 6, 7):
            if k < 7:
                log.record({"forecast": "si", "at": origin, "key": origin, "issuer": "worker", "status": "stale_inputs",
                            "written": False}, None)  # fmt: skip
            continue
        _si_issue(ledger, origin)
    found = forecasts.gaps("si", storage, start=t0, end=t0 + pd.Timedelta(hours=1), limit=pd.Timedelta(minutes=10))
    assert len(found) == 1
    gap = found[0]
    assert (gap.start, gap.end, gap.minutes) == (t0 + pd.Timedelta(minutes=20), t0 + pd.Timedelta(minutes=40), 20.0)
    assert gap.attempts == 2 and gap.statuses == {"stale_inputs": 2}

    spec = importlib.util.spec_from_file_location("forecast_gaps", ROOT / "scripts" / "forecast_gaps.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    args = ["--si-storage", str(storage), "--charged-storage", str(tmp_path / "cc"), "--end", "2026-10-20T01:00:00Z",
            "--days", "0.0417"]  # fmt: skip
    assert script.main(args) == 1  # a Si gap above 10 min
