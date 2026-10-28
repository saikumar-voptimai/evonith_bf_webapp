"""TRACE routes (SV-1.10, #290) on a stored view of the synthetic pipeline and a SQLite twin store: the summary
before and after the first build, per-shift / per-day / per-hour series, burnt against charged coke, units in the
schemas and the role."""

from __future__ import annotations

import dataclasses
import warnings
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import services.coke_energy as ce
from data.coke_energy.sources import freshness
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import store as store_service
from services import trace
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from tests.test_api_twin import UNITS, _numeric
from tests.twin_data import synthetic_frames
from utils.coke_energy.pipeline import PipelineSettings, run_pipeline

IST = "Asia/Kolkata"
SETTINGS = PipelineSettings(history_start="2026-04-01", predict_from="2026-04-01")
CFG = {"freshness_limits": {"online_10min": "30min"}}


@pytest.fixture(scope="module")
def page_data() -> ce.PageData:
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=18)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_pipeline(fd, m10, hm, dpr, SETTINGS)
    samples = ce.source_times(fd, m10, hm, dpr, [], pd.DataFrame())
    return ce.PageData(result, freshness(fd, m10, hm, dpr), [], samples, pd.DataFrame(), {})


@pytest.fixture()
def view_path(tmp_path, monkeypatch):
    path = tmp_path / "trace_view.json"
    monkeypatch.setattr(trace, "STORE_PATH", path)
    return path


@pytest.fixture()
def twin(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store_service.use(store)
    yield store
    store_service.reset()
    store.dispose()


@pytest.fixture()
def client(twin):
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def get(client: TestClient, url: str, user: str = "op", **kw: Any):
    return client.get(url, headers={"Cookie": client.cookies_by_user[user]}, **kw)  # type: ignore[attr-defined]


def ist(text: str) -> datetime:
    return pd.Timestamp(text, tz=IST).tz_convert("UTC").to_pydatetime()


def snapshot(end: datetime, **values: Any) -> dict[str, Any]:
    return {
        "window_start": end - timedelta(hours=8),
        "window_end": end,
        "evaluated_at": end + timedelta(minutes=10),
        "status": "complete",
        "k_trace": 311.0,
        "k_charged": 314.0,
        "quantities": {"k_burnt_raw_kg_thm": 310.5, "burnt_offset_kg_thm": 0.5, "hmt_c": 1480.0},
        "flags": {"notes": []},
        "versions": {"model": "test"},
        **values,
    }


def store_view(page_data: ce.PageData, now: pd.Timestamp | None = None) -> dict[str, Any]:
    view = trace.build_view(page_data, CFG, now=now if now is not None else pd.Timestamp("2026-05-19 08:00"))
    trace.save(view)
    return trace.load()


# ---------------------------------------------------------------------------------------------- summary
def test_before_the_first_build_the_summary_says_so(client, view_path) -> None:
    body = get(client, "/api/trace").json()
    assert body["available"] is False and body["stale"] is True and body["job_kind"] == "twin.trace"
    assert body["latest_shift"] is None and body["feeds"] == [] and body["feeds_at"] is None
    assert body["hour_available"] is True
    for url in ("/api/trace/series?resolution=shift", "/api/trace/series?resolution=day", "/api/trace/agreement"):
        r = get(client, url)
        assert r.status_code == 503 and "twin.trace" in r.json()["detail"], url
    assert get(client, "/api/trace/series?resolution=hour").json()["count"] == 0  # hours come from the twin store


def test_summary_serves_the_latest_shift_hour_correction_and_feeds(client, twin, view_path, page_data) -> None:
    view = store_view(page_data)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    twin.snapshots.upsert(snapshot(now - timedelta(hours=2)))
    twin.snapshots.upsert(snapshot(now - timedelta(hours=1), k_trace=None, status="failed", quantities={}))
    body = get(client, "/api/trace").json()
    assert body["available"] and body["shifts"] == len(view["shifts"]) and body["errors"] == 0
    assert body["stale"] and body["age_s"] > trace.STALE_AFTER_S  # built for 19 May
    latest = body["latest_shift"]
    assert latest["shift"] == {**view["shifts"][-1], "start": latest["shift"]["start"], "end": latest["shift"]["end"]}
    assert pd.Timestamp(latest["shift"]["end"]) == pd.Timestamp(view["shifts"][-1]["end"])
    assert body["previous_shift"]["skey"] == view["shifts"][-2]["skey"]
    assert [i["key"] for i in latest["inputs"]] == [i[1] for i in trace.INPUTS]
    assert latest["feeds"] and body["feeds"]
    hour = body["latest_hour"]  # the failed window has no burnt coke: the one before it
    assert hour["k_burnt"] == 311.0 and hour["k_burnt_raw"] == 310.5 and hour["burnt_offset"] == 0.5
    assert pd.Timestamp(hour["end"]) == pd.Timestamp(now - timedelta(hours=2))
    c = body["correction"]
    assert c["skey"] == view["correction"]["skey"] and c["offset"] == pytest.approx(view["correction"]["offset"])
    assert c["n_shifts"] == len(c["pool"]) >= c["min_shifts"]
    online = next(f for f in body["feeds"] if f["source"] == "online_10min")
    assert pd.Timestamp(body["feeds_at"]) == pd.Timestamp(body["built_at"])  # aged at the build, not at the request
    age = (pd.Timestamp(body["feeds_at"]) - pd.Timestamp(online["last"])).total_seconds()
    assert online["age_s"] == pytest.approx(age, abs=1) and online["stale"] and online["limit_s"] == 1800


def test_feeds_are_aged_at_the_build_not_at_the_request(client, view_path, page_data) -> None:
    """A view built 45 min before the request on a 10-min feed that was 10 min old at the build: the feed reads
    current (limit 30 min), and the view's own age is 45 min."""
    built = pd.Timestamp.now(tz=IST).tz_localize(None).floor("min") - pd.Timedelta(minutes=45)
    samples = page_data.samples.copy()
    online = samples["source"].eq("online_10min")
    times = pd.to_datetime(samples.loc[online, "t"])
    samples.loc[online, "t"] = times + (built - pd.Timedelta(minutes=10) - times.max())
    store_view(dataclasses.replace(page_data, samples=samples), now=built)
    body = get(client, "/api/trace").json()
    assert body["available"] and not body["stale"] and body["age_s"] == pytest.approx(45 * 60, abs=60)
    assert pd.Timestamp(body["feeds_at"]) == pd.Timestamp(body["built_at"])
    row = next(f for f in body["feeds"] if f["source"] == "online_10min")
    assert row["age_s"] == pytest.approx(600) and row["stale"] is False


def test_a_fresh_view_is_not_stale(client, view_path, page_data) -> None:
    store_view(page_data, now=pd.Timestamp.now(tz=IST).tz_localize(None) - pd.Timedelta(minutes=30))
    body = get(client, "/api/trace").json()
    assert body["available"] and not body["stale"] and body["age_s"] == pytest.approx(1800, abs=60)


def test_the_summary_answers_when_the_twin_store_fails(client, twin, view_path, page_data, monkeypatch) -> None:
    store_view(page_data)

    def broken(start, end):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(twin.snapshots, "series", broken)
    body = get(client, "/api/trace").json()
    assert body["available"] and body["latest_hour"] is None and body["hour_available"] is False
    store_service.use(None)
    assert get(client, "/api/trace").json()["hour_available"] is False
    r = get(client, "/api/trace/series?resolution=hour")
    assert r.status_code == 503 and "BF_TWIN_DB_URL" in r.json()["detail"]


# ---------------------------------------------------------------------------------------------- series
def test_series_per_shift_and_day_come_from_the_view(client, view_path, page_data) -> None:
    view = store_view(page_data)
    params = {"start": "2026-05-10T00:30:00Z", "end": "2026-05-12T00:30:00Z"}  # 06:00 IST to 06:00 IST
    shifts = get(client, "/api/trace/series", params={"resolution": "shift", **params}).json()
    assert shifts["count"] == 7 and shifts["rows"][0]["label"] == "2026-05-09_C"
    by_key = {r["skey"]: r for r in view["shifts"]}
    for row in shifts["rows"]:
        expected = by_key[row["label"]]
        for field in trace.FIELDS.values():
            assert row[field] == pytest.approx(expected[field]) if expected[field] is not None else row[field] is None
        assert row["stable"] == expected["stable"] and row["n_shifts"] == 1
    days = get(client, "/api/trace/series", params={"resolution": "day", **params}).json()
    assert [r["label"] for r in days["rows"]] == ["2026-05-09", "2026-05-10", "2026-05-11"]
    assert days["rows"][1]["n_shifts"] == 3 and days["built_at"] is not None


def test_series_per_hour_comes_from_the_twin_windows(client, twin, view_path) -> None:
    for h in (6, 7, 8):
        twin.snapshots.upsert(snapshot(ist(f"2026-05-20 {h:02d}:00"), k_trace=300.0 + h))
    hours = {"resolution": "hour", "start": "2026-05-20T00:00:00Z", "end": "2026-05-20T03:00:00Z"}
    body = get(client, "/api/trace/series", params=hours).json()
    assert body["count"] == 3 and [r["k_burnt"] for r in body["rows"]] == [306.0, 307.0, 308.0]
    assert body["rows"][0]["label"] == "2026-05-19_C" and body["rows"][1]["status"] == "complete"
    assert body["rows"][2]["k_burnt_raw"] == 310.5 and body["built_at"] is None


def test_series_limits(client, view_path, page_data) -> None:
    store_view(page_data)
    r = get(client, "/api/trace/series", params={"resolution": "shift", "start": "2026-05-12T00:00:00Z",
                                                  "end": "2026-05-10T00:00:00Z"})  # fmt: skip
    assert r.status_code == 422 and "start lies after end" in r.json()["detail"]
    r = get(client, "/api/trace/series", params={"resolution": "hour", "start": "2026-01-01T00:00:00Z",
                                                  "end": "2026-05-10T00:00:00Z"})  # fmt: skip
    assert r.status_code == 422 and "92 days" in r.json()["detail"]
    assert get(client, "/api/trace/series", params={"resolution": "week"}).status_code == 422


# ---------------------------------------------------------------------------------------------- agreement
def test_agreement_serves_the_view(client, view_path, page_data) -> None:
    view = store_view(page_data)
    body = get(client, "/api/trace/agreement").json()
    agree = view["agreement"]
    assert body["shift"] == trace.basis_stats(agree, "shift") and body["week"] == trace.basis_stats(agree, "week")
    assert body["table"] == agree["table"] and len(body["points_shift"]) == body["shift"]["n"]
    assert len(body["points_week"]) == agree["table"][-1]["n"]
    assert body["shift"]["loa_lo"] < body["shift"]["bias"] < body["shift"]["loa_hi"]
    assert body["split"].keys() == {"transit_sd", "random_sd", "floor_sd"}


# ---------------------------------------------------------------------------------------------- schema and role
def test_the_routes_need_a_session(client, view_path) -> None:
    for url in ("/api/trace", "/api/trace/series?resolution=shift", "/api/trace/agreement"):
        assert client.get(url).status_code == 401, url
    assert get(client, "/api/trace", user="op").status_code == 200  # role user


TRACE_UNITS = (*UNITS, "t/h", "d")


@pytest.mark.parametrize(
    "schema",
    [
        "TraceShift",
        "TraceLatestHour",
        "TraceCorrectionShift",
        "TraceCorrection",
        "TraceSettings",
        "TraceSummary",
        "TraceFeed",
        "TraceSeriesRow",
        "TraceSeries",
        "AgreementStats",
        "AgreementPoint",
        "AgreementSplit",
        "AgreementClosure",
        "TraceAgreement",
    ],
)
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    missing = []
    for name, prop in spec.get("properties", {}).items():
        if not _numeric(prop):
            continue
        unit = (prop.get("description") or "").rsplit(",", 1)[-1].strip()
        if not (prop.get("description") and "," in prop["description"] and unit.split(" ")[0] in TRACE_UNITS):
            missing.append(f"{name}: {prop.get('description')!r}")
    assert not missing, missing


def test_the_input_value_names_its_unit_field() -> None:
    prop = build().openapi()["components"]["schemas"]["TraceInput"]["properties"]
    assert "unit field" in prop["value"]["description"] and prop["digits"]["description"].endswith(", count")
