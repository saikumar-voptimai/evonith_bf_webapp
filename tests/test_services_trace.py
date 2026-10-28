"""TRACE (SV-1.10, #290): the view of ``services.trace`` against page 10's Streamlit path on the same synthetic data.

The sources are the twin tests' synthetic frames (26 days of shifts made from three real shift rows,
``tests/twin_data``) behind the source functions of ``data.coke_energy.sources``, so the Streamlit loaders
(``ui.coke_energy.loaders``) and the service (``services.coke_energy.page_data``) each run the real pipeline on them.
Acceptance: the per-shift values of the view equal page 10's shift table (relative difference <= 1e-6), and CCC and
limits of agreement equal ``transit.summary`` on the same table.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

import services.coke_energy as svc
import ui.coke_energy.loaders as st_loaders
from data.coke_energy import sources
from services import trace
from tests.twin_data import synthetic_frames
from ui.coke_energy.sections import freshness_table, shift_window
from utils.coke_energy import transit as tr
from utils.coke_energy.freshness import assess, in_use_between, parse_limits
from utils.coke_energy.pipeline import PipelineSettings

REL = 1e-6
CFG = {
    "history_start": "2026-04-01",
    "predict_from": "2026-04-01",
    "rolling_window_days": 14,
    "rolling_min_shifts": 10,
    "freshness_limits": {"hm_slag": "3h", "rm_chemistry": "1D", "ash_chemistry": "15D", "online_10min": "30min"},
}
SETTINGS = PipelineSettings.from_config(CFG)
KEYS = svc.DataKeys("synthetic.csv", 1, "2026-05-27T08:00:00+05:30", 0, "2026-04-01", "{}", "", 30.0)
BUILT = pd.Timestamp("2026-05-27 08:00")  # IST: after the last shift (26 May C ends 27 May 06:00)
NAMES = {"ore_1": "Lloyds CLO", "coke_1": "Coke", "sinter_1": "Sinter"}


def lab_and_charged(fd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lab-sample times (one stamped after the build: a daily ash analysis is stamped at the end of its day) and the
    tonnes charged per hour, from the synthetic hourly record."""
    lab = pd.DataFrame(
        {
            "source": [
                "rm_chemistry",
                "rm_chemistry",
                "rm_chemistry",
                "rm_chemistry",
                "ash_chemistry",
                "ash_chemistry",
            ],
            "material": ["ore_1", "ore_1", "coke_1", "sinter_1", "coke", "coke"],
            "t": [BUILT - pd.Timedelta(hours=h) for h in (60, 30, 12, 5)]
            + [pd.Timestamp("2026-05-20"), pd.Timestamp("2026-05-28")],
        }
    )
    charged = pd.DataFrame(
        {
            "ore_1": fd["ORE_CALC_MT"].to_numpy(),
            "coke_1": fd["COKE_CALC_MT"].to_numpy(),
            "sinter_1": fd["SINTER_CALC_MT"].to_numpy(),
            "charges": fd["CHARGES/HRS."].to_numpy(),
        },
        index=pd.DatetimeIndex(fd["time"]),
    )
    return lab, charged


def clear_all() -> None:
    """Forget the service's and page 10's caches, the day-long material names included (``clear_caches`` keeps
    them, and a later module would read these synthetic names)."""
    svc.clear()
    st_loaders.clear_caches()
    st_loaders.names.clear()


@pytest.fixture(scope="module")
def world():
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=26)
    lab, charged = lab_and_charged(fd)
    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp.setattr(sources, "load_hourly_dataset", lambda path: fd.copy())
        mp.setattr(sources, "fetch_10min", lambda start, end, fields, cache_dir, refresh_minutes: m10.copy())
        mp.setattr(sources, "fetch_hm_slag", lambda start, end: hm.copy())
        mp.setattr(sources, "fetch_dpr_daily", lambda start, end: dpr.copy())
        mp.setattr(svc, "fetch_offline_samples", lambda start: (lab.copy(), charged.copy()))
        mp.setattr(svc, "material_names", lambda: dict(NAMES))
        clear_all()
        service = svc.page_data(KEYS, SETTINGS)  # the API's and the worker's path
        streamlit = st_loaders.page_data(KEYS, SETTINGS)  # page 10's path
        clear_all()
    assert not service.result.errors and len(service.result.shifts) >= 70
    return service, streamlit, trace.build_view(service, CFG, now=BUILT)


def close(a, b) -> bool:
    """Relative difference <= REL; a missing value (None in the view, NaN in the table) only against a missing one."""
    if a is None or b is None or (isinstance(b, float) and math.isnan(b)):
        return a is None and (b is None or math.isnan(b))
    return abs(a - b) <= REL * max(1.0, abs(b))


# ---------------------------------------------------------------------------------------------- per shift and day
def test_shift_values_equal_page_10(world) -> None:
    _, streamlit, view = world
    T = streamlit.result.shifts  # what page 10 draws from
    assert [r["skey"] for r in view["shifts"]] == list(T.sort_values("t_mid").index)
    bad = []
    for r in view["shifts"]:
        row = T.loc[r["skey"]]
        for col, field in trace.FIELDS.items():
            if not close(r[field], float(row[col])):
                bad.append((r["skey"], field, r[field], row[col]))
        start, end = shift_window(row)  # page 10's shift window
        assert pd.Timestamp(r["start"]) == start.tz_localize("Asia/Kolkata")
        assert pd.Timestamp(r["end"]) == end.tz_localize("Asia/Kolkata")
        assert r["stable"] == bool(row["stable"]) and close(r["hm_t"], float(row["hm_t"]))
    assert not bad, bad[:5]
    calibrated = [r for r in view["shifts"] if r["k_burnt"] is not None]
    assert len(calibrated) >= 50  # the 14-day correction exists from the fifth day on
    for r in calibrated:  # calibrated = raw + correction
        assert r["k_burnt"] == pytest.approx(r["k_burnt_raw"] + r["burnt_offset"], rel=REL)


def test_latest_shift_inputs_and_feeds_equal_page_10(world) -> None:
    _, streamlit, view = world
    T = streamlit.result.shifts
    latest = view["latest"]
    assert latest["skey"] == T.index[-1] == "2026-05-26_C"
    row = T.loc[latest["skey"]]
    inputs = {i["key"]: i for i in latest["inputs"]}
    assert close(inputs["blast_air"]["value"], float(row["blast_air_nm3thm"]))
    assert close(inputs["eta_co"]["value"], float(row["eta_meas"])) and inputs["eta_co"]["unit"] == "%"
    assert close(inputs["coke_charged"]["value"], float(row["K_charged"]))
    assert {i["source"] for i in latest["inputs"]} <= {"online_10min", "hourly_dataset", "hm_slag", "open_meteo",
                                                      "plant_rule"}  # fmt: skip
    # page 10: assess at the shift end with the materials charged in the shift, shown by freshness_table
    start, end = shift_window(row)
    limits = parse_limits(CFG["freshness_limits"])
    page = assess(streamlit.samples, end, limits, in_use_between(streamlit.charged, start, end))
    table = freshness_table(page, streamlit.names)
    assert [f["item"] for f in latest["feeds"]] == list(table["Data"])
    assert [f["stale"] for f in latest["feeds"]] == list(page["stale"])
    assert [f["last"] for f in latest["feeds"]] == [trace.utc_text(t) for t in page["last"]]


def test_day_values_are_production_weighted_means_of_the_three_shifts(world) -> None:
    _, streamlit, view = world
    T = streamlit.result.shifts
    days = {d["day"]: d for d in view["days"]}
    assert len(days) == 26
    for day in ("2026-05-10", "2026-05-20", "2026-05-26"):
        part = T[T["sdate"] == pd.Timestamp(day)]
        assert len(part) == 3 and days[day]["n_shifts"] == 3
        for col, field in trace.FIELDS.items():
            expected = float((part[col] * part["hm_t"]).sum() / part["hm_t"].sum())
            assert close(days[day][field], expected), (day, field)
        assert pd.Timestamp(days[day]["start"]) == pd.Timestamp(f"{day} 06:00", tz="Asia/Kolkata")
    assert days["2026-05-02"]["k_burnt"] is None  # no correction yet: fewer than 10 stable shifts before
    assert days["2026-05-02"]["k_burnt_raw"] is not None


def test_correction_is_the_median_over_its_window(world) -> None:
    _, streamlit, view = world
    c = view["correction"]
    T = streamlit.result.shifts
    assert c["skey"] == "2026-05-26_C" and close(c["offset"], float(T.loc[c["skey"], "burnt_offset"]))
    assert (c["window_start"], c["window_end"], c["window_days"]) == ("2026-05-12", "2026-05-26", 14)
    assert c["n_shifts"] >= c["min_shifts"] == 10
    gaps = [p["gap"] for p in c["pool"]]
    assert len(gaps) == c["n_shifts"]
    assert float(np.nanmedian(gaps)) == pytest.approx(c["offset"], rel=REL)
    assert all("2026-05-12" <= p["skey"][:10] < "2026-05-26" for p in c["pool"])
    assert all(bool(T.loc[p["skey"], "stable"]) for p in c["pool"])


# ---------------------------------------------------------------------------------------------- agreement
def test_ccc_and_limits_equal_transit_summary(world) -> None:
    _, streamlit, view = world
    S = tr.summary(streamlit.result.shifts)  # on page 10's table
    table = S["table"].set_index("window")
    for basis, label in trace.BASES.items():
        stats = trace.basis_stats(view["agreement"], basis)
        expected = table.loc[label]
        assert stats["n"] == int(expected["n"]) and stats["n"] >= 3, basis
        for field, col in (("ccc", "ccc"), ("bias", "bias"), ("sd", "sd"), ("loa_lo", "loa_lo"), ("loa_hi", "loa_hi"),
                           ("r", "r"), ("mean_charged", "mean_x"), ("mean_burnt", "mean_y")):  # fmt: skip
            assert close(stats[field], float(expected[col])), (basis, field)
        # the pairs drawn in the Bland-Altman scatter are the ones the statistics come from
        points = view["agreement"]["points"][basis]
        assert len(points) == stats["n"]
        c = tr.concordance([p["charged"] for p in points], [p["burnt"] for p in points])
        assert close(stats["ccc"], c["ccc"]) and close(stats["loa_hi"], c["loa_hi"])
        assert np.mean([p["diff"] for p in points]) == pytest.approx(stats["bias"], rel=REL, abs=1e-9)
        for p in points:
            assert p["diff"] == pytest.approx(p["burnt"] - p["charged"])
            assert p["mean"] == pytest.approx((p["burnt"] + p["charged"]) / 2)
    assert view["agreement"]["closure"]["n"] == S["closure"]["n"]
    assert close(view["agreement"]["closure"]["ratio"], S["closure"]["ratio"])


def test_window_agreement_is_unchanged_by_window_pairs(world) -> None:
    """``window_agreement`` now reads its pairs from ``window_pairs``; the result equals the concordance of the pairs
    computed the way it was computed before (``accuracy.window_means`` on the rows with both values)."""
    from utils.coke_energy.accuracy import window_means

    T = world[1].result.shifts
    P = T[T["K_burnt"].notna() & T["K_charged"].notna()]
    for label, w in tr.WINDOWS.items():
        W = window_means(P, w, ["K_burnt", "K_charged"]).dropna(subset=["K_burnt", "K_charged"])
        before = tr.concordance(W["K_charged"], W["K_burnt"])
        after = tr.window_agreement(T).set_index("window").loc[label]
        assert after["n"] == before["n"] and close(float(after["ccc"]), before["ccc"]), label
    empty = tr.window_agreement(pd.DataFrame())
    assert list(empty["n"]) == [0] * len(tr.WINDOWS)


# ---------------------------------------------------------------------------------------------- feeds
def test_feeds_aged_at_a_later_request_equal_page_10_at_that_time(world) -> None:
    service, _, view = world
    stored = json.loads(json.dumps(view["feeds"]))  # as read back from the file
    assert len(stored["samples"]) < len(service.samples) / 50  # last times only
    limits = parse_limits(CFG["freshness_limits"])
    in_use = trace.in_use_now(service.charged, BUILT)
    assert in_use == {"ore_1", "coke_1", "sinter_1"}
    for later in (BUILT, BUILT + pd.Timedelta(hours=3), BUILT + pd.Timedelta(days=2)):
        expected = trace.feed_rows(assess(service.samples, later, limits, in_use), service.names)
        assert trace.feeds_at(stored, CFG, later) == expected, later
    rows = {f["item"]: f for f in trace.feeds_at(stored, CFG, BUILT + pd.Timedelta(hours=3))}
    assert rows["Lloyds CLO chemistry"]["age_s"] == pytest.approx(33 * 3600)
    assert rows["Lloyds CLO chemistry"]["stale"] is True and rows["Coke chemistry"]["stale"] is False
    assert rows["Coke ash"]["last"] == trace.utc_text(pd.Timestamp("2026-05-20"))  # the 28 May stamp is later


# ---------------------------------------------------------------------------------------------- store and refresh
def test_the_view_is_json_and_round_trips(world, tmp_path) -> None:
    view = world[2]
    path = trace.save(view, tmp_path / "trace_view.json")
    assert trace.load(path) == json.loads(json.dumps(view))
    assert trace.load(tmp_path / "missing.json") is None
    path.write_text(json.dumps({**view, "version": trace.VERSION + 1}), encoding="utf-8")
    assert trace.load(path) is None  # a view of another format is not served
    assert not list(tmp_path.glob(".trace_view_*"))  # no temporary file left behind


def test_refresh_builds_from_the_memoised_page_data(world, tmp_path, monkeypatch) -> None:
    service = world[0]
    seen = []
    monkeypatch.setattr(trace.ce, "data_keys", lambda cfg, nonce: KEYS)
    monkeypatch.setattr(trace.ce, "page_data", lambda keys, settings: seen.append((keys, settings)) or service)
    view = trace.refresh(path=tmp_path / "v.json", cfg=CFG)
    assert seen == [(KEYS, SETTINGS)]
    assert trace.load(tmp_path / "v.json")["shifts"] == view["shifts"]
    head = trace.headline(view)
    assert head["latest_shift"] == "2026-05-26_C" and head["shifts"] == len(view["shifts"])
    assert head["k_burnt"] == view["shifts"][-1]["k_burnt"]


def test_series_filters_on_the_window_end(world) -> None:
    view = world[2]
    start = pd.Timestamp("2026-05-20 06:00", tz="Asia/Kolkata").to_pydatetime()
    end = pd.Timestamp("2026-05-21 06:00", tz="Asia/Kolkata").to_pydatetime()
    shifts = trace.shift_series(view, start, end)
    assert [r["label"] for r in shifts] == ["2026-05-19_C", "2026-05-20_A", "2026-05-20_B", "2026-05-20_C"]
    assert [r["label"] for r in trace.day_series(view, start, end)] == ["2026-05-19", "2026-05-20"]  # ends 06:00


def test_hour_series_reads_the_twin_windows() -> None:
    end = pd.Timestamp("2026-05-20 14:00", tz="Asia/Kolkata").tz_convert("UTC").to_pydatetime()
    rows = [
        {"window_end": end, "status": "complete", "k_trace": 310.0, "k_charged": 312.0,
         "quantities": {"k_burnt_raw_kg_thm": 309.0, "burnt_offset_kg_thm": 1.0}},
        {"window_end": end + pd.Timedelta(hours=1), "status": "failed", "k_trace": None, "quantities": {}},
    ]  # fmt: skip
    out = trace.hour_series(rows)
    assert out[0]["label"] == "2026-05-20_A" and out[0]["k_burnt"] == 310.0 and out[0]["k_burnt_raw"] == 309.0
    assert out[0]["burnt_offset"] == 1.0 and pd.Timestamp(out[0]["start"]) == pd.Timestamp(end) - pd.Timedelta(hours=8)
    assert out[1]["label"] == "2026-05-20 15:00" and out[1]["k_burnt"] is None
    assert trace.latest_hour(rows)["end"] == out[0]["end"]  # the failed window has no burnt coke


def test_the_worker_job_stores_the_view(world, tmp_path, monkeypatch) -> None:
    import asyncio

    from services.events import MemoryEventBus
    from services.jobs import CATALOG, NoParams
    from worker import runtime
    from worker.kinds import trace as trace_kind

    kind = CATALOG["twin.trace"]
    assert (
        kind.schedule == (None, {20})
        and kind.role == "supervisor"
        and runtime.HANDLERS["twin.trace"] is trace_kind.build
    )
    built = []
    monkeypatch.setattr(trace, "refresh", lambda: built.append(1) or world[2])

    class Jobs:
        updates: list = []

        async def update(self, job_id, **changes):
            self.updates.append(changes)

    jobs = Jobs()
    job = runtime.JobContext(job_id="j1", kind=kind, attempt=1, store=jobs, bus=MemoryEventBus(), pool=None)
    out = asyncio.run(trace_kind.build(job, NoParams()))
    assert built == [1] and out == trace.headline(world[2]) and out["latest_shift"] == "2026-05-26_C"
    assert jobs.updates[-1] == {"progress": 1.0, "message": "TRACE view stored"}
