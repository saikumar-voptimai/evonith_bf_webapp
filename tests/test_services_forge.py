"""FORGE page data (SV-1.11, #291) against the Coke Energy page on the same pipeline result.

The pipeline runs once on the synthetic sources of ``tests/twin_data.py`` (18 days of shifts made from three real shift
rows). The view is built as the ``twin.forge`` job builds it, stored and read back (JSON round trip), and every answer
of the API's service functions is compared with page 10's path on the in-memory result:

* statistics of a period: page 10's selection and ``accuracy`` (relative difference <= 1e-6), for the default period,
  shorter ones, with the non-stable shifts and per day over the trailing window;
* Rist frames: the stored lines equal ``coke_energy.build_rist_frames`` (``rist_diagram.shift_rist``) and the points
  A, D, E, W, T, P, P' and E' derived from them equal the builder's, point for point;
* calibration of a day: the medians equal the shift table's fixed calibration of the shifts of that day.
"""

from __future__ import annotations

import asyncio
import json
import math
import warnings
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from data.coke_energy import sources
from services import coke_energy as ce
from services import forge
from services.events import MemoryEventBus
from services.jobs import CATALOG, NoParams
from tests.twin_data import synthetic_frames
from utils.coke_energy.accuracy import accuracy, period_rows, scored_shifts
from utils.coke_energy.pipeline import PipelineSettings, run_pipeline
from utils.coke_energy.rist_diagram import OperatingLine, RistShift, rist_dict
from worker.kinds import forge as forge_kind
from worker.runtime import JobContext, TransientJobError

SETTINGS = PipelineSettings(history_start="2026-04-01", predict_from="2026-04-01")
FORGE_SETTINGS = forge.ForgeSettings(stats_days=30, rist_days=30)
BUILT = datetime(2026, 5, 19, 2, 50, tzinfo=UTC)
REL = 1e-6


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=18)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_pipeline(fd, m10, hm, dpr, SETTINGS)
    assert not result.errors
    data = ce.PageData(
        result=result,
        fresh=sources.freshness(fd, m10, hm, dpr),
        provisional=[],
        samples=pd.DataFrame(columns=["source", "material", "t"]),
        charged=pd.DataFrame(),
    )
    view = forge.build_view(data, SETTINGS, FORGE_SETTINGS, built_at=BUILT, job_id="job-1")
    directory = tmp_path_factory.mktemp("forge")
    forge.save(view, directory)
    return data, view, forge.load(directory)


def close(a: float | None, b: float) -> bool:
    if a is None:
        return not np.isfinite(b)
    return math.isclose(a, b, rel_tol=REL, abs_tol=1e-9)


def page_10_rows(T: pd.DataFrame, start: date, end: date, with_unstable: bool) -> pd.DataFrame:
    """Page 10's selection as ``render_prediction`` wrote it before it moved to ``accuracy.period_rows``."""
    scored = T[T.K_obal.notna() & T.K_energy.notna() & T.K_burnt.notna()]
    rows = scored[(scored.sdate.dt.date >= start) & (scored.sdate.dt.date <= end)]
    if not with_unstable:
        rows = rows[rows.stable]
    return rows


def assert_stats_equal(got: dict, rows: pd.DataFrame) -> None:
    ref = accuracy(rows["K_energy"], rows["K_burnt"])
    assert got["n"] == ref["n"] == len(rows)
    for key in ("bias", "sd", "mae", "r"):
        assert close(got[key], ref[key]), (key, got[key], ref[key])
    assert close(got["within_5_pct"], 100 * ref["within_5"]) and close(got["within_10_pct"], 100 * ref["within_10"])


# ---------------------------------------------------------------------------------------------- statistics
def test_the_synthetic_record_is_scored(world) -> None:
    data, _, stored = world
    T = data.result.shifts
    scored = scored_shifts(T)
    assert len(scored) >= 30 and (~T.stable).sum() >= 1  # stable and non-stable shifts to tell apart
    assert len(stored.shifts) == len(T)


@pytest.mark.parametrize(
    ("back", "length", "unstable"),
    [(None, None, False), (None, None, True), (0, 4, False), (3, 6, True), (8, 0, False)],
    ids=["default", "default-unstable", "last-5-days", "6-days-earlier", "one-day"],
)
def test_statistics_equal_page_10_for_the_same_window(world, back, length, unstable) -> None:
    data, _, stored = world
    T = data.result.shifts
    scored = scored_shifts(T)
    last = scored.sdate.max().date()
    if back is None:
        start = end = None
        ref_end = last
        ref_start = max(scored.sdate.min().date(), last - timedelta(days=30))  # page 10's default period
    else:
        ref_end = end = last - timedelta(days=back)
        ref_start = start = end - timedelta(days=length)
    got = forge.prediction(stored, start, end, unstable)
    rows = page_10_rows(T, ref_start, ref_end, unstable)
    assert (got["start"], got["end"]) == (ref_start, ref_end)
    pd.testing.assert_frame_equal(period_rows(scored, ref_start, ref_end, unstable), rows)
    assert_stats_equal(got["stats"], rows)
    assert [s["shift_key"] for s in got["shifts"]] == list(rows.index)
    for s in got["shifts"]:
        assert close(s["k_forge"], rows.loc[s["shift_key"], "K_energy"])
        assert close(s["k_burnt"], rows.loc[s["shift_key"], "K_burnt"])
        assert close(s["error"], rows.loc[s["shift_key"], "K_energy"] - rows.loc[s["shift_key"], "K_burnt"])


def test_the_record_and_the_rolling_statistics_equal_page_10(world) -> None:
    data, _, stored = world
    T = data.result.shifts
    scored = scored_shifts(T)
    first, last = scored.sdate.min().date(), scored.sdate.max().date()
    got = forge.prediction(stored, first, last, False)
    assert_stats_equal(got["record"], page_10_rows(T, first, last, False))
    assert [p["day"] for p in got["rolling"]] == sorted(set(page_10_rows(T, first, last, False).sdate.dt.date))
    for p in got["rolling"]:
        assert_stats_equal(p, page_10_rows(T, p["day"] - timedelta(days=30), p["day"], False))
        assert p["t"].tzinfo is not None


def test_a_period_without_shifts_has_empty_statistics(world) -> None:
    _, _, stored = world
    got = forge.prediction(stored, date(2026, 1, 1), date(2026, 1, 31))
    assert got["shifts"] == [] and got["stats"]["n"] == 0 and got["stats"]["bias"] is None
    assert got["record"]["n"] > 0


# ---------------------------------------------------------------------------------------------- Rist diagram
def _line(d: dict) -> OperatingLine:
    return OperatingLine(**d)


def _points(rs: RistShift) -> dict:
    a = rs.actual
    out = {"a": a.a, "d": a.d, "e": a.e, "w": a.w, "top": a.top, "w_full": rs.w_full, "p": rs.p_actual}
    if rs.model is not None:
        out.update(e_model=rs.model.e, p_model=rs.p_model)
    return out


def test_the_rist_lines_equal_the_builder_point_for_point(world) -> None:
    data, view, stored = world
    T = data.result.shifts
    last = ce.rist_candidates(T).sdate.max()
    keys = ce.rist_keys(T, (last - pd.Timedelta(days=30)).date(), last.date())
    built = ce.build_rist_frames(data, SETTINGS, keys)
    assert len(built) == len(keys) == view["rist_requested"] and len(built) >= 30
    got = forge.rist(stored)
    assert [f["skey"] for f in got["frames"]] == [f.skey for f in built]
    for frame, rs in zip(got["frames"], built, strict=True):
        assert {k: frame[k] for k in rist_dict(rs)} == rist_dict(rs)  # exact after the JSON round trip
        assert frame["model"] is not None and frame["ideal"] is not None
        line, model = _line(frame["actual"]), _line(frame["model"])
        ref = _points(rs)
        assert (line.a, line.d, line.e, line.w, line.top) == (ref["a"], ref["d"], ref["e"], ref["w"], ref["top"])
        assert (line.w[0], frame["y_w"]) == ref["w_full"]
        assert tuple(frame["p_actual"]) == ref["p"] and tuple(frame["p_model"]) == ref["p_model"]
        assert model.e == ref["e_model"]
        assert frame["k_burnt"] == T.loc[rs.skey, "K_burnt"] and frame["k_forge"] == T.loc[rs.skey, "K_energy"]


def test_the_rist_frames_of_a_period(world) -> None:
    _, _, stored = world
    every = forge.rist(stored)["frames"]
    day = every[len(every) // 2]["day"]
    got = forge.rist(stored, day, day)
    assert got["frames"] and all(f["day"] == day for f in got["frames"])
    assert (got["first_day"], got["last_day"]) == (day, day)


def test_a_frame_that_is_not_finite_is_left_out(world) -> None:
    data, _, _ = world
    T = data.result.shifts
    rs = ce.build_rist_frames(data, SETTINGS, [ce.rist_keys(T)[-1]])[0]
    bad_model = RistShift(
        rs.skey, rs.actual, rs.p_actual, OperatingLine(**{**rist_dict(rs)["model"], "mu": float("nan")}), (1.0, np.nan),
        rs.ideal_mu, rs.ideal_y_a, rs.T_R,
    )  # fmt: skip
    payload = forge.frame_payload(bad_model)
    assert payload["model"] is None and payload["p_model"] is None and payload["actual"] == rist_dict(rs)["actual"]
    bad_shift = RistShift(rs.skey, OperatingLine(**{**rist_dict(rs)["actual"], "y_a": float("inf")}), None, None, None,
                          None, None, rs.T_R)  # fmt: skip
    assert forge.frame_payload(bad_shift) is None


# ---------------------------------------------------------------------------------------------- calibration
def test_the_calibration_of_a_day_equals_the_shift_tables_fixed_calibration(world) -> None:
    data, _, stored = world
    T = data.result.shifts
    predicted = T[T.omega_fixed.notna()]
    assert len(predicted) >= 30
    for day, rows in predicted.groupby(predicted.sdate.dt.date):
        cal = forge.calibration(stored, day)
        row = rows.iloc[0]
        assert cal["complete"] and cal["n_shifts"] >= SETTINGS.min_shifts
        assert close(cal["omega_cal"], row.omega_fixed) and close(cal["r_h"], row.r_H_fixed)
        assert close(cal["q_lz_gj_h"], row.Q_LZ_GJh_fixed) and close(cal["q_uz_gj_h"], row.Q_UZ_GJh_fixed)
        lo = pd.Timestamp(day) - pd.Timedelta(days=14)
        used = T[T.stable & (T.sdate >= lo) & (T.sdate < pd.Timestamp(day))]
        assert [s["shift_key"] for s in cal["shifts"]] == list(used.sort_values("t_mid").index)
        assert np.median([s["omega"] for s in cal["shifts"]]) == pytest.approx(row.omega_fixed, rel=1e-12)


def test_the_default_calibration_is_the_newest_14_days_and_short_windows_have_none(world) -> None:
    data, _, stored = world
    T = data.result.shifts
    cal = forge.calibration(stored)
    assert cal["day"] == (T.sdate.max() + pd.Timedelta(days=1)).date()
    assert cal["window_end"] == T.sdate.max().date() and cal["window_days"] == 14
    early = forge.calibration(stored, date(2026, 5, 3))  # two days of shifts before it
    assert not early["complete"] and early["n_shifts"] == 6 and early["omega_cal"] is None


# ---------------------------------------------------------------------------------------------- store and job
def test_the_view_is_read_back_unchanged(world, tmp_path) -> None:
    data, view, stored = world
    T = data.result.shifts
    pd.testing.assert_series_equal(stored.shifts["K_energy"], T["K_energy"], check_names=False)
    pd.testing.assert_series_equal(stored.shifts["omega_shift"], T["omega_shift"], check_names=False)
    assert stored.built_at == BUILT and stored.info()["sources"]["hourly_end"].tzinfo is not None
    with pytest.raises(forge.ViewUnavailable, match="not been built"):
        forge.load(tmp_path / "missing")
    forge.save({**view, "schema": 0}, tmp_path)
    with pytest.raises(forge.ViewUnavailable, match="schema"):
        forge.load(tmp_path)


def test_refresh_builds_from_the_memoised_pipeline(world, tmp_path, monkeypatch) -> None:
    data, view, _ = world
    calls = []
    monkeypatch.setattr(ce, "data_keys", lambda cfg, nonce: "keys")
    monkeypatch.setattr(ce, "page_data", lambda keys, settings: data)

    def frames(keys, settings, skeys):
        calls.append((keys, skeys))
        return ce.build_rist_frames(data, settings, skeys)

    monkeypatch.setattr(ce, "rist_frames", frames)
    cfg = {"history_start": "2026-04-01", "predict_from": "2026-04-01", "twin": {"forge": {"rist_days": 30}}}
    out = forge.refresh(job_id="job-2", cfg=cfg, directory=tmp_path, now=BUILT)
    assert calls and calls[0][0] == "keys" and len(calls[0][1]) == view["rist_requested"]
    assert forge.load(tmp_path).meta["job_id"] == "job-2" and out["rist"] == view["rist"]
    assert forge.headline(out)["rist_frames"] == len(view["rist"])

    def down(keys, settings):
        raise ConnectionError("InfluxDB unreachable")

    monkeypatch.setattr(ce, "page_data", down)
    with pytest.raises(forge.PlantDataUnavailable, match="InfluxDB unreachable"):
        forge.refresh(cfg=cfg, directory=tmp_path)


def test_settings_come_from_the_config() -> None:
    assert forge.ForgeSettings.from_config(ce.config()) == forge.ForgeSettings(stats_days=30, rist_days=30)
    assert forge.ForgeSettings.from_config({"ui": {"timeline_default_days": 21}}).stats_days == 21


def test_the_job_is_scheduled_hourly_and_retries_when_the_sources_fail(monkeypatch) -> None:
    kind = CATALOG["twin.forge"]
    assert kind.schedule == (None, {20}) and kind.role == "supervisor"
    assert (kind.max_tries - 1) * kind.retry_delay_s + kind.timeout_s < 3600

    class Records:  # the job store: progress only
        async def update(self, job_id, **changes):
            return None

    def context(attempt: int) -> JobContext:
        return JobContext(job_id="j1", kind=kind, attempt=attempt, store=Records(), bus=MemoryEventBus(), pool=None)

    monkeypatch.setattr(forge, "refresh", lambda job_id: {"built_at": "t", "shifts": [{"skey": "k", "stable": 1}]})
    assert asyncio.run(forge_kind.build(context(1), NoParams()))["latest_shift"] == "k"

    def down(job_id):
        raise forge.PlantDataUnavailable("Neon down")

    monkeypatch.setattr(forge, "refresh", down)
    with pytest.raises(TransientJobError, match="Neon down"):
        asyncio.run(forge_kind.build(context(1), NoParams()))
    with pytest.raises(RuntimeError, match="stored FORGE view is kept"):
        asyncio.run(forge_kind.build(context(kind.max_tries), NoParams()))


def test_the_stored_view_is_strict_json(world, tmp_path) -> None:
    _, view, _ = world
    path = forge.save(view, tmp_path)
    json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(f"non-finite {c}"))
