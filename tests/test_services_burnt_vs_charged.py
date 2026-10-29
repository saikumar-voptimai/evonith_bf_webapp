"""Burnt against charged coke, admin analysis (SV-4.4, #307): ``services.burnt_vs_charged`` against
``utils.coke_energy.transit`` and the leak-detection model on the same shift table.

Acceptance: the window table, the variance split, the lag scan and the closure equal ``transit.window_agreement``,
``decompose_window_sd``, ``lag_scan`` and ``closure`` on the same data (relative difference <= 1e-6), on every
balanced shift and on the stable shifts; the leak check's model numbers equal ``plant_effects.leak_for_gap``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from services import burnt_vs_charged as bvc
from services import trace
from utils.coke_energy import leak_check as lc
from utils.coke_energy import transit as tr
from utils.coke_energy.pipeline import PipelineResult, PipelineSettings
from utils.strata_flow.plant_effects import leak_for_gap

REL = 1e-6
BUILT = pd.Timestamp("2026-10-20 07:20", tz="Asia/Kolkata")
CFG = {"twin": {"burnt_vs_charged": {"leak_check": {"recent_days": 3, "baseline_days": 14}}}}


def shift_table(days: int = 150, seed: int = 7, *, gap_step: float = 0.0, h2_step: float = 0.0,
                raft_step: float = 0.0, step_days: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """A pipeline-like shift table (charged coke as a random walk, burnt = charged + noise, raw burnt below it by a
    drifting offset, some unstable and missing shifts) and its input frame with top-gas H2 and RAFT. The steps are
    added over the last ``step_days`` days: the gap by lowering the raw burnt coke."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-05-01", periods=days, freq="D")
    keys = [(d, s) for d in dates for s in "ABC"]
    n = len(keys)
    k = 305 + np.cumsum(rng.normal(0, 2.5, n))
    k = k - (k.mean() - 305)
    burnt = k + rng.normal(0, 9, n) + 0.4
    raw = burnt - (6 + np.linspace(0, 3, n)) + rng.normal(0, 1, n)
    h2 = 2.6 + rng.normal(0, 0.12, n)
    raft = 2150 + rng.normal(0, 8, n)
    last = np.array([d >= dates[-step_days] for d, _ in keys])
    raw = raw - gap_step * last
    h2 = h2 + h2_step * last
    raft = raft + raft_step * last
    idx = [f"{d:%Y-%m-%d}_{s}" for d, s in keys]
    T = pd.DataFrame(
        {
            "sdate": [d for d, _ in keys],
            "shift": [s for _, s in keys],
            "t_mid": [d + pd.Timedelta(hours={"A": 10, "B": 18, "C": 26}[s]) for d, s in keys],
            "stable": rng.random(n) > 0.2,
            "provisional": False,
            "K_charged": k,
            "K_burnt": burnt,
            "K_burnt_raw": raw,
            "burnt_offset": burnt - raw,
            "hm_t": 730 + rng.normal(0, 25, n),
            "prod_th": 92 + rng.normal(0, 3, n),
            "eta_meas": 47 + rng.normal(0, 0.5, n),
            "humidity_g": 15 + rng.normal(0, 0.5, n),
        },
        index=idx,
    )
    if n > 90:
        drop = rng.choice(n - 30, size=25, replace=False)  # missing shifts (not in the last days)
        T = T.drop(T.index[drop])
    S = pd.DataFrame({"h2_pct": h2, "body_raft": raft}, index=idx)
    return T, S


def result_of(T: pd.DataFrame, S: pd.DataFrame) -> PipelineResult:
    empty = pd.DataFrame()
    return PipelineResult(S, empty, empty, empty, empty, empty, T, PipelineSettings())


def close(a, b) -> bool:
    if a is None or b is None or (isinstance(b, float) and math.isnan(b)):
        return a is None and (b is None or math.isnan(b))
    return abs(a - b) <= REL * max(1.0, abs(b))


@pytest.fixture(scope="module")
def world():
    T, S = shift_table()
    stored = json.loads(json.dumps(bvc.build(result_of(T, S), CFG, now=BUILT)))  # as the API reads it
    full = bvc.shift_table(result_of(T, S))
    return full, stored


def bases(full: pd.DataFrame):
    return {"all": full, "stable": full[full["stable"]]}


# ---------------------------------------------------------------------------------------------- windows
def test_the_windows_run_from_one_shift_to_one_month(world) -> None:
    _, stored = world
    assert [w["window"] for w in stored["windows"]][0] == "Shift (8 h)"
    assert stored["windows"][-1] == {"window": "1 month", "shifts": 90, "hours": 720.0}
    assert {k: v for k, v in bvc.WINDOWS.items() if k in tr.WINDOWS} == tr.WINDOWS  # page 11's windows included


@pytest.mark.parametrize("basis", ["all", "stable"])
def test_window_table_equals_window_agreement(world, basis) -> None:
    full, stored = world
    T = bases(full)[basis]
    expected = tr.window_agreement(T, bvc.WINDOWS).to_dict("records")
    if basis == "stable":  # the stable basis is window_agreement's stable_only on the whole table
        alt = tr.window_agreement(full, bvc.WINDOWS, stable_only=True).to_dict("records")
        assert [r["n"] for r in alt] == [r["n"] for r in expected]
    got = stored["bases"][basis]["table"]
    assert [r["window"] for r in got] == list(bvc.WINDOWS) and len(got) == len(expected)
    names = {"mean_charged": "mean_x", "mean_burnt": "mean_y"}
    bad = []
    for g, e in zip(got, expected, strict=True):
        assert g["n"] == e["n"] and g["shifts"] == e["shifts"] and g["hours"] == e["hours"]
        for key in ("ccc", "r", "cb", "bias", "sd", "loa_lo", "loa_hi", "mean_charged", "mean_burnt", "slope"):
            if not close(g[key], float(e[names.get(key, key)])):
                bad.append((g["window"], key, g[key], e[names.get(key, key)]))
    assert not bad, bad
    assert got[-1]["n"] >= 3 and got[0]["n"] == len(T)  # a month has windows; the shift row has every shift


def test_points_are_the_window_pairs(world) -> None:
    full, stored = world
    for label, w in bvc.WINDOWS.items():
        W = tr.window_pairs(full, w)
        pts = stored["bases"]["all"]["points"][label]
        assert len(pts) == len(W)
        for p, (t, r) in zip(pts, W.iterrows(), strict=True):
            assert pd.Timestamp(p["t"]) == t.tz_localize("Asia/Kolkata")
            assert close(p["burnt"], r["K_burnt"]) and close(p["charged"], r["K_charged"]) and p["n"] == r["n"]
            assert close(p["diff"], r["K_burnt"] - r["K_charged"])


def test_trace_page_values_are_unchanged() -> None:
    """``services.trace`` now takes its rows and pairs from the shared helpers: still ``transit.summary``'s."""
    T, _ = shift_table(days=60)
    out = trace.agreement(T)
    S = tr.summary(T)
    for g, e in zip(out["table"], S["table"].to_dict("records"), strict=True):
        assert g["n"] == e["n"] and close(g["ccc"], e["ccc"]) and close(g["loa_hi"], e["loa_hi"])
    assert len(out["points"]["week"]) == len(tr.window_pairs(T, 21))


# ---------------------------------------------------------------------------------------------- split and lag
@pytest.mark.parametrize("basis", ["all", "stable"])
def test_split_equals_decompose_window_sd(world, basis) -> None:
    full, stored = world
    T = bases(full)[basis]
    d = tr.decompose_window_sd(tr.window_agreement(T, bvc.WINDOWS))
    got = stored["bases"][basis]["split"]
    for key in ("transit_sd", "random_sd", "floor_sd"):
        assert close(got[key], d[key]), key
    fitted = {r["window"]: r for r in got["windows"]}
    assert set(d["fitted"]) == {k for k, r in fitted.items() if r["in_fit"]}
    for label, sd in d["fitted"].items():
        r = fitted[label]
        assert close(r["fitted_sd"], sd)
        parts = r["transit_sd"] ** 2 + r["random_sd"] ** 2 + r["floor_sd"] ** 2
        assert close(parts, sd**2)


def test_lag_equals_lag_scan(world) -> None:
    full, stored = world
    scan = tr.lag_scan(full, bvc.LAGS, detrend_shifts=bvc.DETREND_SHIFTS)
    got = stored["bases"]["all"]["lag"]
    assert [r["lag_shifts"] for r in got["rows"]] == list(bvc.LAGS)
    for g, e in zip(got["rows"], scan.to_dict("records"), strict=True):
        assert g["n"] == e["n"] and close(g["r"], e["r"]) and close(g["ccc"], e["ccc"]) and g["lag_h"] == e["lag_h"]
    assert close(got["peak_lag_h"], tr.peak_lag_hours(scan))


# ---------------------------------------------------------------------------------------------- closure
@pytest.mark.parametrize("basis", ["all", "stable"])
def test_closure_equals_transit_closure(world, basis) -> None:
    full, stored = world
    T = bases(full)[basis]
    c = tr.closure(T)
    got = stored["bases"][basis]["closure"]
    assert got["n"] == c["n"]
    for key in ("ratio", "burnt_t", "charged_t"):
        assert close(got[key], c[key]), key
    assert close(got["gap_t"], c["charged_t"] - c["burnt_t"])
    assert [p["skey"] for p in got["cumulative"]] == list(c["cumulative"].index)
    assert all(close(p["cumulative_t"], v) for p, v in zip(got["cumulative"], c["cumulative"], strict=True))
    assert close(got["cumulative"][-1]["cumulative_t"], got["gap_t"])
    for m in got["months"]:
        part = T[pd.to_datetime(T["sdate"]).dt.strftime("%Y-%m") == m["month"]]
        cm = tr.closure(part)
        assert m["n"] == cm["n"] and close(m["ratio"], cm["ratio"]) and close(m["burnt_t"], cm["burnt_t"])
    assert sum(m["n"] for m in got["months"]) == c["n"]


def test_closure_needs_the_raw_burnt_coke() -> None:
    T, _ = shift_table(days=30)
    assert bvc.closure(T.drop(columns=["K_burnt_raw"])) is None
    assert bvc.closure(pd.DataFrame()) is None


# ---------------------------------------------------------------------------------------------- leak water
def _check(**steps) -> dict:
    T, S = shift_table(**steps)
    return lc.check(bvc.shift_table(result_of(T, S)), bvc.settings_from(CFG))


def test_no_step_reads_none(world) -> None:
    leak = world[1]["bases"]["all"]["leak"]
    assert leak["state"] == "none"
    assert leak["recent_last"] == "2026-09-27" and leak["recent_first"] == "2026-09-25"
    assert leak["baseline_first"] == "2026-09-11"
    periods = {s["period"] for s in leak["shifts"]}
    assert periods == {"recent", "baseline"} and len([s for s in leak["shifts"] if s["period"] == "recent"]) == 9


def test_a_leak_shows_as_coke_gap_and_h2_rise_with_raft_unchanged() -> None:
    out = _check(gap_step=11.0, h2_step=1.0)  # 2 m3/h: +11 kg/tHM, about +1 pt H2
    assert out["state"] == "leak"
    gap = out["steps"]["gap"]
    recent = out["frame"][out["frame"]["period"] == "recent"]
    expected = leak_for_gap(gap["step"], production_t_h=recent["prod_th"].mean(),
                            eta_co=recent["eta_co_pct"].mean() / 100)  # fmt: skip
    assert out["model"]["leak_kg_h"] == pytest.approx(expected["leak_kg_h"], rel=REL)
    assert out["model"]["top_gas_h2_rise_pt"] == pytest.approx(expected["top_gas_h2_rise_pt"], rel=REL)
    assert out["model"]["leak_kg_h"] > 1500  # 11 kg/tHM is about 2 m3/h at 92 t/h


def test_the_other_signatures() -> None:
    assert _check(h2_step=0.8, raft_step=-45.0)["state"] == "moisture"
    assert _check(gap_step=11.0)["state"] == "gap_only"
    assert _check(gap_step=11.0, h2_step=1.0, raft_step=40.0)["state"] == "mixed"
    assert _check(h2_step=0.8)["state"] == "h2_only"


def test_too_few_shifts_read_unknown() -> None:
    T, S = shift_table(days=6)
    assert lc.check(bvc.shift_table(result_of(T, S)))["state"] == "unknown"
    assert lc.check(pd.DataFrame())["state"] == "unknown"
    no_h2 = bvc.shift_table(result_of(*shift_table(days=40)))
    assert lc.check(no_h2.assign(h2_pct=np.nan))["state"] == "unknown"


def test_step_is_welch() -> None:
    r, b = pd.Series([3.0, 5.0, 4.0]), pd.Series([1.0, 2.0, 0.0, 1.0])
    s = lc.step(r, b)
    se = math.sqrt(r.var() / 3 + b.var() / 4)
    assert s["step"] == pytest.approx(3.0) and s["se"] == pytest.approx(se) and s["z"] == pytest.approx(3.0 / se)
    assert math.isnan(lc.step(pd.Series([1.0]), b)["step"])


def test_settings_come_from_the_config() -> None:
    s = bvc.settings_from({"twin": {"burnt_vs_charged": {"leak_check": {"recent_days": "2", "z_min": 3}}}})
    assert s.recent_days == 2 and s.z_min == 3.0 and s.baseline_days == 14
    assert bvc.settings_from(None) == lc.LeakCheckSettings()
    shipped = bvc.settings_from(bvc_config())
    assert shipped.baseline_days == 14 and shipped.recent_days == 3


def bvc_config() -> dict:
    from services import coke_energy as ce

    return ce.config()


# ---------------------------------------------------------------------------------------------- store and build
def test_the_inputs_come_from_the_pipeline_frame() -> None:
    T, S = shift_table(days=20)
    full = bvc.shift_table(result_of(T, S))
    assert full["h2_pct"].equals(S["h2_pct"].reindex(T.index).astype(float))
    assert full["raft"].equals(S["body_raft"].reindex(T.index).astype(float))
    assert bvc.shift_table(result_of(T, pd.DataFrame()))["raft"].isna().all()


def test_store_round_trips(world, tmp_path) -> None:
    _, stored = world
    path = bvc.save(stored, tmp_path / bvc.STORE_NAME)
    assert bvc.load(path) == stored
    assert bvc.load(tmp_path / "missing.json") is None
    path.write_text(json.dumps({**stored, "version": bvc.VERSION + 1}), encoding="utf-8")
    assert bvc.load(path) is None
    assert not list(tmp_path.glob(".burnt_vs_charged_*"))
    assert bvc.path_beside(tmp_path / "trace_view.json") == tmp_path / bvc.STORE_NAME
    assert bvc.path_beside(None) == bvc.STORE_PATH
    assert stored["built_at"] == BUILT.tz_convert("UTC").isoformat()


class _Data:
    def __init__(self, result: PipelineResult) -> None:
        self.result = result


def test_the_trace_build_stores_the_analysis_beside_the_view(tmp_path, monkeypatch) -> None:
    T, S = shift_table(days=40)
    data = _Data(result_of(T, S))
    view = {"version": trace.VERSION, "built_at": "2026-10-20T01:50:00+00:00", "shifts": []}
    monkeypatch.setattr(trace.ce, "data_keys", lambda cfg, nonce: "keys")
    monkeypatch.setattr(trace.ce, "page_data", lambda keys, settings: data)
    monkeypatch.setattr(trace, "build_view", lambda d, cfg: dict(view))
    trace.refresh(path=tmp_path / "trace_view.json", cfg=CFG)
    stored = bvc.load(tmp_path / bvc.STORE_NAME)
    assert stored is not None and stored["built_at"] == view["built_at"]
    assert stored["bases"]["all"]["shifts"] == len(T)

    def broken(*args, **kwargs):
        raise RuntimeError("analysis failed")

    (tmp_path / bvc.STORE_NAME).unlink()
    monkeypatch.setattr(bvc, "build", broken)
    assert trace.refresh(path=tmp_path / "trace_view.json", cfg=CFG) == view  # the view is still stored
    assert trace.load(tmp_path / "trace_view.json") is not None and not (tmp_path / bvc.STORE_NAME).exists()
