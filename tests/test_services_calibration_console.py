"""Calibration console (SV-4.6, #309): the stored day calibration as ``live_calibration`` made it (unrounded), the
stable-shift mask of its window, the stored days, the Jacobian cache status and the nightly job's runs.

The window is built on the Coke Energy fixture shifts: three stable copies of 2026-10-14_A in the 14 days before
15 Oct 2026 and one shift that fails the blast-meter test; the pipeline's shift balances are run on them."""

from __future__ import annotations

import functools
import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.burden_engine.calibration import CalibrationInfo, fallback_calibration, live_calibration
from services import calibration_console as console
from services import coke_energy as ce
from services import day_calibration
from services.jobs import JobRecord
from utils.burden_engine import strata_network
from utils.coke_energy.pipeline import PipelineSettings, run_shift_balances
from utils.coke_energy.whatif import stable_mask, stable_tests

FIX = "tests/fixtures/coke_energy/shift_inputs.csv"
DAY = date(2026, 10, 15)
CFG = {"calibration": {"window_days": 14, "min_shifts": 2}}
USED = ["2026-10-12_B", "2026-10-13_C", "2026-10-14_A"]
UNSTABLE = "2026-10-13_A"
VALUES = ("omega", "Q_LZ_GJh", "Q_UZ_GJh", "r_H", "K", "coke_model_cal")
FIELDS = ("omega_cal", "q_lz_gj_h", "q_uz_gj_h", "r_h", "coke_kg_thm", "coke_model_kg_thm")


@functools.cache
def window_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(S, R): the fixture shifts plus three stable shifts and one unstable shift in the window of ``DAY``."""
    S = pd.read_csv(FIX, index_col=0)
    base = S.loc["2026-10-14_A"]
    extra = {}
    for key, blast in (("2026-10-12_B", 1.01), ("2026-10-13_C", 0.99), (UNSTABLE, 1.35)):
        row = base.copy()
        row["sdate"], row["shift"] = key[:10], key[-1]
        row["hot_blast_vol_nm3h"] = base["hot_blast_vol_nm3h"] * blast
        extra[key] = row
    S = pd.concat([S, pd.DataFrame(extra).T.astype(S.dtypes.to_dict())]).sort_index()
    R, errors = run_shift_balances(S, S.index)
    assert not errors
    return S, R


def result() -> SimpleNamespace:
    S, R = window_frames()
    return SimpleNamespace(S=S, R=R, settings=PipelineSettings())


def reference() -> CalibrationInfo:
    """``data.burden_engine.calibration.live_calibration`` on the window: what the console must show."""
    return live_calibration(result(), CFG, as_of=pd.Timestamp(DAY))


def nightly(directory, *, network=None, made_at=None) -> CalibrationInfo:
    """The nightly job's path: ``day_calibration.compute`` on the pipeline result, then ``save``."""
    info = day_calibration.compute(
        DAY, CFG, page_data_for=lambda keys, settings: SimpleNamespace(result=result()), ce_config=ce.config()
    )
    day_calibration.save(
        info,
        DAY,
        made_at=made_at or datetime(2026, 10, 14, 21, 0, 5, tzinfo=UTC),  # 02:30 IST
        job_id="job-1",
        network=network if network is not None else {"jacobian": "computed (17 network solves, 19 s)"},
        settings=CFG["calibration"],
        directory=directory,
    )
    return info


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


# --------------------------------------------------------------------------------------------- stability filter


def test_the_stability_filter_is_the_conjunction_of_its_band_tests() -> None:
    """``stable_mask`` built from ``STABLE_BOUNDS`` gives what the literal filter gave, NaN failing."""
    rng = np.random.default_rng(7)
    n = 3000
    R = pd.DataFrame(
        {
            "coke_kgthm_dry": rng.uniform(200, 420, n),
            "C_closure_pct": rng.uniform(-25, 8, n),
            "blast_meter_vs_reconciled_pct": rng.uniform(-12, 18, n),
            "R_out_cooling_total": rng.uniform(30, 420, n),
            "Fe_closure_pct": rng.uniform(-8, 8, n),
            "prod_th": rng.uniform(70, 115, n),
        },
        index=[f"k{i}" for i in range(n)],
    )
    R.iloc[::11, 3] = np.nan
    R.iloc[5] = [220, 5, -9, 400, 6, 75]  # every bound is inclusive
    literal = (
        R.coke_kgthm_dry.between(220, 400)
        & R.C_closure_pct.between(-20, 5)
        & R.blast_meter_vs_reconciled_pct.between(-9, 15)
        & R.R_out_cooling_total.between(40, 400)
        & R.Fe_closure_pct.between(-6, 6)
        & R.prod_th.between(75, 110)
    )
    assert stable_mask(R).equals(literal) and bool(stable_mask(R).iloc[5])
    assert stable_tests(R).all(axis=1).equals(literal)


def test_live_calibration_reports_the_mask_of_its_window() -> None:
    info = reference()
    assert info.source == "live" and info.shifts == USED and info.window == ("2026-10-01", "2026-10-14")
    assert [m["shift_key"] for m in info.mask] == ["2026-10-12_B", UNSTABLE, "2026-10-13_C", "2026-10-14_A"]
    assert [m["shift_key"] for m in info.mask if m["used"]] == info.shifts
    unstable = next(m for m in info.mask if m["shift_key"] == UNSTABLE)
    _, R = window_frames()
    tests = stable_tests(R).loc[UNSTABLE]
    assert not unstable["stable"] and unstable["failed"] == [c for c in tests.index if not tests[c]]
    assert "blast_meter_vs_reconciled_pct" in unstable["failed"]
    assert unstable["values"]["prod_th"] == pytest.approx(R.at[UNSTABLE, "prod_th"])


# ----------------------------------------------------------------------------------------------------- parity


def test_the_console_shows_the_calibration_of_the_window_unrounded(tmp_path) -> None:
    """Acceptance: the console's values equal ``data/burden_engine/calibration.py`` on the same window."""
    nightly(tmp_path)
    ref = reference()
    out = console.view(directory=tmp_path, cache_dir=tmp_path / "jac", config=CFG["calibration"], today=DAY)
    cal = out["calibration"]
    assert out["day"] == DAY and out["current"] and cal["source"] == "live" and cal["complete"]
    for field, key in zip(FIELDS, VALUES, strict=True):
        assert rel(cal[field], ref.cal[key]) <= 1e-6, field
    assert (cal["window_start"], cal["window_end"]) == (date(2026, 10, 1), date(2026, 10, 14))
    assert (cal["window_days"], cal["min_shifts"], cal["n_shifts"]) == (14, 2, ref.n_shifts)
    assert cal["shifts"] == ref.shifts and cal["message"] == ref.message
    # the mask: every shift of the 14 days; the used ones are the calibration's shifts
    mask = cal["mask"]
    assert len(mask) == 14 * 3 and mask[0]["shift_key"] == "2026-10-01_A" and mask[-1]["shift_key"] == "2026-10-14_C"
    states = {m["shift_key"]: m["state"] for m in mask}
    assert [k for k, s in states.items() if s == "used"] == ref.shifts
    assert states[UNSTABLE] == "unstable" and list(states.values()).count("missing") == 42 - 4
    row = next(m for m in mask if m["shift_key"] == UNSTABLE)
    assert "blast_meter_pct" in row["failed"]
    assert rel(row["blast_meter_pct"], window_frames()[1].at[UNSTABLE, "blast_meter_vs_reconciled_pct"]) <= 1e-6
    assert len(out["tests"]) == 6 and out["jacobian_cache"]["status"] == "unknown"


def test_the_tests_carry_the_filter_bands() -> None:
    tests = console.stability_tests()
    assert [t["name"] for t in tests] == [
        "coke_kg_thm",
        "c_closure_pct",
        "blast_meter_pct",
        "cooling_mj_thm",
        "fe_closure_pct",
        "production_t_h",
    ]
    assert (tests[0]["lo"], tests[0]["hi"], tests[0]["unit"]) == (220.0, 400.0, "kg/tHM")
    assert (tests[-1]["lo"], tests[-1]["hi"], tests[-1]["unit"]) == (75.0, 110.0, "t/h")


def test_a_calibration_made_before_the_console_has_no_mask(tmp_path) -> None:
    nightly(tmp_path)
    path = day_calibration.path_for(DAY, tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["info"]["mask"], doc["settings"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    cal = console.view(directory=tmp_path, cache_dir=tmp_path, config={"min_shifts": 10}, today=DAY)["calibration"]
    assert cal["mask"] is None and cal["min_shifts"] == 10 and not cal["complete"]  # 3 shifts against today's 10
    assert cal["window_days"] == 14  # from the stored window
    assert day_calibration.load(DAY, tmp_path).mask == []


def test_the_newest_day_is_shown_and_a_day_not_stored_is_refused(tmp_path) -> None:
    info = CalibrationInfo(fallback_calibration(), None, "fallback", ("", ""), 0, "No balanced shifts")
    day_calibration.save(
        info, DAY - timedelta(days=3), made_at=datetime(2026, 10, 11, 21, tzinfo=UTC), directory=tmp_path
    )
    nightly(tmp_path)
    out = console.view(directory=tmp_path, cache_dir=tmp_path, config=CFG["calibration"], today=DAY + timedelta(days=1))
    assert out["day"] == DAY and not out["current"]
    assert [d["day"] for d in out["days"]] == [DAY, DAY - timedelta(days=3)]
    older = console.view(DAY - timedelta(days=3), directory=tmp_path, cache_dir=tmp_path, config={}, today=DAY)
    assert older["calibration"]["source"] == "fallback" and older["calibration"]["mask"] == []
    assert older["calibration"]["window_start"] is None and not older["calibration"]["complete"]
    with pytest.raises(console.DayNotFound):
        console.view(DAY - timedelta(days=1), directory=tmp_path, cache_dir=tmp_path, config={}, today=DAY)


def test_without_a_stored_calibration_the_view_is_empty(tmp_path) -> None:
    out = console.view(directory=tmp_path / "none", cache_dir=tmp_path / "none", config={}, today=DAY)
    assert out["day"] is None and out["calibration"] is None and out["days"] == []
    assert out["jacobian_cache"]["status"] == "none" and out["jacobian_cache"]["n_files"] == 0
    assert out["job"]["kind"] == "calibration.nightly" and out["job"]["schedule"] == "02:30"
    assert out["job"]["next_slot"] is not None and len(out["tests"]) == 6


# --------------------------------------------------------------------------------------------------- jacobians


def jacobian_file(folder, name: str, net: str = strata_network.NET_VERSION) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps({"inputs": [], "J": [[0.0]], "net": net}), encoding="utf-8")


def test_the_jacobian_cache_status_of_the_shown_calibration(tmp_path) -> None:
    cache = tmp_path / "strata_flow"
    jacobian_file(cache, "jacobian_aaaa000011112222.json")
    jacobian_file(cache, "jacobian_bbbb000011112222.json", net="STRATA-Flow v0.3")
    nightly(tmp_path / "cal", network={"jacobian": "cache jacobian_aaaa000011112222.json", "solves": 1})
    out = console.view(directory=tmp_path / "cal", cache_dir=cache, config=CFG["calibration"], today=DAY)
    jac = out["jacobian_cache"]
    assert (jac["status"], jac["day_file"], jac["n_files"], jac["stale_files"]) == (
        "cached",
        "jacobian_aaaa000011112222.json",
        2,
        1,
    )
    files = {f["name"]: f for f in jac["files"]}
    assert files["jacobian_aaaa000011112222.json"]["current"] and files["jacobian_aaaa000011112222.json"]["days"] == [
        DAY
    ]
    assert not files["jacobian_bbbb000011112222.json"]["current"]
    assert out["days"][0]["jacobian_cached"] is True and out["calibration"]["network"]["solves"] == 1
    # the file gone: the next STRATA-ω run solves the Jacobian again
    (cache / "jacobian_aaaa000011112222.json").unlink()
    out = console.view(directory=tmp_path / "cal", cache_dir=cache, config=CFG["calibration"], today=DAY)
    assert out["jacobian_cache"]["status"] == "missing" and out["days"][0]["jacobian_cached"] is False
    # the job's statistics name the file (SV-4.6) or only say "computed": unknown
    nightly(
        tmp_path / "cal", network={"jacobian": "computed (17 network solves, 19 s)", "cache_file": "jacobian_c.json"}
    )
    assert console.view(directory=tmp_path / "cal", cache_dir=cache, today=DAY)["jacobian_cache"]["day_file"] == (
        "jacobian_c.json"
    )
    nightly(tmp_path / "cal")
    out = console.view(directory=tmp_path / "cal", cache_dir=cache, today=DAY)
    assert out["jacobian_cache"]["status"] == "unknown" and out["days"][0]["jacobian_cached"] is None


def test_network_stats_name_the_cache_file(tmp_path) -> None:
    net = strata_network.NetworkParts({}, cache_dir=tmp_path)
    net.cache_file = "jacobian_0123456789abcdef.json"
    assert net.stats()["cache_file"] == "jacobian_0123456789abcdef.json"
    assert console.cache_file_of(net.stats()) == "jacobian_0123456789abcdef.json"
    assert console.cache_file_of({"jacobian": "cache jacobian_0123456789abcdef.json"}) == net.cache_file
    assert console.cache_file_of({"jacobian": "none (exact mode)"}) is None and console.cache_file_of(None) is None


# ------------------------------------------------------------------------------------------------------- runs


def test_runs_of_the_nightly_job() -> None:
    t0 = datetime(2026, 10, 14, 21, 0, tzinfo=UTC)
    scheduled = JobRecord(
        id="j1",
        kind="calibration.nightly",
        status="done",
        input_hash="h",
        submitted_by="scheduler",
        submitted_at=t0,
        started_at=t0,
        finished_at=t0 + timedelta(seconds=40),
        tries=1,
        result={"day": "2026-10-15"},
        scheduled_for=t0,
        run_s=40.0,
    )
    manual = JobRecord(
        id="j2",
        kind="calibration.nightly",
        status="failed",
        input_hash="h",
        submitted_by="adm",
        submitted_at=t0 + timedelta(hours=5),
        tries=3,
        error="no live calibration",
    )
    rows = console.run_rows([manual, scheduled])
    assert [(r["id"], r["scheduled"], r["day"]) for r in rows] == [("j2", False, None), ("j1", True, DAY)]
    assert rows[1]["run_s"] == 40.0 and rows[0]["error"] == "no live calibration"
