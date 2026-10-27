"""HMT forecast one descent time ahead and its gate (SV-1.4, #284): ``utils.coke_energy.hmt_forecast``, the forecast
quantities of the window evaluation and ``scripts/hmt_gate.py`` on the synthetic sources."""

from __future__ import annotations

import importlib.util
import json
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pandas as pd
import pytest

from services import coke_energy as ce
from services import twin
from tests.twin_data import synthetic_frames
from utils.coke_energy import hmt_forecast as hf
from utils.coke_energy.pipeline import PipelineSettings, run_pipeline
from utils.coke_energy.twin import Window, evaluate_window

ROOT = Path(__file__).resolve().parents[1]
CFG = {"history_start": "2026-04-01", "predict_from": "2026-04-01", "rolling_window_days": 14, "rolling_min_shifts": 10}


def test_the_forecast_formula() -> None:
    assert hf.forecast(1480.0, 310.0, 300.0, 0.5) == pytest.approx(1500.0)  # 10 kg/tHM excess at 0.5 kg per degC
    assert hf.forecast(1480.0, 290.0, 300.0, 0.5) == pytest.approx(1460.0)
    assert hf.forecast(1480.0, 310.0, 300.0, 0.0) is None and hf.forecast(np.nan, 310.0, 300.0, 0.5) is None


def test_taps_latest_and_target() -> None:
    hm = pd.DataFrame(
        {
            "t": pd.to_datetime(["2026-05-01 01:00", "2026-05-01 03:00", "2026-05-01 05:00", "2026-05-01 06:00"]),
            "hmt_gt_1480c": [1470.0, 1490.0, 1300.0, 1500.0],
        }
    )
    temps = hf.taps(hm)
    assert list(temps) == [1470.0, 1490.0, 1500.0]  # below 1350 degC dropped, as the pipeline's tap filter
    assert hf.latest(temps, "2026-05-01 04:00") == (1490.0, 1.0)
    assert hf.latest(temps, "2026-05-01 03:00") == (1470.0, 2.0) and hf.latest(temps, "2026-05-01 10:00") is None
    assert hf.target(temps, "2026-05-01 02:00") == pytest.approx(1480.0)  # between the taps of 01:00 and 03:00
    assert hf.target(temps, "2026-05-01 08:00") is None  # 2 h after the last tap: none within 1 h
    assert hf.target(temps, "2026-05-01 06:30") == pytest.approx(1500.0)


def test_k_arr_uses_the_coke_to_ferrous_ratio_of_the_last_hour() -> None:
    t = pd.date_range("2026-05-01 00:00", periods=3, freq="h")
    rec = pd.DataFrame(
        {
            "COKE_CALC_MT": [27.0, 27.3, 31.5],
            "COKE_MOIST%": 1.0,
            "SINTER_CALC_MT": 96.0,
            "ORE_CALC_MT": 45.0,
            "TOTAL_PELLET_CALC_MT": 15.0,
        },
        index=t,
    )
    assert hf.k_arr(rec, "2026-05-01 02:00", 1700.0) == pytest.approx(27.3 * 0.99 / 156.0 * 1700.0)
    assert hf.k_arr(rec, "2026-05-01 05:00", 1700.0) is None  # no record of the last hour


def _rows(model_err, persistence_err) -> pd.DataFrame:
    target = np.full(len(model_err), 1480.0)
    return pd.DataFrame({"hmt_now": target - persistence_err, "hmt_forecast": target - model_err, "hmt_target": target})


def test_the_gate_against_persistence() -> None:
    p = np.tile([10.0, -10.0], 20)  # persistence RMSE 10 degC, no bias
    passed = hf.gate(_rows(np.tile([5.0, -5.0], 20), p))
    assert passed.passed and passed.ratio == pytest.approx(0.5) and passed.rmse_persistence == pytest.approx(10.0)
    close = hf.gate(_rows(np.tile([9.5, -9.5], 20), p))
    assert not close.passed and close.ratio == pytest.approx(0.95)  # only 5 % below persistence
    biased = hf.gate(_rows(np.full(40, 6.0), p))
    assert not biased.passed and biased.bias_model == pytest.approx(6.0) and "bias +6.0" in biased.reason
    few = hf.gate(_rows(np.zeros(5), np.ones(5)))
    assert not few.passed and few.n == 5 and "need 10" in few.reason


def test_the_register_fact_of_a_gate_run_follows_the_schema() -> None:
    schema = json.loads((ROOT / "src/config/schemas/facts_register.schema.json").read_text(encoding="utf-8"))
    fact_schema = schema["properties"]["facts"]["items"]
    p = np.tile([10.0, -10.0], 20)
    ok = hf.register_fact(
        hf.gate(_rows(np.tile([5.0, -5.0], 20), p)),
        "COKE-18",
        "2026 expanding test",
        "Jan-Oct 2026",
        hf.GateCriteria(),
        "2026-10-27",
        "1.9",
    )
    bad = hf.register_fact(
        hf.gate(_rows(np.full(40, 6.0), p)),
        "COKE-19",
        "2023-24 test",
        "2023-24",
        hf.GateCriteria(),
        "2026-10-27",
        "1.9",
    )
    for fact in (ok, bad):
        jsonschema.validate(fact, fact_schema)
    assert (ok["status"], bad["status"]) == ("Validated", "Disputed") and ok["value"] == pytest.approx(5.0)
    assert "measured HMT only" in bad["note"]


# ---------------------------------------------------------------------------------------------- window and service
@pytest.fixture(scope="module")
def world():
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=18)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_pipeline(fd, m10, hm, dpr, PipelineSettings.from_config(CFG))
    return twin.TwinFrames(fd, m10, hm, dpr, ()), result


def test_every_window_carries_the_forecast_quantities(world) -> None:
    frames, result = world
    ev = evaluate_window(
        frames.fd,
        frames.m10,
        frames.hm,
        frames.dpr,
        Window.ending("2026-05-16 14:00"),
        result,
        PipelineSettings.from_config(CFG),
        strata=False,
    )
    q = ev.quantities
    assert 0.05 < q["dk_dhmt_kg_thm_per_c"] < 0.3  # sensible heat of metal and slag in the balance
    assert q["hmt_forecast_horizon_h"] == pytest.approx(q["tau_raceway_h"]) and q["hmt_now_age_h"] <= 3.0
    excess = q["k_arr_kg_thm"] - q["k_forge_hmt_now_kg_thm"]
    assert q["hmt_forecast_c"] == pytest.approx(q["hmt_now_c"] + excess / q["dk_dhmt_kg_thm_per_c"])


def test_the_gate_status_comes_from_the_configuration() -> None:
    assert twin.hmt_gate_status({}) == "pending"
    assert twin.hmt_gate_status({"twin": {"hmt_forecast": {"gate": {"status": "passed"}}}}) == "passed"
    assert twin.hmt_gate_status({"twin": {"hmt_forecast": {"gate": {"status": "maybe"}}}}) == "pending"


def test_the_gate_script_runs_and_records_the_result(world, monkeypatch, tmp_path, capsys) -> None:
    frames, result = world
    monkeypatch.setattr(ce, "config", lambda: CFG)
    monkeypatch.setattr(ce, "data_keys", lambda cfg, nonce: "keys")
    monkeypatch.setattr(ce, "page_data", lambda keys, settings: SimpleNamespace(result=result))
    monkeypatch.setattr(twin, "load_frames", lambda keys: frames)
    spec = importlib.util.spec_from_file_location("hmt_gate", ROOT / "scripts" / "hmt_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    register = tmp_path / "facts_register.json"
    shutil.copy(ROOT / "src/config/facts_register.json", register)
    before = json.loads(register.read_text(encoding="utf-8"))["version"]
    out = tmp_path / "forecasts.csv"
    code = gate.main(
        ["--from", "2026-05-15", "--to", "2026-05-17", "--out", str(out), "--write", "--register", str(register)]
    )
    text = capsys.readouterr().out
    assert code in (0, 2) and "48 hourly windows 2026-05-15 to 2026-05-17" in text and "gate (2026 expanding" in text
    rows = pd.read_csv(out)
    assert len(rows) == 48 and rows.hmt_forecast.notna().sum() > 30
    reg = json.loads(register.read_text(encoding="utf-8"))
    fact = next(f for f in reg["facts"] if f["id"] == "COKE-18")
    major, minor = before.split(".")
    assert reg["version"] == f"{major}.{int(minor) + 1}" and fact["version"] == reg["version"]
    assert fact["status"] in ("Validated", "Disputed") and reg["revision_log"][0]["change"].startswith("SV-1.4")
