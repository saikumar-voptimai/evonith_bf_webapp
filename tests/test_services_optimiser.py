"""services.optimiser (#271): yard, calibration and run dispatch without Streamlit."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import services.optimiser as svc
import ui.burden_optimiser.loaders as st_loaders

CFG = {
    "prices": {"materials": {}, "fluxes": {}, "fuel": {}, "slag_price_rs_per_t": 457.85},
    "calibration": {"window_days": 14},
}


@pytest.fixture(autouse=True)
def _clean():
    svc.clear()
    st_loaders.yard.clear()
    st_loaders.run_mode.clear()
    yield
    svc.clear()


def test_reference_yard_applies_the_page_price_basis() -> None:
    base = svc.reference_yard(None)
    m0 = base["materials"][0]
    priced = svc.reference_yard({"prices": {"materials": {m0["id"]: 12345.0}, "slag_price_rs_per_t": 457.85}})
    assert priced["materials"][0]["price"] == 12345.0
    assert priced["fuel"]["slag_price"] == 457.85
    assert all(m["chem_source"] == "reference 16 Oct 2026" for m in priced["materials"])


def test_yard_falls_back_to_the_reference_when_the_database_fails(monkeypatch) -> None:
    monkeypatch.setattr(svc, "load_yard", lambda cfg: (_ for _ in ()).throw(ConnectionError("no neon")))
    y, source, message = svc.read_yard(CFG)
    assert source == "reference"
    assert "ConnectionError: no neon" in message
    assert y == svc.reference_yard(CFG)


def test_yard_without_lab_chemistry_is_not_live(monkeypatch) -> None:
    monkeypatch.setattr(svc, "load_yard", lambda cfg: {"materials": [{"id": "ore_1", "carried": False}]})
    assert svc.read_yard(CFG)[1] == "reference"


def test_memoised_yard_and_streamlit_yard_agree(monkeypatch) -> None:
    live = {"materials": [{"id": "ore_1", "carried": True}], "fluxes": []}
    calls = []
    monkeypatch.setattr(svc, "load_yard", lambda cfg: calls.append(1) or live)
    a = svc.yard(json.dumps(CFG), 0)
    b = svc.yard(json.dumps(CFG), 0)
    c = st_loaders.yard(json.dumps(CFG), 0)
    assert a == b == c == (live, "live", "")
    assert len(calls) == 2  # one load per cache (service memo, Streamlit cache)


def test_calibration_uses_the_given_page_data_and_falls_back(monkeypatch) -> None:
    data = SimpleNamespace(result=object())
    monkeypatch.setattr(svc.ce.sources, "static_dataset_status", lambda ds: {"path": "x.csv", "mtime_ns": 1})
    monkeypatch.setattr(svc, "live_calibration", lambda result, cfg: ("live", result))
    info, got = svc.calibration(CFG, 0, page_data_for=lambda keys, settings: data)
    assert got is data and info == ("live", data.result)

    def down(keys, settings):
        raise RuntimeError("influx down")

    info, got = svc.calibration(CFG, 0, page_data_for=down)
    assert got is None
    assert info.source == "fallback"
    assert info.message.startswith("Plant data unavailable (RuntimeError: influx down)")
    assert info.cal  # the shipped Jul-Oct 2026 calibration


def test_live_calibration_is_reused_for_the_same_result(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(svc, "live_calibration", lambda result, cfg: calls.append(1) or "info")
    data = SimpleNamespace(result=object())
    assert svc.live_calibration_cached(CFG, data) == svc.live_calibration_cached(CFG, data) == "info"
    assert len(calls) == 1
    svc.live_calibration_cached(CFG, SimpleNamespace(result=object()))
    assert len(calls) == 2


def _fake_scenarios(monkeypatch, record: list) -> None:
    class FakeEngine:
        def __init__(self, inputs, cal):
            self.inputs, self.cal = inputs, cal

    def fn(name):
        def run(*args, **kw):
            record.append(name)
            out = {"fn": name}
            if name == "run_fixed":
                out["with_fuel"] = {"total": 1.0}
            if name == "run_cheapest_level":
                out["best"] = {"settings": {}, "limits": {}, "P_tday": 2300.0}
            if name == "run_max":
                out["result"] = {"settings": {}, "limits": {}, "P_tday": 2400.0}
            return out

        return run

    monkeypatch.setattr(svc, "Engine", FakeEngine)
    monkeypatch.setattr(svc, "Settings", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(svc, "Limits", lambda **kw: SimpleNamespace(**kw))
    for name in ("run_fixed", "value_of_limits", "run_today", "run_candidates", "run_cheapest_level", "run_max"):
        monkeypatch.setattr(svc, name, fn(name))


ARGS = dict(
    inputs_json="{}",
    cal_json="{}",
    settings={},
    limits={},
    prod_t_day=2270.0,
    search_o2=True,
    n_o2=9,
    P_lo=2100.0,
    step=20.0,
)


@pytest.mark.parametrize(
    "mode,today,auto,expected",
    [
        ("fixed", '{"ore_1": 0.5}', True, ["run_fixed", "value_of_limits", "run_today", "run_candidates"]),
        ("fixed", "{}", False, ["run_fixed", "value_of_limits"]),
        ("cheapest", "{}", True, ["run_cheapest_level", "run_candidates"]),
        ("max", "{}", False, ["run_max"]),
    ],
)
def test_compute_mode_dispatch(monkeypatch, mode, today, auto, expected) -> None:
    record: list = []
    _fake_scenarios(monkeypatch, record)
    out = svc.compute_mode(mode, today_json=today, auto=auto, **ARGS)
    assert record == expected
    if mode == "fixed":
        assert out["today"] is None if today == "{}" else out["today"] == {"fn": "run_today"}


def test_run_mode_is_memoised_on_its_inputs(monkeypatch) -> None:
    record: list = []
    _fake_scenarios(monkeypatch, record)
    svc.run_mode("max", **ARGS)
    svc.run_mode("max", **ARGS)
    assert record == ["run_max"]
    svc.run_mode("max", **{**ARGS, "prod_t_day": 2300.0})
    assert record == ["run_max", "run_max"]


def test_engine_keeps_only_the_latest(monkeypatch) -> None:
    built = []

    class FakeEngine:
        def __init__(self, inputs, cal):
            built.append(inputs)

    monkeypatch.setattr(svc, "Engine", FakeEngine)
    a = svc.engine({"x": 1}, {"c": 1})
    assert svc.engine({"x": 1}, {"c": 1}) is a
    svc.engine({"x": 2}, {"c": 1})
    assert svc.engine({"x": 1}, {"c": 1}) is not a
    assert len(built) == 3


def test_streamlit_loader_keeps_the_page_interface() -> None:
    for name in (
        "page_config",
        "coke_energy_config",
        "reference_yard",
        "yard",
        "calibration",
        "engine",
        "charged_blend",
        "run_mode",
        "as_json",
        "clear",
    ):
        assert hasattr(st_loaders, name), name
