"""services.coke_energy (#271): one composition for the twin-api, the worker and the Streamlit page.

Sources are synthetic (``services_fakes``) and ``run_pipeline`` is a recording stand-in, so these tests check the
composition that moved out of ``ui/coke_energy/loaders.py``: what each source contributes, the provisional DPR
handling, the freshness inputs, caching and the equivalence of the Streamlit path and the service path.
"""

from __future__ import annotations

import pandas as pd
import pytest
from services_fakes import dpr_frame, patch_sources

import services.coke_energy as svc
import ui.coke_energy.loaders as st_loaders
from utils.coke_energy.pipeline import PipelineSettings

SETTINGS = PipelineSettings()


def _keys(nonce: int = 0) -> svc.DataKeys:
    return svc.DataKeys(
        hourly_path="furnace_dataset.csv",
        hourly_mtime_ns=1,
        clock="2026-10-26T21:30:00+05:30",
        nonce=nonce,
        history_start="2025-08-01",
        fields_json="{}",
        cache_dir="unused",
        online_refresh_minutes=30.0,
    )


@pytest.fixture()
def recorder(monkeypatch):
    rec = patch_sources(monkeypatch, svc)
    svc.clear()
    st_loaders.clear_caches()
    yield rec
    svc.clear()
    st_loaders.clear_caches()


def _assert_same(a: svc.PageData, b: svc.PageData) -> None:
    assert a.provisional == b.provisional
    assert a.names == b.names
    assert a.fresh == b.fresh
    pd.testing.assert_frame_equal(a.samples.reset_index(drop=True), b.samples.reset_index(drop=True))
    pd.testing.assert_frame_equal(a.charged, b.charged)
    pd.testing.assert_frame_equal(a.result.shifts, b.result.shifts)


def test_streamlit_path_equals_service_path(recorder) -> None:
    via_service = svc.page_data(_keys(), SETTINGS)
    via_streamlit = st_loaders.page_data(_keys(), SETTINGS)
    _assert_same(via_service, via_streamlit)
    assert len(recorder.calls) == 2  # one pipeline run per cache
    for k in ("fd", "m10", "hm", "dpr"):
        pd.testing.assert_frame_equal(recorder.calls[0][k], recorder.calls[1][k])


def test_provisional_dpr_dates_reach_the_pipeline_but_not_dpr_freshness(recorder) -> None:
    data = svc.page_data(_keys(), SETTINGS)
    last_real = dpr_frame().index.max()
    assert data.provisional, "the synthetic DPR lags two days, so provisional dates are expected"
    assert all(pd.Timestamp(d) > last_real for d in data.provisional)
    assert data.fresh.dpr_end == last_real
    assert recorder.calls[-1]["provisional"] == data.provisional
    assert len(recorder.calls[-1]["dpr"]) == len(dpr_frame()) + len(data.provisional)
    dpr_times = data.samples.loc[data.samples.source == "dpr", "t"]
    assert dpr_times.max() == last_real + pd.Timedelta(days=1)  # end of the last real DPR day, provisional excluded


def test_every_source_contributes_its_last_valid_times(recorder) -> None:
    data = svc.page_data(_keys(), SETTINGS)
    assert set(data.samples.source) == {
        "ore_chemistry",
        "fuel_chemistry",
        "sinter_chemistry",
        "hourly_dataset",
        "online_10min",
        "hm_slag",
        "dpr",
        "charge_data",
    }
    charge_times = data.samples.loc[data.samples.source == "charge_data", "t"]
    assert len(charge_times) == int((data.charged["charges"] > 0).sum())


def test_lab_failure_leaves_the_page_working(recorder, monkeypatch) -> None:
    monkeypatch.setattr(svc, "fetch_offline_samples", lambda start: (_ for _ in ()).throw(RuntimeError("no db")))
    data = svc.page_data(_keys(), SETTINGS)
    assert data.charged.empty
    assert "charge_data" not in set(data.samples.source)
    assert {"hourly_dataset", "online_10min", "hm_slag", "dpr"} <= set(data.samples.source)


def test_memoised_until_a_token_changes(recorder) -> None:
    svc.page_data(_keys(0), SETTINGS)
    svc.page_data(_keys(0), SETTINGS)
    assert len(recorder.calls) == 1
    svc.page_data(_keys(1), SETTINGS)
    assert len(recorder.calls) == 2
    svc.clear()
    svc.page_data(_keys(1), SETTINGS)
    assert len(recorder.calls) == 3


def test_build_page_data_passes_the_keys_to_each_loader() -> None:
    seen: dict[str, tuple] = {}
    empty_dpr = pd.DataFrame(columns=["hm_t"], index=pd.DatetimeIndex([], name="date"))

    def rec(name, value):
        def fn(*args):
            seen[name] = args
            return value

        return fn

    loaders = svc.Loaders(
        hourly=rec("hourly", pd.DataFrame()),
        online=rec("online", pd.DataFrame()),
        neon=rec("neon", (pd.DataFrame(), empty_dpr, [])),
        lab=rec("lab", (pd.DataFrame(columns=["source", "material", "t"]), pd.DataFrame())),
        names=rec("names", {}),
    )
    keys = _keys(3)
    import utils.coke_energy.pipeline as pl

    result = pl.PipelineResult(*(pd.DataFrame(),) * 7, settings=SETTINGS)
    original = svc.run_pipeline
    svc.run_pipeline = lambda *a, **k: result
    try:
        data = svc.build_page_data(keys, SETTINGS, loaders)
    finally:
        svc.run_pipeline = original
    assert data.result is result
    assert seen["hourly"] == (keys.hourly_path, keys.hourly_mtime_ns)
    assert seen["online"] == (
        keys.history_start,
        keys.clock,
        keys.nonce,
        keys.fields_json,
        keys.cache_dir,
        keys.online_refresh_minutes,
    )
    assert seen["neon"] == seen["lab"] == (keys.history_start, keys.clock, keys.nonce)
    assert seen["names"] == ()


def test_streamlit_loader_keeps_the_page_interface() -> None:
    for name in ("clear_caches", "data_keys", "page_data", "rist_frames", "DataKeys", "PageData"):
        assert hasattr(st_loaders, name), name
    assert st_loaders.data_keys is svc.data_keys
