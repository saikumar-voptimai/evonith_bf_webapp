"""services.twin (SV-1.1, #281): the window evaluation through the Coke Energy service and its snapshot row."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime

import pandas as pd
import pytest

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import coke_energy as ce
from services import twin
from tests.twin_data import synthetic_frames
from utils.coke_energy.twin import Window, evaluate_window

CFG = {"history_start": "2026-04-01", "predict_from": "2026-04-01", "rolling_window_days": 14, "rolling_min_shifts": 10}
KEYS = ce.DataKeys(
    hourly_path="synthetic.csv",
    hourly_mtime_ns=1,
    clock="2026-05-15T10:00:00+05:30",
    nonce=0,
    history_start="2026-04-01",
    fields_json="{}",
    cache_dir="",
    online_refresh_minutes=30.0,
)


@pytest.fixture(scope="module")
def sources():
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=18)
    provisional = [pd.Timestamp("2026-05-18").date()]
    loaders = ce.Loaders(
        hourly=lambda path, mtime: fd,
        online=lambda start, clock, nonce, fields, cache, minutes: m10,
        neon=lambda start, clock, nonce: (hm, dpr, provisional),
        lab=lambda start, clock, nonce: (pd.DataFrame(columns=["source", "material", "t"]), pd.DataFrame()),
        names=dict,
    )
    calls = []

    def page_data_for(keys, settings):
        calls.append(keys)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ce.build_page_data(keys, settings, loaders)

    return (fd, m10, hm, dpr, provisional), loaders, page_data_for, calls


def test_evaluate_runs_the_window_ending_at_the_full_hour(sources) -> None:
    (fd, m10, hm, dpr, provisional), loaders, page_data_for, calls = sources
    ev = twin.evaluate("2026-05-15 10:20", cfg=CFG, keys=KEYS, loaders=loaders, page_data_for=page_data_for)
    assert ev.window == Window(pd.Timestamp("2026-05-15 02:00"), pd.Timestamp("2026-05-15 10:00"))
    assert calls[-1] == KEYS
    pool = page_data_for(KEYS, ce.PipelineSettings.from_config(CFG)).result
    direct = evaluate_window(fd, m10, hm, dpr, ev.window, pool, ce.PipelineSettings.from_config(CFG), provisional)
    assert (ev.status, ev.k_trace, ev.k_forge, ev.omega_imp) == (
        direct.status,
        direct.k_trace,
        direct.k_forge,
        direct.omega_imp,
    )
    assert ev.status == "complete"


def test_the_provisional_dpr_of_the_loaders_is_flagged(sources) -> None:
    _, loaders, page_data_for, _ = sources
    ev = twin.evaluate("2026-05-18 14:00", cfg=CFG, keys=KEYS, loaders=loaders, page_data_for=page_data_for)
    assert ev.flags["provisional_dpr"] is True


def test_snapshot_values_are_in_utc_and_round_trip_through_the_twin_store(sources, tmp_path) -> None:
    _, loaders, page_data_for, _ = sources
    ev = twin.evaluate("2026-05-15 10:00", cfg=CFG, keys=KEYS, loaders=loaders, page_data_for=page_data_for)
    at = datetime(2026, 5, 15, 4, 35, tzinfo=UTC)
    values = twin.snapshot_values(ev, evaluated_at=at, versions={"pipeline_window_days": 14})
    assert values["window_start"] == datetime(2026, 5, 14, 20, 30, tzinfo=UTC)  # 02:00 IST
    assert values["window_end"] == datetime(2026, 5, 15, 4, 30, tzinfo=UTC)  # 10:00 IST
    assert values["k_strata"] is None and values["omega_strata"] is None
    assert values["flags"]["notes"] == [] and values["versions"]["model"] == twin.MODEL

    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store.snapshots.upsert(values)
    row = store.snapshots.get(values["window_end"])
    assert row.status == "complete"
    assert row.k_trace == pytest.approx(ev.k_trace) and row.omega_imp == pytest.approx(ev.omega_imp)
    assert row.quantities["hmt_c"] == pytest.approx(ev.quantities["hmt_c"])
    assert row.flags["n_hours"] == 8 and row.versions["pipeline_window_days"] == 14
    engine.dispose()


def test_a_failed_window_still_gives_a_snapshot_row(sources) -> None:
    _, loaders, page_data_for, _ = sources
    ev = twin.evaluate("2026-07-01 10:00", cfg=CFG, keys=KEYS, loaders=loaders, page_data_for=page_data_for)
    values = twin.snapshot_values(ev, evaluated_at=datetime(2026, 7, 1, 4, 35, tzinfo=UTC))
    assert values["status"] == "failed" and values["k_trace"] is None
    assert values["flags"]["notes"] == ["no hourly data in the window"]
