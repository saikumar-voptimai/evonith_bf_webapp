"""Retrain, export, accept: the 1-5 hour path model end to end (small settings)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.bmo.charged_coke import deployments
from utils.bmo.charged_coke.ledger import ForecastLedger, PlannedChangeRegister
from utils.bmo.charged_coke.physics import load_physics_config
from utils.bmo.charged_coke.policy import PolicyConfig
from utils.bmo.charged_coke.service import ChargedCokeEngine, forecast_hours, issue_latest, recent_forecasts
from utils.bmo.charged_coke.training import TrainingSettings, train, write_bundle

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "charged_coke"
REFERENCE = ROOT / "src/assets/models/bmo_charged_coke/20260915"
END = pd.Timestamp("2026-10-04 23:00")
# Tiny on purpose; the evidence minimum is lowered so its few calibration days
# still let hours be shown.
SMALL = TrainingSettings(train_start="2026-09-05", folds=2, fold_days=3, seeds=(7,), attention_epochs=4,
                         minimum_train_rows=50, minimum_calibration_days=2, minimum_calibration_hours=10)


@pytest.fixture(scope="module")
def source() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "source_window.csv.gz", parse_dates=["time"]).set_index("time")


@pytest.fixture(scope="module")
def trained(source, tmp_path_factory):
    result = train(source, load_physics_config(REFERENCE), PolicyConfig.load(REFERENCE), end=END, settings=SMALL)
    folder = write_bundle(result, tmp_path_factory.mktemp("bundle") / result.version, base_bundle=REFERENCE)
    return result, folder


def test_the_model_forecasts_five_hourly_values(trained) -> None:
    result, folder = trained
    engine = ChargedCokeEngine.load(folder)

    assert engine.model.horizons == [1, 2, 3, 4, 5]
    assert engine.model.beta_by_horizon.shape == (5, 3)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["trained_until"] == str(END - pd.Timedelta(days=3))
    # No pickle anywhere in a deployable bundle.
    assert not list(folder.glob("*.pkl"))


def test_exported_model_reproduces_the_training_code_exactly(trained, source) -> None:
    result, folder = trained
    engine = ChargedCokeEngine.load(folder)
    hold = result.holdout

    replay = forecast_hours(source, engine, start=hold.index.min(), end=hold.index.max())

    common = hold.index.intersection(replay.index)
    assert len(common) == len(hold)
    for h in range(1, 6):
        np.testing.assert_allclose(replay.loc[common, f"prediction_h{h}"], hold.loc[common, f"prediction_h{h}"], atol=1e-9)
    # Same states as the review saw: novelty is measured the live way.
    assert (replay.loc[common, "state"] == hold.loc[common, "state"]).all()


def test_holdout_is_never_seen_by_the_deployed_model(trained) -> None:
    result, _ = trained

    assert result.model.train_rows.max() < result.holdout.index.min() - pd.Timedelta(hours=4)
    report = result.report
    assert [r["horizon"] for r in report["per_horizon"]] == [1, 2, 3, 4, 5]
    assert set(report["checks"]) >= {"Beats persistence at every horizon",
                                     "PCI and nut coke lower coke, slag raises it (every horizon)"}
    for b in report["beta"]:
        assert b["pci"] <= 0 and b["nut"] <= 0 and b["slag"] >= 0


def test_ranges_are_calibrated_per_condition_and_horizon(trained) -> None:
    result, _ = trained
    bins = result.parameters["bins"]

    for severity in ("cruise", "mild", "moderate"):
        assert set(bins[severity]["radius_by_horizon"]) == {"1", "2", "3", "4", "5"}
    for h in "12345":  # a less settled furnace never gets a narrower range
        assert bins["cruise"]["radius_by_horizon"][h] <= bins["mild"]["radius_by_horizon"][h] <= bins["moderate"]["radius_by_horizon"][h]


def test_seven_day_chart_keeps_only_out_of_sample_hours(trained, source) -> None:
    _, folder = trained
    engine = ChargedCokeEngine.load(folder)

    recent = recent_forecasts(source, engine, end=END, days=7)

    trained_until = pd.Timestamp(engine.model.manifest["trained_until"])
    assert recent["issue_row"].min() >= trained_until
    assert set(recent["horizon"]) == {1, 2, 3, 4, 5}
    assert {"prediction", "actual", "state", "shown", "anchor", "at"} <= set(recent.columns)
    one = recent[recent["horizon"] == 3].iloc[0]
    assert one["at"] == one["issue_row"] + pd.Timedelta(hours=3)


def test_issued_record_carries_the_five_hour_path(trained, source, tmp_path) -> None:
    _, folder = trained
    engine = ChargedCokeEngine.load(folder)
    row = pd.Timestamp("2026-10-03 10:00")
    published, built = source.loc[: row + pd.Timedelta(hours=1)], row + pd.Timedelta(hours=1, minutes=6)

    record = issue_latest(published, engine, ledger=ForecastLedger(tmp_path), plans=PlannedChangeRegister(tmp_path),
                          now=built, built_at=built)

    path = record["path"]
    assert [p["horizon"] for p in path] == [1, 2, 3, 4, 5]
    assert pd.Timestamp(path[0]["at"]) == row + pd.Timedelta(hours=1)
    assert pd.Timestamp(path[4]["coke_window"][0]) == row + pd.Timedelta(hours=1)
    if record["shown"]:
        assert record["prediction_kg_thm"] == pytest.approx(path[4]["prediction_kg_thm"])


def test_accept_and_roll_back_are_recorded(trained, tmp_path) -> None:
    _, folder = trained
    import shutil

    for version in ("20261001_0000", "20261002_0000"):
        shutil.copytree(folder, deployments.models_dir(tmp_path) / version)
    default = Path("bundled")

    assert deployments.active_bundle(tmp_path, default) == default
    with pytest.raises(ValueError, match="name"):
        deployments.activate(tmp_path, "20261002_0000", accepted_by=" ", now=datetime(2026, 10, 7, 9))
    deployments.activate(tmp_path, "20261002_0000", accepted_by="R. Supervisor", now=datetime(2026, 10, 7, 9))
    deployments.activate(tmp_path, "20261001_0000", accepted_by="A. Admin", now=datetime(2026, 10, 7, 10), note="roll back")

    assert deployments.active_bundle(tmp_path, default).name == "20261001_0000"
    audit = [json.loads(line) for line in (deployments.models_dir(tmp_path) / "activations.jsonl").read_text().splitlines()]
    assert [(a["version"], a["accepted_by"], a["previous"]) for a in audit] == [
        ("20261002_0000", "R. Supervisor", None), ("20261001_0000", "A. Admin", "20261002_0000")]
    assert [v["version"] for v in deployments.versions(tmp_path)] == ["20261002_0000", "20261001_0000"]


def test_trust_view_colours_by_condition_and_leaves_out_paused_hours(trained, source) -> None:
    from ui.bmo.charged_coke import (
        STATE_COLORS,
        STATE_LABELS,
        trust_figure,
        trust_summary,
    )

    _, folder = trained
    recent = recent_forecasts(source, ChargedCokeEngine.load(folder), end=END, days=7)

    figure = trust_figure(recent, 5)
    names = [trace.name for trace in figure.data]
    assert names[0] == "Measured (4-h charged)"
    shown_states = set(recent.loc[(recent["horizon"] == 5) & recent["shown"], "state"])
    assert {n.split("· ")[-1] for n in names[1:]} == {
        STATE_LABELS[state] for state in shown_states
    }
    plotted = sum(len(trace.x) for trace in figure.data[1:])
    assert plotted == int(((recent["horizon"] == 5) & recent["shown"]).sum())
    assert set(STATE_COLORS) == {"cruise", "unsettled", "off_cruise"}
    assert plotted > 0
    summary = trust_summary(recent, 5)
    assert summary.iloc[-1]["Condition"] == "All shown"
