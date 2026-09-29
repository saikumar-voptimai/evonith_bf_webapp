from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest

from utils.bmo import direct_coke_model as module
from utils.bmo.coke_model_pipeline import pipeline
from utils.bmo.direct_coke_model import (
    DEFAULT_BUNDLE_DIR,
    DirectCokeModelService,
    RetrainReport,
    _apply_current_fuel_overrides,
    auto_retrain_due,
    maybe_retrain_in_background,
    retrain_and_maybe_deploy,
    retrain_kwargs_from_config,
)
from utils.bmo.robust_coke_target import NUT_COKE_FEATURE, PCI_FEATURE


def test_bundled_model_is_the_validated_robust_target_model():
    schema = json.loads(
        (DEFAULT_BUNDLE_DIR / "coke_context_schema.json").read_text(encoding="utf-8")
    )
    config = json.loads((DEFAULT_BUNDLE_DIR / "config.json").read_text(encoding="utf-8"))
    features = schema["features"]

    assert schema["task"] == "coke_robust_window"
    assert config["robust_target"]["window_hours"] == 24
    assert features[:2] == [PCI_FEATURE, NUT_COKE_FEATURE]
    assert schema["monotone_constraints"] == {PCI_FEATURE: -1, NUT_COKE_FEATURE: -1}
    # No coke mass, coke rate, fuel rate, or per-THM burden rate can leak in.
    assert not any("COKE_CALC" in name for name in features)
    assert not any(name.startswith(("COKE RATE", "ACT. FUEL")) for name in features)
    assert not any(name.endswith("_KG_THM__win") and "ORE" in name for name in features)
    assert schema["slag_coke_basis_kg_thm"] == 300
    validation = schema["validation"]
    assert validation["random_day"]["r2"] >= 0.70
    assert validation["later_time"]["mae"] <= 10.0
    assert validation["later_time"]["skill_vs_last_week_level"] >= 0.0
    assert 0.2 <= validation["fuel_response"]["pci_replacement_kg_per_kg"] <= 1.5
    assert validation["fuel_response"]["coke_change_for_nut_coke_minus_10"] > 0.0


def test_fuel_override_changes_only_latest_mass_row():
    raw = pd.DataFrame(
        {
            "time": ["2026-09-25 09:00:00", "2026-09-25 10:00:00"],
            "PRODUCTIONTONNESPERHR": [100.0, 80.0],
            "PCI_CALC_MT": [15.0, 14.0],
            "NUTCOKE_CALC_MT": [7.0, 6.0],
        }
    )

    changed = _apply_current_fuel_overrides(
        raw,
        {"time_col": "time"},
        pci_kg_per_thm=195.0,
        nut_coke_kg_per_thm=70.0,
    )

    assert changed.loc[0, "PCI_CALC_MT"] == 15.0
    assert changed.loc[0, "NUTCOKE_CALC_MT"] == 7.0
    assert changed.loc[1, "PCI_CALC_MT"] == pytest.approx(15.6)
    assert changed.loc[1, "NUTCOKE_CALC_MT"] == pytest.approx(5.6)


def test_direct_model_accepts_iso_and_published_day_first_timestamps():
    values = pd.Series(["2026-09-24 14:00", "25-09-2026 15:00"])

    parsed = pipeline.utc(values, "Asia/Kolkata")

    assert parsed.notna().all()
    assert parsed.iloc[0] == pd.Timestamp("2026-09-24 08:30:00Z")
    assert parsed.iloc[1] == pd.Timestamp("2026-09-25 09:30:00Z")


def test_fuel_override_finds_latest_day_first_timestamp():
    raw = pd.DataFrame(
        {
            "time": ["13-09-2026 09:00", "25-09-2026 10:00"],
            "PRODUCTIONTONNESPERHR": [100.0, 80.0],
            "PCI_CALC_MT": [15.0, 14.0],
            "NUTCOKE_CALC_MT": [7.0, 6.0],
        }
    )

    changed = _apply_current_fuel_overrides(
        raw,
        {"time_col": "time"},
        pci_kg_per_thm=195.0,
        nut_coke_kg_per_thm=70.0,
    )

    assert changed.loc[0, "PCI_CALC_MT"] == 15.0
    assert changed.loc[1, "PCI_CALC_MT"] == pytest.approx(15.6)
    assert changed.loc[1, "NUTCOKE_CALC_MT"] == pytest.approx(5.6)


def test_fuel_override_skips_trailing_zero_burden_rows():
    raw = pd.DataFrame(
        {
            "time": [
                "2026-09-25 18:00:00",
                "2026-09-25 19:00:00",
                "2026-09-25 20:00:00",
            ],
            "PRODUCTIONTONNESPERHR": [90.0, 100.0, 100.0],
            "ORE_CALC_MT": [30.0, 35.0, 0.0],
            "SINTER_CALC_MT": [90.0, 100.0, 0.0],
            "TOTAL_PELLET_CALC_MT": [20.0, 25.0, 0.0],
            "PCI_CALC_MT": [15.0, 16.0, 17.0],
            "NUTCOKE_CALC_MT": [6.0, 7.0, 8.0],
        }
    )

    changed = _apply_current_fuel_overrides(
        raw,
        {"time_col": "time"},
        pci_kg_per_thm=195.0,
        nut_coke_kg_per_thm=70.0,
    )

    assert changed.loc[1, "PCI_CALC_MT"] == pytest.approx(19.5)
    assert changed.loc[1, "NUTCOKE_CALC_MT"] == pytest.approx(7.0)
    assert changed.loc[2, "PCI_CALC_MT"] == 17.0
    assert changed.loc[2, "NUTCOKE_CALC_MT"] == 8.0


def test_fuel_override_covers_the_selected_lookback_window():
    raw = pd.DataFrame(
        {
            "time": pd.date_range("2026-09-25 12:00:00", periods=4, freq="h"),
            "PRODUCTIONTONNESPERHR": [100.0] * 4,
            "ORE_CALC_MT": [30.0] * 4,
            "PCI_CALC_MT": [10.0] * 4,
            "NUTCOKE_CALC_MT": [5.0] * 4,
        }
    )

    changed = _apply_current_fuel_overrides(
        raw,
        {"time_col": "time"},
        pci_kg_per_thm=195.0,
        nut_coke_kg_per_thm=70.0,
        lookback_hours=2,
    )

    assert changed["PCI_CALC_MT"].tolist() == [10.0, 10.0, 19.5, 19.5]
    assert changed["NUTCOKE_CALC_MT"].tolist() == [5.0, 5.0, 7.0, 7.0]


class _WindowFakeModel:
    def predict(self, matrix):
        values = np.array([290.0, 310.0, 300.0, 350.0, 295.0, 305.0])
        return values[: matrix.num_row()]


def _generic_service(tmp_path) -> DirectCokeModelService:
    """Service with the window-coverage gate off, for aggregation mechanics.

    The toy frames below carry no burden chemistry, so the robust window gate
    (tested on its own further down) would reject every row.
    """

    service = DirectCokeModelService(deployment_dir=tmp_path / "deploy")
    service.target_settings = None
    return service


def test_inference_uses_median_of_eligible_predictions_in_lookback(
    monkeypatch, tmp_path
):
    service = _generic_service(tmp_path)
    service.model = _WindowFakeModel()
    index = pd.date_range("2026-09-25 03:30:00Z", periods=8, freq="h")
    cleaned = pd.DataFrame(
        {
            "ORE_CALC_MT": [10.0] * 8,
            "SINTER_CALC_MT": [50.0] * 8,
            "TOTAL_PELLET_CALC_MT": [10.0] * 8,
            "PRODUCTIONTONNESPERHR": [90.0] * 8,
            "COKE_CALC_MT": [27.0] * 8,
        },
        index=index,
    )
    audit = pd.DataFrame({"normal_eligible": True}, index=index)
    features = pd.DataFrame(
        0.0, index=index, columns=service.schema["features"], dtype=float
    )
    ranged_feature = service.schema["features"][0]
    service.schema["feature_limits"] = {ranged_feature: [-1.0, 1.0]}
    features.loc[index[-1], ranged_feature] = 2.0

    def prepared(*args, **kwargs):
        assert kwargs["lookback_hours"] == 6
        return cleaned, audit, features, {}, pd.DataFrame()

    monkeypatch.setattr(module, "_prepare_context", prepared)
    result = service.predict_from_history(
        pd.DataFrame(), lookback_hours=6, now=index[-1]
    )

    assert result.usable is True
    assert result.value_kg_per_thm == pytest.approx(302.5)
    assert result.hourly_prediction_count == 6
    assert result.aggregation == "median"
    assert result.window_start_utc == str(index[2])
    assert result.window_end_utc == str(index[-1])
    assert result.latest_input_diagnostics["burden_mt"] == 70.0
    detail = next(
        item
        for item in result.outside_training_details
        if item["feature"] == ranged_feature
    )
    assert detail["expected_p01"] == -1.0
    assert detail["expected_p99"] == 1.0
    assert detail["received"] == 2.0


def test_inference_rejects_a_prediction_behind_the_dataset(monkeypatch, tmp_path):
    service = _generic_service(tmp_path)
    index = pd.date_range("2026-09-25 09:30:00Z", periods=2, freq="h")
    cleaned = pd.DataFrame(
        {
            "ORE_CALC_MT": [10.0, 0.0],
            "SINTER_CALC_MT": [50.0, 0.0],
            "TOTAL_PELLET_CALC_MT": [10.0, 0.0],
            "PRODUCTIONTONNESPERHR": [90.0, 90.0],
            "COKE_CALC_MT": [27.0, 0.0],
        },
        index=index,
    )
    audit = pd.DataFrame({"normal_eligible": [True, False]}, index=index)
    features = pd.DataFrame(
        0.0, index=index, columns=service.schema["features"], dtype=float
    )

    monkeypatch.setattr(
        module,
        "_prepare_context",
        lambda *args, **kwargs: (cleaned, audit, features, {}, pd.DataFrame()),
    )
    service.max_stale_hours = 0.5
    result = service.predict_from_history(pd.DataFrame(), now=index[-1])

    assert result.value_kg_per_thm is not None
    assert result.usable is False
    assert result.stale_hours == 1.0
    assert "behind the dataset" in " ".join(result.reasons)


def test_prediction_does_not_need_measured_coke_mass(monkeypatch, tmp_path):
    service = _generic_service(tmp_path)
    service.model = _WindowFakeModel()
    index = pd.date_range("2026-09-25 09:30:00Z", periods=2, freq="h")
    cleaned = pd.DataFrame(
        {
            "ORE_CALC_MT": [10.0, 10.0],
            "SINTER_CALC_MT": [50.0, 50.0],
            "TOTAL_PELLET_CALC_MT": [10.0, 10.0],
            "PRODUCTIONTONNESPERHR": [100.0, 50.0],
            "COKE_CALC_MT": [np.nan, np.nan],
        },
        index=index,
    )
    audit = pd.DataFrame({"normal_eligible": True}, index=index)
    features = pd.DataFrame(
        0.0, index=index, columns=service.schema["features"], dtype=float
    )
    monkeypatch.setattr(
        module,
        "_prepare_context",
        lambda *args, **kwargs: (cleaned, audit, features, {}, pd.DataFrame()),
    )

    result = service.predict_from_history(
        pd.DataFrame(), lookback_hours=2, now=index[-1]
    )

    assert result.usable is True
    assert result.value_kg_per_thm == pytest.approx(300.0)
    assert result.reasons == ()


def test_robust_inference_needs_window_coverage(monkeypatch, tmp_path):
    service = DirectCokeModelService(deployment_dir=tmp_path / "deploy")
    service.model = _WindowFakeModel()
    index = pd.date_range("2026-09-25 03:30:00Z", periods=8, freq="h")
    cleaned = pd.DataFrame({"PRODUCTIONTONNESPERHR": [90.0] * 8}, index=index)
    audit = pd.DataFrame({"normal_eligible": True}, index=index)
    features = pd.DataFrame(
        0.0, index=index, columns=service.schema["features"], dtype=float
    )
    recorded: dict[str, float] = {}

    def prepared(*args, **kwargs):
        recorded["recent_hours"] = kwargs["recent_hours"]
        return cleaned, audit, features, {}, pd.DataFrame()

    coverage = pd.Series(0.5, index=index)
    coverage.iloc[-3:] = 1.0
    monkeypatch.setattr(module, "_prepare_context", prepared)
    monkeypatch.setattr(module, "window_coverage", lambda *a, **k: coverage)

    result = service.predict_from_history(
        pd.DataFrame(), lookback_hours=6, now=index[-1]
    )

    # Only the three fully covered windows are scored.
    assert result.hourly_prediction_count == 3
    assert result.value_kg_per_thm == pytest.approx(300.0)
    assert result.target_window_hours == 24
    assert recorded["recent_hours"] >= 168

    monkeypatch.setattr(module, "window_coverage", lambda *a, **k: coverage * 0.0)
    rejected = service.predict_from_history(pd.DataFrame(), now=index[-1])
    assert rejected.usable is False
    assert "24-hour window coverage" in " ".join(rejected.reasons)


def test_robust_override_covers_every_window_in_the_lookback(monkeypatch):
    recorded: dict[str, float] = {}

    def override(raw, cfg, **kwargs):
        recorded["hours"] = kwargs["lookback_hours"]
        return raw

    frame = pd.DataFrame({"x": [1.0]})
    monkeypatch.setattr(module, "_apply_current_fuel_overrides", override)
    monkeypatch.setattr(
        module.pipeline, "clean_furnace", lambda raw, cfg: (frame, frame, {})
    )
    monkeypatch.setattr(
        module.pipeline, "slag_features", lambda d, cfg: (frame, pd.DataFrame())
    )
    monkeypatch.setattr(module, "window_features", lambda *a, **k: frame)

    module._prepare_context(
        frame,
        {"robust_target": {"window_hours": 24}},
        pci_kg_per_thm=160.0,
        lookback_hours=6,
    )

    assert recorded["hours"] == 6 + 24 - 1


class _ConstantFakeModel:
    def predict(self, matrix):
        return np.full(matrix.num_row(), 280.0)


def test_corrected_history_uses_only_prior_residuals(monkeypatch, tmp_path):
    service = _generic_service(tmp_path)
    service.model = _ConstantFakeModel()
    index = pd.date_range("2026-09-25 09:30:00Z", periods=4, freq="h")
    cleaned = pd.DataFrame(
        {
            "PRODUCTIONTONNESPERHR": [100.0] * 4,
            "COKE_CALC_MT": [30.0, 31.0, 29.0, 30.0],
        },
        index=index,
    )
    audit = pd.DataFrame({"normal_eligible": True}, index=index)
    features = pd.DataFrame(
        0.0, index=index, columns=service.schema["features"], dtype=float
    )
    monkeypatch.setattr(
        module,
        "_prepare_context",
        lambda *args, **kwargs: (cleaned, audit, features, {}, pd.DataFrame()),
    )

    history = service.prediction_history(
        pd.DataFrame(), bias_window_hours=24, bias_min_periods=2
    )

    corrected = history["corrected_predicted_coke_kg_per_thm"]
    assert pd.isna(corrected.iloc[0])
    assert pd.isna(corrected.iloc[1])
    assert corrected.iloc[2] == pytest.approx(305.0)
    assert corrected.iloc[3] == pytest.approx(300.0)


class _SavedFakeModel:
    def save_model(self, path: str) -> None:
        Path(path).write_text("fake-model", encoding="utf-8")


_TRAIN_INDEX = pd.date_range("2026-07-01 00:30:00Z", periods=1_600, freq="h")
_TRAIN_TARGET = pd.Series(
    305.0 + 18.0 * np.sin(np.arange(len(_TRAIN_INDEX)) / 30.0), index=_TRAIN_INDEX
)


def _prepared_retrain_frames():
    cleaned = pd.DataFrame({"PRODUCTIONTONNESPERHR": 90.0}, index=_TRAIN_INDEX)
    audit = pd.DataFrame(
        {"normal_eligible": True, "in_requested_window": True}, index=_TRAIN_INDEX
    )
    features = pd.DataFrame(
        {"signal": _TRAIN_TARGET, PCI_FEATURE: 180.0, NUT_COKE_FEATURE: 75.0},
        index=_TRAIN_INDEX,
    )
    config = {"purge_hours": 12, "min_train_rows": 100, "min_test_rows": 20}
    return cleaned, audit, features, config, pd.DataFrame()


def _stub_training(monkeypatch, predict) -> None:
    monkeypatch.setattr(
        module, "_prepare_context", lambda *a, **k: _prepared_retrain_frames()
    )
    monkeypatch.setattr(module, "_training_target", lambda *a, **k: _TRAIN_TARGET)
    monkeypatch.setattr(
        module, "_select_features", lambda *a, **k: ["signal", PCI_FEATURE]
    )
    monkeypatch.setattr(module, "_train_model", lambda *a, **k: _SavedFakeModel())
    monkeypatch.setattr(module, "_predict_model", lambda model, frame, cols: predict(frame))


def _responsive(frame: pd.DataFrame) -> np.ndarray:
    # Perfect fit, and 0.75 kg coke per kg PCI below the observed 180 kg/THM.
    return (frame["signal"] + 0.75 * (180.0 - frame[PCI_FEATURE])).to_numpy()


def test_retrain_deploys_only_after_every_gate(monkeypatch, tmp_path):
    dataset = tmp_path / "furnace.csv"
    dataset.write_text("test", encoding="utf-8")
    deployment = tmp_path / "deployment"
    _stub_training(monkeypatch, _responsive)

    report = retrain_and_maybe_deploy(
        dataset, deployment_dir=deployment, min_random_r2=0.9, rounds_override=1
    )

    assert report.passed is True
    assert report.deployed is True
    assert report.fuel_response["pci_replacement_kg_per_kg"] == pytest.approx(0.75)
    assert report.later_time_metrics["skill_vs_last_week_level"] == pytest.approx(1.0)
    assert report.random_metrics["held_out_days"] > 0
    pointer = json.loads((deployment / "active.json").read_text(encoding="utf-8"))
    assert pointer["deployment_id"] == report.deployment_id
    version = deployment / "versions" / report.deployment_id
    assert (version / "coke_context.json").is_file()
    assert (version / "deployment_metadata.json").is_file()
    schema = json.loads((version / "coke_context_schema.json").read_text(encoding="utf-8"))
    assert schema["task"] == "coke_robust_window"
    assert schema["monotone_constraints"] == {PCI_FEATURE: -1}
    config = json.loads((version / "config.json").read_text(encoding="utf-8"))
    assert "robust_target" in config


def test_retrain_rejects_a_model_that_ignores_pci(monkeypatch, tmp_path):
    dataset = tmp_path / "furnace.csv"
    dataset.write_text("test", encoding="utf-8")
    deployment = tmp_path / "deployment"
    _stub_training(monkeypatch, lambda frame: frame["signal"].to_numpy())

    report = retrain_and_maybe_deploy(
        dataset, deployment_dir=deployment, rounds_override=1
    )

    assert report.deployed is False
    assert "PCI replacement" in " ".join(report.reasons)
    assert not (deployment / "active.json").exists()


def test_failed_retrain_does_not_replace_active_pointer(monkeypatch, tmp_path):
    dataset = tmp_path / "furnace.csv"
    dataset.write_text("test", encoding="utf-8")
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "active.json").write_text(
        json.dumps({"deployment_id": "known-good"}), encoding="utf-8"
    )
    _stub_training(monkeypatch, lambda frame: np.zeros(len(frame)))

    report = retrain_and_maybe_deploy(
        dataset, deployment_dir=deployment, min_random_r2=0.1, rounds_override=1
    )

    assert report.passed is False
    assert report.deployed is False
    assert any("Later-time MAE" in reason for reason in report.reasons)
    pointer = json.loads((deployment / "active.json").read_text(encoding="utf-8"))
    assert pointer == {"deployment_id": "known-good"}


def test_old_versions_are_pruned_but_the_active_one_survives(tmp_path):
    versions = tmp_path / "versions"
    for name in ["20260901T0", "20260902T0", "20260903T0", "20260904T0"]:
        (versions / name).mkdir(parents=True)
        (versions / name / "coke_context.json").write_text("m", encoding="utf-8")

    module._prune_versions(tmp_path, keep=2, active_id="20260901T0")

    assert sorted(path.name for path in versions.iterdir()) == [
        "20260901T0",
        "20260903T0",
        "20260904T0",
    ]


def test_retrain_settings_come_from_yaml():
    kwargs = retrain_kwargs_from_config(
        {"min_random_r2": 0.75, "pci_replacement_range": [0.3, 1.2], "seed": 7}
    )

    assert kwargs["min_random_r2"] == 0.75
    assert kwargs["pci_replacement_range"] == (0.3, 1.2)
    assert kwargs["max_later_time_mae"] == 10.0
    assert kwargs["seed"] == 7


def test_auto_retrain_waits_for_new_data_and_the_interval(tmp_path):
    dataset = tmp_path / "furnace.csv"
    dataset.write_text("v1", encoding="utf-8")
    deployment = tmp_path / "deployment"
    kwargs = {"dataset_path": dataset, "deployment_dir": deployment, "every_hours": 24}

    assert auto_retrain_due(**kwargs) is True  # never attempted

    deployment.mkdir()
    attempted = datetime(2026, 9, 28, 6, 0, tzinfo=timezone.utc)
    state = {
        "last_attempt_utc": attempted.isoformat(),
        "dataset_mtime_ns": dataset.stat().st_mtime_ns,
    }
    (deployment / "auto_retrain.json").write_text(json.dumps(state), encoding="utf-8")
    later = attempted + timedelta(hours=30)
    assert auto_retrain_due(**kwargs, now=later) is False  # same data

    state["dataset_mtime_ns"] = -1  # the dataset has since been refreshed
    (deployment / "auto_retrain.json").write_text(json.dumps(state), encoding="utf-8")
    assert auto_retrain_due(**kwargs, now=attempted + timedelta(hours=2)) is False
    assert auto_retrain_due(**kwargs, now=later) is True


def test_background_retrain_records_its_outcome(monkeypatch, tmp_path):
    dataset = tmp_path / "furnace.csv"
    dataset.write_text("v1", encoding="utf-8")
    deployment = tmp_path / "deployment"
    monkeypatch.setattr(
        module,
        "retrain_and_maybe_deploy",
        lambda *a, **k: RetrainReport(
            passed=True, deployed=True, deployment_id="new-model"
        ),
    )

    started = maybe_retrain_in_background(
        dataset_path=dataset, deployment_dir=deployment, every_hours=24
    )
    for thread in threading.enumerate():
        if thread.name == "bmo-coke-retrain":
            thread.join(timeout=10)

    assert started is True
    state = json.loads((deployment / "auto_retrain.json").read_text(encoding="utf-8"))
    assert state["deployed"] is True
    assert state["deployment_id"] == "new-model"
    assert state["dataset_mtime_ns"] == dataset.stat().st_mtime_ns
    # Nothing new since: the next page load does not start another run.
    assert (
        maybe_retrain_in_background(
            dataset_path=dataset, deployment_dir=deployment, every_hours=24
        )
        is False
    )
