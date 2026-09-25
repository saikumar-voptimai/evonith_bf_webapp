from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.bmo import direct_coke_model as module
from utils.bmo.coke_model_pipeline import pipeline
from utils.bmo.direct_coke_model import (
    DEFAULT_BUNDLE_DIR,
    DirectCokeModelService,
    _apply_current_fuel_overrides,
    retrain_and_maybe_deploy,
)


def test_bundled_schema_has_no_coke_target_features_and_fixed_slag_basis():
    schema = json.loads(
        (DEFAULT_BUNDLE_DIR / "coke_context_schema.json").read_text(encoding="utf-8")
    )

    assert len(schema["features"]) == 80
    assert not any(name.startswith("COKE_CALC_") for name in schema["features"])
    assert "COKE RATE KG/THM" not in schema["features"]
    assert "ACT. FUEL RATEKG/THM." not in schema["features"]
    assert schema["slag_coke_basis_kg_thm"] == 300
    assert schema["validation"]["test"]["r2"] == pytest.approx(0.7672279018708291)


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


def test_inference_rejects_a_prediction_behind_the_dataset(monkeypatch, tmp_path):
    service = DirectCokeModelService(deployment_dir=tmp_path / "deploy")
    index = pd.date_range("2026-09-25 09:30:00Z", periods=2, freq="h")
    cleaned = pd.DataFrame(
        {
            "ORE_CALC_MT": [10.0, 0.0],
            "SINTER_CALC_MT": [50.0, 0.0],
            "TOTAL_PELLET_CALC_MT": [10.0, 0.0],
            "PRODUCTIONTONNESPERHR": [90.0, 90.0],
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


class _SavedFakeModel:
    def save_model(self, path: str) -> None:
        Path(path).write_text("fake-model", encoding="utf-8")


def _prepared_retrain_frames():
    index = pd.date_range("2026-07-01 00:30:00Z", periods=1_600, freq="h")
    target = 305.0 + 18.0 * np.sin(np.arange(len(index)) / 30.0)
    cleaned = pd.DataFrame({"COKE_CALC_KG_THM": target}, index=index)
    audit = pd.DataFrame(
        {"normal_eligible": True, "in_requested_window": True}, index=index
    )
    features = pd.DataFrame({"signal": target}, index=index)
    config = {"purge_hours": 12, "min_train_rows": 100, "min_test_rows": 20}
    return cleaned, audit, features, config, pd.DataFrame()


def test_retrain_deploys_only_after_both_r2_gates(monkeypatch, tmp_path):
    dataset = tmp_path / "furnace.csv"
    dataset.write_text("test", encoding="utf-8")
    deployment = tmp_path / "deployment"
    monkeypatch.setattr(
        module, "_prepare_context", lambda *a, **k: _prepared_retrain_frames()
    )
    monkeypatch.setattr(module, "_select_features", lambda *a, **k: ["signal"])
    monkeypatch.setattr(module, "_train_model", lambda *a, **k: _SavedFakeModel())
    monkeypatch.setattr(
        module,
        "_predict_model",
        lambda model, frame, columns: frame["signal"].to_numpy(),
    )

    report = retrain_and_maybe_deploy(
        dataset,
        deployment_dir=deployment,
        min_random_r2=0.9,
        min_later_time_r2=0.9,
        rounds_override=1,
    )

    assert report.passed is True
    assert report.deployed is True
    pointer = json.loads((deployment / "active.json").read_text(encoding="utf-8"))
    assert pointer["deployment_id"] == report.deployment_id
    version = deployment / "versions" / report.deployment_id
    assert (version / "coke_context.json").is_file()
    assert (version / "coke_context_schema.json").is_file()
    assert (version / "deployment_metadata.json").is_file()


def test_failed_retrain_does_not_replace_active_pointer(monkeypatch, tmp_path):
    dataset = tmp_path / "furnace.csv"
    dataset.write_text("test", encoding="utf-8")
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "active.json").write_text(
        json.dumps({"deployment_id": "known-good"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        module, "_prepare_context", lambda *a, **k: _prepared_retrain_frames()
    )
    monkeypatch.setattr(module, "_select_features", lambda *a, **k: ["signal"])
    monkeypatch.setattr(module, "_train_model", lambda *a, **k: _SavedFakeModel())
    monkeypatch.setattr(
        module,
        "_predict_model",
        lambda model, frame, columns: np.zeros(len(frame)),
    )

    report = retrain_and_maybe_deploy(
        dataset,
        deployment_dir=deployment,
        min_random_r2=0.1,
        min_later_time_r2=0.1,
        rounds_override=1,
    )

    assert report.passed is False
    assert report.deployed is False
    pointer = json.loads((deployment / "active.json").read_text(encoding="utf-8"))
    assert pointer == {"deployment_id": "known-good"}
