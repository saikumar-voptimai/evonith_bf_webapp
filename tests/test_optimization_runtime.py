from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.objective import BmoObjectiveEvaluator
from utils.bmo.types import OreChemistry, OreInput
from domain.optimization_runtime import (
    FeatureVectorBuilder,
    ModelBundleInfo,
    ModelBundleService,
)


class DummyScaler:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = np.array(feature_names, dtype=object)

    def transform(self, X):
        return X


class DummyModel:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = np.array(feature_names, dtype=object)

    def predict(self, X):
        return np.array([float(np.asarray(X).sum())])


def test_feature_vector_builder_direct_normalized_and_lag_resolution():
    bundle = ModelBundleInfo(
        model=None,
        scaler=None,
        expected_features=[
            "A",
            "HOT BLAST VOLUMENM3/HR.",
            "oxygen_lag1",
            "steam_lag2",
            "missing_feat",
        ],
        lag_map={"lags": {"oxygen_lag1": {"base_feature": "oxygen", "lag_steps": 1}}},
        defaults={"missing_feat": 7.5},
    )
    builder = FeatureVectorBuilder(bundle, missing_feature_policy="default_warn")

    history_df = pd.DataFrame(
        {
            "oxygen": [100.0, 110.0, 120.0],
            "steam": [10.0, 11.0, 12.0],
        },
        index=pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
    )
    base_sample = {
        "A": 2.0,
        "hot_blast_volumenm3_hr": 3200.0,
    }

    result = builder.build(base_sample=base_sample, history_df=history_df)
    row = result.vector_df.iloc[0]
    assert row["A"] == pytest.approx(2.0)
    assert row["HOT BLAST VOLUMENM3/HR."] == pytest.approx(3200.0)
    assert row["oxygen_lag1"] == pytest.approx(110.0)
    assert row["steam_lag2"] == pytest.approx(10.0)
    assert row["missing_feat"] == pytest.approx(7.5)
    assert "missing_feat" in result.imputed_features
    assert result.source_map["oxygen_lag1"].startswith("lag_map")
    assert result.source_map["steam_lag2"].startswith("lag_suffix")


def test_feature_vector_builder_strict_fail_policy_raises():
    bundle = ModelBundleInfo(
        model=None,
        scaler=None,
        expected_features=["feature_a", "never_found"],
    )
    builder = FeatureVectorBuilder(bundle, missing_feature_policy="strict_fail")
    with pytest.raises(KeyError):
        builder.build(base_sample={"feature_a": 1.0}, history_df=pd.DataFrame())


def test_model_bundle_precedence_scaler_over_model_and_manifest(tmp_path: Path):
    scaler_path = tmp_path / "scaler.joblib"
    model_path = tmp_path / "model.joblib"
    manifest_path = tmp_path / "feature_manifest.json"
    lag_map_path = tmp_path / "lag_map.json"
    metrics_path = tmp_path / "training_metrics.json"

    joblib.dump(DummyScaler(["f1", "target_y", "f2"]), scaler_path)
    joblib.dump(DummyModel(["m1", "m2"]), model_path)
    manifest_path.write_text(
        json.dumps({"feature_names": ["x", "y"], "target_name": "target_y"}),
        encoding="utf-8",
    )
    lag_map_path.write_text(json.dumps({"lags": {}}), encoding="utf-8")
    metrics_path.write_text(json.dumps({"mae": 1.2}), encoding="utf-8")

    bundle = ModelBundleService(
        {
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "feature_manifest_path": str(manifest_path),
            "lag_map_path": str(lag_map_path),
            "training_metrics_path": str(metrics_path),
            "target_name": "target_y",
        }
    ).get_bundle()

    # precedence: scaler feature_names_in_ should win, and target removed.
    assert bundle.expected_features == ["f1", "f2"]
    assert bundle.status["model_loaded"] is True
    assert bundle.status["scaler_loaded"] is True


def test_bmo_objective_evaluator_runs_with_fallback_model():
    ores = [
        OreInput(
            ore_id="sinter",
            display_name="SINTER",
            stock_mt=5000.0,
            price_rs_per_mt=7000.0,
            min_share_pct=50.0,
            max_share_pct=80.0,
            chemistry=OreChemistry(
                fe_t_pct=55.0, feo_pct=8.0, sio2_pct=5.0, al2o3_pct=2.0
            ),
        ),
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=6000.0,
            min_share_pct=20.0,
            max_share_pct=50.0,
            chemistry=OreChemistry(
                fe_t_pct=62.0, feo_pct=0.5, sio2_pct=4.0, al2o3_pct=1.8
            ),
        ),
    ]
    model_service = FuelUnitCostModelService(
        bundle_cfg={
            "model_path": "src/assets/models/bmo_fuel/definitely_missing_model.joblib",
            "scaler_path": "src/assets/models/bmo_fuel/definitely_missing_scaler.joblib",
            "feature_manifest_path": "src/assets/models/bmo_fuel/feature_manifest.json",
            "lag_map_path": "src/assets/models/bmo_fuel/lag_map.json",
        },
        fallback_cfg={},
    )
    history_df = pd.DataFrame(
        {
            "TOTAL OXYGENNM3/HR.": [7000.0, 7100.0],
            "HOT BLAST VOLUMENM3/HR.": [9800.0, 9900.0],
            "COKE RATE KG/THM": [340.0, 338.0],
            "ACTUALKG/THM.": [170.0, 171.0],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
    )
    evaluator = BmoObjectiveEvaluator(
        ores=ores,
        target_total_qty_mt=3800.0,
        min_fe_production_mt=2100.0,
        max_fe_production_mt=2600.0,
        target_slag_qty_mt=900.0,
        feo_in_slag_pct=0.4,
        si_in_slag_pct=0.5,
        model_service=model_service,
        process_context={"COKE RATE KG/THM": 338.0, "ACTUALKG/THM.": 171.0},
        history_df=history_df,
        penalty_cfg={},
    )
    result = evaluator.evaluate_shares(np.array([0.6, 0.4], dtype=float))
    assert np.isfinite(result.objective_value)
    assert "base_objective_rs_per_thm" in result.components
    assert "blend" in result.diagnostics
