from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from domain.optimization_runtime import parse_lag_feature_name
from utils.bmo.feature_builder import (
    build_bmo_v4_feature_frame,
    build_feature_payload,
    max_bmo_lag_steps,
)
from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.types import OreChemistry, OreInput


class IdentityScaler:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = np.array(feature_names, dtype=object)
        self.mean_ = np.zeros(len(feature_names), dtype=float)
        self.scale_ = np.ones(len(feature_names), dtype=float)

    def transform(self, X):
        return np.asarray(X[list(self.feature_names_in_)], dtype=float)


class SumModel:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = np.array(feature_names, dtype=object)
        self.n_features_in_ = len(feature_names)

    def predict(self, X):
        return np.array([float(np.asarray(X).sum())])


def _ore(
    ore_id: str,
    name: str,
    material_key: str,
    chemistry: OreChemistry,
) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=name,
        stock_mt=5000.0,
        price_rs_per_mt=1000.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=chemistry,
        metadata={"material_key": material_key},
    )


def test_candidate_blend_payload_maps_ore_slots_and_ratios() -> None:
    ores = [
        _ore(
            "sinter",
            "SINTER (SP-02)",
            "sinter_sp_02",
            OreChemistry(55, moisture_pct=8, sio2_pct=5),
        ),
        _ore(
            "ore3",
            "GEOMIN CLO",
            "ore_3",
            OreChemistry(61, moisture_pct=6, sio2_pct=4),
        ),
        _ore(
            "ore8",
            "ACORE",
            "ore_8",
            OreChemistry(30, moisture_pct=2, sio2_pct=8),
        ),
        _ore("pellet", "LLOYDS PELLET", "lloyds_pellet", OreChemistry(64, sio2_pct=3)),
    ]
    quantities = {"sinter": 60.0, "ore3": 20.0, "ore8": 30.0, "pellet": 10.0}

    payload = build_feature_payload(
        quantities_mt=quantities,
        ore_display_name_by_id={ore.ore_id: ore.display_name for ore in ores},
        ores=ores,
    )

    assert payload["ORE_CALC_MT"] == pytest.approx(50.0)
    assert payload["SINTER_CALC_MT"] == pytest.approx(60.0)
    assert payload["TOTAL_PELLET_CALC_MT"] == pytest.approx(10.0)
    hot_metal_mt = (60.0 * 0.92 * 0.55) + (20.0 * 0.94 * 0.61)
    hot_metal_mt += (30.0 * 0.98 * 0.30) + (10.0 * 1.0 * 0.64)
    assert payload["ORE_CALC_THM"] == pytest.approx(50.0 / hot_metal_mt)
    assert payload["SINTER_CALC_THM"] == pytest.approx(60.0 / hot_metal_mt)
    assert payload["TOTAL_PELLET_CALC_THM"] == pytest.approx(10.0 / hot_metal_mt)
    assert payload["ORE_3_PCT"] == pytest.approx(40.0)
    assert payload["ORE_8_PCT"] == pytest.approx(60.0)
    assert payload["ORE_12_PCT"] == pytest.approx(0.0)
    assert payload["SINTER_CLO_RATIO"] == pytest.approx(1.2)
    assert payload["PELLET_CLO_RATIO"] == pytest.approx(0.2)
    assert payload["PELLET_PCT_SIO2"] == pytest.approx(3.0)
    assert payload["LLOYDS_PELLET_PCT_SIO2"] == pytest.approx(3.0)
    assert payload["ORE_SIO2%"] == pytest.approx(
        ((20.0 * 0.94 * 4.0) + (30.0 * 0.98 * 8.0))
        / ((20.0 * 0.94) + (30.0 * 0.98))
    )
    assert payload["ORE_TM%"] == pytest.approx(
        ((20.0 * 0.94 * 6.0) + (30.0 * 0.98 * 2.0))
        / ((20.0 * 0.94) + (30.0 * 0.98))
    )
    assert payload["SINTER_SP_02_TM%"] == pytest.approx(8.0)


def test_legacy_pellet_model_features_resolve_from_generic_payload() -> None:
    result = build_bmo_v4_feature_frame(
        feature_payload={"PELLET_PCT_CAO": 0.3, "PELLET_CLO_RATIO": 0.2},
        history_df=None,
        expected_features=[
            "LLOYDS_PELLET_PCT_CAO",
            "LLOYDS_PELLET_PCT_CAO_lag4_(MeltImpact)",
            "PELLET_CLO_RATIO_lag1_(GasImpact)",
        ],
        default_values={},
    )

    assert result.vector_df["LLOYDS_PELLET_PCT_CAO"].iloc[0] == pytest.approx(0.3)
    assert result.vector_df["LLOYDS_PELLET_PCT_CAO_lag4_(MeltImpact)"].iloc[
        0
    ] == pytest.approx(0.3)
    assert result.vector_df["PELLET_CLO_RATIO_lag1_(GasImpact)"].iloc[
        0
    ] == pytest.approx(0.2)
    assert result.imputed_features == []


def test_candidate_blend_payload_uses_operator_hm_basis_for_thm_features() -> None:
    ores = [
        _ore("sinter", "SINTER", "sinter_sp_02", OreChemistry(55.0)),
        _ore("ore3", "GEOMIN CLO", "ore_3", OreChemistry(61.0)),
    ]
    quantities = {"sinter": 60.0, "ore3": 40.0}

    payload = build_feature_payload(
        quantities_mt=quantities,
        ore_display_name_by_id={ore.ore_id: ore.display_name for ore in ores},
        ores=ores,
        hot_metal_target_mt=80.0,
    )

    assert payload["SINTER_CALC_THM"] == pytest.approx(60.0 / 80.0)
    assert payload["ORE_CALC_THM"] == pytest.approx(40.0 / 80.0)


def test_bmo_v4_feature_frame_handles_dual_lags_and_candidate_overrides() -> None:
    history_df = pd.DataFrame(
        {
            "STOCKRODLEVEL": [10.0, 11.0, 12.0, 13.0, 14.0],
            "TOPPRESSUREBAR": [1.0, 1.1, 1.2, 1.3, 1.4],
            "ORE_3_PCT": [1.0, 2.0, 3.0, 4.0, 5.0],
            "COKE_CALC_MT": [100.0, 110.0, 120.0, 130.0, 140.0],
            "PRODUCTIONTONNESPERHR": [50.0, 55.0, 60.0, 65.0, 70.0],
        },
        index=pd.date_range("2026-05-21", periods=5, freq="h"),
    )

    result = build_bmo_v4_feature_frame(
        feature_payload={"ORE_3_PCT": 42.0, "ORE_CALC_THM": 1.7, "TOPBAR": 1.7},
        history_df=history_df,
        expected_features=[
            "ORE_3_PCT_lag1_(GasImpact)",
            "ORE_3_PCT_lag4_(MeltImpact)",
            "ORE_CALC_THM_lag1_(GasImpact)",
            "STOCKRODLEVEL_lag1",
            "TOPBAR",
            "COKE_CALC_THM",
            "COKE_CALC_THM_lag4",
            "day_of_year",
            "trend_index",
            "MISSING_FEATURE",
        ],
        default_values={"MISSING_FEATURE": 7.0},
        candidate_lag_bases={"ORE_3_PCT", "ORE_CALC_THM"},
    )

    row = result.vector_df.iloc[0]
    assert row["ORE_3_PCT_lag1_(GasImpact)"] == pytest.approx(42.0)
    assert row["ORE_3_PCT_lag4_(MeltImpact)"] == pytest.approx(42.0)
    assert row["ORE_CALC_THM_lag1_(GasImpact)"] == pytest.approx(1.7)
    assert row["STOCKRODLEVEL_lag1"] == pytest.approx(13.0)
    assert row["TOPBAR"] == pytest.approx(1.7)
    assert row["COKE_CALC_THM"] == pytest.approx(2.0)
    assert row["COKE_CALC_THM_lag4"] == pytest.approx(2.0)
    assert row["day_of_year"] == pytest.approx(141.0)
    assert row["trend_index"] == pytest.approx(4.0)
    assert row["MISSING_FEATURE"] == pytest.approx(7.0)
    assert "MISSING_FEATURE" in result.missing_features


def test_topbar_does_not_alias_to_toppressurebar() -> None:
    history_df = pd.DataFrame(
        {"TOPPRESSUREBAR": [1.3]},
        index=pd.date_range("2026-05-21", periods=1, freq="h"),
    )

    result = build_bmo_v4_feature_frame(
        feature_payload={},
        history_df=history_df,
        expected_features=["TOPBAR"],
        default_values={},
    )

    assert result.vector_df["TOPBAR"].iloc[0] == pytest.approx(0.0)
    assert result.source_map["TOPBAR"] == "default"
    assert "TOPBAR" in result.missing_features


def test_selected_feature_model_service_scales_full_frame_then_slices(
    tmp_path: Path,
) -> None:
    scaler_path = tmp_path / "scaler.joblib"
    model_path = tmp_path / "model.joblib"
    selected_path = tmp_path / "features.json"

    joblib.dump(IdentityScaler(["a", "b", "c"]), scaler_path)
    joblib.dump(SumModel(["a", "c"]), model_path)
    selected_path.write_text(json.dumps(["a", "c"]), encoding="utf-8")

    service = FuelUnitCostModelService(
        bundle_cfg={
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "selected_feature_columns_path": str(selected_path),
        },
        fallback_cfg={},
    )

    prediction = service.predict({"a": 2.0, "c": 3.0}, pd.DataFrame())

    assert prediction.used_fallback is False
    assert prediction.value == pytest.approx(5.0)
    assert prediction.details["raw_feature_count"] == 3
    assert prediction.details["selected_feature_count"] == 2


def test_selected_feature_model_service_rejects_negative_fuel_cost(
    tmp_path: Path,
) -> None:
    scaler_path = tmp_path / "scaler.joblib"
    model_path = tmp_path / "model.joblib"
    selected_path = tmp_path / "features.json"

    joblib.dump(IdentityScaler(["a", "b", "c"]), scaler_path)
    joblib.dump(SumModel(["a", "c"]), model_path)
    selected_path.write_text(json.dumps(["a", "c"]), encoding="utf-8")

    service = FuelUnitCostModelService(
        bundle_cfg={
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "selected_feature_columns_path": str(selected_path),
        },
        fallback_cfg={},
    )

    prediction = service.predict({"a": -2.0, "c": -3.0}, pd.DataFrame())

    assert prediction.used_fallback is True
    assert prediction.value > 0.0
    assert prediction.details["model_value"] == pytest.approx(-5.0)
    assert "not above the minimum fuel cost" in prediction.details["reason"]


def test_blend_fuel_prediction_helper_attaches_model_cost(tmp_path: Path) -> None:
    scaler_path = tmp_path / "scaler.joblib"
    model_path = tmp_path / "model.joblib"
    selected_path = tmp_path / "features.json"

    joblib.dump(IdentityScaler(["a", "b", "c"]), scaler_path)
    joblib.dump(SumModel(["a", "c"]), model_path)
    selected_path.write_text(json.dumps(["a", "c"]), encoding="utf-8")

    service = FuelUnitCostModelService(
        bundle_cfg={
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "selected_feature_columns_path": str(selected_path),
        },
        fallback_cfg={},
    )
    ore = _ore("ore3", "GEOMIN CLO", "ore_3", OreChemistry(50.0))

    blend = evaluate_blend_with_fuel_prediction(
        ores=[ore],
        quantities_mt={"ore3": 100.0},
        feo_in_slag_pct=0.4,
        model_service=service,
        process_context={"a": 2.0, "c": 3.0},
        history_df=pd.DataFrame(),
    )

    assert blend.fuel_cost_per_thm_rs == pytest.approx(5.0)
    assert blend.objective_rs_per_thm == pytest.approx(blend.ore_cost_per_thm_rs + 5.0)
    assert blend.diagnostics["model_prediction"].used_fallback is False


def test_shared_lag_parser_accepts_bmo_dual_impact_suffix() -> None:
    assert parse_lag_feature_name("ORE_8_PCT_lag4_(MeltImpact)") == (
        "ORE_8_PCT",
        4,
    )


def test_bmo_max_lag_steps_uses_model_feature_names() -> None:
    assert (
        max_bmo_lag_steps(
            [
                "HOT BLAST TEMP.OC",
                "STOCKRODLEVEL_lag1",
                "ORE_8_PCT_lag4_(MeltImpact)",
            ]
        )
        == 4
    )
