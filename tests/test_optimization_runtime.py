from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from domain.optimization_runtime import (
    FeatureVectorBuilder,
    ModelBundleInfo,
    ModelBundleService,
)
from utils.bmo.calculations import evaluate_blend
from utils.bmo.constraints import check_blend_constraints
from utils.bmo.constraints import validate_selected_pellet_inputs
import utils.bmo.lp_solver as lp_solver
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.nonlinear_optimizer import run_nonlinear_optimizer
from utils.bmo.objective import BmoObjectiveEvaluator
from utils.bmo.slag_balance import calculate_full_slag_balance
from utils.bmo.types import (
    BlendEvaluation,
    DustInput,
    FluxInput,
    FuelAshInput,
    OreChemistry,
    OreInput,
    SlagBalanceSettings,
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


def test_blend_evaluation_uses_dry_weight_for_final_fe():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=60.0, moisture_pct=10.0),
        ),
        OreInput(
            ore_id="ore_b",
            display_name="ORE B",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0, moisture_pct=0.0),
        ),
    ]

    blend = evaluate_blend(
        ores=ores,
        quantities_mt={"ore_a": 100.0, "ore_b": 100.0},
        feo_in_slag_pct=0.0,
    )

    assert blend.diagnostics["total_dry_qty_mt"] == pytest.approx(190.0)
    assert blend.fe_production_mt == pytest.approx(104.0)
    assert blend.fe_t_pct == pytest.approx((104.0 / 190.0) * 100.0)
    assert blend.diagnostics["dry_weight_mt_by_ore"]["ore_a"] == pytest.approx(90.0)
    assert blend.diagnostics["fe_contribution_mt_by_ore"]["ore_a"] == pytest.approx(
        54.0
    )


def test_blend_evaluation_uses_oxide_sum_for_slag_and_rate():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=60.0,
                moisture_pct=10.0,
                sio2_pct=5.0,
                al2o3_pct=2.0,
                cao_pct=1.0,
                mgo_pct=1.0,
                tio2_pct=1.0,
                mno_pct=1.0,
                na2o_pct=0.5,
                k2o_pct=0.5,
            ),
        ),
        OreInput(
            ore_id="ore_b",
            display_name="ORE B",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=50.0,
                sio2_pct=3.0,
                al2o3_pct=1.0,
                cao_pct=1.0,
                mgo_pct=0.5,
                tio2_pct=0.25,
                mno_pct=0.25,
            ),
        ),
    ]

    blend = evaluate_blend(
        ores=ores,
        quantities_mt={"ore_a": 100.0, "ore_b": 100.0},
        feo_in_slag_pct=0.0,
    )

    expected_slag_mt = (90.0 * 12.0 / 100.0) + (100.0 * 6.0 / 100.0)
    assert blend.slag_mt == pytest.approx(expected_slag_mt)
    assert blend.slag_pct == pytest.approx((expected_slag_mt / 190.0) * 100.0)
    assert blend.slag_rate_kg_per_thm == pytest.approx(
        (expected_slag_mt / 104.0) * 1000.0
    )
    assert blend.diagnostics["slag_contribution_mt_by_ore"]["ore_a"] == pytest.approx(
        10.8
    )


def test_blend_evaluation_uses_operator_hm_basis_for_thm_metrics():
    ore = OreInput(
        ore_id="ore_a",
        display_name="ORE A",
        stock_mt=5000.0,
        price_rs_per_mt=1000.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(fe_t_pct=50.0, sio2_pct=10.0),
    )

    blend = evaluate_blend(
        ores=[ore],
        quantities_mt={"ore_a": 200.0},
        feo_in_slag_pct=0.0,
        hot_metal_target_mt=100.0,
    )

    assert blend.fe_production_mt == pytest.approx(100.0)
    assert blend.ore_cost_total_rs == pytest.approx(200_000.0)
    assert blend.ore_cost_per_thm_rs == pytest.approx(2_000.0)
    assert blend.slag_rate_kg_per_thm == pytest.approx(200.0)
    assert blend.diagnostics["slag_rate_denominator"] == "hot_metal_target_mt"


def test_blend_evaluation_adds_fuel_ash_slag_contribution():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=60.0, sio2_pct=10.0),
        )
    ]
    fuels = [
        FuelAshInput(
            fuel_id="coke",
            display_name="Coke",
            rate_kg_per_thm=100.0,
            ash_pct=10.0,
            sio2_pct=50.0,
        )
    ]

    blend = evaluate_blend(
        ores=ores,
        quantities_mt={"ore_a": 100.0},
        feo_in_slag_pct=0.0,
        fuel_ash_inputs=fuels,
    )

    expected_ore_slag_mt = 100.0 * 10.0 / 100.0
    expected_fuel_ash_slag_mt = (100.0 * 60.0 / 100.0) * 0.1 * 0.1 * 0.5
    assert blend.diagnostics["ore_slag_mt"] == pytest.approx(expected_ore_slag_mt)
    assert blend.diagnostics["fuel_ash_slag_mt"] == pytest.approx(
        expected_fuel_ash_slag_mt
    )
    assert blend.slag_mt == pytest.approx(
        expected_ore_slag_mt + expected_fuel_ash_slag_mt
    )


def test_blend_evaluation_adds_fixed_flux_slag_contribution():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=60.0, sio2_pct=10.0),
        )
    ]
    fluxes = [
        FluxInput(
            flux_id="limestone",
            display_name="Limestone",
            wet_qty_mt=10.0,
            moisture_pct=10.0,
            sio2_pct=5.0,
            cao_pct=50.0,
        )
    ]

    blend = evaluate_blend(
        ores=ores,
        quantities_mt={"ore_a": 100.0},
        feo_in_slag_pct=0.0,
        flux_inputs=fluxes,
    )

    expected_ore_slag_mt = 100.0 * 10.0 / 100.0
    expected_flux_slag_mt = 9.0 * 55.0 / 100.0
    assert blend.diagnostics["ore_slag_mt"] == pytest.approx(expected_ore_slag_mt)
    assert blend.diagnostics["flux_slag_mt"] == pytest.approx(expected_flux_slag_mt)
    assert blend.diagnostics["flux_dry_weight_mt_by_flux"]["limestone"] == (
        pytest.approx(9.0)
    )
    assert blend.slag_mt == pytest.approx(expected_ore_slag_mt + expected_flux_slag_mt)


def test_full_slag_balance_calculates_workbook_component_sequence():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=60.0,
                sio2_pct=10.0,
                al2o3_pct=5.0,
                cao_pct=2.0,
                mgo_pct=1.0,
                na2o_pct=1.0,
                k2o_pct=1.0,
            ),
        )
    ]
    settings = SlagBalanceSettings(
        enabled=True,
        carbon_pct=4.0,
        silicon_pct=0.0,
        sulphur_pct=0.0,
        other_pct=0.0,
        fe_to_pig_iron_fraction=0.999,
        mn_recovery_pct=0.0,
        sulphur_gas_loss_pct=0.0,
        alkali_to_slag_fraction=0.8,
        fe_to_feo_factor=72.0 / 56.0,
    )

    result = calculate_full_slag_balance(
        ores=ores,
        quantities_mt={"ore_a": 100.0},
        hot_metal_mt=60.0,
        settings=settings,
    )

    expected_feo_mt = (60.0 - (60.0 * 0.999)) * (72.0 / 56.0)
    expected_slag_mt = 10.0 + 5.0 + 2.0 + 1.0 + expected_feo_mt + (2.0 * 0.8)
    assert result.theoretical_pig_iron_mt == pytest.approx(
        (60.0 * 0.999) * 100.0 / 96.0
    )
    assert result.actual_pig_iron_mt == pytest.approx(
        (60.0 * 0.999) * 100.0 / 96.0 * 0.998
    )
    assert result.slag_components_mt["sio2"] == pytest.approx(10.0)
    assert result.slag_components_mt["feo"] == pytest.approx(expected_feo_mt)
    assert result.slag_components_mt["alkali"] == pytest.approx(1.6)
    assert result.total_slag_mt == pytest.approx(expected_slag_mt)


def test_full_slag_balance_uses_new_burden_pig_iron_split():
    mn_from_mno_factor = 54.938 / 70.937
    ti_from_tio2_factor = 47.867 / 79.866
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=50.0,
                mno_pct=10.0 / mn_from_mno_factor,
                tio2_pct=10.0 / ti_from_tio2_factor,
                p_pct=1.0,
                zn_pct=2.0,
            ),
        )
    ]
    settings = SlagBalanceSettings(
        enabled=True,
        carbon_pct=4.0,
        silicon_pct=0.0,
        sulphur_pct=0.0,
        other_pct=1.0,
        pi_loss_pct=0.2,
        fe_to_pig_iron_fraction=0.999,
        mn_recovery_pct=60.0,
        sulphur_gas_loss_pct=10.0,
        fe_to_feo_factor=72.0 / 56.0,
        mn_to_mno_factor=1.291,
    )

    result = calculate_full_slag_balance(
        ores=ores,
        quantities_mt={"ore_a": 100.0},
        hot_metal_mt=60.0,
        settings=settings,
    )

    expected_fe_pi_mt = 50.0 * 0.999
    expected_mn_pi_mt = 10.0 * 0.6
    expected_ti_pi_mt = 10.0 * 0.6
    expected_metallic_mass_mt = (
        expected_fe_pi_mt + expected_mn_pi_mt + expected_ti_pi_mt + 1.0 + 2.0
    )
    expected_theoretical_pi_mt = expected_metallic_mass_mt * 100.0 / 95.0
    expected_actual_pi_mt = expected_theoretical_pi_mt * 0.998
    expected_feo_mt = (50.0 - expected_fe_pi_mt) * (72.0 / 56.0)
    expected_mno_mt = (10.0 - expected_mn_pi_mt) * 1.291

    assert result.diagnostics["fe_to_pig_iron_mt"] == pytest.approx(expected_fe_pi_mt)
    assert result.diagnostics["mn_to_pig_iron_mt"] == pytest.approx(expected_mn_pi_mt)
    assert result.diagnostics["ti_to_pig_iron_mt"] == pytest.approx(expected_ti_pi_mt)
    assert result.diagnostics["metallic_mass_mt"] == pytest.approx(
        expected_metallic_mass_mt
    )
    assert result.theoretical_pig_iron_mt == pytest.approx(expected_theoretical_pi_mt)
    assert result.actual_pig_iron_mt == pytest.approx(expected_actual_pi_mt)
    assert result.slag_components_mt["feo"] == pytest.approx(expected_feo_mt)
    assert result.slag_components_mt["mno"] == pytest.approx(expected_mno_mt)
    assert result.total_slag_mt == pytest.approx(expected_feo_mt + expected_mno_mt)


def test_full_slag_balance_treats_fuel_s_and_p_on_dry_fuel_basis():
    fuel = FuelAshInput(
        fuel_id="coke",
        display_name="Coke",
        rate_kg_per_thm=100.0,
        moisture_pct=10.0,
        ash_pct=10.0,
        sio2_pct=50.0,
        s_pct=2.0,
        p_pct=1.0,
    )

    result = calculate_full_slag_balance(
        ores=[],
        quantities_mt={},
        hot_metal_mt=10.0,
        settings=SlagBalanceSettings(enabled=True, sulphur_pct=0.0),
        fuel_ash_inputs=[fuel],
    )

    assert result.fuel_ash_components_mt["sio2"] == pytest.approx(0.045)
    assert result.fuel_ash_components_mt["s"] == pytest.approx(0.018)
    assert result.fuel_ash_components_mt["p"] == pytest.approx(0.009)


def test_full_slag_balance_applies_slag_correction_factor():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=0.0, sio2_pct=10.0),
        )
    ]

    result = calculate_full_slag_balance(
        ores=ores,
        quantities_mt={"ore_a": 100.0},
        hot_metal_mt=60.0,
        settings=SlagBalanceSettings(
            enabled=True,
            fe_to_pig_iron_fraction=0.0,
            mn_recovery_pct=0.0,
            sulphur_gas_loss_pct=0.0,
            alkali_to_slag_fraction=0.8,
            slag_correction_factor=0.95,
        ),
    )

    assert result.diagnostics["raw_total_slag_mt"] == pytest.approx(10.0)
    assert result.diagnostics["slag_correction_factor"] == pytest.approx(0.95)
    assert result.slag_components_mt["sio2"] == pytest.approx(9.5)
    assert result.total_slag_mt == pytest.approx(9.5)


def test_blend_evaluation_scales_slag_source_diagnostics_with_correction_factor():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=0.0, sio2_pct=10.0),
        )
    ]

    blend = evaluate_blend(
        ores=ores,
        quantities_mt={"ore_a": 100.0},
        feo_in_slag_pct=0.0,
        slag_balance_settings=SlagBalanceSettings(
            enabled=True,
            fe_to_pig_iron_fraction=0.0,
            mn_recovery_pct=0.0,
            sulphur_gas_loss_pct=0.0,
            slag_correction_factor=0.95,
        ),
    )

    assert blend.slag_mt == pytest.approx(9.5)
    assert blend.diagnostics["ore_slag_mt"] == pytest.approx(9.5)
    assert blend.diagnostics["raw_ore_slag_mt"] == pytest.approx(10.0)
    assert blend.diagnostics["slag_contribution_mt_by_ore"]["ore_a"] == pytest.approx(
        9.5
    )
    assert blend.diagnostics["raw_slag_contribution_mt_by_ore"]["ore_a"] == (
        pytest.approx(10.0)
    )


def test_blend_evaluation_uses_full_slag_balance_when_enabled():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=60.0,
                sio2_pct=10.0,
                al2o3_pct=5.0,
                cao_pct=2.0,
                mgo_pct=1.0,
                na2o_pct=1.0,
                k2o_pct=1.0,
            ),
        )
    ]

    blend = evaluate_blend(
        ores=ores,
        quantities_mt={"ore_a": 100.0},
        feo_in_slag_pct=0.0,
        slag_balance_settings=SlagBalanceSettings(
            enabled=True,
            carbon_pct=4.0,
            silicon_pct=0.0,
            sulphur_pct=0.0,
            other_pct=0.0,
            fe_to_pig_iron_fraction=0.999,
            mn_recovery_pct=0.0,
            sulphur_gas_loss_pct=0.0,
            alkali_to_slag_fraction=0.8,
            fe_to_feo_factor=72.0 / 56.0,
        ),
    )

    expected_feo_mt = (60.0 - (60.0 * 0.999)) * (72.0 / 56.0)
    expected_slag_mt = 10.0 + 5.0 + 2.0 + 1.0 + expected_feo_mt + (2.0 * 0.8)
    assert blend.diagnostics["simplified_slag_mt"] == pytest.approx(20.0)
    assert blend.slag_mt == pytest.approx(expected_slag_mt)
    assert blend.diagnostics["full_slag_balance"]["total_slag_mt"] == pytest.approx(
        expected_slag_mt
    )


def test_full_slag_balance_deducts_bf_gas_dust_components():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=60.0,
                sio2_pct=10.0,
                al2o3_pct=5.0,
                cao_pct=2.0,
                mgo_pct=1.0,
            ),
        )
    ]
    dust = [
        DustInput(
            dust_id="bf_gas_dust",
            display_name="BF Gas Dust",
            wet_qty_mt=10.0,
            moisture_pct=0.0,
            sio2_pct=10.0,
        )
    ]

    result = calculate_full_slag_balance(
        ores=ores,
        quantities_mt={"ore_a": 100.0},
        hot_metal_mt=60.0,
        settings=SlagBalanceSettings(
            enabled=True,
            carbon_pct=4.0,
            silicon_pct=0.0,
            sulphur_pct=0.0,
            other_pct=0.0,
            fe_to_pig_iron_fraction=1.0,
            alkali_to_slag_fraction=0.8,
        ),
        dust_inputs=dust,
    )

    assert result.dust_components_mt["sio2"] == pytest.approx(1.0)
    assert result.net_into_bf_mt["sio2"] == pytest.approx(9.0)
    assert result.total_slag_mt == pytest.approx(9.0 + 5.0 + 2.0 + 1.0)


def test_lp_baseline_uses_dry_weight_fe_constraint():
    ores = [
        OreInput(
            ore_id="cheap_moist",
            display_name="CHEAP MOIST ORE",
            stock_mt=100.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=60.0, moisture_pct=10.0),
        ),
        OreInput(
            ore_id="rich_dry",
            display_name="RICH DRY ORE",
            stock_mt=100.0,
            price_rs_per_mt=5000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=70.0, moisture_pct=0.0),
        ),
    ]

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=55.0,
        target_slag_qty_mt=100.0,
        feo_in_slag_pct=0.0,
    )

    assert errors == []
    assert blend is not None
    assert blend.fe_production_mt >= 55.0 - 1e-6
    assert blend.quantities_mt["cheap_moist"] == pytest.approx(100.0)
    assert blend.quantities_mt["rich_dry"] >= 1.42


def test_lp_baseline_uses_slag_as_hard_cost_constraint():
    ores = [
        OreInput(
            ore_id="cheap_high_slag",
            display_name="CHEAP HIGH SLAG",
            stock_mt=500.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0, sio2_pct=20.0),
        ),
        OreInput(
            ore_id="costly_low_slag",
            display_name="COSTLY LOW SLAG",
            stock_mt=500.0,
            price_rs_per_mt=2000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0, sio2_pct=5.0),
        ),
    ]

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=50.0,
        target_slag_qty_mt=10.0,
        feo_in_slag_pct=0.0,
    )

    assert errors == []
    assert blend is not None
    assert blend.feasible is True
    assert blend.violations == []
    assert blend.fe_production_mt == pytest.approx(50.0, abs=0.5)
    assert blend.slag_mt <= 10.0 + 1e-6
    assert blend.quantities_mt["cheap_high_slag"] < 50.0
    assert blend.quantities_mt["costly_low_slag"] > 50.0


def test_evaluate_blend_reports_plant_slag_basicities():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=500.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=60.0,
                sio2_pct=4.0,
                al2o3_pct=2.0,
                cao_pct=3.0,
                mgo_pct=1.0,
            ),
        )
    ]
    fluxes = [
        FluxInput(
            flux_id="limestone",
            display_name="Limestone",
            wet_qty_mt=10.0,
            moisture_pct=0.0,
            cao_pct=50.0,
            mgo_pct=10.0,
        )
    ]

    blend = evaluate_blend(
        ores,
        {"ore_a": 100.0},
        feo_in_slag_pct=0.0,
        flux_inputs=fluxes,
    )

    assert blend.slag_basicity == pytest.approx(8.0 / 4.0)
    assert blend.slag_t_basicity == pytest.approx(10.0 / 4.0)
    assert blend.diagnostics["slag_basicity_numerator_mt"] == pytest.approx(8.0)
    assert blend.diagnostics["slag_basicity_denominator_mt"] == pytest.approx(4.0)
    assert blend.diagnostics["slag_t_basicity_numerator_mt"] == pytest.approx(10.0)
    assert blend.diagnostics["slag_t_basicity_denominator_mt"] == pytest.approx(4.0)


def test_blend_constraint_check_enforces_basicity_bounds():
    ores = [
        OreInput(
            ore_id="acid_ore",
            display_name="ACID ORE",
            stock_mt=500.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=60.0,
                sio2_pct=8.0,
                al2o3_pct=2.0,
                cao_pct=1.0,
                mgo_pct=1.0,
            ),
        )
    ]
    blend = evaluate_blend(
        ores,
        {"acid_ore": 100.0},
        feo_in_slag_pct=0.0,
    )

    violations = check_blend_constraints(
        blend,
        ores,
        target_production_mt=60.0,
        target_slag_qty_mt=100.0,
        target_slag_basicity_min=0.5,
        target_slag_basicity_max=2.0,
    )

    assert any("Slag basicity below bound" in violation for violation in violations)

    t_violations = check_blend_constraints(
        blend,
        ores,
        target_production_mt=60.0,
        target_slag_qty_mt=100.0,
        target_slag_t_basicity_min=0.5,
        target_slag_t_basicity_max=2.0,
    )

    assert any("Slag T Basicity below bound" in violation for violation in t_violations)


def test_lp_baseline_applies_slag_basicity_min_as_hard_constraint():
    ores = [
        OreInput(
            ore_id="cheap_acid",
            display_name="CHEAP ACID ORE",
            stock_mt=500.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0, sio2_pct=10.0),
        ),
        OreInput(
            ore_id="costly_basic",
            display_name="COSTLY BASIC ORE",
            stock_mt=500.0,
            price_rs_per_mt=2000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0, cao_pct=10.0),
        ),
    ]

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=50.0,
        target_slag_qty_mt=100.0,
        feo_in_slag_pct=0.0,
        target_slag_basicity_min=1.0,
        target_slag_basicity_max=10.0,
    )

    assert errors == []
    assert blend is not None
    assert blend.feasible is True
    assert blend.slag_basicity >= 1.0 - 1e-6
    assert blend.quantities_mt["costly_basic"] >= blend.quantities_mt["cheap_acid"]


def test_lp_baseline_ignores_t_basicity_bounds():
    ores = [
        OreInput(
            ore_id="balanced_ore",
            display_name="BALANCED ORE",
            stock_mt=500.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=50.0,
                sio2_pct=4.0,
                cao_pct=4.0,
                mgo_pct=0.0,
            ),
        )
    ]

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=50.0,
        target_slag_qty_mt=100.0,
        feo_in_slag_pct=0.0,
        target_slag_basicity_min=0.5,
        target_slag_basicity_max=1.5,
        target_slag_t_basicity_min=2.0,
        target_slag_t_basicity_max=3.0,
    )

    assert errors == []
    assert blend is not None
    assert blend.feasible is True
    assert blend.slag_basicity == pytest.approx(1.0)
    assert blend.slag_t_basicity < 2.0


def test_lp_baseline_does_not_return_exact_slag_violating_blend(monkeypatch):
    ores = [
        OreInput(
            ore_id="cheap_high_slag",
            display_name="CHEAP HIGH SLAG",
            stock_mt=500.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0, sio2_pct=20.0),
        )
    ]

    def underestimated_slag_terms(*args, **kwargs):
        zeros = np.zeros(len(ores), dtype=float)
        return zeros, 0.0, zeros, 0.0, zeros, 0.0, zeros, 0.0

    monkeypatch.setattr(
        lp_solver, "_build_linear_slag_and_basicity_terms", underestimated_slag_terms
    )

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=50.0,
        target_slag_qty_mt=10.0,
        feo_in_slag_pct=0.0,
    )

    assert blend is None
    assert any("exact slag" in error for error in errors)


def test_lp_baseline_treats_fuel_ash_slag_as_hard_constraint():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=500.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0),
        ),
        OreInput(
            ore_id="ore_b",
            display_name="ORE B",
            stock_mt=500.0,
            price_rs_per_mt=1200.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0),
        ),
    ]
    fuels = [
        FuelAshInput(
            fuel_id="coke",
            display_name="Coke",
            rate_kg_per_thm=1000.0,
            ash_pct=100.0,
            sio2_pct=1.0,
        )
    ]

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=100.0,
        target_slag_qty_mt=0.4,
        feo_in_slag_pct=0.0,
        fuel_ash_inputs=fuels,
    )

    assert blend is None
    assert any("LP infeasible" in error for error in errors)


def test_lp_baseline_treats_fixed_flux_slag_as_hard_constraint():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=500.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0),
        ),
        OreInput(
            ore_id="ore_b",
            display_name="ORE B",
            stock_mt=500.0,
            price_rs_per_mt=1200.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0),
        ),
    ]
    fluxes = [
        FluxInput(
            flux_id="limestone",
            display_name="Limestone",
            wet_qty_mt=10.0,
            moisture_pct=0.0,
            cao_pct=50.0,
        )
    ]

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=100.0,
        target_slag_qty_mt=4.0,
        feo_in_slag_pct=0.0,
        flux_inputs=fluxes,
    )

    assert blend is None
    assert any("LP infeasible" in error for error in errors)


def test_lp_baseline_explains_stock_and_share_infeasibility():
    # Only the sinter has stock and it is capped below a full burden, while the
    # other ores have zero stock. The opaque HiGHS infeasible should be enriched
    # with the stock/share coverage and Fe-capacity reasons.
    ores = [
        OreInput(
            ore_id="sinter",
            display_name="SINTER (SP-02)",
            stock_mt=2660.0,
            price_rs_per_mt=1000.0,
            min_share_pct=58.0,
            max_share_pct=70.0,
            chemistry=OreChemistry(fe_t_pct=55.0),
        ),
        OreInput(
            ore_id="geomin",
            display_name="GEOMIN CLO",
            stock_mt=0.0,
            price_rs_per_mt=1100.0,
            min_share_pct=0.0,
            max_share_pct=30.0,
            chemistry=OreChemistry(fe_t_pct=61.0),
        ),
    ]

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=2230.0,
        target_slag_qty_mt=750.0,
        feo_in_slag_pct=0.4,
    )

    assert blend is None
    assert any("LP infeasible" in error for error in errors)
    assert any("maximum shares sum to" in error for error in errors)
    assert any("can supply at most" in error and "MT Fe" in error for error in errors)


def test_lp_baseline_explains_min_share_on_zero_stock_ore():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=60.0),
        ),
        OreInput(
            ore_id="ore_b",
            display_name="ORE B",
            stock_mt=0.0,
            price_rs_per_mt=1100.0,
            min_share_pct=20.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=62.0),
        ),
    ]

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=1000.0,
        target_slag_qty_mt=5000.0,
        feo_in_slag_pct=0.4,
    )

    assert blend is None
    assert any(
        "ORE B" in error and "minimum share" in error and "zero stock" in error
        for error in errors
    )


def test_lp_baseline_attributes_infeasibility_to_basicity_bounds():
    # Ample stock and wide shares, but a CaO/SiO2 floor no ore-only blend can
    # reach (no flux supplies CaO). The explainer should name the basicity
    # bounds and report the achievable basicity.
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=5000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=60.0, sio2_pct=5.0, cao_pct=0.5),
        ),
        OreInput(
            ore_id="ore_b",
            display_name="ORE B",
            stock_mt=5000.0,
            price_rs_per_mt=1100.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=62.0, sio2_pct=4.0, cao_pct=0.4),
        ),
    ]

    blend, errors = run_lp_baseline(
        ores,
        target_production_mt=2230.0,
        target_slag_qty_mt=5000.0,
        feo_in_slag_pct=0.4,
        target_slag_basicity_min=1.0,
    )

    assert blend is None
    assert any("basicity bounds are the binding limit" in error for error in errors)


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
        target_production_mt=2100.0,
        target_slag_qty_mt=900.0,
        feo_in_slag_pct=0.4,
        model_service=model_service,
        process_context={"COKE RATE KG/THM": 338.0, "ACTUALKG/THM.": 171.0},
        history_df=history_df,
        penalty_cfg={},
    )
    result = evaluator.evaluate_quantities(np.array([2280.0, 1520.0], dtype=float))
    assert np.isfinite(result.objective_value)
    assert "base_objective_rs_per_thm" in result.components
    assert "blend" in result.diagnostics


def test_bmo_objective_penalizes_slag_basicity_violation():
    ores = [
        OreInput(
            ore_id="acid_ore",
            display_name="ACID ORE",
            stock_mt=500.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=50.0,
                sio2_pct=10.0,
                cao_pct=1.0,
            ),
        )
    ]
    model_service = FuelUnitCostModelService(
        bundle_cfg={
            "model_path": "src/assets/models/bmo_fuel/definitely_missing_model.joblib",
            "scaler_path": "src/assets/models/bmo_fuel/definitely_missing_scaler.joblib",
        },
        fallback_cfg={},
    )
    evaluator = BmoObjectiveEvaluator(
        ores=ores,
        target_production_mt=50.0,
        target_slag_qty_mt=100.0,
        target_slag_basicity_min=0.5,
        target_slag_basicity_max=2.0,
        feo_in_slag_pct=0.0,
        model_service=model_service,
        process_context={},
        history_df=pd.DataFrame(),
        penalty_cfg={"penalty_basicity": 1000.0},
    )

    result = evaluator.evaluate_quantities(np.array([100.0], dtype=float))

    assert result.components["penalty_slag_basicity"] > 0.0
    assert result.feasible is False
    assert any("Slag basicity below bound" in item for item in result.violations)

    t_evaluator = BmoObjectiveEvaluator(
        ores=ores,
        target_production_mt=50.0,
        target_slag_qty_mt=100.0,
        target_slag_t_basicity_min=0.5,
        target_slag_t_basicity_max=2.0,
        feo_in_slag_pct=0.0,
        model_service=model_service,
        process_context={},
        history_df=pd.DataFrame(),
        penalty_cfg={"penalty_basicity": 1000.0},
    )
    t_result = t_evaluator.evaluate_quantities(np.array([100.0], dtype=float))

    assert t_result.components["penalty_slag_t_basicity"] == pytest.approx(0.0)
    assert t_result.feasible is True
    assert not any("Slag T Basicity" in item for item in t_result.violations)


def test_bmo_objective_penalizes_production_above_target():
    ores = [
        OreInput(
            ore_id="ore_a",
            display_name="ORE A",
            stock_mt=1000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0),
        )
    ]
    model_service = FuelUnitCostModelService(
        bundle_cfg={
            "model_path": "src/assets/models/bmo_fuel/definitely_missing_model.joblib",
            "scaler_path": "src/assets/models/bmo_fuel/definitely_missing_scaler.joblib",
        },
        fallback_cfg={},
    )
    evaluator = BmoObjectiveEvaluator(
        ores=ores,
        target_production_mt=50.0,
        target_slag_qty_mt=1000.0,
        feo_in_slag_pct=0.0,
        model_service=model_service,
        process_context={},
        history_df=pd.DataFrame(),
        penalty_cfg={
            "penalty_fe": 1000.0,
            "penalty_production_excess": 1000.0,
        },
    )

    exact_result = evaluator.evaluate_quantities(np.array([100.0], dtype=float))
    over_result = evaluator.evaluate_quantities(np.array([120.0], dtype=float))

    assert exact_result.components["penalty_production_excess"] == pytest.approx(0.0)
    assert over_result.components["penalty_production_excess"] == pytest.approx(9500.0)
    assert over_result.feasible is False
    assert any(
        "Fe production above required target" in violation
        for violation in over_result.violations
    )
    assert over_result.objective_value > exact_result.objective_value


def test_blend_constraint_check_tolerates_small_fe_rounding_delta():
    ore = OreInput(
        ore_id="ore_a",
        display_name="ORE A",
        stock_mt=5000.0,
        price_rs_per_mt=6000.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(fe_t_pct=62.0),
    )
    blend = BlendEvaluation(
        quantities_mt={"ore_a": 3800.0},
        shares_pct={"ore_a": 100.0},
        total_qty_mt=3800.0,
        ore_cost_total_rs=0.0,
        ore_cost_per_thm_rs=0.0,
        fuel_cost_per_thm_rs=0.0,
        objective_rs_per_thm=0.0,
        fe_t_pct=62.0,
        effective_fe_pct=62.0,
        fe_production_mt=2349.9999,
        slag_pct=0.0,
        slag_mt=0.0,
        feasible=True,
        violations=[],
    )

    assert (
        check_blend_constraints(
            blend,
            [ore],
            target_production_mt=2350.0,
            target_slag_qty_mt=750.0,
        )
        == []
    )


def test_selected_pellet_validation_blocks_fallback_and_stale_inputs():
    pellet = OreInput(
        ore_id="pellet_1",
        display_name="LLOYDS PELLET",
        stock_mt=250.0,
        price_rs_per_mt=9650.0,
        min_share_pct=0.0,
        max_share_pct=10.0,
        chemistry=OreChemistry(fe_t_pct=64.0),
        metadata={
            "material_key": "pellet_1",
            "stock_source": "fallback",
            "chemistry_source": "offline_db_latest",
            "chemistry_sample_timestamp": "2026-01-01T00:00:00+00:00",
        },
    )

    issues = validate_selected_pellet_inputs(
        [pellet],
        max_chemistry_age_days=30,
        now=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )

    assert any("stock is not from raw_material_stock" in issue for issue in issues)
    assert any("chemistry sample is" in issue for issue in issues)


def test_selected_pellet_validation_accepts_fresh_database_inputs():
    pellet = OreInput(
        ore_id="pellet_1",
        display_name="LLOYDS PELLET",
        stock_mt=250.0,
        price_rs_per_mt=9650.0,
        min_share_pct=0.0,
        max_share_pct=10.0,
        chemistry=OreChemistry(fe_t_pct=64.0),
        metadata={
            "material_key": "pellet_1",
            "stock_source": "offline_db",
            "chemistry_source": "offline_db_latest",
            "chemistry_sample_timestamp": "2026-02-20T00:00:00+00:00",
        },
    )

    assert (
        validate_selected_pellet_inputs(
            [pellet],
            max_chemistry_age_days=30,
            now=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        == []
    )


def test_blend_constraint_check_flags_hot_metal_above_target():
    ore = OreInput(
        ore_id="ore_a",
        display_name="ORE A",
        stock_mt=5000.0,
        price_rs_per_mt=6000.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(fe_t_pct=62.0),
    )
    blend = BlendEvaluation(
        quantities_mt={"ore_a": 3800.0},
        shares_pct={"ore_a": 100.0},
        total_qty_mt=3800.0,
        ore_cost_total_rs=0.0,
        ore_cost_per_thm_rs=0.0,
        fuel_cost_per_thm_rs=0.0,
        objective_rs_per_thm=0.0,
        fe_t_pct=62.0,
        effective_fe_pct=62.0,
        fe_production_mt=2351.0,
        slag_pct=0.0,
        slag_mt=0.0,
        feasible=True,
        violations=[],
    )

    violations = check_blend_constraints(
        blend,
        [ore],
        target_production_mt=2350.0,
        target_slag_qty_mt=750.0,
    )

    assert any(
        "Fe production above required target" in violation for violation in violations
    )


def test_blend_constraint_check_uses_strict_slag_cap():
    ore = OreInput(
        ore_id="ore_a",
        display_name="ORE A",
        stock_mt=5000.0,
        price_rs_per_mt=6000.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(fe_t_pct=62.0),
    )
    blend = BlendEvaluation(
        quantities_mt={"ore_a": 3800.0},
        shares_pct={"ore_a": 100.0},
        total_qty_mt=3800.0,
        ore_cost_total_rs=0.0,
        ore_cost_per_thm_rs=0.0,
        fuel_cost_per_thm_rs=0.0,
        objective_rs_per_thm=0.0,
        fe_t_pct=62.0,
        effective_fe_pct=62.0,
        fe_production_mt=2350.0,
        slag_pct=0.0,
        slag_mt=800.01,
        feasible=True,
        violations=[],
    )

    violations = check_blend_constraints(
        blend,
        [ore],
        target_production_mt=2350.0,
        target_slag_qty_mt=800.0,
    )

    assert any("Slag exceeds bound" in violation for violation in violations)


def test_nonlinear_optimizer_respects_stock_caps_and_keeps_feasible_baseline():
    ores = [
        OreInput(
            ore_id="sinter",
            display_name="SINTER",
            stock_mt=70.0,
            price_rs_per_mt=7000.0,
            min_share_pct=50.0,
            max_share_pct=70.0,
            chemistry=OreChemistry(fe_t_pct=55.0, sio2_pct=5.0, al2o3_pct=2.0),
        ),
        OreInput(
            ore_id="rich",
            display_name="RICH ORE",
            stock_mt=50.0,
            price_rs_per_mt=8000.0,
            min_share_pct=0.0,
            max_share_pct=50.0,
            chemistry=OreChemistry(fe_t_pct=70.0, sio2_pct=4.0, al2o3_pct=2.0),
        ),
        OreInput(
            ore_id="cheap_empty",
            display_name="CHEAP EMPTY",
            stock_mt=0.0,
            price_rs_per_mt=10.0,
            min_share_pct=0.0,
            max_share_pct=50.0,
            chemistry=OreChemistry(fe_t_pct=80.0, sio2_pct=4.0, al2o3_pct=2.0),
        ),
    ]
    model_service = FuelUnitCostModelService(
        bundle_cfg={
            "model_path": "src/assets/models/bmo_fuel/definitely_missing_model.joblib",
            "scaler_path": "src/assets/models/bmo_fuel/definitely_missing_scaler.joblib",
        },
        fallback_cfg={},
    )

    blend, errors = run_nonlinear_optimizer(
        ores,
        target_production_mt=58.0,
        target_slag_qty_mt=30.0,
        feo_in_slag_pct=0.4,
        model_service=model_service,
        process_context={},
        history_df=pd.DataFrame(),
        de_cfg={"maxiter": 1, "popsize": 2, "polish": False, "seed": 42},
    )

    assert errors == []
    assert blend is not None
    assert blend.feasible is True
    assert blend.violations == []
    assert blend.quantities_mt["cheap_empty"] == pytest.approx(0.0)
    assert blend.fe_production_mt >= 58.0
    runtime = blend.diagnostics["runtime"]["best_solution"]["diagnostics"]
    assert "candidate_shares_pct" in runtime
    assert sum(runtime["candidate_shares_pct"]) == pytest.approx(100.0)


def test_nonlinear_optimizer_skips_de_when_lp_constraints_are_infeasible():
    ores = [
        OreInput(
            ore_id="low_fe",
            display_name="LOW FE",
            stock_mt=100.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=50.0),
        ),
        OreInput(
            ore_id="medium_fe",
            display_name="MEDIUM FE",
            stock_mt=100.0,
            price_rs_per_mt=1200.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=60.0),
        ),
    ]
    model_service = FuelUnitCostModelService(
        bundle_cfg={
            "model_path": "src/assets/models/bmo_fuel/definitely_missing_model.joblib",
            "scaler_path": "src/assets/models/bmo_fuel/definitely_missing_scaler.joblib",
        },
        fallback_cfg={},
    )

    blend, errors = run_nonlinear_optimizer(
        ores,
        target_production_mt=130.0,
        target_slag_qty_mt=100.0,
        feo_in_slag_pct=0.4,
        model_service=model_service,
        process_context={},
        history_df=pd.DataFrame(),
        de_cfg={"maxiter": 50, "popsize": 20, "polish": False, "seed": 42},
    )

    assert blend is None
    assert (
        errors[0]
        == "Total-cost optimizer skipped because hard LP constraints are infeasible."
    )
    assert any("LP infeasible" in error for error in errors)
