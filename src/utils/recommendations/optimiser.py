"""Differential-evolution optimiser for V-OptimAIse recommendations.

This module now uses the shared optimisation runtime for:
- model/scaler bundle loading,
- feature vector assembly contract alignment,
- DE orchestration with structured objective diagnostics.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config.config_loader import load_config
from domain.optimization_runtime import ModelBundleService, OptimizerRunner
from utils.logger import setup_logger
from utils.recommendations.bounds import get_control_bounds
from utils.recommendations.data import DataframesProcessor
from utils.recommendations.dependencies import build_bf_dependency_graph
from utils.recommendations.features import extract_scaler_params
from utils.recommendations.objective import RecommendationObjectiveEvaluator

logger = setup_logger()
config_vsense = load_config("setting_vsense.yml")


def run_optimiser(
    df: pd.DataFrame,
    models_dict: Dict[str, Any],
    user_input: Dict[str, float],
    fixed_cp: Dict[str, float],
    dfprocessor: DataframesProcessor | None = None,
    lambda_reg: float = 0.1,
    impute_lags: bool = True,
) -> Dict[str, float]:
    """Run DE optimisation and return recommended controls + KPI deltas."""
    df_feat_vec = df.copy()
    for model in models_dict.keys():
        if models_dict[model]["Optimised"] is True:
            target_output = models_dict[model]["output_param"]
            optimisation_type = model
            maxmin = models_dict[model]["maxmin"]
            control_params = models_dict[model]["control_params"]

    for key, value in user_input.items():
        if not np.isnan(value):
            df_feat_vec.at[df_feat_vec.index[-1], key] = value

    free_cp = [cp for cp in control_params if cp not in fixed_cp]

    for key, value in fixed_cp.items():
        if not np.isnan(value):
            df_feat_vec.at[df_feat_vec.index[-1], key] = value

    scaler_path = models_dict[optimisation_type]["scaling"]
    bundle = ModelBundleService({"scaler_path": scaler_path}).get_bundle()
    scaler = bundle.scaler
    if scaler is None:
        raise ValueError(f"Scaler missing for optimisation type '{optimisation_type}'")
    models_dict[optimisation_type]["LoadedScaler"] = scaler

    local_feature_names = list(scaler.feature_names_in_)
    if impute_lags:
        for feat in local_feature_names:
            if "_lag" in feat and feat.split("_lag")[0] in control_params:
                free_cp.append(feat)

    # Bounds should be based on historical full dataset, not only one-row feature vector.
    bounds_source_df = dfprocessor.df_full if dfprocessor is not None else df
    bounds = get_control_bounds(bounds_source_df, free_cp, impute_lags=impute_lags)

    local_feat_vec = df_feat_vec.copy()
    loaded_model = models_dict[optimisation_type]["LoadedMLModel"]

    offsets, scales = extract_scaler_params(scaler)

    if target_output in local_feature_names:
        target_idx = local_feature_names.index(target_output)
        local_feature_names.pop(local_feature_names.index(target_output))
        if target_output in local_feat_vec.columns:
            local_feat_vec = local_feat_vec.drop(columns=[target_output])
    else:
        for feature in local_feature_names:
            if target_output in feature:
                feature_idx = local_feature_names.index(feature)
                target_idx = len(offsets)
                offsets = np.append(offsets, offsets[feature_idx])
                scales = np.append(scales, scales[feature_idx])
                break
        else:
            raise KeyError(
                f"Target '{target_output}' not found in feature names for optimisation."
            )

    scaler_index = {name: i for i, name in enumerate(local_feature_names)}
    feature_idx = np.array([scaler_index.get(f, -1) for f in local_feature_names])

    if local_feat_vec.columns.tolist() != local_feature_names:
        raise AssertionError(
            "Feature names mismatch: "
            f"{local_feat_vec.columns.tolist()} vs {local_feature_names}"
        )
    if len(feature_idx) != len(local_feature_names):
        raise AssertionError(
            f"Feature index length mismatch: {len(feature_idx)} vs {len(local_feature_names)}"
        )

    base_row = local_feat_vec.iloc[-1][local_feature_names].to_numpy(float)
    base_scaled = (base_row - offsets[feature_idx]) / scales[feature_idx]

    mask = feature_idx == -1
    base_scaled[mask] = 0.0

    free_idx = [local_feature_names.index(cp) for cp in free_cp]
    control_idx = [local_feature_names.index(cp) for cp in control_params]
    scaled_prev_params = base_scaled[control_idx].copy()

    evaluator = RecommendationObjectiveEvaluator(
        base_row=base_row,
        base_scaled=base_scaled,
        free_idx=free_idx,
        control_idx=control_idx,
        scaled_prev_params=scaled_prev_params,
        offsets=offsets,
        scales=scales,
        feature_idx=feature_idx,
        model=loaded_model,
        lambda_reg=float(lambda_reg),
        maxmin=float(maxmin),
    )

    optimizer_cfg = {
        "strategy": "best1bin",
        "polish": True,
        "popsize": 15,
        "tol": 0.01,
        "maxiter": int(config_vsense.get("OPTIM_STEPS", 20)),
        "seed": 42,
    }
    runner = OptimizerRunner(optimizer_cfg)
    optimization_result = runner.run_differential_evolution(
        bounds=bounds,
        objective_fn=evaluator.evaluate,
    )

    if not optimization_result.best_solution.get("x"):
        raise RuntimeError(
            "Recommendation optimizer failed to find a valid solution: "
            f"{optimization_result.diagnostics}"
        )

    optimal_free_cp = dict(zip(free_cp, optimization_result.best_solution["x"]))
    optimal_cp = {**fixed_cp, **optimal_free_cp}

    row_prev = df_feat_vec.iloc[-1].copy()
    row_curr = row_prev.copy()
    for key, value in optimal_cp.items():
        if key in row_curr.index:
            row_curr[key] = value

    raw_prev = row_prev[local_feature_names].to_numpy(float)
    raw_curr = row_curr[local_feature_names].to_numpy(float)

    scaled_prev = (raw_prev - offsets[feature_idx]) / scales[feature_idx]
    scaled_curr = (raw_curr - offsets[feature_idx]) / scales[feature_idx]

    mask = feature_idx == -1
    scaled_prev[mask] = 0.0
    scaled_curr[mask] = 0.0

    y_pred_scaled_prev = loaded_model.predict(scaled_prev.reshape(1, -1))[0]
    y_pred_scaled_curr = loaded_model.predict(scaled_curr.reshape(1, -1))[0]

    y_pred_prev = y_pred_scaled_prev * scales[target_idx] + offsets[target_idx]
    y_pred_curr = y_pred_scaled_curr * scales[target_idx] + offsets[target_idx]

    optimal_cp[target_output + "_previous"] = y_pred_prev
    optimal_cp[target_output + "_current"] = y_pred_curr

    try:
        dep_graph = build_bf_dependency_graph()
        dep_prev = dep_graph.apply(row_prev)
        dep_curr = dep_graph.apply(row_curr)
        for dep_name in dep_graph.names():
            optimal_cp[f"{dep_name}_dep_previous"] = float(dep_prev[dep_name])
            optimal_cp[f"{dep_name}_dep_current"] = float(dep_curr[dep_name])
    except Exception as err:
        logger.info(f"[dependencies] skipped dependent-variable calc: {err}")

    for model_name, model_dict in models_dict.items():
        if model_dict["Optimised"]:
            continue

        impact_target = model_dict["output_param"]
        impact_model = model_dict["LoadedMLModel"]
        impact_scaler_path = model_dict["scaling"]

        impact_bundle = ModelBundleService({"scaler_path": impact_scaler_path}).get_bundle()
        impact_scaler = impact_bundle.scaler
        if impact_scaler is None:
            logger.info(
                f"[impact] skipped '{model_name}' because scaler is unavailable at "
                f"{impact_scaler_path}"
            )
            continue

        if dfprocessor is None:
            logger.info(
                f"[impact] skipped '{model_name}' because dfprocessor context is unavailable"
            )
            continue
        df_impact_full = dfprocessor.process_dataframe(impact_scaler_path)
        row_prev_imp = df_impact_full.iloc[-1].copy()
        row_curr_imp = row_prev_imp.copy()
        for key, value in optimal_cp.items():
            if key in row_curr_imp.index:
                row_curr_imp[key] = value

        impact_feature_names = list(impact_scaler.feature_names_in_)
        offsets_imp, scales_imp = extract_scaler_params(impact_scaler)

        if impact_target in impact_feature_names:
            target_idx_imp = impact_feature_names.index(impact_target)
            impact_feature_names.remove(impact_target)
        else:
            for feature in impact_feature_names:
                if impact_target in feature:
                    targetproxy_feature_idx = impact_feature_names.index(feature)
                    target_idx_imp = len(offsets_imp)
                    offsets_imp = np.append(
                        offsets_imp, offsets_imp[targetproxy_feature_idx]
                    )
                    scales_imp = np.append(scales_imp, scales_imp[targetproxy_feature_idx])
                    break
            else:
                raise KeyError(
                    f"Impact target '{impact_target}' not found in feature names "
                    f"for model '{model_name}'."
                )

        raw_prev_imp = row_prev_imp[impact_feature_names].to_numpy(float)
        raw_curr_imp = row_curr_imp[impact_feature_names].to_numpy(float)

        impact_index_map = {
            name: i for i, name in enumerate(impact_scaler.feature_names_in_)
        }
        feature_idx_imp = np.array(
            [impact_index_map.get(f, -1) for f in impact_feature_names], dtype=int
        )

        scaled_prev_imp = (raw_prev_imp - offsets_imp[feature_idx_imp]) / scales_imp[
            feature_idx_imp
        ]
        scaled_curr_imp = (raw_curr_imp - offsets_imp[feature_idx_imp]) / scales_imp[
            feature_idx_imp
        ]

        mask_imp = feature_idx_imp == -1
        scaled_prev_imp[mask_imp] = 0.0
        scaled_curr_imp[mask_imp] = 0.0

        y_imp_scaled_prev = impact_model.predict(scaled_prev_imp.reshape(1, -1))[0]
        y_imp_scaled_curr = impact_model.predict(scaled_curr_imp.reshape(1, -1))[0]

        target_offset_imp = offsets_imp[target_idx_imp]
        target_scale_imp = scales_imp[target_idx_imp]

        y_imp_prev = y_imp_scaled_prev * target_scale_imp + target_offset_imp
        y_imp_curr = y_imp_scaled_curr * target_scale_imp + target_offset_imp

        optimal_cp[impact_target + "_previous"] = float(y_imp_prev)
        optimal_cp[impact_target + "_current"] = float(y_imp_curr)

    return optimal_cp


def debug_callback(xc_trial: np.ndarray, convergence: bool) -> None:
    """Debug callback helper for local experiments."""
    print(f"Current trial values: {xc_trial}, Convergence: {convergence}")
