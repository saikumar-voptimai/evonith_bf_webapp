import numpy as np
import pandas as pd
import time
import joblib
from scipy.optimize import differential_evolution
from utils.recommendations.data import DataframesProcessor
from utils.recommendations.bounds import get_control_bounds
from utils.recommendations.features import extract_scaler_params
from utils.recommendations.dependencies import build_bf_dependency_graph
from typing import Dict, List, Any
from config.config_loader import load_config
from utils.logger import setup_logger

logger = setup_logger()

config_vsense = load_config('setting_vsense.yml')

def _log_profile(message: str) -> None:
    """
    Cheap logger to stderr so Streamlit output is not polluted too much.
    """
    logger.info(message)

def objective(
    xc_trial: np.ndarray,
    base_row: pd.Series,
    base_scaled: np.ndarray,
    free_idx: List[int],
    control_idx: List[int],
    scaled_prev_params: np.ndarray,
    offsets: np.ndarray,
    scales: np.ndarray,
    feature_idx: np.ndarray,
    model,
    lambda_reg: float,
    maxmin: float
) -> float:
    """
    Objective function combining the predicted target output and a penalty term
    for deviation from previous control parameters, both in scaled space.

    Args:
        xc_trial (np.ndarray): Array of free control parameter values (to be optimized).
        df_feat_vec (pd.DataFrame): DataFrame with feature vectors.
        fixed_cp (Dict[str, float]): Fixed control parameters. 
        models_dict (Dict): Pre-trained regression model.
        lambda_reg (float): Regularization strength.
        free_cp (List[str]): Names of control params being optimized (precomputed).
        scaled_prev_params (np.ndarray): Scaled previous control parameters for penalty (precomputed).
    Returns:
        float: Combined objective value (predicted output + penalty).
    """
    t0 = time.perf_counter()
    x_raw = base_row.copy()
    x_raw[free_idx] = xc_trial
    x_scaled = base_scaled.copy()

    raw_vals = x_raw[free_idx]
    idx = feature_idx[free_idx]
    scaled_vals = (raw_vals - offsets[idx]) / scales[idx]
    x_scaled[free_idx] = scaled_vals
    t_scale = time.perf_counter()

    y_pred_scaled = model.predict(x_scaled.reshape(1,-1))[0] * maxmin
    t_pred = time.perf_counter()

    # Penalty on scaled control parameters deviation from previous
    penalty = np.sum((x_scaled[control_idx] - scaled_prev_params) ** 2)
    
    t_pen = time.perf_counter()
    _log_profile(
        f"[objective] total={t_pen - t0:8.4f}s | "
        f"scale_full={t_scale - t0:7.4f}s, "
        f"predict={t_pred - t_scale:7.4f}s, "
        f"penalty={t_pen - t_pred:7.4f}s"
    )
    
    return float(y_pred_scaled + lambda_reg * penalty)

def run_optimiser(
    df: pd.DataFrame, # will have feature vectors as per the required scaler.feature_names_in
    models_dict: Dict[str, Any],
    user_input: Dict[str, float],
    fixed_cp: Dict[str, float],
    dfprocessor: DataframesProcessor,
    lambda_reg: float = 0.1,
) -> Dict[str, float]:
    """
    Runs optimization to find control parameters minimizing target output.
    Args:
        df (pd.DataFrame): DataFrame with feature vectors as per the required scaler.feature_names_in.
        models_dict (Dict[str, Any]): Each output-input-control combination and pre-trained regression models.
        user_input (Dict[str, float]): User-specified input parameters.
        fixed_cp (Dict[str, float]): Fixed control parameters.
        lambda_reg (float): Regularization strength for penalty term.
    Returns:
        Dict[str, float]: Optimal control parameters including the predicted target output.
    """
    df_feat_vec = df.copy()
    for model in models_dict.keys():
        if models_dict[model]['Optimised'] == True:
            target_output = models_dict[model]['output_param']
            optimisation_type = model
            maxmin = models_dict[model]['maxmin']
            control_params = models_dict[model]['control_params']
        
    for key, value in user_input.items():
        if not np.isnan(value):
            df_feat_vec.at[df_feat_vec.index[-1], key] = value

    free_cp = [cp for cp in control_params if cp not in fixed_cp]

    for key, value in fixed_cp.items():
        if not np.isnan(value):
            df_feat_vec.at[df_feat_vec.index[-1], key] = value

    # Load scaler for the target output
    scaler_path = models_dict[optimisation_type]['scaling']
    scaler = joblib.load(scaler_path)
    models_dict[optimisation_type]['LoadedScaler'] = scaler
        
    # Bounds
    bounds = get_control_bounds(df, free_cp )
    
    local_feat_vec = df_feat_vec
    loaded_model = models_dict[optimisation_type]['LoadedMLModel']
    
    # Precompute scaled control params
    offsets, scales = extract_scaler_params(scaler)

    local_feature_names = scaler.feature_names_in_.tolist()
    if target_output in local_feature_names:
        target_idx = local_feature_names.index(target_output)
        target_value = local_feat_vec.iloc[-1][target_output]
        local_feature_names.pop(local_feature_names.index(target_output))
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
    scaler_index = {name: i for i, name in enumerate(scaler.feature_names_in_)}
    feature_idx = np.array([scaler_index.get(f, -1) for f in local_feature_names])

    assert local_feat_vec.columns.tolist() == local_feature_names, \
        f"Feature names mismatch: {local_feat_vec.columns.tolist()} vs {local_feature_names}"
    assert len(feature_idx) == len(local_feature_names), \
        f"Feature index length mismatch: {len(feature_idx)} vs {len(local_feature_names)}"
    base_row = local_feat_vec.iloc[-1][local_feature_names].to_numpy(float)
    base_scaled = (base_row - offsets[feature_idx]) / scales[feature_idx]
    
    mask = (feature_idx == -1)
    base_scaled[mask] = 0.0

    free_idx = [local_feature_names.index(cp) for cp in free_cp]
    control_idx = [local_feature_names.index(cp) for cp in control_params]
    
    scaled_prev_params = base_scaled[control_idx].copy()
    
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        args=(
            base_row,
            base_scaled,
            free_idx,
            control_idx,
            scaled_prev_params,
            offsets,
            scales,
            feature_idx,
            loaded_model,
            lambda_reg,
            maxmin
            ),
        strategy='best1bin',
        polish=True,
        popsize=15,
        tol=0.01,
        maxiter=20,
        workers=1
    )

    optimal_free_cp = dict(zip(free_cp, result.x))
    optimal_cp = {**fixed_cp, **optimal_free_cp}

    # Previous row (before optimisation) in raw space
    row_prev = df_feat_vec.iloc[-1].copy()
    # Current row: apply optimal controls
    row_curr = row_prev.copy()
    for key, value in optimal_cp.items():
        if key in row_curr.index:
            row_curr[key] = value
    # Build raw feature vectors in the same order as local_feature_names
    raw_prev = row_prev[local_feature_names].to_numpy(float)
    raw_curr = row_curr[local_feature_names].to_numpy(float)

    scaled_prev = (raw_prev - offsets[feature_idx]) / scales[feature_idx]
    scaled_curr = (raw_curr - offsets[feature_idx]) / scales[feature_idx]

   # Zero out any features not in scaler (feature_idx == -1)
    mask = (feature_idx == -1)
    scaled_prev[mask] = 0.0
    scaled_curr[mask] = 0.0

    # Model predictions in scaled target space
    y_pred_scaled_prev = loaded_model.predict(scaled_prev.reshape(1,-1))[0]
    y_pred_scaled_curr = loaded_model.predict(scaled_curr.reshape(1,-1))[0]

    # Inverse-transform to original target units
    y_pred_prev = y_pred_scaled_prev * scales[target_idx] + offsets[target_idx]
    y_pred_curr = y_pred_scaled_curr * scales[target_idx] + offsets[target_idx]

    optimal_cp[target_output + "_previous"] = y_pred_prev
    optimal_cp[target_output + "_current"] = y_pred_curr

    # ------------------------------------------------------------------
    # Optional: compute dependent variables from the optimal knobs
    # (does not affect optimisation; just for explainability/impact display)
    # ------------------------------------------------------------------
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
        impact_scaler = joblib.load(impact_scaler_path)
        # Build feature dataframe for this impact model
        df_impact_full = dfprocessor.process_dataframe(impact_scaler_path)

        # Previous row in this model's feature space
        row_prev_imp = df_impact_full.iloc[-1].copy()
        # Current row: apply optimal controls
        row_curr_imp = row_prev_imp.copy()
        for key, value in optimal_cp.items():
            if key in row_curr_imp.index:
                row_curr_imp[key] = value

        # Feature names for this model
        impact_feat_vec = df_impact_full.iloc[-1]
        impact_feature_names = impact_scaler.feature_names_in_.tolist()
        offsets_imp, scales_imp = extract_scaler_params(impact_scaler)

        # Drop the impact target for inference
        if impact_target in impact_feature_names:
            target_idx_imp = impact_feature_names.index(impact_target)
            impact_feature_names.remove(impact_target)
        else:
            for feature in impact_feature_names:
                if impact_target in feature:
                    targetproxy_feature_idx = impact_feature_names.index(feature)
                    target_idx_imp = len(offsets_imp)
                    offsets_imp = np.append(offsets_imp, offsets_imp[targetproxy_feature_idx])
                    scales_imp = np.append(scales_imp, scales_imp[targetproxy_feature_idx])
                    break
            else:
                raise KeyError(
                    f"Impact target '{impact_target}' not found in feature names "
                    f"for model '{model_name}'."
                )

        # Raw feature vectors
        raw_prev_imp = row_prev_imp[impact_feature_names].to_numpy(float)
        raw_curr_imp = row_curr_imp[impact_feature_names].to_numpy(float)

        # Scale using this model's scaler
        impact_index_map = {name: i for i, name in enumerate(impact_scaler.feature_names_in_)}
        feature_idx_imp = np.array(
            [impact_index_map.get(f, -1) for f in impact_feature_names], dtype=int
        )

        scaled_prev_imp = (raw_prev_imp - offsets_imp[feature_idx_imp]) / scales_imp[feature_idx_imp]
        scaled_curr_imp = (raw_curr_imp - offsets_imp[feature_idx_imp]) / scales_imp[feature_idx_imp]

        mask_imp = feature_idx_imp == -1
        scaled_prev_imp[mask_imp] = 0.0
        scaled_curr_imp[mask_imp] = 0.0

        # Predict in scaled target space
        y_imp_scaled_prev = impact_model.predict(
            scaled_prev_imp.reshape(1, -1)
            )[0]
        y_imp_scaled_curr = impact_model.predict(
            scaled_curr_imp.reshape(1, -1)
            )[0]

        target_offset_imp = offsets_imp[target_idx_imp]
        target_scale_imp = scales_imp[target_idx_imp]

        y_imp_prev = y_imp_scaled_prev * target_scale_imp + target_offset_imp
        y_imp_curr = y_imp_scaled_curr * target_scale_imp + target_offset_imp

        optimal_cp[impact_target + "_previous"] = float(y_imp_prev)
        optimal_cp[impact_target + "_current"] = float(y_imp_curr)

    return optimal_cp


def debug_callback(xc_trial: np.ndarray, convergence: bool) -> None:
    """
    Debug callback function to print the current trial values.
    
    Args:
        xc_trial (np.ndarray): Current trial values of control parameters.
        convergence (bool): Whether the optimization has converged.
    """
    print(f"Current trial values: {xc_trial}, Convergence: {convergence}")