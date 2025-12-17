import numpy as np
import pandas as pd
import sys, time
from scipy.optimize import differential_evolution
from utils.recommendations import get_control_bounds, load_scaler, scale_features, inverse_transform_output, extract_scaler_params
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
    
    return y_pred_scaled + lambda_reg * penalty * 0.0

def run_optimiser(
    df: pd.DataFrame,
    models_dict: Dict[str, Any],
    user_input: Dict[str, float],
    fixed_cp: Dict[str, float],
    lambda_reg: float = 0.1,
) -> Dict[str, float]:
    """
    Runs optimization to find control parameters minimizing target output.
    Args:
        df (pd.DataFrame): DataFrame with feature vectors.
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
        else:
            df_feat_vec.drop(columns=[col for col in list(df_feat_vec.columns) if models_dict[model]['output_param'] in col], inplace=True)

    for key, value in user_input.items():
        if not np.isnan(value):
            df_feat_vec.at[df_feat_vec.index[-1], key] = value

    free_cp = [cp for cp in control_params if cp not in fixed_cp]

    for key, value in fixed_cp.items():
        if not np.isnan(value):
            df_feat_vec.at[df_feat_vec.index[-1], key] = value

    # Load scaler for the target output
    scaler_path = models_dict[optimisation_type]['scaling']
    scaler = load_scaler(scaler_path)
    models_dict[optimisation_type]['LoadedScaler'] = scaler
    feature_names = df.columns.tolist()
    for model in models_dict.keys():
        local_features = feature_names.copy()
        ip_cp_params = models_dict[model]['input_params_flat'] + \
                        models_dict[model]['control_params'] + \
                        [models_dict[model]['output_param']] + \
                        ['hour', 'month', 'day_of_week']
        # Remove all lagged feature vectors of other impact_targets
        for feature in feature_names:
            non_lagged_feature_name = feature.split('_lag')[0]
            if non_lagged_feature_name not in ip_cp_params:
                local_features.remove(feature)
        models_dict[model]['local_feature_names'] = local_features
    # Bounds
    bounds = get_control_bounds(df, free_cp )
    print(bounds)

    local_feat_vec = df_feat_vec[models_dict[optimisation_type]['local_feature_names']]
    loaded_model = models_dict[optimisation_type]['LoadedMLModel']
    # Precompute scaled control params
    scaler = models_dict[optimisation_type]["LoadedScaler"]
    offsets, scales = extract_scaler_params(scaler)

    local_feature_names = models_dict[optimisation_type]["local_feature_names"]
    scaler_index = {name: i for i, name in enumerate(scaler.feature_names_in_)}
    feature_idx = np.array([scaler_index.get(f, -1) for f in local_feature_names])

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
        workers=1, 
    )

    optimal_free_cp = dict(zip(free_cp, result.x))
    optimal_cp = {**fixed_cp, **optimal_free_cp}

    # Build the feature vector for prediction using the optimal control parameters
    row_prev = df_feat_vec.iloc[-1].copy()
    row_curr = row_prev.copy()
    for key, value in optimal_cp.items():
        row_curr[key] = value
    scaled_features_prev = scale_features(scaler, 
                                          row_prev, 
                                          models_dict[optimisation_type]['local_feature_names']).reshape(1,-1)
    scaled_features_curr = scale_features(scaler, 
                                          row_curr, 
                                          models_dict[optimisation_type]['local_feature_names']).reshape(1,-1)
    
    y_pred_scaled_prev = models_dict[optimisation_type]['LoadedMLModel'].predict(scaled_features_prev)[0]
    y_pred_scaled_curr = models_dict[optimisation_type]['LoadedMLModel'].predict(scaled_features_curr)[0]
    
    y_pred_prev = inverse_transform_output(scaler, y_pred_scaled_prev, target_output)
    y_pred_curr = inverse_transform_output(scaler, y_pred_scaled_curr, target_output)
    
    optimal_cp[target_output + '_previous'] = y_pred_prev
    optimal_cp[target_output + '_current'] = y_pred_curr

    # Predict impact on other outputs
    targets = [models_dict[model]['output_param'] for model in models_dict.keys()]
    for model_name, model_dict in models_dict.items():
        if model_dict['Optimised']:
            continue
        target_output = model_dict['output_param']
        df_local = df.copy()
        # Remove all lagged feature vectors of other impact_targets
        for i, impact_target in enumerate(targets):
            if impact_target != target_output:
                df_local.drop(columns=[col for col in list(df_local.columns) if impact_target in col], inplace=True)
        row_prev = df_local.iloc[-1].copy()
        row_curr = row_prev.copy()
        for key, value in optimal_cp.items():
            row_curr[key] = value
        impact_model = model_dict['LoadedMLModel']
        impact_scaler_path = model_dict['scaling']
        impact_scaler = load_scaler(impact_scaler_path)
        feature_names = df_local.columns.tolist()
        impact_scaled_features_prev = scale_features(impact_scaler, 
                                                     row_prev, 
                                                     models_dict[model_name]['local_feature_names']).reshape(1,-1)
        impact_scaled_features_curr = scale_features(impact_scaler, 
                                                     row_curr, 
                                                     models_dict[model_name]['local_feature_names']).reshape(1,-1)
        
        y_pred_scaled_prev = impact_model.predict(impact_scaled_features_prev)[0]
        y_pred_scaled_curr = impact_model.predict(impact_scaled_features_curr)[0]
        
        y_pred_prev = inverse_transform_output(impact_scaler, y_pred_scaled_prev, target_output)
        y_pred_curr = inverse_transform_output(impact_scaler, y_pred_scaled_curr, target_output)
        
        optimal_cp[target_output + '_previous'] = y_pred_prev
        optimal_cp[target_output + '_current'] = y_pred_curr

    return optimal_cp

def debug_callback(xc_trial: np.ndarray, convergence: bool) -> None:
    """
    Debug callback function to print the current trial values.
    
    Args:
        xc_trial (np.ndarray): Current trial values of control parameters.
        convergence (bool): Whether the optimization has converged.
    """
    print(f"Current trial values: {xc_trial}, Convergence: {convergence}")