import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from utils.recommendations import get_control_bounds, load_scaler, scale_features, inverse_transform_output
from typing import Dict, List, Any
from src.config.config_loader import load_config

config_vsense = load_config('setting_vsense.yml')

def objective(
    xc_trial: np.ndarray,
    df_feat_vec: pd.DataFrame,
    fixed_cp: Dict[str, float],
    model: Any,
    control_params: List[str],
    lambda_reg: float,
    time_idx: int,
    scaler,
    feature_names: List[str],
) -> float:
    """
    Objective function combining the predicted target output and a penalty term
    for deviation from previous control parameters, both in scaled space.

    Args:
        xc_trial (np.ndarray): Array of free control parameter values (to be optimized).
        df_feat_vec (pd.DataFrame): DataFrame with feature vectors.
        fixed_cp (Dict[str, float]): Fixed control parameters. 
        model (Any): Pre-trained regression model.
        control_params (List[str]): List of control parameter names.
        lambda_reg (float): Regularization strength.
        time_idx (int): Index of the timestep (historic) in df_feat_vec.
        scaler: Fitted scaler for features.
        feature_names: List of feature names in correct order.
        output_name: Name of the output parameter.
    Returns:
        float: Combined objective value (predicted output + penalty).
    """
    free_cp = [cp for cp in control_params if cp not in fixed_cp]
    row = df_feat_vec.iloc[time_idx].copy()
    for idx, cp in enumerate(free_cp):
        row[cp] = xc_trial[idx]
    # Scale feature vector
    scaled_features = scale_features(scaler, row, feature_names).reshape(1,-1)
    # Predict in scaled space
    y_pred_scaled = model.predict(scaled_features)[0]
    # Scale control parameters for penalty
    row_cp = row[control_params]
    prev_row_cp = df_feat_vec.iloc[-1][control_params]
    scaled_xc_t_ordered = scale_features(scaler, row_cp, control_params)[0]
    scaled_prev_params = scale_features(scaler, prev_row_cp, control_params)[0]
    penalty = np.sum((scaled_xc_t_ordered - scaled_prev_params) ** 2)
    return y_pred_scaled + lambda_reg * penalty

def run_optimiser(
    df: pd.DataFrame,
    models: Dict[str, Any],
    user_input: Dict[str, float],
    fixed_cp: Dict[str, float],
    control_params: List[str],
    target_output: str,
    optimisation_type: str,
    date_time: pd.Timestamp,
    lambda_reg: float = 0.1,
) -> Dict[str, float]:
    """
    Runs optimization to find control parameters minimizing target output.
    Args:
        df (pd.DataFrame): DataFrame with feature vectors.
        models (Dict[str, Any]): Pre-trained regression models for each target output.
        user_input (Dict[str, float]): User-specified input parameters.
        fixed_cp (Dict[str, float]): Fixed control parameters.
        control_params (List[str]): List of control parameter names.
        target_output (str): Target output variable name.
        optimisation_type (str): Type of optimization - Ex: Coke Rate, Fuel Rate etc
        date_time (pd.Timestamp): Timestamp for which to optimize.
        lambda_reg (float): Regularization strength for penalty term.
    Returns:
        Dict[str, float]: Optimal control parameters including the predicted target output.
    """

    df_feat_vec = df.copy()
    targets = [config_vsense['Optimisation'][model]['output_param'] for model in list(config_vsense['Optimisation'].keys())]
    for i, target in enumerate(targets):
        if target != target_output:
            df_feat_vec.drop(columns=[col for col in list(df_feat_vec.columns) if target in col], inplace=True)

    TIME_IDX = int(np.where(pd.to_datetime(df_feat_vec.index, format="%d/%m/%Y %H:%M") < date_time)[0][-1])

    # Update the raw material input parameters if any are overridden
    for key, value in user_input.items():
        if not np.isnan(value):
            df_feat_vec.at[df_feat_vec.index[TIME_IDX], key] = value

    free_cp = [cp for cp in control_params if cp not in fixed_cp]

    # Update the control parameters if any are overridden
    for key, value in fixed_cp.items():
        if not np.isnan(value):
            df_feat_vec.at[df_feat_vec.index[TIME_IDX], key] = value

    # Load scaler for the target output
    scaler_path = config_vsense['Optimisation'][optimisation_type]['scaling']
    scaler = load_scaler(scaler_path)
    feature_names = df_feat_vec.columns.tolist()

    # Bounds
    bounds = get_control_bounds(df, free_cp, q_low=0.01, q_high=0.99)

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        args=(df_feat_vec, 
              fixed_cp, 
              models[target_output], 
              control_params, 
              lambda_reg, 
              TIME_IDX, 
              scaler,
              feature_names
              ),
        strategy='best1bin',
        popsize=5,
        tol=0.01,
        maxiter=10
    )

    optimal_free_cp = dict(zip(free_cp, result.x))
    optimal_cp = {**fixed_cp, **optimal_free_cp}

    # Build the feature vector for prediction using the optimal control parameters
    row = df_feat_vec.iloc[TIME_IDX].copy()
    for key, value in optimal_cp.items():
        row[key] = value
    scaled_features = scale_features(scaler, row, feature_names).reshape(1,-1)
    y_pred_scaled = models[target_output].predict(scaled_features)[0]
    y_pred = inverse_transform_output(scaler, y_pred_scaled, target_output)
    optimal_cp[target_output] = y_pred

    # Predict impact on other outputs
    for output, impact_model in models.items():
        if output == target_output:
            continue
        df_local = df.copy()
        # Remove all lagged feature vectors of other impact_targets
        for i, impact_target in enumerate(targets):
            if impact_target != output:
                df_local.drop(columns=[col for col in list(df_local.columns) if impact_target in col], inplace=True)
        row = df_local.iloc[TIME_IDX].copy()
        for key, value in optimal_cp.items():
            row[key] = value
        for i, key in enumerate(config_vsense['Optimisation'].keys()):
            if config_vsense['Optimisation'][key]['output_param'] == output:
                impact_scaler_path = config_vsense['Optimisation'][key]['scaling']
        impact_scaler = load_scaler(impact_scaler_path)
        feature_names = df_local.columns.tolist()
        impact_scaled_features = scale_features(impact_scaler, row, feature_names).reshape(1,-1)
        y_pred_scaled = impact_model.predict(impact_scaled_features)[0]
        y_pred = inverse_transform_output(impact_scaler, y_pred_scaled, output)
        optimal_cp[output] = y_pred

    return optimal_cp

def debug_callback(xc_trial: np.ndarray, convergence: bool) -> None:
    """
    Debug callback function to print the current trial values.
    
    Args:
        xc_trial (np.ndarray): Current trial values of control parameters.
        convergence (bool): Whether the optimization has converged.
    """
    print(f"Current trial values: {xc_trial}, Convergence: {convergence}")