import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any

def build_feature_vector(df: pd.DataFrame,
                         user_input: Dict[str, Any],
                         xc_trial: List[float], 
                         input_params: List[str], 
                         control_params: List[str], 
                         target_output: str, 
                         lags=4
                         ) -> np.ndarray:
    """
    Builds the feature vector expected by the ML model.
    Args:
        df (pd.DataFrame): Historical data DataFrame.
        user_input (Dict[str, Any]): Current user input parameters.
        prev_params (Dict[str, float]): Previous control parameters.
        xc_trial (List[float]): Trial values for control parameters.
        input_params (List[str]): List of input parameter names.
        control_params (List[str]): List of control parameter names.
        target_output (str): Name of the target output parameter.
        lags (int): Number of lagged timesteps to include.
    Returns:
        np.ndarray: Feature vector for the model.
    """

    features = []
    
    # Current input params (user input or last available)
    for inp in input_params:
        features.append(user_input.get(inp, df[inp].iloc[-1]))

    # Input params for lags (historical data)
    for lag in range(1, lags+1):
        for inp in input_params:
            features.append(df[inp].iloc[-lag])

    # Current control params (trial values)
    for val in xc_trial:
        features.append(val)

    # Control params for lags (historical data)
    for lag in range(1, lags+1):
        for cp in control_params:
            features.append(df[cp].iloc[-lag])

    # Output param for lags
    for lag in range(1, lags+1):
        features.append(df[target_output].iloc[-lag])

    return np.array(features).reshape(1, -1)

def get_control_bounds(df: pd.DataFrame, 
                       control_params: List[str], 
                       q_low: float=0.01, 
                       q_high: float=0.99
                       ) -> List[tuple]:
    """
    Get quantile-based bounds for control params.
    Args:
        df (pd.DataFrame): DataFrame containing control parameters.
        control_params (List[str]): List of control parameter names.
        q_low (float): Lower quantile for bounds.
        q_high (float): Upper quantile for bounds.
    Returns:
        List[tuple]: List of tuples with lower and upper bounds for each control parameter.
    """
    return [(df[cp].quantile(q_low), df[cp].quantile(q_high)) for cp in control_params]

def process_dataframe(df: pd.DataFrame,
                      target_col: str,
                      targets: List[str],
                      lags: int = 4
                      ) -> pd.DataFrame:
    """
    Process the DataFrame to ensure it has the correct format.
    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): The target column to retain.
        targets (List[str]): List of target columns to drop.
    Returns:
        pd.DataFrame: Processed DataFrame with 'time' index.
    """
    df_work = df.copy()
    
    # Drop other target columns
    # df_work.drop(columns=[col for col in targets if col != target_col], inplace=True)

    df_work.index = pd.to_datetime(df_work.index, errors='coerce', format='%d/%m/%Y %H:%M')

    # Extract datetime features
    df_work['hour'] = df_work.index.to_series().dt.hour
    df_work['day_of_week'] = df_work.index.to_series().dt.dayofweek
    df_work['month'] = df_work.index.to_series().dt.month

    # Generate lagged features (1–5 hours) for all columns except the target
    feature_cols = [col for col in df_work.columns if col not in targets]

    for lag in range(1, lags):
        lagged_features = df_work[feature_cols].shift(lag)
        lagged_features.columns = [f'{col}_lag{lag}' for col in feature_cols]
        df_work = pd.concat([df_work, lagged_features], axis=1)

    for target in targets:
        df_work[target +'_lag1'] = df_work[target].shift(1)
        df_work.drop(columns=target, inplace=True)

    # Remove rows with NaNs after lagging
    df_work.dropna(inplace=True)
    return df_work

def load_scaler(scaler_path):
    """Load a fitted scaler from disk."""
    return joblib.load(scaler_path)

def scale_features(scaler, row, feature_names):
    """Scale a feature vector (row: pd.Series) using the provided scaler and feature order.
    Returns a DataFrame with correct feature names for compatibility with sklearn scalers.
    """
    all_feats = scaler.feature_names_in_.tolist()
    data = {feat: row.get(feat, 0.0) for feat in all_feats}
    df = pd.DataFrame([data], columns=all_feats)
    arr_scaled = scaler.transform(df)
    all_feats = [col.replace('ŋ','ETA') for col in all_feats]  # Replace 'ŋ' with 'ETA' for compatibility
    idxs = [all_feats.index(feat) for feat in feature_names]
    return arr_scaled[0, idxs]

def inverse_transform_output(scaler, y_scaled, output_name):
    """Inverse transform a single output value using the scaler and output name."""

    # If scaler is MultiOutput, handle accordingly
    if hasattr(scaler, 'feature_names_in_'):
        feature_names_list = scaler.feature_names_in_.tolist()
        if 'FurnaceTopGasAnalysis' in output_name and '_lag' not in output_name:
            identifier = 'FurnaceTopGasAnalysis'
            for i, name in enumerate(feature_names_list):
                if identifier in name:
                    output_name = name
                    break
        idx = feature_names_list.index(output_name)
        # Create a zero vector except for the output index
        arr = np.zeros((1, len(scaler.feature_names_in_)))
        arr[0, idx] = y_scaled
        y_unscaled = scaler.inverse_transform(arr)[0, idx]
        return y_unscaled
    else:
        # Fallback for single-output scaler
        return scaler.inverse_transform(np.array([[y_scaled]]))[0, 0]