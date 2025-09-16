import pandas as pd
import numpy as np
import joblib
import warnings
from typing import List, Dict, Any
from data_fetchers.base_data_fetcher import BaseDataFetcher

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
    - Aligns to scaler.feature_names_in_ for transformation.
    - If any requested features (feature_names) are not present in the scaler, fills their
      scaled values with 0.0 while preserving the requested output order.
    This prevents index errors when models/scalers differ in expected inputs (e.g., UnitCost model).
    """
    # If scaler doesn't expose feature names (unlikely for sklearn >=1.0 with pandas),
    # fall back to transforming the requested features directly.
    if not hasattr(scaler, "feature_names_in_"):
        data = {feat: row.get(feat, 0.0) for feat in feature_names}
        df = pd.DataFrame([data], columns=feature_names)
        arr_scaled = scaler.transform(df)
        return np.asarray(arr_scaled[0, :], dtype=float)

    # Use the scaler's native feature names for transform
    scaler_feats = scaler.feature_names_in_.tolist()
    df_input = pd.DataFrame([{feat: row.get(feat, 0.0) for feat in scaler_feats}], columns=scaler_feats)
    arr_scaled_full = scaler.transform(df_input)

    # Build a lookup between sanitized scaler feature names and indices
    def _sanitize(name: str) -> str:
        # Normalize any model-time special characters to runtime names
        return name.replace('ŋ', 'ETA')

    sanitized_scaler_feats = [_sanitize(f) for f in scaler_feats]
    name_to_idx = {name: i for i, name in enumerate(sanitized_scaler_feats)}

    # Construct the result in the exact order of requested feature_names
    result = []
    missing = []
    for req in feature_names:
        req_s = _sanitize(req)
        idx = name_to_idx.get(req_s)
        if idx is None:
            # Feature not used by this scaler/model. Use neutral 0.0 in scaled space
            # so downstream shapes remain consistent and penalties stay inactive.
            result.append(0.0)
            missing.append(req)
        else:
            result.append(arr_scaled_full[0, idx])

    if missing:
        warnings.warn(
            f"scale_features: requested features not found in scaler and set to 0.0: {missing}"
        )

    return np.asarray(result, dtype=float)

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
    
def fetch_live_data(cp_op_ml_dict: Dict[str, Any], paths_set: List[str]) -> pd.DataFrame:
    """
    Fetch latest hourly averaged data for control and input parameters.
    Args:
        cp_op_ml_dict (Dict[str, Any]): Dictionary with control and input parameter metadata.
        paths_set (List[str]): List of unique InfluxDB paths to query.
    Returns:
        pd.DataFrame: DataFrame with latest hourly averaged data.
    """
    now = pd.Timestamp.utcnow()
    this_hour = now.replace(minute=0, second=0, microsecond=0)
    one_hour_ago = this_hour - pd.Timedelta(hours=1)

    values_needed = {cp_op_ml_dict[param]['InfluxName']:cp_op_ml_dict[param]['NameInMLData'] for param in list(cp_op_ml_dict.keys())}
    meas_dict = {}
    paths = []
    for param in list(cp_op_ml_dict.keys()):
        path = cp_op_ml_dict[param]['InfluxBucket'] + '/' + cp_op_ml_dict[param]['InfluxMeasurement']
        if path not in paths:
            paths.append(path)
        if meas_dict.get(path) is None:
            meas_dict[path] = []
        meas_dict[path].append(cp_op_ml_dict[param]['InfluxName'])
        
        if cp_op_ml_dict[param]['InfluxName'] not in values_needed:
            values_needed[cp_op_ml_dict[param]['InfluxName']] = cp_op_ml_dict[param]['NameInMLData']

    combined_df = pd.DataFrame()
    for influx_path, required_vars in meas_dict.items():
        bucket = influx_path.split('/')[0]
        meas = influx_path.split('/')[1]
        datafetcher = BaseDataFetcher(meas, database=bucket)
        df_meas = datafetcher.fetch_averaged_data(recent_data_of='over selected range',
                                        start_time=one_hour_ago,
                                        end_time=this_hour)
        if meas == 'process_params':
            df_meas['coke_rate'] = df_meas['coke_rate'] + df_meas['nut_coke_rate']
        df_meas = df_meas[required_vars + ['time']]
        df_meas['time'] = pd.to_datetime(df_meas['time'], errors='coerce', utc=True)
        df_meas.set_index('time', inplace=True)
        
        combined_df = df_meas if combined_df.empty else combined_df.join(df_meas, how='outer')
    
    hourly_avg = combined_df.resample('1h').mean()
    hourly_avg.index = hourly_avg.index + pd.Timedelta(hours=1)
    hourly_avg = hourly_avg.rename(columns=values_needed)

    return hourly_avg