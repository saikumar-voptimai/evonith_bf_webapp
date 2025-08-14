import re
import pandas as pd
from typing import List, Dict
from data_fetchers import base_data_fetcher

def clean_rm_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Offline Data fetcher:
    Cleans the raw data DataFrame by removing columns with all NaN values and renaming ore prefixes.
    Args:
        df (pd.DataFrame): Raw data DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame with renamed ore prefixes."""
    ore_groups = {}
    for col in df.columns:
        match = re.match(r'(ore_\d+)_', col)
        if match:
            ore_groups.setdefault(match.group(1), []).append(col)

    drop_cols = []
    for ore_key, cols in ore_groups.items():
        if df[cols].isnull().all(axis=1).all():
            drop_cols.extend(cols)

    df_clean = df.drop(columns=drop_cols, errors='ignore')

    # Rename ore prefixes
    ore_name_map = {
        'ore_1': 'NMDC_DONAMALAI',
        'ore_2': 'LLOYDS',
        'ore_3': 'GEOMIN',
        'ore_4': 'JRS_VENTURES',
        'ore_5': 'NMDC_ROM'
    }
    rename_map = {}
    for col in df_clean.columns:
        for ore_prefix, ore_name in ore_name_map.items():
            if col.startswith(ore_prefix):
                rename_map[col] = col.replace(ore_prefix, ore_name)
    df_clean = df_clean.rename(columns=rename_map)

    return df_clean

def fetch_offline_data(measurement: str,
                       time_range: str) -> pd.DataFrame:
    """
    Fetches offline data for a given measurement.
    Args:
        measurement (str): Measurement name to fetch data for.
    Returns:
        pd.DataFrame: DataFrame containing the offline data for the measurement.
    """
    datafetcher = base_data_fetcher.BaseDataFetcher(measurement,
                                                    database="bf2_evonith_offline",
                                                    token="INFLUX_OFFLINE_TOKEN")
    df_meas = datafetcher.fetch_averaged_data(average_by=time_range)
    if 'time' in df_meas.columns:
        df_meas['time'] = pd.to_datetime(df_meas['time'], errors='coerce')
        df_meas.set_index('time', inplace=True, drop=True)
        df_meas.sort_index(inplace=True)
    else:
        if not isinstance(df_meas.index, pd.DatetimeIndex):
            df_meas.index = pd.to_datetime(df_meas.index, errors='coerce')

    return df_meas

def average_data(df, freq):
    """
    Average the data in the DataFrame by the specified frequency.
    Args:
        df (pd.DataFrame): DataFrame indexed by time.
        freq (str): Frequency string for resampling (e.g., '1h', '5min').
    Returns:
        pd.DataFrame: DataFrame with averaged data.
    """
    df_loc = df.copy()
    if not isinstance(df_loc.index, pd.DatetimeIndex):
        df_loc.index = pd.to_datetime(df_loc.index, errors='coerce')
    df_avg = df_loc.resample(freq).mean().dropna(how='all')
    return df_avg

def fetch_online_df(selected_measurements: List[str], 
                    time_range: str, 
                    average_range: str, 
                    datafetchers: Dict,
                    FREQUENCY_TO_TIMEDTA: Dict,
                    MEASUREMENT_LABELS: Dict,
                    FIELD_LABELS: Dict) -> pd.DataFrame:
    """
    Fetches and combines data from multiple data fetchers based on selected measurements and time range.
    Args:
        selected_measurements (List[str]): List of selected measurement keys.
        time_range (str): Time range for data fetching (key from TIME_OPTIONS).
        average_range (str): Averaging window for the data (key from FREQUENCY_TO_TIMEDTA).
    """
    combined_df = pd.DataFrame()
    for meas in selected_measurements:
        df_meas = datafetchers[meas].fetch_averaged_data(average_by=time_range)
        if df_meas.empty:
            continue
        # ensure time index
        if 'time' in df_meas.columns:
            df_meas['time'] = pd.to_datetime(df_meas['time'], errors='coerce')
            df_meas.set_index('time', inplace=True, drop=True)
        else:
            if not isinstance(df_meas.index, pd.DatetimeIndex):
                df_meas.index = pd.to_datetime(df_meas.index, errors='coerce')
        # averaging
        freq = FREQUENCY_TO_TIMEDTA.get(average_range)
        if freq is not None:
            df_meas = average_data(df_meas, freq)
        else:
            df_meas = average_data(df_meas, '1h')
        # rename columns with measurement and field labels
        df_meas = df_meas.rename(columns={
            col: f"{MEASUREMENT_LABELS.get(meas, meas)} - {FIELD_LABELS.get(col, col)}"
            for col in df_meas.columns
        })
        combined_df = df_meas if combined_df.empty else combined_df.join(df_meas, how='outer')
    return combined_df