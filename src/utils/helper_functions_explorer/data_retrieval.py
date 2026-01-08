import re
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from data_fetchers.base_data_fetcher import BaseDataFetcher

TIMEDELTAS = {
        'last 1 minute': timedelta(minutes=1),
        'last 5 minutes': timedelta(minutes=5),
        'last 15 minutes': timedelta(minutes=15),
        'last 30 minutes': timedelta(minutes=30),
        'last 1 hour': timedelta(hours=1),
        'last 6 hours': timedelta(hours=6),
        'last 8 hours': timedelta(hours=8),
        'last 12 hours': timedelta(hours=12),
        'last 1 day': timedelta(days=1),
        'last 3 days': timedelta(days=3),
        'last 1 week': timedelta(weeks=1),
        'last 2 weeks': timedelta(weeks=2),
        'last 1 month': timedelta(days=30),
        'last 2 months': timedelta(days=60),
        'last 3 months': timedelta(days=90)
}

def clean_rm_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Offline Data fetcher:
    Cleans the raw data DataFrame by removing columns with all NaN values and renaming ore prefixes.
    Args:
        df (pd.DataFrame): Raw data DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame with renamed ore prefixes.
    """
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


def fetch_offline_data(measurement: str, time_range, database: str) -> pd.DataFrame:
    """
    Fetch offline data with minimal and necessary time conversions.
    Supports:
    - preset strings (e.g. 'last 1 month')
    - tuple ranges (start, end)
    - full dataset ('full')
    """
    dfetch = BaseDataFetcher(
        variable_tag=measurement,
        database=database,
        token="INFLUX_OFFLINE_TOKEN"
    )

    now = datetime.now(timezone.utc)

    # --- Resolve time range ---
    if isinstance(time_range, str) and time_range.strip().lower() == "full":
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = now
        recent_label = "over selected range"

    elif isinstance(time_range, tuple) and len(time_range) == 2:
        # Convert tuple times once only
        start = pd.to_datetime(time_range[0], utc=True)
        end = pd.to_datetime(time_range[1], utc=True)
        recent_label = "over selected range"

    else:
        # Preset string route → return directly
        df = dfetch.fetch_averaged_data(recent_data_of=time_range)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time").sort_index()

    # --- Fetch for custom/full ranges ---
    df = dfetch.fetch_averaged_data(
        recent_data_of=recent_label,
        start_time=start,
        end_time=end
    )

    # --- Final minimal cleanup ---
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")

    return df.sort_index()



def fetch_online_df(selected_measurements: List[str],
                    time_range: str,
                    MEASUREMENT_LABELS: Dict,
                    FIELD_LABELS: Dict,
                    request_type: str = None,
                    window_by: str = None) -> pd.DataFrame:
    """
    Fetches and combines data from multiple data fetchers based on selected measurements and time range.
    Args:
        selected_measurements (List[str]): List of measurement names to fetch.
        time_range (str): Time range string (e.g., 'last 1 hour').
        average_range (str): Averaging interval (e.g., '1h', '30min').
        FREQUENCY_TO_TIMEDTA (Dict): Mapping of frequency strings to timedelta objects.
        MEASUREMENT_LABELS (Dict): Mapping of measurement names to human-readable labels.
        FIELD_LABELS (Dict): Mapping of field names to human-readable labels.
    Returns:
        pd.DataFrame: Combined DataFrame with data from all selected measurements.
    """
    datafetchers = {}
    for key in MEASUREMENT_LABELS.keys():
        datafetchers[key] = BaseDataFetcher(key)

    if not selected_measurements:
        return pd.DataFrame()

    combined_df = pd.DataFrame()
    now = datetime.now(timezone.utc) 
    start_time = now - TIMEDELTAS[time_range]
    end_time = now
        
    for meas in selected_measurements:
        if meas not in datafetchers:
            continue
        
        df_meas = datafetchers[meas].fetch_averaged_data(recent_data_of='over selected range',
                                                        start_time=start_time, 
                                                        end_time=end_time,
                                                        request_type=request_type,
                                                        window_by=window_by)
        if df_meas is None or df_meas.empty:
            continue

        df_meas['time'] = pd.to_datetime(df_meas['time'], errors='coerce', utc=True)
        df_meas.set_index('time', inplace=True, drop=True)
        df_meas.sort_index(inplace=True)

        def _rename(col: str) -> str:
            human_meas = MEASUREMENT_LABELS.get(meas, meas)
            human_field = FIELD_LABELS.get(col, col)
            return f"{human_meas} - {human_field}"

        df_meas = df_meas.rename(columns={col: _rename(col) for col in df_meas.columns})

        combined_df = df_meas if combined_df.empty else combined_df.join(df_meas, how='outer')

    combined_df = combined_df.select_dtypes(exclude=['object'])
    combined_df = combined_df.rename(index={combined_df.index[-1]: pd.Timestamp(end_time).round('1min')})

    if combined_df.empty:
        return combined_df
    combined_df = combined_df.sort_index()
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    combined_df.index = combined_df.index.tz_convert('Asia/Kolkata')
    combined_df.index.name = 'time (IST)'
    return combined_df