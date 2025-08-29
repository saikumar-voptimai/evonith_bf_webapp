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
        df_meas['time'] = pd.to_datetime(df_meas['time'], errors='coerce', utc=True)
        df_meas.set_index('time', inplace=True, drop=True)
        df_meas.sort_index(inplace=True)
    else:
        if not isinstance(df_meas.index, pd.DatetimeIndex):
            df_meas.index = pd.to_datetime(df_meas.index, errors='coerce', utc=True)
        df_meas.sort_index(inplace=True)

    return df_meas


def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime index (UTC) and sorted."""
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors='coerce', utc=True)
    elif d.index.tz is None:
        d.index = d.index.tz_localize('UTC')
    d.sort_index(inplace=True)
    return d


def average_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Average the data with right-labeled full bins and a custom last partial-window
    aggregation assigned to the current timestamp ("now").

    - Ensures DatetimeIndex (UTC) and sorted.
    - Uses closed='right', label='right' for full bins.
    - If the last bin is partial, compute its mean manually over (last_full_edge, max_ts]
      and label it at now.

    Args:
        df: DataFrame indexed by time.
        freq: Frequency string for resampling (e.g., '1h', '5min').
    Returns:
        Averaged DataFrame with optional last row labeled at now for partial window.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    d = _ensure_dtindex(df)

    # Convert all numeric columns to numeric types where possible
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = pd.to_numeric(d[c], errors='coerce')

    from pandas.tseries.frequencies import to_offset
    off = to_offset(freq)

    max_ts = d.index.max()
    # Full-bin cut: include data up to and including the last full right edge
    last_full_right = max_ts.floor(freq)
    is_on_edge = (max_ts == last_full_right)

    # Full bins (right-labeled, closed-right) up to last_full_right
    d_full = d[d.index <= last_full_right]
    full = d_full.resample(freq, label='right', closed='right').mean().dropna(how='all')

    # Handle partial tail if not exactly on boundary and if there is tail data
    if not is_on_edge:
        tail = d[d.index > last_full_right]
        if not tail.empty:
            tail_mean = tail.mean(numeric_only=True).to_frame().T
            # Label the partial aggregate at "now" in the same tz as data
            now_ts = pd.Timestamp.utcnow().tz_convert(d.index.tz) if d.index.tz is not None else pd.Timestamp.utcnow().tz_localize('UTC')
            tail_mean.index = pd.DatetimeIndex([now_ts])
            out = pd.concat([full, tail_mean], axis=0)
        else:
            out = full
    else:
        out = full

    # Drop columns that are entirely NaN
    out = out.dropna(axis=1, how='all')

    return out


def fetch_online_df(selected_measurements: List[str], 
                    time_range: str, 
                    average_range: str, 
                    datafetchers: Dict,
                    FREQUENCY_TO_TIMEDTA: Dict,
                    MEASUREMENT_LABELS: Dict,
                    FIELD_LABELS: Dict) -> pd.DataFrame:
    """
    Fetches and combines data from multiple data fetchers based on selected measurements and time range.

    - Validates datetime index and sorts.
    - Applies averaging window: if "None", default to 1h averaging.
    - Renames columns to "<Measurement Label> - <Field Label>" when possible.
    - Returns an outer-joined DataFrame across measurements.
    """
    if not selected_measurements:
        return pd.DataFrame()

    combined_df = pd.DataFrame()
    for meas in selected_measurements:
        if meas not in datafetchers:
            continue
        try:
            df_meas = datafetchers[meas].fetch_averaged_data(average_by=time_range)
        except Exception:
            df_meas = pd.DataFrame()
        if df_meas is None or df_meas.empty:
            continue

        # ensure time index
        if 'time' in df_meas.columns:
            df_meas['time'] = pd.to_datetime(df_meas['time'], errors='coerce', utc=True)
            df_meas.set_index('time', inplace=True, drop=True)
        else:
            if not isinstance(df_meas.index, pd.DatetimeIndex):
                df_meas.index = pd.to_datetime(df_meas.index, errors='coerce', utc=True)
        df_meas.sort_index(inplace=True)

        # averaging
        freq = FREQUENCY_TO_TIMEDTA.get(average_range)
        if freq is not None:
            df_meas = average_data(df_meas, freq)
        else:
            # default to 1h
            df_meas = average_data(df_meas, '1h')

        # rename columns with measurement and field labels
        def _rename(col: str) -> str:
            human_meas = MEASUREMENT_LABELS.get(meas, meas)
            human_field = FIELD_LABELS.get(col, col)
            return f"{human_meas} - {human_field}"

        df_meas = df_meas.rename(columns={col: _rename(col) for col in df_meas.columns})

        # combine
        combined_df = df_meas if combined_df.empty else combined_df.join(df_meas, how='outer')

    # final cleanup
    if combined_df.empty:
        return combined_df
    combined_df = combined_df.sort_index()
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    return combined_df