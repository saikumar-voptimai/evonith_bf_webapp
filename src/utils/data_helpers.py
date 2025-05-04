import pandas as pd

def process_heat_load_data(df):
    """
    Process heat load data for visualization.
    
    Args:
        df (pd.DataFrame): Raw heat load data.
    
    Returns:
        pd.DataFrame: Processed DataFrame suitable for plotting.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df
