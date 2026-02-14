import pandas as pd
import matplotlib.pyplot as plt

from utils.helper_functions_explorer import data_retrieval as dr
from config.config_loader import load_config


# ---------------------------------------
# 🔧 CONFIG LOAD (Your Actual Setup)
# ---------------------------------------

config = load_config("setting_ds_dv.yml")

MEASUREMENT_LABELS = {
    "heatload_delta_t": "Heatload Delta T",
    "process_params": "Process Params",
    "temperature_profile": "Temperature Profile",
}

FREQUENCY_TO_TIMEDTA = {
    "None": None,
    "1 minute": "1min",
    "5 minutes": "5min",
    "10 minutes": "10min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "1 hour": "1h",
    "6 hours": "6h",
    "8 hours": "8h",
    "12 hours": "12h",
    "1 day": "1d",
}

FIELD_LABELS = {
    internal_key: human_label
    for mapping in config["data_mapping"].values()
    for human_label, internal_key in mapping.items()
}


# =======================================
# 🔧 MCP TOOL 1 — Influx Data Fetcher
# =======================================

class InfluxDataFetcher:

    def fetch(
        self,
        time_range: str = "last 8 hours",
        window: str = "15 minutes",
        measurements: list[str] | None = None,
    ) -> pd.DataFrame:

        selected_measurements = (
            measurements if measurements else list(MEASUREMENT_LABELS.keys())
        )

        df = dr.fetch_online_df(
            selected_measurements=selected_measurements,
            time_range=time_range,
            average_range=window,
            FREQUENCY_TO_TIMEDTA=FREQUENCY_TO_TIMEDTA,
            MEASUREMENT_LABELS=MEASUREMENT_LABELS,
            FIELD_LABELS=FIELD_LABELS,
        )

        if df is None:
            return pd.DataFrame()

        return df.sort_index()


# =======================================
# 🔧 MCP TOOL 2 — Python Plotter
# =======================================

class PythonPlotter:

    def plot(self, df: pd.DataFrame, columns: list[str], title: str = "Trend"):

        if df is None or df.empty:
            return None

        fig, ax = plt.subplots()

        for col in columns:
            if col in df.columns:
                ax.plot(df.index, df[col])

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        return fig