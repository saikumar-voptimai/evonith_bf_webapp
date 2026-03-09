# FurnaceMind/agents/mcp_tools.py
# Purpose: MCP tools for live data fetching and plotting
# Fixed: Added time range snapping to valid TIMEDELTAS keys so
#        "last 4 hours" doesn't cause KeyError.

import logging
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt


from utils.helper_functions_explorer import data_retrieval as dr
from config.config_loader import load_config

logger = logging.getLogger(__name__)


# Config
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



# Valid time ranges that data_retrieval.TIMEDELTAS supports.
# These are the ONLY strings dr.fetch_online_df accepts as time_range.
# If your TIMEDELTAS dict has different keys, update this list to match.
VALID_TIME_RANGES_MINUTES = [
    (15,    "last 15 minutes"),
    (30,    "last 30 minutes"),
    (60,    "last 1 hours"),
    (120,   "last 2 hours"),
    (360,   "last 6 hours"),
    (480,   "last 8 hours"),
    (720,   "last 12 hours"),
    (1440,  "last 24 hours"),
]


def _snap_time_range(time_range: str) -> str:
    """
    Snap an arbitrary time range string to the nearest valid TIMEDELTAS key.

    Examples:
        "last 4 hours"  → "last 6 hours"   (snap up to nearest valid)
        "last 8 hours"  → "last 8 hours"   (already valid)
        "last 3 hours"  → "last 2 hours"   (snap to nearest)
        "last 10 hours" → "last 12 hours"  (snap up)
        "last 45 minutes" → "last 30 minutes" (snap to nearest)
    """
    # Parse the input
    m = re.match(r"last\s+(\d+)\s+(hours?|minutes?|mins?|hrs?)", time_range.lower())
    if not m:
        logger.warning(f"Cannot parse time range: {time_range!r}, defaulting to 'last 8 hours'")
        return "last 8 hours"

    value = int(m.group(1))
    unit = m.group(2)

    # Convert to minutes
    if unit.startswith("h"):
        requested_minutes = value * 60
    else:
        requested_minutes = value

    # Find the nearest valid time range
    best_key = "last 8 hours"
    best_diff = float("inf")

    for valid_minutes, valid_key in VALID_TIME_RANGES_MINUTES:
        diff = abs(valid_minutes - requested_minutes)
        if diff < best_diff:
            best_diff = diff
            best_key = valid_key
        # If exact match, break early
        if diff == 0:
            break

    # If user asked for more than available, snap UP to next larger
    for valid_minutes, valid_key in VALID_TIME_RANGES_MINUTES:
        if valid_minutes >= requested_minutes:
            best_key = valid_key
            break
    else:
        # If even the largest is smaller, use the largest
        best_key = VALID_TIME_RANGES_MINUTES[-1][1]

    if best_key != time_range:
        logger.info(f"Snapped time range: {time_range!r} → {best_key!r}")

    return best_key



# MCP TOOL 1 — Influx Data Fetcher
class InfluxDataFetcher:

    def fetch(
        self,
        time_range: str = "last 8 hours",
        window: str = "15 minutes",
        measurements: list[str] | None = None,
        fields: list[str] | None = None,
        field_match_mode: str = "contains",
    ) -> pd.DataFrame:

        selected_measurements = (
            measurements if measurements else list(MEASUREMENT_LABELS.keys())
        )

        # Snap to valid time range to prevent KeyError in data_retrieval
        safe_time_range = _snap_time_range(time_range)

        df = dr.fetch_online_df(
            selected_measurements=selected_measurements,
            time_range=safe_time_range,
            average_range=window,
            FREQUENCY_TO_TIMEDTA=FREQUENCY_TO_TIMEDTA,
            MEASUREMENT_LABELS=MEASUREMENT_LABELS,
            FIELD_LABELS=FIELD_LABELS,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.sort_index()

        # Optional: keep only requested fields/columns
        if fields:
            if field_match_mode not in {"contains", "exact"}:
                field_match_mode = "contains"

            if field_match_mode == "exact":
                keep = [c for c in df.columns if c in fields]
            else:
                wanted = [str(f).lower() for f in fields]
                keep = [
                    c for c in df.columns
                    if any(w in str(c).lower() for w in wanted)
                ]

            if keep:
                df = df[keep]

        return df



# MCP TOOL 2 — Python Plotter
class PythonPlotter:

    def plot(
        self,
        df: pd.DataFrame,
        columns: list[str],
        title: str = "Live Furnace Trend",
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 4))

        for col in columns:
            if col in df.columns:
                ax.plot(df.index, df[col], label=col)

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
        if len(columns) > 1:
            ax.legend()

        fig.autofmt_xdate()
        fig.tight_layout()

        # NOTE: caller should call plt.close(fig) after rendering
        return fig