from __future__ import annotations

import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore

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
        measurements: list[str] | None = None,
        fields: list[str] | None = None,
        field_match_mode: str = "contains",  # "contains" | "exact"
        request_type: str = 'windowed-average',
        window_by: str = "15 minutes",
    ) -> pd.DataFrame:
        """
        Fetch data from InfluxDB based on the specified parameters.
        
        Parameters:
        - time_range: A string like "last 8 hours" or a custom range.
        - measurements: List of measurement keys to fetch (e.g., ["heatload_delta_t"]).
        - fields: Optional list of field names to keep in the final DataFrame.
        - field_match_mode: If "exact", fields must match exactly; if "contains", fields that contain the string are kept.
        - request_type: The type of data retrieval (e.g., "windowed-average", "avg-min-max", "ts").
        - window_by: The aggregation window for averaging (e.g., "15 minutes").

        Returns:
        - A pandas DataFrame with the requested data, indexed by time.
        """

        selected_measurements = (
            measurements if measurements else list(MEASUREMENT_LABELS.keys())
        )

        df = dr.fetch_online_df(
            selected_measurements=selected_measurements,
            time_range=time_range,
            FREQUENCY_TO_TIMEDTA=FREQUENCY_TO_TIMEDTA,
            MEASUREMENT_LABELS=MEASUREMENT_LABELS,
            FIELD_LABELS=FIELD_LABELS,
            request_type = request_type,
            window_by = window_by,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.sort_index()

        # ✅ Optional: keep only requested fields/columns
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

            # If we matched something, filter; otherwise leave df unchanged for UI fallback.
            if keep:
                df = df[keep]

        return df


# =======================================
# 🔧 MCP TOOL 2 — Python Plotter
# =======================================

class PythonPlotter:

    def __init__(self):
        self.last_plot_code: str | None = None
        self.last_plot_error: str | None = None

    def _safe_exec(self, code: str, local_vars: dict[str, Any]) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Empty code string")

        banned = [
            r"\bimport\b",
            r"\bopen\s*\(",
            r"__import__",
            r"\bos\b",
            r"\bsys\b",
            r"\bsubprocess\b",
            r"\beval\b",
            r"\bexec\b",
        ]
        for pat in banned:
            if re.search(pat, code):
                raise ValueError(f"Disallowed token in code: {pat}")

        safe_builtins = {
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "float": float,
            "int": int,
            "str": str,
            "print": print,
        }

        exec(code, {"__builtins__": safe_builtins}, local_vars)

    def generate_plotly_code(
        self,
        df: pd.DataFrame,
        columns: list[str],
        title: str = "Live Furnace Trend",
    ) -> str:
        """Return Python code that produces a Plotly figure named `fig`.

        Notes:
        - Code assumes `df` is already available as a DataFrame.
        - Code must be safe for restricted exec (no imports / no IO).
        """
        cols = [c for c in (columns or []) if isinstance(c, str) and c in df.columns]
        if not cols:
            cols = list(df.columns[:2])

        # Escape title for embedding in Python code
        safe_title = title.replace("\\", "\\\\").replace("\"", "\\\"")

        # We plot df.index as time. Keep it robust even if index is not datetime.
        code = (
            "_df = df.copy()\n"
            "_df = _df.reset_index().rename(columns={'index': 'timestamp'})\n"
            "try:\n"
            "    _df['timestamp'] = pd.to_datetime(_df['timestamp'])\n"
            "except Exception:\n"
            "    pass\n"
            f"_cols = {cols!r}\n"
            "_cols = [c for c in _cols if c in _df.columns]\n"
            "if not _cols:\n"
            "    _cols = [c for c in _df.columns if c != 'timestamp'][:1]\n"
            "_long = _df.melt(id_vars=['timestamp'], value_vars=_cols, var_name='signal', value_name='value')\n"
            f"fig = px.line(_long, x='timestamp', y='value', color='signal', title=\"{safe_title}\")\n"
            "fig.update_layout(legend_title_text='Signal')\n"
        )
        return code

    def compile_plotly_figure(self, code: str, df: pd.DataFrame) -> go.Figure:
        """Execute `code` in a restricted environment and return `fig`."""
        local_vars: dict[str, Any] = {"pd": pd, "px": px, "go": go, "df": df}
        self._safe_exec(code, local_vars)
        fig = local_vars.get("fig")
        if fig is None:
            raise ValueError("Code executed but did not define `fig`.")
        return fig

    def plot(
        self,
        df: pd.DataFrame,
        columns: list[str],
        title: str = "Live Furnace Trend",
    ) -> go.Figure:
        """
        Generate Plotly code, compile it, and return a Plotly figure.

        Streamlit Cloud expectation:
        - The caller should render with `st.plotly_chart(fig, use_container_width=True)`.
        - The generated code is stored in `self.last_plot_code` and (if available) `st.session_state`.
        """
        self.last_plot_error = None

        code = self.generate_plotly_code(df=df, columns=columns, title=title)
        self.last_plot_code = code

        if st is not None:
            try:
                st.session_state["mcp_last_plot_code"] = code
            except Exception:
                pass

        try:
            fig = self.compile_plotly_figure(code=code, df=df)
        except Exception as e:
            self.last_plot_error = str(e)
            if st is not None:
                try:
                    st.session_state["mcp_last_plot_error"] = self.last_plot_error
                except Exception:
                    pass
            raise

        if st is not None:
            try:
                st.session_state["mcp_fig"] = fig
            except Exception:
                pass

        return fig
