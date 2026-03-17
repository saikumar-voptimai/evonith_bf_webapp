import pandas as pd
import streamlit as st
from langchain.tools import tool
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils.helper_functions_explorer import data_retrieval as dr
from config.config_loader import load_config

# CONFIG
config = load_config("setting_ds_dv.yml")

MEASUREMENT_LABELS = {
    "heatload_delta_t": "Heatload Delta T",
    "process_params": "Process Params",
    "temperature_profile": "Temperature Profile",
    "cooling_water": "Cooling Water",
    "delta_t": "Delta T",
    "miscellaneous": "Miscellaneous",
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


_TOOL_ERRORS_PATH = Path(__file__).resolve().parent / "tool_errors.md"


def _append_tool_error(*, tool_name: str, params: Dict[str, Any], error: str) -> None:
    """Append tool failure details to tool_errors.md (best-effort, never raises)."""
    try:
        _TOOL_ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        entry = (
            f"\n\n## {ts}\n"
            f"**Tool:** {tool_name}\n\n"
            f"**Params:** `{json.dumps(params, ensure_ascii=False)}`\n\n"
            f"**Error:**\n\n```\n{error}\n```\n"
        )
        if _TOOL_ERRORS_PATH.exists():
            _TOOL_ERRORS_PATH.write_text(_TOOL_ERRORS_PATH.read_text(encoding="utf-8") + entry, encoding="utf-8")
        else:
            _TOOL_ERRORS_PATH.write_text(
                "# FurnaceMind Tool Errors & Learnings\n\n"
                "This file is auto-updated when tool execution fails during AI Co-Operate sessions.\n\n---\n"
                + entry,
                encoding="utf-8",
            )
    except Exception:
        return


def _normalize_time_range(user_time_range: str) -> str:
    """Normalize natural language into dr.TIMEDELTAS-compatible keys, extending TIMEDELTAS as needed."""
    tr = (user_time_range or "").strip().lower()
    if not tr:
        return "last 8 hours"

    # Already supported
    if hasattr(dr, "TIMEDELTAS") and tr in dr.TIMEDELTAS:
        return tr

    m = re.search(r"last\s+(\d+)\s*(minute|minutes|min|mins)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} minutes" if n != 1 else "last 1 minute"
        if hasattr(dr, "TIMEDELTAS"):
            dr.TIMEDELTAS.setdefault(key, timedelta(minutes=n))
        return key

    m = re.search(r"last\s+(\d+)\s*(hour|hours|hr|hrs|h)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} hours" if n != 1 else "last 1 hour"
        if hasattr(dr, "TIMEDELTAS"):
            dr.TIMEDELTAS.setdefault(key, timedelta(hours=n))
        return key

    m = re.search(r"last\s+(\d+)\s*(day|days|d)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} days" if n != 1 else "last 1 day"
        if hasattr(dr, "TIMEDELTAS"):
            dr.TIMEDELTAS.setdefault(key, timedelta(days=n))
        return key

    m = re.search(r"last\s+(\d+)\s*(week|weeks|w)\b", tr)
    if m:
        n = int(m.group(1))
        key = f"last {n} weeks" if n != 1 else "last 1 week"
        if hasattr(dr, "TIMEDELTAS"):
            dr.TIMEDELTAS.setdefault(key, timedelta(weeks=n))
        return key

    # Fallback to safe default
    return "last 8 hours"


def _safe_exec(code: str, local_vars: Dict[str, Any]) -> None:
    """Execute plotting code with a restricted builtins set and basic static checks."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Empty code string")

    # Basic static blocks
    banned = [
        r"\bimport\b",
        r"\bopen\s*\(",
        r"__import__",
        r"\bos\b",
        r"\bsubprocess\b",
        r"\bsys\b",
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


@tool
def fetch_and_summarize_data(time_range: str, window: str = "15 minutes") -> str:
    """
    Fetch data and save to a temp file. 
    Returns the first 5 rows and column names so the Python tool knows how to code.
    """
    
    try:
        normalized_time_range = _normalize_time_range(time_range)

        df = dr.fetch_online_df(
            selected_measurements=[
                "process_params",
                "cooling_water",
                "heatload_delta_t",
                "delta_t",
                "temperature_profile",
                "miscellaneous",
            ],
            time_range=normalized_time_range,
            average_range=window,
            FREQUENCY_TO_TIMEDTA=FREQUENCY_TO_TIMEDTA,
            MEASUREMENT_LABELS=MEASUREMENT_LABELS,
            FIELD_LABELS=FIELD_LABELS,
        )
    except Exception as e:
        _append_tool_error(
            tool_name="fetch_and_summarize_data",
            params={"time_range": time_range, "window": window},
            error=str(e),
        )
        return f"Fetch Error: {str(e)}"
    
    if df is None or df.empty:
        return "No data found."

    # Save to a fixed path for the Python executor to find
    # Keep index so timestamps remain available
    df.to_csv("current_furnace_data.csv", index=True)
    st.session_state.copilot_df = df
    
    # Give the LLM a 'peek' at the data
    summary = (
        "Data saved to 'current_furnace_data.csv'.\n"
        "Note: Columns are renamed as 'Measurement - Field'.\n"
        f"Time range: {time_range} | Window: {window}\n"
        f"Columns ({len(df.columns)}): {list(df.columns)}\n\n"
        f"Preview:\n{df.head(2).to_string()}"
    )
    return summary

@tool
def execute_python_plot(code: str) -> str:
    """
    Execute python code to create a Plotly figure. 
    The code MUST:
    1. Read data from 'current_furnace_data.csv'.
    2. Create a plotly figure named 'fig'.
    3. Not use 'fig.show()'.
    Example: fig = px.scatter(pd.read_csv('current_furnace_data.csv'), x='A', y='B')
    """
    try:
        # Preload the dataframe for convenience (LLM may still choose to read CSV explicitly)
        try:
            df = pd.read_csv("current_furnace_data.csv", index_col=0, parse_dates=True)
        except Exception:
            df = None

        # Create a local environment for execution
        local_vars = {"pd": pd, "px": px, "go": go, "df": df}

        # Execute the LLM-generated code (restricted)
        _safe_exec(code, local_vars)
        
        if "fig" in local_vars:
            # Save the figure object to session state for the UI to pick up
            st.session_state.copilot_fig = local_vars["fig"]
            st.session_state.last_plot_code = code
            return "Successfully generated Plotly figure."
        else:
            return "Code executed but no variable named 'fig' was found."
            
    except Exception as e:
        _append_tool_error(
            tool_name="execute_python_plot",
            params={"code": (code[:2000] + "…") if isinstance(code, str) and len(code) > 2000 else code},
            error=str(e),
        )
        st.session_state.last_plot_error = str(e)
        return f"Python Error: {str(e)}"


@tool
def search_shift_history(query: str) -> str:
    """
    Search past shift summaries using semantic similarity.
    Use for questions about past shifts, stability, anomalies, or shift performance.
    """
    shift_store = st.session_state.get("shift_store")
    if shift_store is None:
        return "Shift store not initialized."

    results = shift_store.search_similar_windows(query_text=query, top_k=5)

    if not results:
        return "No shift summaries found for this query."

    parts = []
    for i, r in enumerate(results, 1):
        payload = r.get("payload", {})
        text = payload.get("summary_text", "No summary.")
        window_id = payload.get("window_id", "unknown")
        parts.append(f"[{i}] Shift: {window_id}\n{text}")

    return "\n\n".join(parts)


@tool
def search_knowledge_docs(query: str) -> str:
    """
    Search uploaded knowledge documents (SOPs, manuals, specs, policies).
    Use for questions about procedures, specifications, or reference material.
    """
    knowledge_store = st.session_state.get("knowledge_store")
    if knowledge_store is None:
        return "Knowledge store not initialized."

    results = knowledge_store.search(query, top_k=5)

    if not results:
        return "No knowledge documents found for this query."

    parts = []
    for i, r in enumerate(results, 1):
        payload = r.get("payload", {})
        content = payload.get("content", "No content.")
        source = payload.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{content}")

    return "\n\n".join(parts)