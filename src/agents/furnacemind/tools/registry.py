"""OpenAI tool schema registry and dispatcher for FurnaceMind."""

from __future__ import annotations

from typing import Any, Dict

from agents.furnacemind.tools.artifact_store import ArtifactStore, set_artifact_store
from agents.furnacemind.tools.data_tool_adapters import (
    concat_datasets,
    fetch_ml_data,
    fetch_offline_data,
    fetch_online_data,
    load_static_shift_data,
    merge_furnace_data,
)
from agents.furnacemind.tools.memory_tool_adapters import (
    search_knowledge_docs,
    search_shift_history,
)
from agents.furnacemind.tools.plotting_tool_adapters import execute_python_plot
from furnace_data.neon_db.offline import NEON_OFFLINE_TABLES


def configure_artifact_store(store: ArtifactStore) -> None:
    """Configure the artifact store used by tool adapters."""
    set_artifact_store(store)


def get_openai_tool_schemas() -> list[dict]:
    """Return OpenAI/OpenRouter tool schemas for LLM function-calling."""
    from utils.settings import settings as _settings  # noqa: PLC0415

    tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_online_data",
                "description": (
                    "Fetch live InfluxDB telemetry. Use either lookback "
                    "(for example '8h', '2d', '30m') or start_time_utc plus "
                    "end_time_utc for exact windows. Stores a dataframe artifact "
                    "and returns dataset_id plus column preview."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lookback": {"type": "string"},
                        "window": {"type": "string"},
                        "start_time_utc": {"type": "string"},
                        "end_time_utc": {"type": "string"},
                        "measurement_groups": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "process_params",
                                    "cooling_water",
                                    "heatload_delta_t",
                                    "delta_t",
                                    "temperature_profile",
                                    "miscellaneous",
                                ],
                            },
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_offline_data",
                "description": (
                    "Fetch offline report datasets: HM/Slag, Charge, DPR, "
                    "raw material composition, burden distribution, or hopper management."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report_type": {
                            "type": "string",
                            "enum": [
                                "HM_SLAG",
                                "CHARGE",
                                "RAW_MATERIAL_COMPOSITION",
                                "RM_COMPOSITION",
                                "DPR",
                                "BURDEN_DISTRIBUTION",
                                "HOPPER_MANAGEMENT",
                            ],
                        },
                        "table_name": {
                            "type": "string",
                            "enum": sorted(NEON_OFFLINE_TABLES.keys()),
                        },
                        "start_time_utc": {"type": "string"},
                        "end_time_utc": {"type": "string"},
                        "lookback_days": {"type": "integer", "minimum": 1, "maximum": 365},
                        "cadence": {"type": "string", "enum": ["1h", "8h", "1d"]},
                    },
                    "required": ["report_type"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "merge_furnace_data",
                "description": (
                    "Merge offline datasets onto an online dataset by aligning "
                    "offline rows to online timestamps."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "online_dataset_id": {"type": "string"},
                        "offline_dataset_ids": {"type": "array", "items": {"type": "string"}},
                        "fill_method": {"type": "string", "enum": ["ffill", "none"], "default": "ffill"},
                    },
                    "required": ["online_dataset_id", "offline_dataset_ids"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_shift_history",
                "description": "Search past shift summaries.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_docs",
                "description": "Search uploaded knowledge documents.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python_plot",
                "description": (
                    "Execute restricted Python to create a Plotly figure named fig "
                    "using the active dataframe artifact as df."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_ml_data",
                "description": (
                    "Fetch historical static ML data. Prefer this for historical "
                    "queries spanning more than two days."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "string"},
                        "end_time": {"type": "string"},
                        "resample": {"type": "string", "enum": ["1h", "4h", "8h", "1d"]},
                        "columns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["start_time"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "concat_datasets",
                "description": "Concatenate dataframe artifacts vertically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["dataset_ids"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_static_shift_data",
                "description": "Load one 8-hour shift from the static ML dataset.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shift_date": {"type": "string"},
                        "shift_label": {"type": "string", "enum": ["A", "B", "C"]},
                    },
                    "required": ["shift_date", "shift_label"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    if not _settings.enable_shift_history_vector:
        tools = [tool for tool in tools if tool["function"]["name"] != "search_shift_history"]
    return tools


def execute_openai_tool_call(*, name: str, arguments: Dict[str, Any]) -> str:
    """Dispatch a tool call by name."""
    if name == "fetch_ml_data":
        return fetch_ml_data(**arguments)
    if name == "concat_datasets":
        return concat_datasets(**arguments)
    if name == "fetch_online_data":
        return fetch_online_data(**arguments)
    if name == "fetch_offline_data":
        return fetch_offline_data(**arguments)
    if name == "merge_furnace_data":
        return merge_furnace_data(**arguments)
    if name == "load_static_shift_data":
        return load_static_shift_data(**arguments)
    if name == "search_shift_history":
        return search_shift_history(**arguments)
    if name == "search_knowledge_docs":
        return search_knowledge_docs(**arguments)
    if name == "execute_python_plot":
        if not arguments.get("code"):
            return "Error: execute_python_plot requires a non-empty 'code' argument."
        return execute_python_plot(arguments["code"])
    return f"Unknown tool: {name}"
