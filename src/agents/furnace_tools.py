"""Deprecated compatibility facade for FurnaceMind tools.

New code should import from ``agents.furnacemind.tools`` directly.
"""

from __future__ import annotations

from typing import Any, Dict

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
from agents.furnacemind.tools.registry import (
    execute_openai_tool_call as _execute_openai_tool_call,
    get_openai_tool_schemas,
)

__all__ = [
    "fetch_online_data",
    "fetch_offline_data",
    "merge_furnace_data",
    "fetch_ml_data",
    "concat_datasets",
    "load_static_shift_data",
    "execute_python_plot",
    "get_openai_tool_schemas",
    "execute_openai_tool_call",
    "search_shift_history",
    "search_knowledge_docs",
]


def execute_openai_tool_call(*, name: str, arguments: Dict[str, Any]) -> str:
    """Dispatch a tool call by name."""
    return _execute_openai_tool_call(name=name, arguments=arguments)
