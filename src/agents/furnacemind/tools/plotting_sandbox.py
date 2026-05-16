"""Agent-only sandbox for LLM-generated Plotly code."""

from __future__ import annotations

import io
import re
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_BANNED_PATTERNS = [
    r"__",
    r"\bimport\b",
    r"\bopen\s*\(",
    r"\bos\b",
    r"\bsubprocess\b",
    r"\bsys\b",
    r"\beval\b",
    r"\bexec\b",
]

_SAFE_BUILTINS: dict[str, Any] = {
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
    "bool": bool,
    "round": round,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "None": None,
    "True": True,
    "False": False,
}


def safe_exec(
    code: str,
    local_vars: dict[str, Any],
    stdout_buf: Optional[io.StringIO] = None,
) -> None:
    """Execute code in a restricted namespace."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Empty code string")

    for pattern in _BANNED_PATTERNS:
        if re.search(pattern, code):
            raise ValueError(f"Disallowed token in code: {pattern}")

    if stdout_buf is not None:

        def _buffered_print(*args, **kwargs):  # noqa: ANN202
            kwargs.setdefault("file", stdout_buf)
            print(*args, **kwargs)  # noqa: T201

        captured_print = _buffered_print
    else:
        captured_print = print

    local_vars["__builtins__"] = {**_SAFE_BUILTINS, "print": captured_print}
    exec(code, local_vars)  # noqa: S102


def execute_plot_code(
    code: str,
    df: Optional[pd.DataFrame],
) -> tuple[Any | None, str]:
    """Run LLM-generated Plotly code and return ``(fig, captured_stdout)``."""
    import numpy as np  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    stdout_buf = io.StringIO()
    local_vars: dict[str, Any] = {
        "pd": pd,
        "px": px,
        "go": go,
        "df": df,
        "np": np,
        "make_subplots": make_subplots,
    }

    safe_exec(code, local_vars, stdout_buf=stdout_buf)
    return local_vars.get("fig"), stdout_buf.getvalue().strip()
