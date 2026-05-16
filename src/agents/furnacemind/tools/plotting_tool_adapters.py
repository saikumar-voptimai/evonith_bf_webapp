"""Agent adapter for the ``execute_python_plot`` tool."""

from __future__ import annotations

from agents.furnacemind.tools._utils import log_tool_error
from agents.furnacemind.tools.artifact_store import get_artifact_store
from agents.furnacemind.tools.plotting_sandbox import execute_plot_code


def execute_python_plot(code: str) -> str:
    """Execute restricted Python code to create a Plotly figure."""
    store = get_artifact_store()
    try:
        fig, captured_output = execute_plot_code(code, store.get_active_df())

        if fig is not None:
            store.save_figure(fig, code)
            return "Successfully generated Plotly figure."
        if captured_output:
            return f"Diagnostic output (no figure created):\n{captured_output}"
        return "Code executed but no variable named 'fig' was found."

    except Exception as exc:
        store.append_plot_error(str(exc))
        logged_code = (code[:2000] + "...") if isinstance(code, str) and len(code) > 2000 else code
        log_tool_error(
            tool_name="execute_python_plot",
            params={"code": logged_code},
            error=str(exc),
        )
        return f"Python Error: {exc}"
