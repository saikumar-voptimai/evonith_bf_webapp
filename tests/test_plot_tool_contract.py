from __future__ import annotations

import types

# Plotly and LangChain are dependencies of the project: the real modules are imported. Stubs placed in sys.modules
# here remained for every later test (tests/module_stubs.py).
from agents.furnace_tools import (
    apply_default_plot_style,
    get_openai_tool_schemas,
)


class _FakeLine:
    def __init__(self, color=None):  # noqa: ANN001
        self.color = color


class _FakeTrace:
    def __init__(self, color=None):  # noqa: ANN001
        self.line = _FakeLine(color)
        self.marker = types.SimpleNamespace(color=None)

    def update(self, **kwargs):  # noqa: ANN003
        if "line" in kwargs and "color" in kwargs["line"]:
            self.line.color = kwargs["line"]["color"]
        if "marker" in kwargs and "color" in kwargs["marker"]:
            self.marker.color = kwargs["marker"]["color"]


class _FakeFigure:
    def __init__(self, colors):  # noqa: ANN001
        self.data = [_FakeTrace(color) for color in colors]
        self.layout_updates = []

    def update_layout(self, **kwargs):  # noqa: ANN003
        self.layout_updates.append(kwargs)


def test_heatload_plot_tool_schema_is_available_to_model() -> None:
    """Heatload questions should have a deterministic plotting tool available."""
    tool_names = {schema["function"]["name"] for schema in get_openai_tool_schemas()}

    assert "render_heatload_plot" in tool_names


def test_plot_tool_schema_exposes_no_import_contract_to_model() -> None:
    """The OpenAI tool schema should tell the model how to call the plot tool."""
    plot_schema = next(
        schema
        for schema in get_openai_tool_schemas()
        if schema["function"]["name"] == "execute_python_plot"
    )

    description = plot_schema["function"]["description"]
    assert "Do not include import statements" in description
    assert "preloaded df, pd, px, go, make_subplots, and np" in description


def test_default_plot_style_assigns_distinct_colors_to_plain_multiseries() -> None:
    """Plain model-generated multi-line figures should still be readable."""
    fig = _FakeFigure([None, None, None])

    styled = apply_default_plot_style(fig)

    colors = [trace.line.color for trace in styled.data]
    assert len(set(colors)) == 3
    assert styled.layout_updates[0]["template"] == "plotly_white"
    assert styled.layout_updates[0]["hovermode"] == "x unified"


def test_default_plot_style_recolors_repeated_trace_color() -> None:
    """Repeated explicit colors are usually model mistakes, so diversify them."""
    fig = _FakeFigure(["#0070c0", "#0070c0", "#0070c0"])

    styled = apply_default_plot_style(fig)

    assert len({trace.line.color for trace in styled.data}) == 3


def test_default_plot_style_preserves_existing_distinct_colors() -> None:
    """Hand-authored charts with distinct colors should not be overwritten."""
    fig = _FakeFigure(["red", "green"])

    styled = apply_default_plot_style(fig)

    assert [trace.line.color for trace in styled.data] == ["red", "green"]
