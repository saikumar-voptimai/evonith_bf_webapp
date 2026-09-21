from __future__ import annotations

import sys
import types

import pandas as pd


def _install_plotly_stubs() -> None:
    """Install lightweight Plotly stubs before importing furnace_tools."""
    plotly = types.ModuleType("plotly")
    express = types.ModuleType("plotly.express")
    graph_objects = types.ModuleType("plotly.graph_objects")
    subplots = types.ModuleType("plotly.subplots")
    subplots.make_subplots = lambda *args, **kwargs: None
    plotly.express = express
    plotly.graph_objects = graph_objects
    plotly.subplots = subplots
    sys.modules.setdefault("plotly", plotly)
    sys.modules.setdefault("plotly.express", express)
    sys.modules.setdefault("plotly.graph_objects", graph_objects)
    sys.modules.setdefault("plotly.subplots", subplots)


def _install_langchain_tool_stub() -> None:
    """Install a no-op LangChain tool decorator for import-only tests."""
    langchain = types.ModuleType("langchain")
    tools = types.ModuleType("langchain.tools")

    def tool(func=None, *args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        if func is None:
            return lambda wrapped: wrapped
        return func

    tools.tool = tool
    langchain.tools = tools
    sys.modules.setdefault("langchain", langchain)
    sys.modules.setdefault("langchain.tools", tools)


_install_plotly_stubs()
_install_langchain_tool_stub()

from agents import furnace_tools  # noqa: E402
from agents.furnace_tools import (  # noqa: E402
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


def test_charge_fetch_sums_mt_columns_into_hour_ending_bucket(monkeypatch) -> None:
    index = pd.DatetimeIndex(
        [
            "2026-01-01T19:40:00Z",
            "2026-01-01T19:45:00Z",
            "2026-01-01T20:00:00Z",
            "2026-01-01T20:10:00Z",
            "2026-01-01T20:20:00Z",
        ],
        name="time",
    )
    raw = pd.DataFrame(
        {"sinter_1_mt": [5, 6, 7, 6, 5], "charges_per_hour": [4, 5, 6, 7, 8]},
        index=index,
    )
    captured = {}

    monkeypatch.setattr(furnace_tools, "_fetch_offline_table_df", lambda **_: raw)
    monkeypatch.setattr(furnace_tools, "_new_dataset_id", lambda _: "offline_test")
    monkeypatch.setattr(
        furnace_tools,
        "_save_dataset",
        lambda *, df, **_: captured.update(df=df),
    )

    furnace_tools.fetch_offline_data(
        report_type="CHARGE",
        table_name="charge_data",
        start_time_utc="2026-01-01T19:30:00Z",
        end_time_utc="2026-01-01T20:30:00Z",
    )

    hourly = captured["df"]
    expected_hour = pd.Timestamp("2026-01-02T02:00:00", tz="Asia/Kolkata")
    assert hourly.loc[expected_hour, "Offline[Charge] - sinter_1_mt"] == 29
    assert hourly.loc[expected_hour, "Offline[Charge] - charges_per_hour"] == 6
