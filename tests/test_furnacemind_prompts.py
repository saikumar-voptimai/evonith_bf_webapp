from __future__ import annotations

from agents.furnacemind import prompts


def test_tool_policy_routes_document_charts_to_knowledge_docs() -> None:
    """Prompt policy should keep uploaded-document visuals in MRAG retrieval."""
    assert "chart/figure/table inside an uploaded document" in prompts.TOOL_POLICY
    assert "use search_knowledge_docs" in prompts.TOOL_POLICY
    assert "Do not create a new furnace telemetry plot" in prompts.TOOL_POLICY


def test_tool_policy_keeps_generic_prompts_out_of_mrag() -> None:
    """Prompt policy should prevent generic help answers from citing documents."""
    assert (
        "Generic assistant capability questions do not require tools"
        in prompts.TOOL_POLICY
    )
    assert "without citing uploaded documents" in prompts.TOOL_POLICY


def test_tool_policy_tells_plot_code_not_to_import() -> None:
    """Prompt policy should match the execute_python_plot sandbox contract."""
    assert "never include import statements" in prompts.TOOL_POLICY
    assert "df, pd, px, go, make_subplots, and np" in prompts.TOOL_POLICY


def test_tool_policy_routes_heatload_charts_to_standard_tool() -> None:
    """Heatload plots should use the deterministic chart tool, not ad hoc code."""
    assert "heatload checks/trends" in prompts.TOOL_POLICY
    assert "render_heatload_plot" in prompts.TOOL_POLICY
