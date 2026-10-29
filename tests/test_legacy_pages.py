"""The SPA's links to the Streamlit pages under /legacy follow ``config/page_registry.py`` (#279), and each linked
section opens on its page (SV-3.6, #303).

``frontend/src/lib/legacy_pages.ts`` is written by ``python scripts/export_legacy_pages.py`` (``make fixtures``);
``frontend/src/lib/legacy.test.ts`` checks the hrefs the SPA builds from it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_legacy_pages", ROOT / "scripts" / "export_legacy_pages.py")
assert spec is not None and spec.loader is not None
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_generated_page_map_is_current() -> None:
    committed = exporter.OUT.read_text(encoding="utf-8")
    assert committed == exporter.render(exporter.legacy_pages(), exporter.legacy_sections()), (
        "Regenerate with: make fixtures"
    )


def test_url_names_are_those_of_st_navigation() -> None:
    pages = dict(exporter.legacy_pages())
    assert pages["Welcome"] == ""  # the default page is the Streamlit home, /legacy/
    assert pages["Coke Energy Model"] == "Coke_Energy"
    assert pages["Burden & Fuel Optimiser"] == "Burden_Fuel_Optimiser"
    assert all("/" not in p and " " not in p for p in pages.values())


# ---------------------------------------------------------------------------------------------- page 9 sections
def _page9_sections() -> dict[str, Any]:
    from config.page_registry import get_navigation_pages

    page = next(p for p in get_navigation_pages() if p.title == "Blend Mix Optimiser")
    return {s.key: s for s in page.sections}


def test_the_blend_mix_optimiser_links_de_search_replay_and_sandbox() -> None:
    sections = _page9_sections()
    assert {k: (s.via, s.params) for k, s in sections.items()} == {
        "de-search": ("anchor", ()),
        "replay": ("query", ("snapshot",)),
        "sandbox": ("query", ()),
    }


def test_the_de_search_anchor_is_the_heading_of_the_run_form() -> None:
    """``#de-search`` scrolls to the heading above "Run Intensive Optimizer" (Streamlit scrolls to a heading whose
    anchor is the URL fragment, and keeps the fragment while the page sets no query parameter)."""
    from ui.bmo import snapshot_panel

    source = (ROOT / "src" / "custom_pages" / "9_Blend_Optimizer.py").read_text(encoding="utf-8")
    heading = source.index("anchor=DE_SEARCH_ANCHOR")
    assert snapshot_panel.DE_SEARCH_ANCHOR == "de-search" and _page9_sections()["de-search"].via == "anchor"
    assert heading < source.index('"Run Intensive Optimizer"') and source.index("render_mode_gate(") < heading
    assert "st.query_params" not in source  # a query change would drop the fragment before the heading is drawn


class QueryParams(dict):
    """``st.query_params`` as the gate uses it: get, membership, deletion."""


@pytest.mark.parametrize(
    ("params", "opened", "sandbox", "loaded"),
    [
        ({"section": "sandbox"}, "sandbox", True, None),
        ({"section": "replay", "snapshot": "20260925_135718_blend_mix_optimiser"}, "replay", True,
         "20260925_135718_blend_mix_optimiser"),
        ({"section": "replay"}, None, None, None),  # no snapshot named: nothing opens
        ({"section": "unknown"}, None, None, None),
        ({}, None, None, None),
    ],
)  # fmt: skip
def test_a_query_section_opens_once(params, opened, sandbox, loaded) -> None:
    from ui.bmo import snapshot_panel as panel

    state: dict[str, Any] = {}
    query = QueryParams(params)
    calls: list[str] = []

    def loader(snapshot_id: str) -> dict[str, Any]:
        calls.append(snapshot_id)
        return {"id": snapshot_id, "inputs": {"target_production_mt": 2250.0}, "summary": {}}

    assert panel.open_linked_section(state, query, loader) == opened
    assert state.get(panel.MODE_KEY) is sandbox
    assert calls == ([loaded] if loaded else [])
    if loaded:
        assert state[f"{panel.SB_UI}saved_choice"] == loaded and state[panel.LOADED_KEY]["id"] == loaded
        assert state[f"{panel.SANDBOX_PREFIX}target_production_mt"] == 2250.0
    assert "section" not in query and "snapshot" not in query  # consumed: the switch works as usual afterwards
    if opened:
        assert opened in _page9_sections() and _page9_sections()[opened].via == "query"


def test_a_snapshot_that_does_not_load_opens_the_sandbox_with_the_error() -> None:
    from ui.bmo import snapshot_panel as panel

    def loader(snapshot_id: str) -> dict[str, Any]:
        raise FileNotFoundError(snapshot_id)

    state: dict[str, Any] = {}
    assert panel.open_linked_section(state, QueryParams(section="replay", snapshot="gone"), loader) == "replay"
    assert state[panel.MODE_KEY] is True and "gone" in state[panel.LOAD_ERROR_KEY]


def test_every_query_section_of_the_registry_is_opened_by_the_gate() -> None:
    from ui.bmo import snapshot_panel as panel

    for key, section in _page9_sections().items():
        if section.via != "query":
            continue
        params = QueryParams(section=key, **{p: "20260925_135718_blend_mix_optimiser" for p in section.params})
        state: dict[str, Any] = {}
        assert panel.open_linked_section(state, params, lambda sid: {"id": sid, "inputs": {}}) == key, key
