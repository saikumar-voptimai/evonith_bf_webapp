"""The SPA's links to the Streamlit pages under /legacy follow ``config/page_registry.py`` (#279).

``frontend/src/lib/legacy_pages.ts`` is written by ``python scripts/export_legacy_pages.py`` (``make fixtures``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_legacy_pages", ROOT / "scripts" / "export_legacy_pages.py")
assert spec is not None and spec.loader is not None
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_generated_page_map_is_current() -> None:
    committed = exporter.OUT.read_text(encoding="utf-8")
    assert committed == exporter.render(exporter.legacy_pages()), "Regenerate with: make fixtures"


def test_url_names_are_those_of_st_navigation() -> None:
    pages = dict(exporter.legacy_pages())
    assert pages["Welcome"] == ""  # the default page is the Streamlit home, /legacy/
    assert pages["Coke Energy Model"] == "Coke_Energy"
    assert pages["Burden & Fuel Optimiser"] == "Burden_Fuel_Optimiser"
    assert all("/" not in p and " " not in p for p in pages.values())
