"""Write ``frontend/src/lib/legacy_pages.ts``: the Streamlit pages under /legacy with their URL names (SV-0.10, #279).

    python scripts/export_legacy_pages.py        (part of ``make fixtures``)

The pages and their titles come from ``config/page_registry.py``; the URL name of each is the one ``st.navigation``
gives it (``streamlit.source_util.page_icon_and_name``: ``10_Coke_Energy.py`` → ``Coke_Energy``). The first page is
the default and is served at ``/legacy/`` itself. The sections the SPA links to (``PageSection``, SV-3.6) are written
as ``LEGACY_SECTIONS``: how the page opens each (query or heading anchor) and the further query parameters it reads.
The SPA links to these pages (``#lib/legacy.ts``); ``tests/test_legacy_pages.py`` fails when the committed file
differs from this output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
OUT = ROOT / "frontend" / "src" / "lib" / "legacy_pages.ts"

#: (key, title, via, params) of a linked section.
Section = tuple[str, str, str, tuple[str, ...]]


def legacy_pages() -> list[tuple[str, str]]:
    from streamlit.source_util import page_icon_and_name

    from config.page_registry import get_navigation_pages

    pages = get_navigation_pages()
    return [(p.title, "" if i == 0 else page_icon_and_name(Path(p.file_path))[1]) for i, p in enumerate(pages)]


def legacy_sections() -> list[tuple[str, list[Section]]]:
    """Pages with linked sections: (page title, its sections)."""
    from config.page_registry import get_navigation_pages

    return [
        (p.title, [(s.key, s.title, s.via, s.params) for s in p.sections]) for p in get_navigation_pages() if p.sections
    ]


def _key(name: str) -> str:
    return name if name.isidentifier() else json.dumps(name, ensure_ascii=False).replace('"', "'")


def render(pages: list[tuple[str, str]], sections: list[tuple[str, list[Section]]]) -> str:
    lines = [
        "// Streamlit pages under /legacy: title in config/page_registry.py → URL name given by st.navigation.",
        "// Written by scripts/export_legacy_pages.py (make fixtures), checked by tests/test_legacy_pages.py.",
        "export const LEGACY_PAGES = {",
    ]
    for title, path in pages:
        lines.append(f"\t{_key(title)}: '{path}',")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("} as const;")
    lines += [
        "",
        "// Sections the SPA links to (PageSection): via 'query' = ?section=<key> and the params; 'anchor' = #<key>.",
        "export const LEGACY_SECTIONS = {",
    ]
    for page_index, (title, items) in enumerate(sections):
        lines.append(f"\t{_key(title)}: {{")
        for index, (key, name, via, params) in enumerate(items):
            listed = ", ".join(f"'{p}'" for p in params)
            comma = "," if index < len(items) - 1 else ""
            lines.append(f"\t\t{_key(key)}: {{ title: '{name}', via: '{via}', params: [{listed}] }}{comma}")
        lines.append("\t}" + ("," if page_index < len(sections) - 1 else ""))
    lines.append("} as const;")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.write_text(render(legacy_pages(), legacy_sections()), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
