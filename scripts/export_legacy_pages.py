"""Write ``frontend/src/lib/legacy_pages.ts``: the Streamlit pages under /legacy with their URL names (SV-0.10, #279).

    python scripts/export_legacy_pages.py        (part of ``make fixtures``)

The pages and their titles come from ``config/page_registry.py``; the URL name of each is the one ``st.navigation``
gives it (``streamlit.source_util.page_icon_and_name``: ``10_Coke_Energy.py`` → ``Coke_Energy``). The first page is
the default and is served at ``/legacy/`` itself. The SPA links to these pages (``#lib/legacy.ts``);
``tests/test_legacy_pages.py`` fails when the committed file differs from this output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
OUT = ROOT / "frontend" / "src" / "lib" / "legacy_pages.ts"


def legacy_pages() -> list[tuple[str, str]]:
    from streamlit.source_util import page_icon_and_name

    from config.page_registry import get_navigation_pages

    pages = get_navigation_pages()
    return [(p.title, "" if i == 0 else page_icon_and_name(Path(p.file_path))[1]) for i, p in enumerate(pages)]


def render(pages: list[tuple[str, str]]) -> str:
    lines = [
        "// Streamlit pages under /legacy: title in config/page_registry.py → URL name given by st.navigation.",
        "// Written by scripts/export_legacy_pages.py (make fixtures), checked by tests/test_legacy_pages.py.",
        "export const LEGACY_PAGES = {",
    ]
    for title, path in pages:
        key = title if title.isidentifier() else json.dumps(title, ensure_ascii=False).replace('"', "'")
        lines.append(f"\t{key}: '{path}',")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("} as const;")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.write_text(render(legacy_pages()), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
