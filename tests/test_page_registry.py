"""Tests for shared page registry metadata and the retirement of Streamlit pages (SV-6.7, #321)."""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from config import page_registry as registry
from config.page_registry import (
    PAGE_REGISTRY,
    Retirement,
    get_expired_pages,
    get_feedback_page_options,
    get_navigation_pages,
    get_retired_pages,
    retirement_notice,
    served_pages,
)

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "retirement.md"
ROUTES = ROOT / "frontend" / "src" / "routes"
RETIRED_ON = date(2026, 11, 2)


def _exporter():
    spec = importlib.util.spec_from_file_location("export_legacy_pages", ROOT / "scripts" / "export_legacy_pages.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _retire(
    monkeypatch: pytest.MonkeyPatch, routes: dict[str, tuple[str, ...]]
) -> dict[str, registry.AppPageDescriptor]:
    """The registry with the titled pages retired on ``RETIRED_ON``."""
    pages = tuple(
        dataclasses.replace(p, retirement=Retirement(RETIRED_ON, routes[p.title])) if p.title in routes else p
        for p in PAGE_REGISTRY
    )
    monkeypatch.setattr(registry, "PAGE_REGISTRY", pages)
    return {p.title: p for p in pages}


def _titles(pages) -> list[str]:
    return [p.title for p in pages]


def test_feedback_options_come_from_navigation_registry() -> None:
    """Feedback dropdown options should mirror navigation titles."""
    descriptors = get_navigation_pages()
    expected_titles = [descriptor.title for descriptor in descriptors if descriptor.include_in_feedback]
    assert get_feedback_page_options() == expected_titles


# ------------------------------------------------------------------------------------------------- the mechanism
def test_a_retired_page_leaves_the_navigation_and_stays_hidden_for_administrators_for_30_days(monkeypatch) -> None:
    pages = _retire(monkeypatch, {"V-Board": ("/status",)})
    v_board = pages["V-Board"]
    assert "V-Board" not in _titles(get_navigation_pages())
    assert "V-Board" not in get_feedback_page_options()

    for day in (RETIRED_ON, date(2026, 12, 1)):  # the first and the 30th day
        assert v_board not in [p for p, _ in served_pages(admin=False, today=day)]
        served = served_pages(admin=True, today=day)
        assert served[-1] == (v_board, False)  # registered for the administrator, hidden from the menu
        assert all(shown for p, shown in served if p is not v_board)
        assert get_retired_pages(day) == (v_board,) and get_expired_pages(day) == ()

    deleted = date(2026, 12, 2)
    assert v_board.retirement is not None and v_board.retirement.delete_on == deleted
    assert v_board not in [p for p, _ in served_pages(admin=True, today=deleted)]
    assert get_retired_pages(deleted) == () and get_expired_pages(deleted) == (v_board,)


def test_the_default_page_is_the_first_navigation_page_for_every_role_and_in_the_spa_map(monkeypatch) -> None:
    _retire(monkeypatch, {"Welcome": ("/", "/admin/users")})
    for admin in (False, True):
        first, shown = served_pages(admin=admin, today=RETIRED_ON)[0]
        assert (first.title, shown) == ("Data Explorer", True)
    pages = dict(_exporter().legacy_pages())
    assert "Welcome" not in pages and pages["Data Explorer"] == ""


def test_a_retired_page_and_its_sections_leave_the_spa_page_map(monkeypatch) -> None:
    _retire(monkeypatch, {"Blend Mix Optimiser": ("/bmo",)})
    exporter = _exporter()
    rendered = exporter.render(exporter.legacy_pages(), exporter.legacy_sections())
    assert "Blend Mix Optimiser" not in rendered and "Blend_Optimizer" not in rendered
    assert "export const LEGACY_SECTIONS = {\n} as const;" in rendered


def test_the_banner_names_the_replacement_and_the_last_day() -> None:
    page = dataclasses.replace(PAGE_REGISTRY[2], retirement=Retirement(RETIRED_ON, ("/status",)))
    notice = retirement_notice(page)
    assert notice is not None
    assert "retired on 02 Nov 2026" in notice and "/status" in notice and "until 01 Dec 2026" in notice
    assert retirement_notice(PAGE_REGISTRY[2]) is None


@pytest.mark.parametrize("routes", [(), ("status",), ("/status", "admin")])
def test_a_retirement_names_the_spa_routes_that_replace_the_page(routes) -> None:
    with pytest.raises(ValueError, match="SPA routes"):
        Retirement(RETIRED_ON, routes)


# -------------------------------------------------------------------------------------------- the real registry
def test_no_retired_page_is_past_its_deletion_date() -> None:
    expired = [(p.title, p.file_path) for p in get_expired_pages()]
    assert not expired, f"30 days are over: delete the page file, its registry entry and its code: {expired}"


def _route_exists(route: str) -> bool:
    return (ROUTES / route.strip("/") / "+page.svelte").is_file()


def test_every_page_file_exists_and_a_retired_page_names_existing_spa_routes() -> None:
    for page in PAGE_REGISTRY:
        assert (ROOT / "src" / page.file_path).is_file(), page.file_path
        if page.retirement is not None:
            assert all(_route_exists(r) for r in page.retirement.replaced_by), page.retirement.replaced_by


def test_the_welcome_tiles_link_registered_pages_and_are_filtered_by_the_navigation() -> None:
    source = (ROOT / "src" / "custom_pages" / "1_Welcome.py").read_text(encoding="utf-8")
    linked = set(re.findall(r'"(custom_pages/[^"]+\.py)"', source))
    assert linked and linked <= {p.file_path for p in PAGE_REGISTRY}
    assert "if page_path not in _IN_NAVIGATION" in source


@pytest.mark.parametrize(("role", "opened", "banner"), [("admin", "retired page", True), ("user", "live page", False)])
def test_the_app_serves_a_retired_page_to_administrators_only_with_its_banner(
    tmp_path, monkeypatch, role, opened, banner
) -> None:
    """``src/app.py`` under AppTest, signed in, with a registry of a live page and a page retired 3 days ago: the
    administrator opens the retired page at its URL with the banner; another role falls back to the default page."""
    from streamlit.testing.v1 import AppTest

    for name, text in (("live.py", "live page"), ("old.py", "retired page")):
        (tmp_path / name).write_text(f"import streamlit as st\nst.write({text!r})\n", encoding="utf-8")
    retired = Retirement(registry.ist_today() - timedelta(days=3), ("/status",))
    monkeypatch.setattr(
        registry,
        "PAGE_REGISTRY",
        (
            registry.AppPageDescriptor(str(tmp_path / "live.py"), "Live", "🏭"),
            registry.AppPageDescriptor(str(tmp_path / "old.py"), "Old", "📈", retirement=retired),
        ),
    )
    monkeypatch.chdir(ROOT)  # the logger reads its configuration relative to the repository root (run_streamlit.py)
    # A fresh utils.session on the real Streamlit (tests/test_session_auth.py leaves one bound to its stub).
    monkeypatch.delitem(sys.modules, "utils.session", raising=False)

    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=60)
    at.session_state["auth_user"] = "operator"
    at.session_state["role"] = role
    at.run()
    assert [m.value for m in at.markdown] == ["live page"] and not at.exception

    at.switch_page(str(tmp_path / "old.py")).run()
    assert [m.value for m in at.markdown] == [opened] and not at.exception
    assert [w.value.startswith("Old was retired on") for w in at.warning] == ([True] if banner else [])


# --------------------------------------------------------------------------------------- docs/retirement.md
def _retirement_rows() -> list[dict[str, str]]:
    """Rows of the table under "## Retirement list", by column header."""
    lines = DOC.read_text(encoding="utf-8").splitlines()
    after = itertools.dropwhile(lambda line: not line.startswith("|"), lines[lines.index("## Retirement list") :])
    table = list(itertools.takewhile(lambda line: line.startswith("|"), after))
    header = [c.strip() for c in table[0].strip("|").split("|")]
    return [dict(zip(header, (c.strip() for c in row.strip("|").split("|")), strict=True)) for row in table[2:]]


def test_the_retirement_list_has_a_row_per_registered_page_with_its_dates_and_routes() -> None:
    rows = _retirement_rows()
    assert [r["Streamlit page"] for r in rows] == _titles(PAGE_REGISTRY)
    for row, page in zip(rows, PAGE_REGISTRY, strict=True):
        routes = re.findall(r"`(/[^`]*)`", row["Svelte replacement"])
        assert all(_route_exists(r) for r in routes), (page.title, routes)
        retired = page.retirement
        if retired is None:
            assert (row["Retired on"], row["Deleted on"]) == ("not retired", "—"), page.title
        else:
            assert (row["Retired on"], row["Deleted on"]) == (str(retired.retired_on), str(retired.delete_on))
            assert set(routes) == set(retired.replaced_by), page.title
            assert "pending" not in (row["Shadow use from"], row["Operator feedback"], row["Sign-off"]), page.title


def test_every_parity_test_named_in_the_retirement_list_exists() -> None:
    named = re.findall(r"`(tests/[\w/]+\.py)::(test_\w+)`", DOC.read_text(encoding="utf-8"))
    assert len(named) >= 30
    for path, name in named:
        assert re.search(rf"^def {name}\(", (ROOT / path).read_text(encoding="utf-8"), re.M), f"{path}::{name}"
