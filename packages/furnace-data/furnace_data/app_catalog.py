"""Frontend-neutral Evonith application catalog."""

from __future__ import annotations

from dataclasses import dataclass


CATALOG_VERSION = "app-catalog-v1"


@dataclass(frozen=True)
class AppCatalogPage:
    """Stable page identifier and display label."""

    id: str
    label: str


APP_PAGES: tuple[AppCatalogPage, ...] = (
    AppCatalogPage("welcome", "Welcome"),
    AppCatalogPage("data_explorer", "Data Explorer"),
    AppCatalogPage("vboard", "V-Board"),
    AppCatalogPage("vsense", "V-Sense"),
    AppCatalogPage("copilot", "CoPilot"),
    AppCatalogPage("material_balance", "Material Balance"),
    AppCatalogPage("furnacemind", "FurnaceMind"),
    AppCatalogPage("blend_optimizer", "Blend Optimizer"),
    AppCatalogPage("feedback", "Feedback"),
)

APP_PAGE_BY_ID = {page.id: page for page in APP_PAGES}
APP_PAGE_BY_LABEL = {page.label.casefold(): page for page in APP_PAGES}


def page_options() -> list[dict[str, str]]:
    """Return JSON-safe page records."""
    return [{"id": page.id, "label": page.label} for page in APP_PAGES]


def page_label(page_id: str | None) -> str | None:
    """Return the label for a stable page id."""
    if not page_id:
        return None
    page = APP_PAGE_BY_ID.get(str(page_id).strip())
    return page.label if page else None


def canonical_page_id(value: str | None) -> str | None:
    """Return a stable page id from either an id or display label."""
    clean = str(value or "").strip()
    if not clean:
        return None
    if clean in APP_PAGE_BY_ID:
        return clean
    label_match = APP_PAGE_BY_LABEL.get(clean.casefold())
    return label_match.id if label_match else None
