"""Central page registry used by app navigation and feedback dropdowns.

Retirement (SV-6.7, #321; ``docs/retirement.md``): a page whose descriptor carries a :class:`Retirement` leaves the
Streamlit navigation (and the SPA's ``LEGACY_PAGES``, so every link to it fails the type check). Administrators still
open it at its /legacy URL, hidden from the menu, for :data:`RETIREMENT_GRACE_DAYS` days from the recorded date; from
the deletion date it is served to nobody, and ``tests/test_page_registry.py`` fails until its file and its entry are
deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

#: Days a retired page stays reachable for administrators under /legacy before it is deleted.
RETIREMENT_GRACE_DAYS = 30
PLANT_TZ = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class PageSection:
    """A section of a Streamlit page that the SPA links to (``legacyHref(page, section)`` in ``#lib/legacy.ts``).

    ``via="query"``: the page opens the section from ``?section=<key>`` plus ``params`` (and removes them);
    ``via="anchor"``: a heading of the page carries ``anchor=<key>`` and Streamlit scrolls to ``#<key>``.
    """

    key: str
    title: str
    via: Literal["query", "anchor"] = "query"
    params: tuple[str, ...] = ()


@dataclass(frozen=True)
class Retirement:
    """Retirement of a page replaced by the Svelte app (SV-6.7): the IST date it left the navigation (the date of the
    change, never a later one) and the SPA routes that replace it (``"/status"``, ``"/admin/users"``)."""

    retired_on: date
    replaced_by: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.replaced_by or not all(r.startswith("/") for r in self.replaced_by):
            raise ValueError(f"a retired page names the SPA routes that replace it, got {self.replaced_by!r}")

    @property
    def delete_on(self) -> date:
        """First IST day the page is no longer served: ``retired_on`` + 30 days."""
        return self.retired_on + timedelta(days=RETIREMENT_GRACE_DAYS)


@dataclass(frozen=True)
class AppPageDescriptor:
    """Metadata describing one Streamlit page."""

    file_path: str
    title: str
    icon: str
    include_in_feedback: bool = True
    sections: tuple[PageSection, ...] = ()
    retirement: Retirement | None = None


PAGE_REGISTRY: tuple[AppPageDescriptor, ...] = (
    AppPageDescriptor("custom_pages/1_Welcome.py", title="Welcome", icon="🏭"),
    AppPageDescriptor(
        "custom_pages/2_Data_Explorer.py",
        title="Data Explorer",
        icon="📓",
    ),
    AppPageDescriptor(
        "custom_pages/3_Data_Visualisation.py",
        title="V-Board",
        icon="📈",
    ),
    AppPageDescriptor(
        "custom_pages/4_Recommendations.py",
        title="V-Sense",
        icon="💡",
    ),
    AppPageDescriptor("custom_pages/5_AI_Copilot.py", title="CoPilot", icon="🤖"),
    AppPageDescriptor(
        "custom_pages/6_Material_Balance.py",
        title="Material Balance",
        icon="⚖️",
    ),
    AppPageDescriptor(
        "custom_pages/7_FurnaceMind.py",
        title="FurnaceMind",
        icon="🧠",
    ),
    AppPageDescriptor(
        "custom_pages/9_Blend_Optimizer.py",
        title="Blend Mix Optimiser",
        icon=":material/science:",
        # Linked from /bmo (SV-3.6); opened by ui.bmo.snapshot_panel.open_linked_section and the run form's heading.
        sections=(
            PageSection("de-search", "Intensive Optimizer (DE search)", via="anchor"),
            PageSection("replay", "Snapshot replay", params=("snapshot",)),
            PageSection("sandbox", "Sandbox"),
        ),
    ),
    AppPageDescriptor(
        "custom_pages/10_Coke_Energy.py",
        title="Coke Energy Model",
        icon=":material/local_fire_department:",
    ),
    AppPageDescriptor(
        "custom_pages/11_Burden_Fuel_Optimiser.py",
        title="Burden & Fuel Optimiser",
        icon=":material/tune:",
    ),
    AppPageDescriptor("custom_pages/8_Feedback.py", title="Feedback", icon="📮"),
)


def ist_today() -> date:
    """Today's date at the plant (IST)."""
    return datetime.now(PLANT_TZ).date()


def get_navigation_pages() -> tuple[AppPageDescriptor, ...]:
    """Pages in the Streamlit navigation, for every role: those not retired. The first is the default (/legacy/)."""
    return tuple(p for p in PAGE_REGISTRY if p.retirement is None)


def get_retired_pages(today: date | None = None) -> tuple[AppPageDescriptor, ...]:
    """Retired pages still served on ``today`` (IST, default now): to administrators only, hidden from the menu."""
    day = today or ist_today()
    return tuple(p for p in PAGE_REGISTRY if p.retirement is not None and day < p.retirement.delete_on)


def get_expired_pages(today: date | None = None) -> tuple[AppPageDescriptor, ...]:
    """Retired pages on or past their deletion date: served to nobody; their file and entry are to be deleted."""
    day = today or ist_today()
    return tuple(p for p in PAGE_REGISTRY if p.retirement is not None and day >= p.retirement.delete_on)


def served_pages(admin: bool, today: date | None = None) -> tuple[tuple[AppPageDescriptor, bool], ...]:
    """Pages ``st.navigation`` registers for a session, each with whether it is shown in the menu: the navigation
    pages first (so the default page is the same for every role), then, for an administrator, the retired pages in
    their 30 days, hidden. At the URL of a page not registered Streamlit shows the default page."""
    shown = tuple((p, True) for p in get_navigation_pages())
    return shown + (tuple((p, False) for p in get_retired_pages(today)) if admin else ())


def retirement_notice(descriptor: AppPageDescriptor) -> str | None:
    """Banner of a retired page: its replacement and the date it is deleted; ``None`` for a page in service."""
    retired = descriptor.retirement
    if retired is None:
        return None
    routes = ", ".join(retired.replaced_by)
    return (
        f"{descriptor.title} was retired on {retired.retired_on:%d %b %Y}; use {routes} in the BF-02 app. "
        f"Administrators can open this page until {retired.delete_on - timedelta(days=1):%d %b %Y}; "
        "it is deleted after that."
    )


def get_feedback_page_options() -> list[str]:
    """Return page titles available in the feedback form dropdown: the navigation pages."""
    return [descriptor.title for descriptor in get_navigation_pages() if descriptor.include_in_feedback]
