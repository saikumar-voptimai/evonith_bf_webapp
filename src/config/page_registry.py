"""Central page registry used by app navigation and feedback dropdowns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
class AppPageDescriptor:
    """Metadata describing one Streamlit page."""

    file_path: str
    title: str
    icon: str
    include_in_feedback: bool = True
    sections: tuple[PageSection, ...] = ()


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


def get_navigation_pages() -> tuple[AppPageDescriptor, ...]:
    """Return the full navigation registry."""
    return PAGE_REGISTRY


def get_feedback_page_options() -> list[str]:
    """Return page titles available in the feedback form dropdown."""
    return [
        descriptor.title
        for descriptor in PAGE_REGISTRY
        if descriptor.include_in_feedback
    ]
