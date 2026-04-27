"""Central page registry used by app navigation and feedback dropdowns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppPageDescriptor:
    """Metadata describing one Streamlit page."""

    file_path: str
    title: str
    icon: str
    include_in_feedback: bool = True


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
