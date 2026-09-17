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
    AppPageDescriptor(
        "custom_pages/9_Blend_Optimizer.py",
        title="Blend Optimizer",
        icon=":material/science:",
    ),
    AppPageDescriptor(
        "custom_pages/10_Scheduled_Jobs.py",
        title="Scheduled Jobs",
        icon="⏱️",
    ),
    AppPageDescriptor("custom_pages/8_Feedback.py", title="Feedback", icon="📮"),
)

# Keep hidden pages registered so they can be restored without recreating their
# metadata or page implementations.
HIDDEN_PAGE_PATHS = frozenset(
    {
        "custom_pages/4_Recommendations.py",
        "custom_pages/6_Material_Balance.py",
        "custom_pages/8_Feedback.py",
        "custom_pages/9_Blend_Optimizer.py",
    }
)
_HIDDEN_PAGE_PATHS_CASEFOLD = {path.casefold() for path in HIDDEN_PAGE_PATHS}


def is_page_visible(file_path: str) -> bool:
    """Return whether a registered page should appear in the application UI."""
    return file_path.casefold() not in _HIDDEN_PAGE_PATHS_CASEFOLD



def get_navigation_pages() -> tuple[AppPageDescriptor, ...]:
    """Return page descriptors that should appear in sidebar navigation."""
    return tuple(
        descriptor
        for descriptor in PAGE_REGISTRY
        if is_page_visible(descriptor.file_path)
    )


def get_feedback_page_options() -> list[str]:
    """Return page titles available in the feedback form dropdown."""
    return [
        descriptor.title
        for descriptor in PAGE_REGISTRY
        if descriptor.include_in_feedback and is_page_visible(descriptor.file_path)
    ]
