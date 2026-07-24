"""Central page registry used by app navigation and feedback dropdowns."""

from __future__ import annotations

from dataclasses import dataclass

from furnace_data.app_catalog import APP_PAGE_BY_ID, APP_PAGES


@dataclass(frozen=True)
class AppPageDescriptor:
    """Metadata describing one Streamlit page."""

    page_id: str
    file_path: str
    title: str
    icon: str
    include_in_feedback: bool = True


def _label(page_id: str) -> str:
    return APP_PAGE_BY_ID[page_id].label


PAGE_REGISTRY: tuple[AppPageDescriptor, ...] = (
    AppPageDescriptor("welcome", "custom_pages/1_Welcome.py", title=_label("welcome"), icon=":material/home:"),
    AppPageDescriptor("data_explorer", "custom_pages/2_Data_Explorer.py", title=_label("data_explorer"), icon=":material/table_view:"),
    AppPageDescriptor("vboard", "custom_pages/3_Data_Visualisation.py", title=_label("vboard"), icon=":material/monitoring:"),
    AppPageDescriptor("vsense", "custom_pages/4_Recommendations.py", title=_label("vsense"), icon=":material/lightbulb:"),
    AppPageDescriptor("copilot", "custom_pages/5_AI_Copilot.py", title=_label("copilot"), icon=":material/smart_toy:"),
    AppPageDescriptor("material_balance", "custom_pages/6_Material_Balance.py", title=_label("material_balance"), icon=":material/balance:"),
    AppPageDescriptor("furnacemind", "custom_pages/7_FurnaceMind.py", title=_label("furnacemind"), icon=":material/psychology:"),
    AppPageDescriptor("blend_optimizer", "custom_pages/9_Blend_Optimizer.py", title=_label("blend_optimizer"), icon=":material/science:"),
    AppPageDescriptor("feedback", "custom_pages/8_Feedback.py", title=_label("feedback"), icon=":material/mail:"),
)


def get_navigation_pages() -> tuple[AppPageDescriptor, ...]:
    """Return the full navigation registry."""
    return PAGE_REGISTRY


def get_feedback_page_options() -> list[str]:
    """Return page titles available in the feedback form dropdown."""
    return [descriptor.title for descriptor in PAGE_REGISTRY if descriptor.include_in_feedback]


def get_feedback_page_catalog() -> list[dict[str, str]]:
    """Return stable page IDs and labels for feedback controls."""
    included = {descriptor.page_id for descriptor in PAGE_REGISTRY if descriptor.include_in_feedback}
    return [{"id": page.id, "label": page.label} for page in APP_PAGES if page.id in included]
