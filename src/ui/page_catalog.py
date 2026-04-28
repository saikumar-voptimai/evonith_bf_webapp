"""Shared Streamlit page catalog for navigation and feedback metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppPage:
    """Navigation metadata for one Streamlit page."""

    path: str
    title: str
    icon: str


APP_PAGE_CATALOG: tuple[AppPage, ...] = (
    AppPage("custom_pages/1_🏭_Welcome.py", "Welcome", "🏭"),
    AppPage("custom_pages/2_📓_Data_Explorer.py", "Data Explorer", "📓"),
    AppPage("custom_pages/3_📈_Data_Visualisation.py", "V-Board", "📈"),
    AppPage("custom_pages/4_💡_Recommendations.py", "V-Sense", "💡"),
    AppPage("custom_pages/5_🤖_AI_Copilot.py", "CoPilot", "🤖"),
    AppPage("custom_pages/6_⚖️_Material_Balance.py", "Material Balance", "⚖️"),
    AppPage("custom_pages/7_🧠_FurnaceMind.py", "FurnaceMind", "🧠"),
    AppPage("custom_pages/8_Feedback.py", "Feedback", "💬"),
)


def get_feedback_page_options() -> list[str]:
    """Return user-facing page names for the feedback issue dropdown."""
    return [page.title for page in APP_PAGE_CATALOG]
