"""FurnaceMind navigation helpers.

Provides the top-level tab list and the nav-tab widget used by
``custom_pages/7_FurnaceMind.py``.
"""

from __future__ import annotations

import streamlit as st

NAV_TABS: list[str] = [
    "🤖 AI Co-Operate",
    "📊 Reports",
]


def select_nav_tab() -> str:
    """Render a segmented-control for top-level navigation.

    Returns the label of the selected tab.
    """
    return st.segmented_control(
        "Navigation",
        NAV_TABS,
        default=NAV_TABS[0],
        key="furnacemind_nav",
    )
