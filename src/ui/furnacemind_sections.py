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
    """Render a segmented-control (or radio fallback) for top-level navigation.

    Returns the label of the selected tab.
    """
    try:
        if hasattr(st, "segmented_control"):
            return st.segmented_control(
                "Navigation",
                NAV_TABS,
                default=NAV_TABS[0],
                key="furnacemind_nav",
            )
    except TypeError:
        pass
    return st.radio(
        "Navigation", NAV_TABS, horizontal=True, index=0, key="furnacemind_nav"
    )
