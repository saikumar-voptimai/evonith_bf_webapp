"""Canonical Streamlit multi-page application entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from apps.frontend_streamlit.config.page_registry import get_navigation_pages
from apps.frontend_streamlit.ui.backend_status_badge import render_backend_status_badge
from apps.frontend_streamlit.utils.logger import setup_logger
from apps.frontend_streamlit.utils.session import is_logged_in
from furnace_data.runtime_paths import ensure_runtime_dirs, get_runtime_dir

APP_ROOT = Path(__file__).resolve().parent

# Must be first Streamlit call.
st.set_page_config(page_title="Manufacturing Dashboard", layout="wide")


def _page_path(relative_path: str) -> str:
    canonical_path = APP_ROOT / relative_path
    if canonical_path.exists():
        return str(canonical_path)
    return str(Path(relative_path))


ensure_runtime_dirs()

# Initialize logging once.
setup_logger()
if not st.session_state.get("_evonith_runtime_dir_logged"):
    logging.getLogger(__name__).info("Evonith runtime directory: %s", get_runtime_dir())
    st.session_state["_evonith_runtime_dir_logged"] = True


# ------------------------------------------------------
# Authentication Gate
# ------------------------------------------------------
if not is_logged_in():
    from apps.frontend_streamlit.ui.login_page import LoginPage  # lazy import avoids partial-module cache on rerun

    # Hide sidebar completely during login.
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    LoginPage().run()
    st.stop()

render_backend_status_badge()

# ------------------------------------------------------
# Page Registration
# ------------------------------------------------------
pages = [
    st.Page(_page_path(descriptor.file_path), title=descriptor.title, icon=descriptor.icon)
    for descriptor in get_navigation_pages()
]

pg = st.navigation(pages)

# Run active page.
pg.run()
