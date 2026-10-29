"""Streamlit multi-page application entry point for the BF2 dashboard.

Handles:
- ``st.set_page_config`` (must be the first Streamlit call)
- Authentication gate: redirects unauthenticated users to the login page
- Page registration via :func:`streamlit.navigation`

Run via ``python run_streamlit.py`` (never invoke ``streamlit run`` directly
to avoid Windows DLL ordering issues with PyTorch).
"""

import streamlit as st

# Must be first Streamlit call
st.set_page_config(page_title="Manufacturing Dashboard", layout="wide")

from config.page_registry import get_navigation_pages
from utils.logger import setup_logger
from utils.session import is_logged_in, sync_api_session

# Initialize logging once
setup_logger()


# ------------------------------------------------------
#  Authentication Gate
# ------------------------------------------------------
# Single sign-on: a valid twin-api session cookie (Svelte frontend, same origin under /legacy) signs this session in
# with its user and role; the login form below stays the fallback (SV-6.6, #320).
sync_api_session()
if not is_logged_in():
    from ui.login_page import LoginPage  # lazy import avoids partial-module cache on rerun

    # Hide sidebar completely during login
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
    # Run login page
    LoginPage().run()
    st.stop()

# ------------------------------------------------------
# PAGE REGISTRATION
# ------------------------------------------------------
pages = [
    st.Page(descriptor.file_path, title=descriptor.title, icon=descriptor.icon)
    for descriptor in get_navigation_pages()
]

pg = st.navigation(pages)

# Run active page
pg.run()
