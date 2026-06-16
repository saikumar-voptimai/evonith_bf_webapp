"""Streamlit multi-page application entry point for the BF2 dashboard.

Handles:
- ``st.set_page_config`` (must be the first Streamlit call)
- Authentication gate: redirects unauthenticated users to the login page
- Page registration via :func:`streamlit.navigation`

Run via ``python run_streamlit.py`` (never invoke ``streamlit run`` directly
to avoid Windows DLL ordering issues with PyTorch).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_FURNACE_DATA = _REPO_ROOT / "furnace_data"
_LOCAL_FURNACE_DATA_PATH = str(_LOCAL_FURNACE_DATA)
if _LOCAL_FURNACE_DATA.exists() and _LOCAL_FURNACE_DATA_PATH not in sys.path:
    sys.path.insert(0, _LOCAL_FURNACE_DATA_PATH)

_loaded_furnace_data = sys.modules.get("furnace_data")
_loaded_path = getattr(_loaded_furnace_data, "__file__", "") if _loaded_furnace_data else ""
if _loaded_path and not str(Path(_loaded_path).resolve()).startswith(
    str(_LOCAL_FURNACE_DATA.resolve())
):
    for _module_name in list(sys.modules):
        if _module_name == "furnace_data" or _module_name.startswith("furnace_data."):
            del sys.modules[_module_name]

import streamlit as st

# Must be first Streamlit call
st.set_page_config(page_title="Manufacturing Dashboard", layout="wide")

from config.page_registry import get_navigation_pages
from utils.logger import setup_logger
from utils.session import is_logged_in

# Initialize logging once
setup_logger()


# ------------------------------------------------------
#  Authentication Gate
# ------------------------------------------------------
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
