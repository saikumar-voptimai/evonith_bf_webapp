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
from utils.session import is_logged_in

# Initialize logging once
setup_logger()

import sys
from pathlib import Path

def _prefer_local_furnace_data() -> None:
    """Use the repo package during Streamlit reruns, not a stale wheel."""
    repo_root = Path(__file__).resolve().parents[1]
    local_package_root = repo_root / "furnace_data"
    if not (local_package_root / "furnace_data").exists():
        return

    package_path = str(local_package_root)
    if package_path in sys.path:
        sys.path.remove(package_path)
    sys.path.insert(0, package_path)

    for module_name, module in list(sys.modules.items()):
        if module_name != "furnace_data" and not module_name.startswith("furnace_data."):
            continue
        module_file = getattr(module, "__file__", "") or ""
        if "site-packages" in module_file.lower():
            sys.modules.pop(module_name, None)


_prefer_local_furnace_data()


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
