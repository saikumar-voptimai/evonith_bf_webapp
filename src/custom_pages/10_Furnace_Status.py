"""Furnace Status — live BF2 status board with per-parameter trend view.

Thin entry point: authentication gate, then hand over to the UI module.  Data
fetching/formatting lives in ``data/furnace_status.py`` and rendering in
``ui/furnace_status_page.py``.
"""

import streamlit as st

from utils.session import is_logged_in

if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()

from ui.furnace_status_page import render_furnace_status_page

render_furnace_status_page()
