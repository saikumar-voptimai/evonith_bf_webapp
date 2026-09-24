"""TestBMO - an independent sandbox copy of the Blend Mix Optimiser.

Load any snapshot or hand-written JSON, change anything, run LP/DE. The page
below the loader IS the Blend Mix Optimiser, run from its own source with every
``bmo_*`` state key renamed to ``testbmo_*`` (see ``utils/bmo/sandbox.py``), so
nothing done here reaches the live page and any change to the live page shows up
here without maintenance.
"""

import streamlit as st

from utils.session import is_logged_in

if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()

from ui.bmo.snapshot_panel import render_sandbox_loader  # noqa: E402
from utils.bmo.sandbox import run_sandbox_page  # noqa: E402

render_sandbox_loader()
run_sandbox_page()
