"""Provide the authenticated Streamlit entry point for Scheduled Tasks."""

import streamlit as st

from ui.scheduled_tasks import render_scheduled_tasks_page
from utils.session import is_logged_in

if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()

render_scheduled_tasks_page()
