"""Global custom CSS injection helpers for the Streamlit dashboard.

Call :func:`apply_styles` once per page to apply the shared stylesheet.
"""

# ui/styles.py
# Purpose: Custom CSS styles for Streamlit UI

import streamlit as st


def apply_styles() -> None:
    """Inject the shared custom CSS into the Streamlit app."""
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            font-family: monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
