# ui/styles.py
# Purpose: Custom CSS styles for Streamlit UI

import streamlit as st


def apply_styles():
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