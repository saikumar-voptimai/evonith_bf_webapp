"""Reusable Streamlit UI components for the FurnaceMind dashboard.

All functions accept plain Python values and render directly to the
current Streamlit container; no return values.
"""

# ui/components.py
# Purpose: Reusable Streamlit UI components

import streamlit as st


def show_shift_summary(title: str, text: str) -> None:
    """Render a shift summary as a non-editable text area.

    Args:
        title: Subheader label shown above the text area.
        text:  Summary text to display.
    """
    st.subheader(title)
    st.text_area(
        label="",
        value=text,
        height=250,
    )


def show_report(title: str, summary_text: str) -> None:
    """Render a Markdown report under a subheader.

    Args:
        title:        Subheader label shown above the report.
        summary_text: Markdown-formatted report text.
    """
    st.subheader(title)
    st.markdown(summary_text)
