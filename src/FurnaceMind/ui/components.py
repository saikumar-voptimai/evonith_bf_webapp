# ui/components.py
# Purpose: Reusable Streamlit UI components

import streamlit as st


def show_shift_summary(title: str, text: str):
    st.subheader(title)
    st.text_area(
        label="",
        value=text,
        height=250,
    )


def show_report(title: str, summary_text: str):
    st.subheader(title)
    st.markdown(summary_text)
