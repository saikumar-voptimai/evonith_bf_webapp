"""Reusable Streamlit UI components for the FurnaceMind dashboard."""

import re

import streamlit as st

_TABLE_SEP = re.compile(r"^\|[-:| ]+\|", re.MULTILINE)


def show_shift_summary(title: str, text: str) -> None:
    st.subheader(title)
    st.text_area(label="", value=text, height=250)


def show_report(title: str, summary_text: str) -> None:
    """Render a shift report: header + watchlist full-width, tables in 2 columns.

    First markdown table → left column (Shift Report).
    All remaining tables → stacked in right column.
    Non-table content before tables → full-width header.
    Non-table content after tables → full-width watchlist.
    """
    st.subheader(title)

    blocks = [b for b in re.split(r"\n{2,}", summary_text.strip()) if b.strip()]

    tables, pre, post = [], [], []
    seen = False
    for block in blocks:
        if _TABLE_SEP.search(block):
            tables.append(block)
            seen = True
        elif not seen:
            pre.append(block)
        else:
            post.append(block)

    if pre:
        st.markdown("\n\n".join(pre))

    if len(tables) >= 2:
        col_l, col_r = st.columns([52, 48])
        with col_l:
            st.markdown(tables[0])
        with col_r:
            st.markdown("\n\n".join(tables[1:]))
    elif tables:
        st.markdown("\n\n".join(tables))

    if post:
        st.markdown("\n\n".join(post))
