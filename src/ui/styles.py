"""Shared CSS injection for the Streamlit dashboard.

Call :func:`apply_styles` once per page to apply the project-wide stylesheet.
Individual helpers are also available for targeted injection.

Usage::

    from ui.styles import apply_styles
    apply_styles()
"""

import streamlit as st

_BASE_CSS = """
<style>
/* ── Typography ─────────────────────────────────────────────────────────── */
/* Code/data areas use monospace */
.stTextArea textarea,
.stCodeBlock code          { font-family: "JetBrains Mono", "Fira Code",
                             "Cascadia Code", monospace !important; }

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1rem 0.7rem;
}
[data-testid="stMetricValue"]  { font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { font-size: 0.75rem !important; color: #64748b !important;
                                  text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricDelta"]  { font-size: 0.82rem !important; }

/* ── Dataframe tweaks ────────────────────────────────────────────────────── */
/* Tighter row height for compact tables */
[data-testid="stDataFrame"] thead th { font-size: 0.78rem !important; }
[data-testid="stDataFrame"] td       { font-size: 0.82rem !important; }

/* ── Horizontal rule ────────────────────────────────────────────────────── */
hr { border-color: #e2e8f0 !important; margin: 0.6rem 0 !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] .stButton button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] summary { font-weight: 600; }

/* ── Form submit button primary ─────────────────────────────────────────── */
[data-testid="stForm"] [data-testid="stBaseButton-primary"] button {
    min-width: 120px;
}

/* ── Caption / help text ─────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #64748b !important;
    font-size: 0.78rem !important;
}

/* ── FurnaceMind chat ────────────────────────────────────────────────────── */
/* Tighter spacing between messages */
[data-testid="stChatMessage"] { margin-bottom: 0.25rem !important; }

/* Subtle background on assistant bubbles */
[data-testid="stChatMessageContent"] {
    border-radius: 8px;
    padding: 0.5rem 0.75rem !important;
}

/* Slightly smaller avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
}

/* Chart inside a chat message — remove default bottom margin */
[data-testid="stChatMessageContent"] [data-testid="stPlotlyChart"] {
    margin-bottom: 0 !important;
}
</style>
"""


def apply_styles() -> None:
    """Inject the shared project CSS into the current page."""
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
