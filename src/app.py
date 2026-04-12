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
from ui.login_page import LoginPage
from utils.logger import setup_logger
from utils.session import is_logged_in

# Initialize logging once
setup_logger()


# ------------------------------------------------------
#  Authentication Gate
# ------------------------------------------------------
if not is_logged_in():
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
# Base pages (visible to all)
pages = [
    st.Page("custom_pages/1_🏭_Welcome.py", title="Welcome", icon="🏭"),
    st.Page("custom_pages/2_📓_Data_Explorer.py", title="Data Explorer", icon="📓"),
    st.Page("custom_pages/3_📈_Data_Visualisation.py", title="V-Board", icon="📈"),
    st.Page("custom_pages/4_💡_Recommendations.py", title="V-Sense", icon="💡"),
    st.Page("custom_pages/5_🤖_AI_Copilot.py", title="CoPilot", icon="🤖"),
    # st.Page("custom_pages/6_📊_Reports.py", title="Reports", icon="📊" ),
    st.Page("custom_pages/6_⚖️_Material_Balance.py", title="Material Balance", icon="⚖️"),
    st.Page("custom_pages/7_🧠_FurnaceMind.py", title="FurnaceMind", icon="🧠"),
]


pg = st.navigation(pages)

# Run active page
pg.run()
