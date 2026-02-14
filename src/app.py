import streamlit as st

# Must be first Streamlit call
st.set_page_config(page_title="Manufacturing Dashboard", layout="wide")
from utils.logger import setup_logger
from ui.login_page import LoginPage
from utils.session import is_logged_in

# Initialize logging once
setup_logger()


import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

root = logging.getLogger()
print("=== ROOT HANDLERS ===")
for h in root.handlers:
    print(type(h).__name__, getattr(h, "baseFilename", None))
print("=====================")

# Also check a typical app logger if you use get_logger("...")
test = logging.getLogger("FurnaceMind")
print("=== FurnaceMind logger handlers ===")
for h in test.handlers:
    print(type(h).__name__, getattr(h, "baseFilename", None))
print("=====================")


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
    st.Page("custom_pages/7_🧠_FurnaceMind.py", title="FurnaceMind", icon="🧠"),
]



pg = st.navigation(pages)

# Run active page
pg.run()