import streamlit as st
from utils.logger import setup_logger

# Setup the logger at the start of the application
setup_logger()

# Set app-wide page config
st.set_page_config(page_title="Manufacturing Dashboard", layout="wide")

# Define your pages
page1 = st.Page("custom_pages/1_🏭_Welcome.py", title="Welcome", icon="🏭")
page2 = st.Page("custom_pages/2_📓_Data_Submission.py", title="Data Submission", icon="📓")
page3 = st.Page("custom_pages/3_📈_Data_Visualisation.py", title="V-Board", icon="📈")
page4 = st.Page("custom_pages/4_💡_Recommendations.py", title="V-Sense", icon="💡")
page5 = st.Page("custom_pages/5_🤖_AI_Copilot.py", title="CoPilot", icon="🤖")
# page6 = st.Page("custom_pages/6_📝_Reports.py", title="Reporter", icon="📝")

# Set up navigation
pg = st.navigation([page1, page2, page3, page4, page5])

# Run the selected page
pg.run()