import streamlit as st

st.title("Welcome to the Manufacturing Dashboard")
st.write("This dashboard provides tools for data submission, visualization, and recommendations.")

import streamlit as st

def load_images():
    """
    Helper function to load and return images.
    Adjust paths based on where your image files are located.
    """
    v_optimAIse_logo = "src/data/VOPTIMAISELOGO.png"
    evonith_logo = "src/data/evonith.png"
    
    return v_optimAIse_logo, evonith_logo

# Load images
v_optimAIse_logo, evonith_logo = load_images()

# Place images in the layout
col1, col2 = st.columns([2,2])
with col1:
    st.image(v_optimAIse_logo, width=400)
with col2:
    st.image(evonith_logo, width=500)

st.markdown(
    """
    **BlastFurnace WebApp** serves as a comprehensive digital platform for monitoring 
    and optimizing the Blast Furnace operations at Evonith. Through the combined efforts 
    of V-OptimAIse and Evonith, we aim to improve operational efficiency, streamline data 
    analytics, and provide actionable insights.
    """
)

st.markdown("---")
st.header("Explore Our Features:")

# Data Submission Page
st.subheader("1. Data Submission Page (Data Governance)")
st.write(
    """
    - **Purpose**: Consolidate both offline and online BF data in one place, making it accessible 
      to operations teams at EML, our AI models for inference, and for V-OptimAIse analytics & training. 
    - **Key Functions**:
      - Upload and download offline/online data.
      - Ensure data consistency and availability for downstream AI/analytics systems.
    """
)

# V-Sense
st.subheader("2. V-Sense (AI Recommendation System)")
st.write(
    """
    - **Purpose**: Provide real-time, self-learning AI-driven recommendations to operators.
    - **Key Functions**:
      - Generate textual recommendations along with reasoning, explaining the motivation 
        behind each suggested action.
      - Continually refine recommendations based on evolving furnace conditions 
        and historical outcomes.
    """
)

# V-Board
st.subheader("3. V-Board (Data Visualization)")
st.write(
    """
    - **Purpose**: Deliver tailored data visualizations to facilitate quick insights 
      into real-time operations.
    - **Key Functions**:
      - Visualize 2D heat load or temperature distributions in real-time, aiding 
        in stave analysis and quick decision-making.
      - Customize dashboards to meet the specific needs of various user groups.
    """
)

# Reporter
st.subheader("4. Reporter (GenAI Reporter)")
st.write(
    """
    - **Purpose**: Automate weekly/daily reporting for various teams, saving time 
      and reducing human error.
    - **Key Functions**:
      - Generate comprehensive reports on furnace performance.
      - Compare current performance with historical benchmarks, offering deep 
        insights into trends and anomalies.
    """
)

# Chatbot
st.subheader("Chatbot (Steel Manufacturing Expert)")
st.write(
    """
    - **Purpose**: Provide answers, guidance, and recommendations to teams in the plant, 
      leveraging knowledge from local production data.
    - **Key Functions**:
      - Runs locally with limited internet access, ensuring data privacy and protection.
      - Offers accurate, tailor-cut responses based on the plant's own production 
        and technical data.
    """
)

st.markdown("---")
st.markdown(
    """
    **We hope you find this platform useful and intuitive!** 
    Use the sidebar to navigate through the pages and access each feature.
    """
)
