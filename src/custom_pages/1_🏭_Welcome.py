# src/custom_pages/1_Welcome.py
import streamlit as st
from utils.session import logout_user, is_admin

# ---------------------------------------------------
#  AUTH CHECK
# ---------------------------------------------------
if "auth_user" not in st.session_state:
    st.warning("Please login to access this page.")
    st.stop()

# ----------------------------------------------------
#  SIDEBAR (Admin tools + logout)
# ----------------------------------------------------
with st.sidebar:
    st.markdown(f"👋 Logged in as: `{st.session_state['auth_user']}` ")
    st.markdown("---")

    # Admin-only quick links (only on Welcome page)
    if is_admin():
        st.markdown("### 🔧 Admin Tools")

        # Initialize session key for tool tracking
        if "admin_tool_selection" not in st.session_state:
            st.session_state["admin_tool_selection"] = None

        # Admin Tool Buttons
        if st.button("🛠 Hopper Mapping"):
            st.session_state["admin_tool_selection"] = "hopper"
            st.rerun()

        if st.button("📝 Register Page"):
            st.session_state["admin_tool_selection"] = "register"
            st.rerun()

        # Back to Dashboard button (visible only when inside tool)
        if st.session_state.get("admin_tool_selection") in ["hopper", "register"]:
            if st.button("⬅ Back to Dashboard"):
                st.session_state["admin_tool_selection"] = None
                st.rerun()

        st.markdown("---")

    #  Common logout for all users
    if st.button("🚪 Logout"):
        logout_user()
        st.stop()

# ----------------------------------------------------
#  RENDER ADMIN TOOLS INLINE IF SELECTED
# ----------------------------------------------------
if is_admin() and st.session_state.get("admin_tool_selection") == "hopper":
    # st.title("🛠 Hopper Material Mapping (Admin Tool)")
    from ui.hopper_admin_page import hopper_admin_page
    hopper_admin_page(st.session_state.get("auth_user", "Unknown"))

    st.stop()

elif is_admin() and st.session_state.get("admin_tool_selection") == "register":
    # st.title("📝 Register Page (Admin Tool)")
    from ui.register_page import register_page
    register_page()
    st.stop()

# ----------------------------------------------------
#  MAIN PAGE CONTENT (Default welcome content)
# ----------------------------------------------------
st.title("Welcome to the Manufacturing Dashboard")
st.write("This dashboard provides tools for data submission, visualization, and recommendations.")

# ---- Image Loading ----
def load_images():
    """Helper function to load and return images."""
    v_optimAIse_logo = "src/data/VOPTIMAISELOGO.png"
    evonith_logo = "src/data/evonith.png"
    return v_optimAIse_logo, evonith_logo

v_optimAIse_logo, evonith_logo = load_images()

col1, col2 = st.columns([2, 2])
with col1:
    st.image(v_optimAIse_logo, width=400)
with col2:
    st.image(evonith_logo, width=500)

# ---- Intro ----
st.markdown(
    """
    **BlastFurnace WebApp** serves as a comprehensive digital platform for monitoring 
    and optimizing the Blast Furnace operations at Evonith. Through the combined efforts 
    of **V-OptimAIse** and **Evonith**, we aim to improve operational efficiency, 
    streamline data analytics, and provide actionable insights.
    """
)

st.markdown("---")
st.header("Explore Our Features:")

# ---- Data Submission ----
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

# ---- V-Sense ----
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

# ---- V-Board ----
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

# ---- Reporter ----
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

# ---- Chatbot ----
st.subheader("5. Chatbot (Steel Manufacturing Expert)")
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