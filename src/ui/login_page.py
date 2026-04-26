"""Streamlit login page for the BF2 Dashboard.

Handles credential collection, validation via :class:`~domain.auth_service.AuthService`,
and session state initialisation via :func:`~utils.session.login_user`.
"""

# # ui/login_page.py
import streamlit as st

from data.db import Database
from domain.auth_service import AuthService
from utils.session import login_user


class LoginPage:
    """Login page component for the Blast Furnace Dashboard.

    Collects username and password, validates them through
    :class:`~domain.auth_service.AuthService`, and writes login state to
    session via :func:`~utils.session.login_user`.

    Attributes:
        db:           :class:`~data.db.Database` instance for credential lookup.
        auth_service: :class:`~domain.auth_service.AuthService` facade.
    """

    def __init__(self) -> None:
        """Initialise the login page with a fresh database connection."""
        self.db = Database()
        self.auth_service = AuthService(self.db)

    # ---------------------------------------------------
    #  UI Layout
    # ---------------------------------------------------
    def render_layout(self) -> None:
        st.markdown(
            """
            <h1 style='text-align:center;'>🔐 Blast Furnace Dashboard Access</h1>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <style>
                div[data-testid="stVerticalBlock"] div[data-testid="stForm"] {
                    border-radius: 10px;
  
                }
                input[type="text"], input[type="password"] {
                    border-radius: 6px;
                    padding: 8px;
                }
                .stButton>button {
                    background-color: #004578;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 0.3rem 1rem;
                    font-weight: 600;
                    cursor: pointer;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------
    #  Login Form
    # ----------------------------------------------------
    def render_form(self) -> None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Please log in to continue")

            # Use a FORM to prevent auto-rerun on typing
            with st.form(key="login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input(
                    "Password", placeholder="Enter your password", type="password"
                )
                submit = st.form_submit_button("Login")

                if submit:
                    self.handle_login(username, password)

    # ----------------------------------------------------
    #  Authentication Logic
    # ----------------------------------------------------
    def handle_login(self, username, password) -> None:
        """
        Handles user authentication flow.

        Validates input, authenticates user credentials via AuthService,
        and manages session creation upon successful login.
        """
        if not username or not password:
            st.warning("Please enter both username and password.")
            return

        result = self.auth_service.authenticate(username, password)

        if result:
            login_user(result[0], result[1])
            st.success(f"✅ Welcome, {result[0]}!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

    # ----------------------------------------------------
    #  Entry Point
    # ----------------------------------------------------
    def run(self) -> None:
        """
        Entry point for rendering the Login Page.

        Combines layout setup and form rendering to provide
        a cohesive user authentication interface.
        """
        self.render_layout()
        self.render_form()
