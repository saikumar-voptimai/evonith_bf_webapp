# register_page.py
import streamlit as st
from domain.auth_service import AuthService
from data.db import Database


class RegisterPage:
    """
    Handles the user registration interface and logic for the application.
    This class manages the form UI, field validation, and interaction
    with the AuthService for creating new users.
    """

    def __init__(self):
        """Initialize database and authentication service."""
        self.db = Database()
        self.auth_service = AuthService(self.db)

    # -------------------------------
    # Helpers
    # -------------------------------
    def show_success_message(self):
        """
        Displays a success message stored in session state and then clears it.

        This method ensures that the success message appears only once after
        a successful registration and is removed on subsequent reruns.
        """
        if st.session_state.get("registration_success", False):
            st.success(
                st.session_state.get(
                    "registration_success_message", "✅ User registered successfully."
                )
            )
            # Reset success message state so it shows only once
            st.session_state["registration_success"] = False
            st.session_state.pop("registration_success_message", None)
            # ✅ Stop rendering further (prevents form flicker)
            # st.stop()
            return

    def handle_registration(self, username: str, password: str, role: str):
        """
        Handles registration logic and interacts with AuthService.

        Args:
            username (str): The new user's username.
            password (str): The new user's password.
            role (str): The role assigned to the user ("admin" or "user").
        """
        if not username.strip() or not password.strip():
            st.warning("⚠️ Please fill in all fields.")
            return

        try:
            # Register new user
            self.auth_service.register(username.strip(), password.strip(), role)

            # Store success state for message
            st.session_state["registration_success"] = True
            st.session_state[
                "registration_success_message"
            ] = f"✅ User '{username}' registered successfully."

            # Trigger rerun to refresh and show message
            st.rerun()

        except ValueError:
            st.error("❌ Username already exists.")
        except Exception as e:
            st.error(f"⚠️ Error during registration: {e}")

    # -------------------------------
    # UI
    # -------------------------------
    def render(self):
        """
        Renders the user registration page UI in Streamlit.

        Displays a registration form for entering username, password, and role.
        Also handles form submission and displays feedback messages.
        """
        st.subheader("🧾 Register New User")

        # ✅ Show success message and prevent form re-render
        self.show_success_message()

        # Streamlit form (clears inputs automatically)
        with st.form(key="register_form", clear_on_submit=True):
            username = st.text_input("👤 New Username")
            password = st.text_input("🔑 New Password", type="password")
            role = st.selectbox("🧩 Role", ["user", "admin"])

            # Submit button
            submitted = st.form_submit_button("✅ Register")

            if submitted:
                self.handle_registration(username, password, role)


# -------------------------------
# Entry Point
# -------------------------------
def register_page():
    """
    Streamlit entry function for the Register Page.
    Instantiates and renders the RegisterPage class.
    """
    RegisterPage().render()
