"""User registration and management UI for the BF2 dashboard.

Provides :class:`RegisterPage` which wraps the :class:`~domain.auth_service.AuthService`
to add new users with role assignment from an admin interface.
"""

# register_page.py
import streamlit as st

from config.frontend_settings import is_backend_api_enabled
from data.db import UserDataService
from domain.auth_service import AuthService
from services.admin_api import create_user as backend_create_user
from services.admin_api import list_users as backend_list_users
from services.api_errors import FrontendApiError


class RegisterPage:
    """User registration interface and validation logic.

    Manages the registration form UI, field validation, and interaction
    with :class:`~domain.auth_service.AuthService` for creating new users.

    Attributes:
        db:           :class:`~data.db.UserDataService` instance for persistence.
        auth_service: :class:`~domain.auth_service.AuthService` facade.
    """

    def __init__(self) -> None:
        """Initialise the database connection and authentication service."""
        self.db = None
        self.auth_service = None
        self.use_backend_api_admin = is_backend_api_enabled("admin")

    def _direct_auth_service(self) -> AuthService:
        """Create direct-mode user management objects lazily."""
        if self.db is None or self.auth_service is None:
            self.db = UserDataService()
            self.auth_service = AuthService(self.db)
        return self.auth_service

    # ------------------------------
    # Helpers
    # ------------------------------
    def show_success_message(self) -> None:
        """
        Displays a success message stored in session state and then clears it.
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
            # Stop rendering further (prevents form flicker)
            # st.stop()
            return

    def handle_registration(self, username: str, password: str, role: str) -> None:
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
            if self.use_backend_api_admin:
                access_token = str(st.session_state.get("auth_access_token") or "")
                backend_create_user(
                    access_token,
                    {
                        "username": username.strip(),
                        "password": password.strip(),
                        "role": role,
                    },
                )
            else:
                # Register new user
                self._direct_auth_service().register(
                    username.strip(),
                    password.strip(),
                    role,
                )

            # Store success state for message
            st.session_state["registration_success"] = True
            st.session_state["registration_success_message"] = (
                f"✅ User '{username}' registered successfully."
            )

            # Trigger rerun to refresh and show message
            st.rerun()

        except ValueError:
            st.error("❌ Username already exists.")
        except FrontendApiError as e:
            st.error(f"Backend user registration failed: {e}")
        except Exception as e:
            st.error(f"⚠️ Error during registration: {e}")

    def render_backend_user_list(self) -> None:
        """Render backend-managed users when admin API mode is enabled."""
        if not self.use_backend_api_admin:
            return
        access_token = str(st.session_state.get("auth_access_token") or "")
        if not access_token:
            st.info("Backend admin mode requires a backend login token.")
            return
        try:
            users = backend_list_users(access_token, limit=100)
        except FrontendApiError as exc:
            st.warning(f"Could not load backend users: {exc}")
            return
        rows = [
            {
                "username": item.get("username"),
                "role": item.get("role"),
                "active": item.get("is_active"),
            }
            for item in users.get("items", [])
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # -------------------------------
    # UI
    # -------------------------------
    def render(self) -> None:
        """
        Displays a registration form for entering username, password, and role.
        Also handles form submission and displays feedback messages.
        """
        st.subheader("🧾 Register New User")

        # Show success message and prevent form re-render
        self.show_success_message()
        self.render_backend_user_list()

        # Streamlit form (clears inputs automatically)
        with st.form(key="register_form", clear_on_submit=True):
            username = st.text_input("👤 New Username")
            password = st.text_input("🔑 New Password", type="password")
            role = st.selectbox("🧩 Role", ["user", "supervisor", "admin"])

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
