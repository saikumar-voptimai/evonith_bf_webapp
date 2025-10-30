# utils/session.py
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

# Initialize cookies (stored per browser tab)
cookies = EncryptedCookieManager(
    prefix="bf_dashboard_",  # Unique prefix
    password="use_a_random_secret_key_here"  # change this to a secure key
)

if not cookies.ready():
    st.stop()


# ---------------------------
# AUTH STATE HANDLERS
# ---------------------------
def is_logged_in() -> bool:
    """Check if the user is logged in (session or cookie)."""
    # If session lost (reload), restore from cookie
    if "auth_user" not in st.session_state and cookies.get("auth_user"):
        st.session_state["auth_user"] = cookies.get("auth_user")
        st.session_state["role"] = cookies.get("role")
    return "auth_user" in st.session_state


def is_admin() -> bool:
    return is_logged_in() and st.session_state.get("role") == "admin"


def login_user(username, role) -> None:
    """Login user: set session and short-lived cookie."""
    st.session_state["auth_user"] = username
    st.session_state["role"] = role
    cookies["auth_user"] = username
    cookies["role"] = role
    cookies.save()  # persist in this tab until closed


def logout_user() -> None:
    """Logout only when explicitly clicked."""
    # Clear session state
    for key in ["auth_user", "role"]:
        if key in st.session_state:
            del st.session_state[key]

    # Clear cookie (so reload doesn't restore login)
    cookies["auth_user"] = ""
    cookies["role"] = ""
    cookies.save()

    st.rerun()
