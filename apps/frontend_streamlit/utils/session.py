"""Streamlit session-state authentication helpers.

User credentials and roles are persisted in the database through
``furnace_data.relational``. This module only keeps the current browser
session in ``st.session_state`` and derives permissions from the persisted
role returned at login time.
"""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st


ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "admin": frozenset(
        {
            "hopper:write",
            "burden:write",
            "users:write",
            "feedback:moderate",
        }
    ),
    "supervisor": frozenset(
        {
            "hopper:write",
            "feedback:moderate",
        }
    ),
    "user": frozenset(),
}


def permissions_for_role(role: str | None) -> frozenset[str]:
    """Return deterministic app permissions for a stored role."""
    return ROLE_PERMISSIONS.get(str(role or "").strip().lower(), frozenset())


def _set_permissions(role: str | None) -> None:
    st.session_state["permissions"] = sorted(permissions_for_role(role))


def is_logged_in() -> bool:
    """Check if the current Streamlit session has an authenticated user."""
    if "auth_user" in st.session_state:
        _set_permissions(st.session_state.get("role"))
        return True
    return False


def current_user_id() -> str | None:
    """
    Return the database UUID for the authenticated Streamlit user.

    The UI stores ``auth_user`` as the login name for display and older pages.
    FurnaceMind persistence in UAT needs the UUID from ``identity.users``.
    This helper first uses the UUID already stored at login time, then lazily
    resolves older sessions that only have the username.

    Args:
         - None

    Returns:
         - return: str | None - Authenticated user's UUID string, or None when
           the session user cannot be resolved.
    """
    user_id = str(st.session_state.get("auth_user_id") or "").strip()
    if user_id:
        return user_id

    username = str(st.session_state.get("auth_user") or "").strip()
    if not username:
        return None

    try:
        from apps.frontend_streamlit.data.db import UserDataService

        resolved_user_id = UserDataService().get_user_id(username)
    except Exception:
        return None

    if resolved_user_id:
        st.session_state["auth_user_id"] = resolved_user_id
    return resolved_user_id


def current_permissions() -> set[str]:
    """Return permissions for the current session role."""
    _set_permissions(st.session_state.get("role"))
    return set(st.session_state.get("permissions", []))


def has_permission(permission: str) -> bool:
    """Return ``True`` when the logged-in user has *permission*."""
    return is_logged_in() and permission in current_permissions()


def has_any_permission(permissions: Iterable[str]) -> bool:
    """Return ``True`` when the logged-in user has any requested permission."""
    return is_logged_in() and bool(current_permissions().intersection(permissions))


def is_admin() -> bool:
    """Return ``True`` when the logged-in user holds the ``admin`` role."""
    return is_logged_in() and st.session_state.get("role") == "admin"


def is_supervisor() -> bool:
    """Return ``True`` when the logged-in user holds the ``supervisor`` role."""
    return is_logged_in() and st.session_state.get("role") == "supervisor"


def login_user(
    username: str,
    role: str,
    user_id: str | None = None,
    *,
    access_token: str | None = None,
    token_expires_at: str | None = None,
) -> None:
    """
    Store the authenticated user, database UUID, and permissions in session.

    Args:
         - username: str - Login username used by UI display and legacy pages.
         - role: str - Authenticated role name.
         - user_id: str | None - Optional ``identity.users.id`` UUID string.
         - access_token: str | None - Optional backend API bearer token.
         - token_expires_at: str | None - Optional token expiry timestamp.

    Returns:
         - return: None - This function does not return a value.
    """
    role = str(role).strip().lower()
    st.session_state["auth_user"] = username
    if user_id:
        st.session_state["auth_user_id"] = str(user_id)
    else:
        st.session_state.pop("auth_user_id", None)
    st.session_state["role"] = role
    if access_token:
        st.session_state["auth_access_token"] = access_token
        st.session_state["auth_backend_mode"] = True
    else:
        st.session_state.pop("auth_access_token", None)
        st.session_state.pop("auth_backend_mode", None)
    if token_expires_at:
        st.session_state["auth_token_expires_at"] = str(token_expires_at)
    else:
        st.session_state.pop("auth_token_expires_at", None)
    _set_permissions(role)


def logout_user() -> None:
    """Clear authenticated session state and rerun the app."""
    access_token = str(st.session_state.get("auth_access_token") or "").strip()
    if access_token:
        try:
            from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
            from apps.frontend_streamlit.services.auth_api import logout

            if is_backend_api_enabled("auth"):
                logout(access_token)
        except Exception:
            pass

    for key in (
        "auth_user",
        "auth_user_id",
        "auth_access_token",
        "auth_token_expires_at",
        "auth_backend_mode",
        "role",
        "permissions",
        "admin_tool_selection",
    ):
        st.session_state.pop(key, None)
    st.rerun()
