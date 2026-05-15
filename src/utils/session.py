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


def login_user(username: str, role: str) -> None:
    """Store the authenticated user and derived permissions for this session."""
    role = str(role).strip().lower()
    st.session_state["auth_user"] = username
    st.session_state["role"] = role
    _set_permissions(role)


def logout_user() -> None:
    """Clear authenticated session state and rerun the app."""
    for key in ("auth_user", "role", "permissions", "admin_tool_selection"):
        st.session_state.pop(key, None)
    st.rerun()
