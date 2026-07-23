"""User registration and management UI for the BF2 dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.services import admin_api
from apps.frontend_streamlit.services.api_errors import FrontendApiError


def _api_error_text(exc: FrontendApiError) -> str:
    parts = [exc.message]
    if exc.error_code:
        parts.append(f"code={exc.error_code}")
    if exc.request_id:
        parts.append(f"request_id={exc.request_id}")
    return " ".join(parts)


class RegisterPage:
    """User registration and management interface."""

    def __init__(self) -> None:
        self.db = None
        self.auth_service = None
        self.use_backend_api_admin = is_backend_api_enabled("admin") or is_backend_api_enabled("welcome")

    def _direct_auth_service(self):
        if self.db is None or self.auth_service is None:
            from apps.frontend_streamlit.data.db import UserDataService
            from apps.frontend_streamlit.domain.auth_service import AuthService

            self.db = UserDataService()
            self.auth_service = AuthService(self.db)
        return self.auth_service

    @staticmethod
    def _access_token() -> str:
        return str(st.session_state.get("auth_access_token") or "")

    def show_success_message(self) -> None:
        if st.session_state.get("registration_success", False):
            st.success(
                st.session_state.get(
                    "registration_success_message", "User operation completed successfully."
                )
            )
            st.session_state["registration_success"] = False
            st.session_state.pop("registration_success_message", None)

    def _roles(self) -> list[str]:
        if self.use_backend_api_admin:
            payload = admin_api.list_roles(self._access_token())
            return [str(item["role"]) for item in payload.get("roles", [])]
        from apps.frontend_streamlit.utils.session import ROLE_PERMISSIONS

        return sorted(ROLE_PERMISSIONS)

    def _set_success(self, message: str) -> None:
        st.session_state["registration_success"] = True
        st.session_state["registration_success_message"] = message
        st.rerun()

    def handle_registration(
        self,
        *,
        username: str,
        password: str,
        role: str,
        email: str | None = None,
        full_name: str | None = None,
    ) -> None:
        if not username.strip() or not password.strip():
            st.warning("Please fill in username and password.")
            return
        try:
            if self.use_backend_api_admin:
                admin_api.create_user(
                    self._access_token(),
                    {
                        "username": username.strip(),
                        "password": password,
                        "role": role,
                        "email": email.strip() if email else None,
                        "full_name": full_name.strip() if full_name else None,
                    },
                )
            else:
                self._direct_auth_service().register(username.strip(), password, role)
            self._set_success(f"User '{username.strip()}' registered successfully.")
        except ValueError:
            st.error("Username already exists.")
        except FrontendApiError as exc:
            st.error(f"Backend user registration failed: {_api_error_text(exc)}")
        except Exception as exc:
            st.error(f"Error during registration: {exc}")

    def _load_users(self, *, limit: int, offset: int) -> dict[str, Any] | None:
        try:
            return admin_api.list_users(self._access_token(), limit=limit, offset=offset)
        except FrontendApiError as exc:
            st.warning(f"Could not load backend users: {_api_error_text(exc)}")
            return None

    def render_user_list(self, users_payload: dict[str, Any], *, limit: int, offset: int) -> None:
        users = users_payload.get("items", [])
        rows = [
            {
                "id": item.get("id"),
                "username": item.get("username"),
                "role": item.get("role"),
                "email": item.get("email"),
                "full_name": item.get("full_name"),
                "active": item.get("is_active"),
            }
            for item in users
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No users found.")

        total = int(users_payload.get("total", 0))
        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            if st.button("Previous", disabled=offset <= 0, key="users_prev"):
                st.session_state["users_offset"] = max(0, offset - limit)
                st.rerun()
        with c2:
            if st.button("Next", disabled=offset + limit >= total, key="users_next"):
                st.session_state["users_offset"] = offset + limit
                st.rerun()
        with c3:
            if total:
                st.caption(f"Showing {offset + 1}-{min(offset + len(rows), total)} of {total}")

    def render_create_form(self, roles: list[str]) -> None:
        with st.form(key="register_form", clear_on_submit=True):
            username = st.text_input("New Username")
            password = st.text_input("New Password", type="password")
            role = st.selectbox("Role", roles)
            email = st.text_input("Email")
            full_name = st.text_input("Full Name")
            submitted = st.form_submit_button("Create User")
            if submitted:
                self.handle_registration(
                    username=username,
                    password=password,
                    role=role,
                    email=email,
                    full_name=full_name,
                )

    def render_edit_panel(self, users: list[dict[str, Any]], roles: list[str]) -> None:
        if not users:
            return
        st.markdown("### Edit User")
        selected_id = st.selectbox(
            "User",
            [str(item["id"]) for item in users],
            format_func=lambda user_id: next(
                (
                    f"{item.get('username')} ({item.get('role')})"
                    for item in users
                    if str(item.get("id")) == str(user_id)
                ),
                str(user_id),
            ),
            key="selected_user_id",
        )
        user = next(item for item in users if str(item.get("id")) == str(selected_id))
        current_role = str(user.get("role") or roles[0])
        role_index = roles.index(current_role) if current_role in roles else 0

        with st.form("edit_user_form"):
            role = st.selectbox("Role", roles, index=role_index, key="edit_role")
            email = st.text_input("Email", value=str(user.get("email") or ""), key="edit_email")
            full_name = st.text_input("Full Name", value=str(user.get("full_name") or ""), key="edit_full_name")
            submitted = st.form_submit_button("Save User")
            if submitted:
                try:
                    admin_api.update_user(
                        self._access_token(),
                        selected_id,
                        {
                            "role": role,
                            "email": email.strip() or None,
                            "full_name": full_name.strip() or None,
                        },
                    )
                    self._set_success("User updated successfully.")
                except FrontendApiError as exc:
                    st.error(f"Could not update user: {_api_error_text(exc)}")

        with st.form("reset_password_form"):
            new_password = st.text_input("New Password", type="password", key="reset_password")
            reset_submitted = st.form_submit_button("Reset Password")
            if reset_submitted:
                try:
                    admin_api.reset_password(self._access_token(), selected_id, new_password)
                    self._set_success("Password reset successfully.")
                except FrontendApiError as exc:
                    st.error(f"Could not reset password: {_api_error_text(exc)}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Activate User", disabled=bool(user.get("is_active")), key="activate_user"):
                try:
                    admin_api.activate_user(self._access_token(), selected_id)
                    self._set_success("User activated successfully.")
                except FrontendApiError as exc:
                    st.error(f"Could not activate user: {_api_error_text(exc)}")
        with c2:
            if st.button("Deactivate User", disabled=not bool(user.get("is_active")), key="deactivate_user"):
                try:
                    admin_api.deactivate_user(self._access_token(), selected_id)
                    self._set_success("User deactivated successfully.")
                except FrontendApiError as exc:
                    st.error(f"Could not deactivate user: {_api_error_text(exc)}")

    def render_backend(self) -> None:
        if not self._access_token():
            st.info("Backend user management requires a backend login token.")
            return
        try:
            roles = self._roles()
        except FrontendApiError as exc:
            st.error(f"Could not load roles: {_api_error_text(exc)}")
            return
        if not roles:
            st.error("No backend roles are available.")
            return

        limit = st.selectbox("Users per page", [10, 25, 50, 100], index=1, key="users_limit")
        offset = int(st.session_state.get("users_offset", 0))
        users_payload = self._load_users(limit=int(limit), offset=offset)
        if users_payload is None:
            return

        self.render_user_list(users_payload, limit=int(limit), offset=offset)
        st.markdown("### Create User")
        self.render_create_form(roles)
        self.render_edit_panel(users_payload.get("items", []), roles)

    def render_direct(self) -> None:
        st.info("Direct legacy mode supports user creation only. Enable backend admin API for full management.")
        roles = self._roles()
        self.render_create_form(roles)

    def render(self) -> None:
        st.subheader("User Management")
        self.show_success_message()
        if self.use_backend_api_admin:
            self.render_backend()
        else:
            self.render_direct()


def register_page():
    RegisterPage().render()
