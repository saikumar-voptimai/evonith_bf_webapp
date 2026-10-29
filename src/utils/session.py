"""Streamlit session-state authentication helpers.

User credentials and roles are persisted in the database through
``furnace_data.relational``. This module only keeps the current browser
session in ``st.session_state`` and derives permissions from the persisted
role returned at login time.

Single sign-on with the twin-api (SV-6.6, #320): behind Caddy the Streamlit
pages run under ``/legacy`` on the origin of the Svelte frontend, so the
browser sends the API session cookie to Streamlit. :func:`sync_api_session`
verifies it with ``services.session_tokens`` (the API's rules: signature,
expiry, issuer, role, revoked ids in Redis when reachable) and signs the
Streamlit session in with the token's user and role. The Streamlit login
form stays the fallback.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache

import streamlit as st
from pydantic import ValidationError

from services.session_tokens import RevocationStore, Session, SessionSettings, verify_session

log = logging.getLogger("utils.session")


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
        from data.db import UserDataService

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


def login_user(username: str, role: str, user_id: str | None = None) -> None:
    """
    Store the authenticated user, database UUID, and permissions in session.

    Args:
         - username: str - Login username used by UI display and legacy pages.
         - role: str - Authenticated role name.
         - user_id: str | None - Optional ``identity.users.id`` UUID string.

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
    _set_permissions(role)


def _clear_auth() -> None:
    for key in AUTH_KEYS:
        st.session_state.pop(key, None)


def logout_user() -> None:
    """Clear authenticated session state and rerun the app.

    A session signed in from the twin-api cookie also ends the API session (its token id is revoked in Redis, when
    configured, so the Svelte frontend signs out too) and is not signed in again from the same cookie.
    """
    state = st.session_state
    if state.get(AUTH_SOURCE_KEY) == AUTH_SOURCE_API:
        jti = state.get(AUTH_JTI_KEY)
        if jti:
            state[SSO_DECLINED_KEY] = jti
            sso = api_sso()
            if sso is not None and sso.revocations is not None:
                sso.revocations.add(str(jti), float(state.get(AUTH_EXPIRES_KEY) or 0))
            log.info("sign-out %s (twin-api session)", state.get("auth_user"))
    _clear_auth()
    st.rerun()


# ---------------------------------------------------------------- single sign-on with the twin-api (SV-6.6, #320)

AUTH_SOURCE_KEY = "auth_source"  # "api": signed in from the twin-api session cookie
AUTH_SOURCE_API = "api"
AUTH_JTI_KEY = "auth_jti"  # token id of that cookie
AUTH_EXPIRES_KEY = "auth_expires_at"  # end of the API session, s since the Unix epoch (UTC)
SSO_DECLINED_KEY = "sso_declined_jti"  # token id this Streamlit session signed out of
AUTH_KEYS = (
    "auth_user",
    "auth_user_id",
    "role",
    "permissions",
    "admin_tool_selection",
    AUTH_SOURCE_KEY,
    AUTH_JTI_KEY,
    AUTH_EXPIRES_KEY,
)


@dataclass(frozen=True)
class ApiSso:
    """How Streamlit verifies the twin-api session cookie."""

    key: bytes
    cookie_name: str = "bf_session"
    revocations: RevocationStore | None = None
    clock: Callable[[], float] = field(default=time.time)


@lru_cache(maxsize=1)
def api_sso() -> ApiSso | None:
    """Single sign-on settings from the twin-api's environment (``BF_API_JWT_SECRET``, ``BF_API_COOKIE_NAME``,
    ``BF_REDIS_URL``, also from the repository ``.env``); None without a signing key: Streamlit login only."""
    try:
        settings = SessionSettings()
    except ValidationError as exc:
        log.warning("single sign-on with the twin-api is off: %s", exc.errors()[0].get("msg"))
        return None
    key = settings.signing_key()
    if key is None:
        return None
    return ApiSso(key=key, cookie_name=settings.cookie_name, revocations=settings.revocations())


def api_cookie_session(sso: ApiSso) -> Session | None:
    """The valid, unrevoked twin-api session of this browser session's cookie, or None."""
    cookies = getattr(getattr(st, "context", None), "cookies", None) or {}
    revocations = sso.revocations
    return verify_session(
        cookies.get(sso.cookie_name),
        sso.key,
        now=int(sso.clock()),
        is_revoked=revocations.contains if revocations is not None else None,
    )


def sync_api_session() -> bool:
    """Follow the twin-api session cookie; True when the Streamlit session is signed in from it.

    Called on every run of the app, before the login gate:

    - signed in from the cookie earlier: the cookie is checked again, so an expired or revoked API session (sign-out
      in the Svelte frontend, seen through Redis) ends the Streamlit session at its next run;
    - signed in with the Streamlit login form: unchanged;
    - not signed in: a valid cookie signs in with the token's user, user id and role, unless this Streamlit session
      signed out of that token.
    """
    sso = api_sso()
    if sso is None:
        return False
    state = st.session_state
    from_api = state.get(AUTH_SOURCE_KEY) == AUTH_SOURCE_API
    if "auth_user" in state and not from_api:
        return False
    session = api_cookie_session(sso)
    if from_api:
        if session is not None and session.jti == state.get(AUTH_JTI_KEY):
            return True
        _clear_auth()
    if session is None or session.jti == state.get(SSO_DECLINED_KEY):
        return False
    principal = session.principal
    login_user(principal.username, principal.role, user_id=principal.user_id)
    state[AUTH_SOURCE_KEY] = AUTH_SOURCE_API
    state[AUTH_JTI_KEY] = session.jti
    state[AUTH_EXPIRES_KEY] = session.expires_at
    return True
