"""Sessions and role checks of the twin-api (SV-0.4, #273).

A successful sign-in sets an HS256 session token (12 h) in an httpOnly, Secure, SameSite=Strict cookie with path ``/``:
the Streamlit pages under ``/legacy`` on the same origin read it for single sign-on (SV-6.6, #320) and verify it with
``services.session_tokens``, the rules used here. Until SV-6.6 the cookie path was ``/api``; sign-out also expires a
cookie left at that path. Every router declares its minimum role with ``require_role``; ``tests/test_api_auth.py`` fails
when a route outside ``PUBLIC_PATHS`` has none. Roles rank admin > supervisor > user.
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends, Request

from api.errors import ApiError
from api.security.limits import DenyList, FailureLimiter
from api.security.tokens import encode
from api.settings import ApiSettings
from services.auth import ROLES, IdentityStore, NeonIdentityStore, Principal, Role, has_role
from services.session_tokens import ISSUER, verify_session

log = logging.getLogger("api.auth")

COOKIE_PATH = "/"
# Paths of earlier session cookies, expired at sign-out (``/api`` until SV-6.6).
STALE_COOKIE_PATHS = ("/api",)
# Routes that answer without a session: liveness, version, sign-in and sign-out.
PUBLIC_PATHS = frozenset({"/api/health", "/api/version", "/api/auth/login", "/api/auth/logout"})


@dataclass
class AuthState:
    """Per-application authentication state (``app.state.auth``); tests replace ``identity`` and the clocks."""

    settings: ApiSettings
    key: bytes
    identity: IdentityStore
    limiter: FailureLimiter
    denylist: DenyList
    clock: Callable[[], float] = field(default=time.time)


def build_auth_state(settings: ApiSettings) -> AuthState:
    if settings.jwt_secret is not None:
        key = settings.jwt_secret.get_secret_value().encode()
    elif settings.environment == "prod":
        raise RuntimeError("BF_API_JWT_SECRET is required in prod (at least 32 characters; openssl rand -hex 32).")
    else:
        key = secrets.token_bytes(32)
        log.warning("BF_API_JWT_SECRET is not set: sessions use a random key and end when the API restarts.")
    return AuthState(
        settings=settings,
        key=key,
        identity=NeonIdentityStore(settings.identity_db_url),
        limiter=FailureLimiter(settings.login_max_failures, settings.login_window_s),
        denylist=DenyList(shared=settings.revocations()),
    )


def auth_state(request: Request) -> AuthState:
    return request.app.state.auth


def client_address(request: Request, trusted: frozenset[str]) -> str:
    """Client address; behind a trusted proxy, the right-most untrusted ``X-Forwarded-For`` hop."""
    peer = request.client.host if request.client else "unknown"
    if peer not in trusted:
        return peer
    hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
    for hop in reversed(hops):
        if hop not in trusted:
            return hop
    return peer


def issue_token(state: AuthState, principal: Principal) -> tuple[str, dict[str, Any]]:
    now = int(state.clock())
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "sub": principal.username,
        "role": principal.role,
        "uid": principal.user_id,
        "iat": now,
        "exp": now + int(state.settings.session_hours * 3600),
        "jti": uuid.uuid4().hex,
    }
    return encode(claims, state.key), claims


def read_session(request: Request) -> tuple[Principal, dict[str, Any]] | None:
    """The principal and claims of a valid, unrevoked session cookie; None otherwise.

    With Redis configured the revocation check is a Redis round trip: call it from a worker thread in async code.
    """
    state = auth_state(request)
    session = verify_session(
        request.cookies.get(state.settings.cookie_name),
        state.key,
        now=int(state.clock()),
        is_revoked=state.denylist.contains,
    )
    if session is None:
        return None
    return session.principal, dict(session.claims)


def current_principal(request: Request) -> Principal:
    session = read_session(request)
    if session is None:
        raise ApiError(401, "Sign in to continue.")
    principal, claims = session
    request.state.principal = principal
    request.state.claims = claims
    return principal


def _role_dependency(role: Role) -> Callable[[Principal], Principal]:
    def dependency(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
        if not has_role(principal.role, role):
            raise ApiError(403, f"This action needs the {role} role; signed in as {principal.role}.")
        return principal

    dependency.__name__ = f"require_{role}"
    dependency.required_role = role  # type: ignore[attr-defined]
    return dependency


_ROLE_DEPENDENCIES = {role: _role_dependency(role) for role in ROLES}


def require_role(role: Role) -> Callable[[Principal], Principal]:
    """Dependency that admits ``role`` and higher roles: ``dependencies=[Depends(require_role("admin"))]``."""
    return _ROLE_DEPENDENCIES[role]


def required_role_of(call: Any) -> Role | None:
    return getattr(call, "required_role", None)
