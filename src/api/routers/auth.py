"""Sign-in, current session and sign-out."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from api.auth import COOKIE_PATH, auth_state, client_address, issue_token, read_session, require_role
from api.errors import ApiError
from services.auth import IdentityUnavailable, Principal, Role

log = logging.getLogger("api.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128, description="Account name, as in the Streamlit app")
    password: str = Field(..., min_length=1, max_length=256)


class SessionUser(BaseModel):
    username: str
    role: Role = Field(..., description="user, supervisor or admin")
    user_id: str | None = Field(None, description="identity.users id")


class SessionInfo(BaseModel):
    user: SessionUser
    expires_at: datetime = Field(..., description="End of the session (UTC)")


def _session_info(principal: Principal, exp: int) -> SessionInfo:
    return SessionInfo(
        user=SessionUser(username=principal.username, role=principal.role, user_id=principal.user_id),
        expires_at=datetime.fromtimestamp(exp, UTC),
    )


@router.post("/login", response_model=SessionInfo, summary="Sign in")
def login(body: LoginRequest, request: Request, response: Response) -> SessionInfo:
    """Check the account against ``identity.users`` and start a 12 h session (cookie).

    After ``login_max_failures`` failed attempts from one address within ``login_window_s``, further attempts are
    refused with 429 and ``Retry-After`` until the oldest failure leaves the window.
    """
    state = auth_state(request)
    address = client_address(request, state.settings.trusted_proxy_set)
    wait = state.limiter.retry_after(address)
    if wait > 0:
        seconds = math.ceil(wait)
        log.warning("sign-in refused for %s: failure limit reached", address)
        raise ApiError(
            429,
            f"Too many failed sign-ins from this address. Try again in {seconds} s.",
            headers={"Retry-After": str(seconds)},
        )
    username = body.username.strip()
    try:
        principal = state.identity.authenticate(username, body.password)
    except IdentityUnavailable as exc:
        raise ApiError(503, "The user directory is not reachable. Try again in a few minutes.") from exc
    if principal is None:
        state.limiter.record_failure(address)
        log.info("failed sign-in for %r from %s", username, address)
        raise ApiError(401, "Unknown user name or wrong password.")

    token, claims = issue_token(state, principal)
    response.set_cookie(
        state.settings.cookie_name,
        token,
        max_age=claims["exp"] - claims["iat"],
        path=COOKIE_PATH,
        secure=state.settings.cookie_secure,
        httponly=True,
        samesite="strict",
    )
    log.info("sign-in %s (%s) from %s", principal.username, principal.role, address)
    return _session_info(principal, claims["exp"])


@router.get("/me", response_model=SessionInfo, summary="Current session")
def me(request: Request, principal: Annotated[Principal, Depends(require_role("user"))]) -> SessionInfo:
    """The signed-in user and the end of the session; 401 without a valid session."""
    return _session_info(principal, request.state.claims["exp"])


@router.post("/logout", status_code=204, response_class=Response, summary="Sign out")
def logout(request: Request) -> Response:
    """End the session: its token id is refused until it would have expired, and the cookie is removed."""
    state = auth_state(request)
    session = read_session(request)
    if session is not None:
        principal, claims = session
        state.denylist.add(claims["jti"], claims["exp"])
        log.info("sign-out %s", principal.username)
    response = Response(status_code=204)
    response.delete_cookie(
        state.settings.cookie_name,
        path=COOKIE_PATH,
        secure=state.settings.cookie_secure,
        httponly=True,
        samesite="strict",
    )
    return response
