"""FastAPI dependencies for backend auth and RBAC."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.errors import ApiError
from app.services.admin_service import AdminService
from app.services.auth_service import AuthService
from app.services.password_service import PasswordService

_bearer = HTTPBearer(auto_error=False)


def get_auth_service(request: Request) -> AuthService:
    """Return the request app's auth service or a lazy default."""
    service = getattr(request.app.state, "auth_service", None)
    if service is not None:
        return service
    settings = getattr(request.app.state, "backend_settings", None)
    service = AuthService(settings=settings)
    request.app.state.auth_service = service
    return service


def get_admin_service(request: Request) -> AdminService:
    """Return the request app's admin service or a lazy default."""
    service = getattr(request.app.state, "admin_service", None)
    if service is not None:
        return service
    settings = getattr(request.app.state, "backend_settings", None)
    service = AdminService(
        password_service=PasswordService(settings) if settings is not None else None
    )
    request.app.state.admin_service = service
    return service


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    """Return the current user profile from the bearer token."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)

    token = credentials.credentials.strip()
    if not token:
        raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)

    user = get_auth_service(request).current_user_from_token(token)
    request.state.current_user = user
    return user


def get_optional_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any] | None:
    """Return current user when a bearer token is present, otherwise None."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None
    token = credentials.credentials.strip()
    if not token:
        return None
    user = get_auth_service(request).current_user_from_token(token)
    request.state.current_user = user
    return user


def require_authenticated_user(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Require any authenticated user."""
    return current_user


def require_role(*roles: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a dependency requiring one of the supplied roles."""
    allowed = {role.strip().lower() for role in roles if role.strip()}

    def _dependency(
        current_user: dict[str, Any] = Depends(require_authenticated_user),
    ) -> dict[str, Any]:
        role = str(current_user.get("role") or "").strip().lower()
        if role not in allowed:
            raise ApiError("FORBIDDEN", "Insufficient permissions.", status_code=403)
        return current_user

    return _dependency


def require_admin_user(
    current_user: dict[str, Any] = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Require the admin role."""
    role = str(current_user.get("role") or "").strip().lower()
    if role != "admin":
        raise ApiError("FORBIDDEN", "Admin role is required.", status_code=403)
    return current_user
