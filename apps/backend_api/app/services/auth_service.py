"""Backend authentication service."""

from __future__ import annotations

from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.rbac import permissions_for_role
from apps.backend_api.app.repositories.user_repository import UserRecord, UserRepository
from apps.backend_api.app.services.password_service import (
    PasswordConfigurationError,
    PasswordService,
    PasswordValidationError,
)
from apps.backend_api.app.services.token_service import TokenError, TokenService


class AuthService:
    """Authenticate users and create backend access tokens."""

    def __init__(
        self,
        *,
        repository: UserRepository | None = None,
        password_service: PasswordService | None = None,
        token_service: TokenService | None = None,
        settings: BackendSettings | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.repository = repository or UserRepository()
        self.password_service = password_service or PasswordService(self.settings)
        self.token_service = token_service or TokenService(self.settings)

    def _ensure_enabled(self) -> None:
        if not self.settings.auth_enabled:
            raise ApiError(
                code="AUTH_DISABLED",
                message="Backend auth is disabled.",
                status_code=503,
            )

    @staticmethod
    def profile_for_user(user: UserRecord) -> dict[str, Any]:
        """Return the API user profile shape."""
        permissions = sorted(permissions_for_role(user.role))
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "roles": [user.role],
            "permissions": permissions,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "last_login_at": user.last_login_at,
        }

    def login(self, *, username: str, password: str) -> dict[str, Any]:
        """Authenticate credentials and return an access token response."""
        self._ensure_enabled()
        identifier = str(username or "").strip()
        if not identifier or not password:
            raise ApiError(
                code="INVALID_CREDENTIALS",
                message="Invalid username or password.",
                status_code=401,
            )

        user = self.repository.find_by_username_or_email(identifier)
        if user is None:
            raise ApiError(
                code="INVALID_CREDENTIALS",
                message="Invalid username or password.",
                status_code=401,
            )

        verification = self.password_service.verify_password(
            password=password,
            stored_hash=user.password_hash,
        )
        if not verification.valid:
            raise ApiError(
                code="INVALID_CREDENTIALS",
                message="Invalid username or password.",
                status_code=401,
            )

        if not user.is_active:
            raise ApiError(
                code="USER_INACTIVE",
                message="User account is inactive.",
                status_code=403,
            )

        if verification.needs_rehash:
            try:
                upgraded_hash = self.password_service.hash_password(
                    password,
                    validate=False,
                )
            except PasswordConfigurationError as exc:
                raise ApiError(
                    code="AUTH_CONFIG_ERROR",
                    message=str(exc),
                    status_code=500,
                ) from exc
            self.repository.update_password_hash(user.id, upgraded_hash)
            user = self.repository.find_by_id(user.id) or user

        self.repository.record_login(user.id)
        profile = self.profile_for_user(user)
        try:
            token = self.token_service.create_access_token(
                user_id=user.id,
                username=user.username,
                role=user.role,
                permissions=profile["permissions"],
            )
        except TokenError as exc:
            raise ApiError(
                code=exc.code,
                message=exc.message,
                status_code=exc.status_code,
            ) from exc
        return {
            "access_token": token.token,
            "token_type": "bearer",
            "expires_at": token.expires_at,
            "expires_in": token.expires_in,
            "user": profile,
        }

    def current_user_from_token(self, token: str) -> dict[str, Any]:
        """Return current user profile for a bearer token."""
        self._ensure_enabled()
        try:
            claims = self.token_service.verify_access_token(token)
        except TokenError as exc:
            raise ApiError(
                code=exc.code,
                message=exc.message,
                status_code=exc.status_code,
            ) from exc

        user = self.repository.find_by_id(str(claims.get("sub") or ""))
        if user is None:
            raise ApiError(
                code="INVALID_TOKEN",
                message="Token user no longer exists.",
                status_code=401,
            )
        if not user.is_active:
            raise ApiError(
                code="USER_INACTIVE",
                message="User account is inactive.",
                status_code=403,
            )
        return self.profile_for_user(user)

    def change_password(
        self,
        *,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> dict[str, Any]:
        """Change the authenticated user's password."""
        self._ensure_enabled()
        user = self.repository.find_by_id(user_id)
        if user is None or not user.is_active:
            raise ApiError("USER_INACTIVE", "User account is inactive.", status_code=403)

        verification = self.password_service.verify_password(
            password=current_password,
            stored_hash=user.password_hash,
        )
        if not verification.valid:
            raise ApiError(
                code="INVALID_CREDENTIALS",
                message="Current password is incorrect.",
                status_code=401,
            )

        try:
            password_hash = self.password_service.hash_password(new_password)
        except PasswordValidationError as exc:
            raise ApiError(
                code="PASSWORD_POLICY_FAILED",
                message=str(exc),
                status_code=422,
            ) from exc
        except PasswordConfigurationError as exc:
            raise ApiError(
                code="AUTH_CONFIG_ERROR",
                message=str(exc),
                status_code=500,
            ) from exc

        self.repository.update_password_hash(user.id, password_hash)
        return {"changed": True}
