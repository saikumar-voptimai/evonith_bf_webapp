"""Admin service for backend user management."""

from __future__ import annotations

from typing import Any

from sqlalchemy.exc import IntegrityError

from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.rbac import ALL_PERMISSIONS, roles_payload
from apps.backend_api.app.repositories.admin_repository import AdminRepository
from apps.backend_api.app.repositories.user_repository import UserRecord
from apps.backend_api.app.services.auth_service import AuthService
from apps.backend_api.app.services.password_service import (
    PasswordConfigurationError,
    PasswordService,
    PasswordValidationError,
)


class AdminService:
    """Business logic for admin-owned user management endpoints."""

    def __init__(
        self,
        *,
        repository: AdminRepository | None = None,
        password_service: PasswordService | None = None,
    ) -> None:
        self.repository = repository or AdminRepository()
        self.password_service = password_service or PasswordService()

    @staticmethod
    def _profile(user: UserRecord) -> dict[str, Any]:
        return AuthService.profile_for_user(user)

    @staticmethod
    def _require_username(username: str) -> str:
        normalized = str(username or "").strip()
        if not normalized:
            raise ApiError(
                "ADMIN_USER_INVALID",
                "Username is required.",
                status_code=422,
            )
        return normalized

    def _count_active_admins(self) -> int:
        counter = getattr(self.repository, "count_active_admins", None)
        if counter is not None:
            return int(counter())
        users = self.repository.list_users(limit=500, offset=0)
        return sum(
            1
            for user in users
            if str(user.role).lower() == "admin" and bool(user.is_active)
        )

    def _guard_admin_update(
        self,
        user_id: str,
        changes: dict[str, Any],
        actor_user: dict[str, Any] | None,
    ) -> UserRecord:
        target = self.repository.find_by_id(user_id)
        if target is None:
            raise ApiError("ADMIN_USER_NOT_FOUND", "User not found.", status_code=404)

        deactivating = changes.get("is_active") is False
        actor_id = str((actor_user or {}).get("id") or "")
        if deactivating and actor_id and actor_id == str(target.id):
            raise ApiError(
                "FORBIDDEN",
                "The currently authenticated admin cannot be deactivated.",
                status_code=403,
            )

        new_role = changes.get("role")
        demoting_admin = (
            new_role is not None
            and str(target.role).lower() == "admin"
            and str(new_role).strip().lower() != "admin"
        )
        if (deactivating or demoting_admin) and str(target.role).lower() == "admin" and target.is_active:
            if self._count_active_admins() <= 1:
                raise ApiError(
                    "FORBIDDEN",
                    "The final active admin cannot be deactivated or demoted.",
                    status_code=403,
                )
        return target

    def list_users(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """Return a paginated user list."""
        users = self.repository.list_users(limit=limit, offset=offset)
        return {
            "items": [self._profile(user) for user in users],
            "total": self.repository.count_users(),
            "limit": max(1, min(500, limit)),
            "offset": max(0, offset),
        }

    def get_user(self, user_id: str) -> dict[str, Any]:
        """Return one user profile."""
        user = self.repository.find_by_id(user_id)
        if user is None:
            raise ApiError("ADMIN_USER_NOT_FOUND", "User not found.", status_code=404)
        return self._profile(user)

    def create_user(
        self,
        *,
        username: str,
        password: str,
        role: str = "user",
        email: str | None = None,
        full_name: str | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        """Create a managed user account."""
        username = self._require_username(username)
        try:
            password_hash = self.password_service.hash_password(password)
            user = self.repository.create_user(
                username=username,
                password_hash=password_hash,
                role=role,
                email=email,
                full_name=full_name,
                is_active=is_active,
            )
        except PasswordValidationError as exc:
            raise ApiError(
                "PASSWORD_POLICY_FAILED",
                str(exc),
                status_code=422,
            ) from exc
        except PasswordConfigurationError as exc:
            raise ApiError("AUTH_CONFIG_ERROR", str(exc), status_code=500) from exc
        except IntegrityError as exc:
            raise ApiError(
                "ADMIN_USER_EXISTS",
                "User already exists.",
                status_code=409,
            ) from exc
        except ValueError as exc:
            raise ApiError("ADMIN_USER_INVALID", str(exc), status_code=422) from exc
        return self._profile(user)

    def update_user(
        self,
        user_id: str,
        *,
        actor_user: dict[str, Any] | None = None,
        **changes: Any,
    ) -> dict[str, Any]:
        """Patch allowed user metadata with admin safety guards."""
        if "username" in changes and changes["username"] is not None:
            changes["username"] = self._require_username(changes["username"])

        self._guard_admin_update(user_id, changes, actor_user)

        try:
            user = self.repository.update_user(user_id, **changes)
        except IntegrityError as exc:
            raise ApiError(
                "ADMIN_USER_EXISTS",
                "User already exists.",
                status_code=409,
            ) from exc
        except ValueError as exc:
            raise ApiError("ADMIN_USER_INVALID", str(exc), status_code=422) from exc

        if user is None:
            raise ApiError("ADMIN_USER_NOT_FOUND", "User not found.", status_code=404)
        return self._profile(user)

    def reset_password(self, user_id: str, new_password: str) -> dict[str, Any]:
        """Set a new password for a user."""
        if self.repository.find_by_id(user_id) is None:
            raise ApiError("ADMIN_USER_NOT_FOUND", "User not found.", status_code=404)
        try:
            password_hash = self.password_service.hash_password(new_password)
        except PasswordValidationError as exc:
            raise ApiError(
                "PASSWORD_POLICY_FAILED",
                str(exc),
                status_code=422,
            ) from exc
        except PasswordConfigurationError as exc:
            raise ApiError("AUTH_CONFIG_ERROR", str(exc), status_code=500) from exc

        self.repository.update_password_hash(user_id, password_hash)
        return {"reset": True}

    def set_active(
        self,
        user_id: str,
        is_active: bool,
        *,
        actor_user: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Activate or deactivate a user."""
        return self.update_user(user_id, actor_user=actor_user, is_active=is_active)

    @staticmethod
    def list_roles() -> dict[str, Any]:
        """Return supported roles and permissions."""
        return {
            "roles": roles_payload(),
            "permissions": sorted(ALL_PERMISSIONS),
        }

    @staticmethod
    def list_permissions() -> dict[str, Any]:
        """Return the complete permission catalog."""
        return {"permissions": sorted(ALL_PERMISSIONS)}
