"""Admin repository facade for user-management operations."""

from __future__ import annotations

from apps.backend_api.app.repositories.user_repository import UserRepository


class AdminRepository(UserRepository):
    """Use the identity user repository for admin management in Phase 5."""

