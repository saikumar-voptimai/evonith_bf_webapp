"""Backend-only role and permission helpers."""

from __future__ import annotations

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

VALID_ROLES = frozenset(ROLE_PERMISSIONS)
ALL_PERMISSIONS = frozenset().union(*ROLE_PERMISSIONS.values())


def normalize_role(role: str | None) -> str:
    """Return a normalized role name or raise ValueError."""
    normalized = str(role or "").strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError("Invalid role")
    return normalized


def permissions_for_role(role: str | None) -> frozenset[str]:
    """Return deterministic permissions for a role."""
    return ROLE_PERMISSIONS.get(str(role or "").strip().lower(), frozenset())


def roles_payload() -> list[dict[str, object]]:
    """Return role metadata suitable for API responses."""
    return [
        {"role": role, "permissions": sorted(permissions)}
        for role, permissions in sorted(ROLE_PERMISSIONS.items())
    ]
