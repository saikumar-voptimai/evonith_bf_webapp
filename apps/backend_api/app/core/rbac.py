"""Backend-only role and permission helpers."""

from __future__ import annotations

# Permissions are issued by the backend token service; routes never infer them
# from a browser-supplied role. Data Explorer keeps read/export capabilities
# separate from dataset mutation capabilities.
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "admin": frozenset(
        {
            "hopper:write",
            "hopper:history:delete",
            "burden:write",
            "burden:history:delete",
            "users:write",
            "feedback:moderate",
            "data:read",
            "data:export",
            "data:export:any",
            "datasets:build",
            "datasets:refresh",
            "datasets:override",
            "vboard:read",
            "vsense:read",
            "vsense:run",
            "vsense:bounds:write",
            "vsense:diagnostics",
            "vsense:llm",
            "vsense:runs:read:any",
            "vsense:runs:cancel:any",
            "material_balance:read",
            "material_balance:run",
            "material_balance:export",
            "material_balance:config:write",
            "material_balance:diagnostics",
            "material_balance:artifacts:read:any",
            "furnacemind:read",
            "furnacemind:run",
            "furnacemind:runs:cancel:any",
            "furnacemind:documents:write",
            "furnacemind:skills:write",
            "furnacemind:reports:run",
            "furnacemind:feedback:write",
            "furnacemind:artifacts:read",
            "furnacemind:artifacts:read:any",
        }
    ),
    "supervisor": frozenset(
        {
            "hopper:write",
            "feedback:moderate",
            "data:read",
            "data:export",
            "datasets:build",
            "datasets:refresh",
            "vboard:read",
            "vsense:read",
            "vsense:run",
            "vsense:bounds:write",
            "vsense:llm",
            "material_balance:read",
            "material_balance:run",
            "material_balance:export",
            "material_balance:config:write",
            "furnacemind:read",
            "furnacemind:run",
            "furnacemind:documents:write",
            "furnacemind:skills:write",
            "furnacemind:reports:run",
            "furnacemind:feedback:write",
            "furnacemind:artifacts:read",
        }
    ),
    "user": frozenset(
        {
            "data:read",
            "data:export",
            "vboard:read",
            "vsense:read",
            "vsense:run",
            "material_balance:read",
            "material_balance:run",
            "material_balance:export",
            "furnacemind:read",
            "furnacemind:run",
            "furnacemind:documents:write",
            "furnacemind:reports:run",
            "furnacemind:feedback:write",
            "furnacemind:artifacts:read",
        }
    ),
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