"""Pure permission and ownership policy for interactive scheduled-job access."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

VIEW_ALL = "scheduled_jobs:view_all"
CREATE = "scheduled_jobs:create"
UPDATE_ALL = "scheduled_jobs:update_all"
UPDATE_OWN = "scheduled_jobs:update_own"
DELETE_ALL = "scheduled_jobs:delete_all"


class ScheduledJobAuthorizationError(PermissionError):
    """Raised when an interactive principal cannot perform a requested action."""


@dataclass(frozen=True, slots=True)
class ScheduledJobPrincipal:
    """Trusted authenticated identity and its derived application permissions."""

    username: str
    permissions: frozenset[str]

    @classmethod
    def from_permissions(
        cls, username: str, permissions: Iterable[str]
    ) -> ScheduledJobPrincipal:
        """Build a normalized immutable principal from trusted session values."""
        return cls(
            username=str(username or "").strip(),
            permissions=frozenset(str(permission) for permission in permissions),
        )


def can_view_job(
    principal: ScheduledJobPrincipal, job: Mapping[str, Any] | None = None
) -> bool:
    """Return whether *principal* may view scheduled jobs and their histories."""
    del job
    return VIEW_ALL in principal.permissions


def can_create_job(principal: ScheduledJobPrincipal) -> bool:
    """Return whether *principal* may create scheduled jobs."""
    return CREATE in principal.permissions


def owns_job(principal: ScheduledJobPrincipal, job: Mapping[str, Any]) -> bool:
    """Return whether persisted ownership exactly matches the authenticated user."""
    return bool(principal.username) and job.get("created_by") == principal.username


def can_update_job(principal: ScheduledJobPrincipal, job: Mapping[str, Any]) -> bool:
    """Allow global updates or owner-only updates for the persisted job."""
    return UPDATE_ALL in principal.permissions or (
        UPDATE_OWN in principal.permissions and owns_job(principal, job)
    )


def can_delete_job(
    principal: ScheduledJobPrincipal, job: Mapping[str, Any] | None = None
) -> bool:
    """Return whether *principal* has global permanent-delete permission."""
    del job
    return DELETE_ALL in principal.permissions


def require_authorized(allowed: bool, action: str) -> None:
    """Raise a non-disclosing authorization error when *allowed* is false."""
    if not allowed:
        raise ScheduledJobAuthorizationError(
            f"You are not authorized to {action} scheduled jobs."
        )
