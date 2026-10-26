"""Users and roles for the twin-api, from Neon ``identity.users`` (the accounts of the Streamlit app).

Credentials are checked by ``domain.auth_service.AuthService`` over ``data.db.UserDataService``, with the same stored
password form as the Streamlit login. The API never seeds default users.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Literal, Protocol, cast

log = logging.getLogger("api.auth")

Role = Literal["user", "supervisor", "admin"]
ROLES: tuple[Role, ...] = ("user", "supervisor", "admin")
ROLE_RANK: dict[Role, int] = {"user": 0, "supervisor": 1, "admin": 2}


def normalise_role(value: object) -> Role | None:
    text = str(value or "").strip().lower()
    return cast(Role, text) if text in ROLE_RANK else None


def has_role(role: Role, required: Role) -> bool:
    """``role`` is at least ``required`` (admin > supervisor > user)."""
    return ROLE_RANK[role] >= ROLE_RANK[required]


@dataclass(frozen=True)
class Principal:
    """The signed-in user of a request."""

    username: str
    role: Role
    user_id: str | None = None


class IdentityUnavailable(RuntimeError):
    """The user directory could not be reached."""


class IdentityStore(Protocol):
    def authenticate(self, username: str, password: str) -> Principal | None:
        """The user when the credentials match an active account, otherwise None."""
        ...


class NeonIdentityStore:
    """``AuthService`` over ``UserDataService(seed_defaults=False)``; the engine is created on first use."""

    def __init__(self, db_url: str | None = None) -> None:
        self._db_url = db_url
        self._service = None
        self._lock = threading.Lock()

    def _auth(self):
        with self._lock:
            if self._service is None:
                from data.db import UserDataService
                from domain.auth_service import AuthService

                self._service = AuthService(UserDataService(self._db_url, seed_defaults=False))
            return self._service

    def authenticate(self, username: str, password: str) -> Principal | None:
        from sqlalchemy.exc import SQLAlchemyError

        try:
            auth = self._auth()
            result = auth.authenticate(username, password)
            if result is None:
                return None
            name, stored_role = result
            user_id = auth.db.get_user_id(name)
        except (SQLAlchemyError, OSError, ValueError) as exc:
            log.error("identity store unavailable: %s", type(exc).__name__)
            raise IdentityUnavailable("The user directory is not reachable.") from exc
        role = normalise_role(stored_role)
        if role is None:
            log.warning("user %s has an unknown role %r; sign-in refused", name, stored_role)
            return None
        return Principal(username=name, role=role, user_id=user_id)
