"""User repository for backend auth and admin APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from apps.backend_api.app.core.rbac import normalize_role
from furnace_data.relational import (
    User,
    UserRoleAssignment,
    build_relational_engine,
    build_relational_session_factory,
)
from furnace_data.relational.models import utc_now


@dataclass(frozen=True)
class UserRecord:
    """Backend-safe user shape detached from the ORM session."""

    id: str
    username: str
    password_hash: str
    role: str
    is_active: bool
    email: str | None = None
    full_name: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_login_at: datetime | None = None


class UserRepository:
    """Lazy SQLAlchemy repository for identity users."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Build the default session factory only when a request needs it."""
        if self._session_factory is None:
            engine = build_relational_engine()
            self._session_factory = build_relational_session_factory(engine)
        return self._session_factory

    @staticmethod
    def _to_record(user: User) -> UserRecord:
        return UserRecord(
            id=str(user.id),
            username=str(user.username),
            password_hash=str(user.password_hash),
            role=str(user.role),
            is_active=bool(user.is_active),
            email=getattr(user, "email", None),
            full_name=getattr(user, "full_name", None),
            created_at=getattr(user, "created_at", None),
            updated_at=getattr(user, "updated_at", None),
            last_login_at=getattr(user, "last_login_at", None)
            or getattr(user, "last_login", None),
        )

    @staticmethod
    def _user_id(value: str) -> UUID | str:
        try:
            return UUID(str(value))
        except (TypeError, ValueError):
            return str(value)

    def find_by_username_or_email(self, identifier: str) -> UserRecord | None:
        """Return a user by username, or email if that column exists later."""
        identifier = str(identifier or "").strip()
        if not identifier:
            return None

        with self.session_factory() as session:
            conditions = [User.username == identifier]
            email_column = getattr(User, "email", None)
            if email_column is not None:
                conditions.append(email_column == identifier)
            user = session.execute(
                select(User).where(or_(*conditions)).limit(1)
            ).scalar_one_or_none()
            if user is None:
                return None
            session.expunge(user)
            return self._to_record(user)

    def find_by_id(self, user_id: str) -> UserRecord | None:
        """Return a user by database id."""
        with self.session_factory() as session:
            user = session.get(User, self._user_id(user_id))
            if user is None:
                return None
            session.expunge(user)
            return self._to_record(user)

    def list_users(self, *, limit: int = 100, offset: int = 0) -> list[UserRecord]:
        """Return users ordered by creation time and username."""
        with self.session_factory() as session:
            stmt = (
                select(User)
                .order_by(User.created_at.desc(), User.username.asc())
                .offset(max(0, offset))
                .limit(max(1, min(500, limit)))
            )
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return [self._to_record(row) for row in rows]

    def count_users(self) -> int:
        """Return total user count."""
        with self.session_factory() as session:
            return int(session.execute(select(func.count()).select_from(User)).scalar_one())

    def create_user(
        self,
        *,
        username: str,
        password_hash: str,
        role: str,
        email: str | None = None,
        full_name: str | None = None,
        is_active: bool = True,
    ) -> UserRecord:
        """Create and return a user."""
        role = normalize_role(role)
        payload: dict[str, Any] = {
            "username": username,
            "password_hash": password_hash,
            "role": role,
            "is_active": is_active,
        }
        if hasattr(User, "email"):
            payload["email"] = email
        if hasattr(User, "full_name"):
            payload["full_name"] = full_name

        with self.session_factory() as session:
            user = User(**payload)
            session.add(user)
            try:
                session.flush()
                session.add(UserRoleAssignment(user_id=user.id, role=role))
                session.commit()
                session.refresh(user)
                session.expunge(user)
                return self._to_record(user)
            except IntegrityError:
                session.rollback()
                raise

    def update_user(self, user_id: str, **changes: Any) -> UserRecord | None:
        """Patch simple user fields and return the updated row."""
        values: dict[str, Any] = {}
        for key in ("username", "email", "full_name", "is_active"):
            if key in changes and changes[key] is not None and hasattr(User, key):
                values[key] = changes[key]

        if "role" in changes and changes["role"] is not None:
            values["role"] = normalize_role(changes["role"])

        if hasattr(User, "updated_at"):
            values["updated_at"] = utc_now()

        with self.session_factory() as session:
            user = session.get(User, self._user_id(user_id))
            if user is None:
                return None

            for key, value in values.items():
                setattr(user, key, value)

            if "role" in values:
                self._replace_role_assignment(session, user.id, values["role"])

            try:
                session.commit()
                session.refresh(user)
                session.expunge(user)
                return self._to_record(user)
            except IntegrityError:
                session.rollback()
                raise

    def update_password_hash(self, user_id: str, password_hash: str) -> None:
        """Update a user's stored password hash."""
        values: dict[str, Any] = {"password_hash": password_hash}
        if hasattr(User, "updated_at"):
            values["updated_at"] = utc_now()
        with self.session_factory() as session:
            session.execute(
                update(User).where(User.id == self._user_id(user_id)).values(**values)
            )
            session.commit()

    def record_login(self, user_id: str) -> None:
        """Record last-login metadata if the deployed schema supports it."""
        values: dict[str, Any] = {}
        if hasattr(User, "last_login_at"):
            values["last_login_at"] = utc_now()
        elif hasattr(User, "last_login"):
            values["last_login"] = utc_now()
        if not values:
            return
        with self.session_factory() as session:
            session.execute(
                update(User).where(User.id == self._user_id(user_id)).values(**values)
            )
            session.commit()

    @staticmethod
    def _replace_role_assignment(session: Session, user_id: UUID, role: str) -> None:
        """Keep the role assignment table aligned with the primary role column."""
        session.query(UserRoleAssignment).filter(
            UserRoleAssignment.user_id == user_id
        ).delete()
        session.add(UserRoleAssignment(user_id=user_id, role=role))
