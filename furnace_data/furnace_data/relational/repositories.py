"""Repository classes for Neon-backed relational persistence."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    BURDEN_VALUE_COLUMNS,
    HOPPER_COLUMNS,
    BurdenHistory,
    Hopper,
    HopperRawMaterialHistory,
    Material,
    User,
    UserRole,
    UserRoleAssignment,
)


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class UserRepository:
    """User/auth repository operations."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def seed_admin_user(self, *, password_hash: str) -> None:
        """Seed default admin user if missing."""
        with self._session_factory() as session:
            exists_stmt = select(User).where(User.username == "admin").limit(1)
            if session.execute(exists_stmt).scalar_one_or_none():
                return

            user = User(
                username="admin",
                password_hash=password_hash,
                role=UserRole.ADMIN.value,
            )
            session.add(user)
            session.flush()
            session.add(
                UserRoleAssignment(user_id=user.id, role=UserRole.ADMIN.value)
            )
            session.commit()

    def add_user(self, username: str, password_hash: str, role: str) -> None:
        """Create a user row."""
        role = UserRole(role).value
        with self._session_factory() as session:
            user = User(username=username, password_hash=password_hash, role=role)
            session.add(user)
            try:
                session.flush()
                session.add(UserRoleAssignment(user_id=user.id, role=role))
                session.commit()
            except IntegrityError:
                session.rollback()
                raise

    def validate_user(
        self, username: str, password_hash: str
    ) -> tuple[str, str] | None:
        """Return ``(username, role)`` when credentials are valid."""
        with self._session_factory() as session:
            stmt = select(User.username, User.role).where(
                User.username == username,
                User.password_hash == password_hash,
                User.is_active.is_(True),
            )
            row = session.execute(stmt).first()
            return (row[0], row[1]) if row else None

    def get_user_id(self, username: str | None) -> UUID | None:
        """Return the identity UUID for *username*, if present."""
        if not username:
            return None
        with self._session_factory() as session:
            stmt = select(User.id).where(User.username == username, User.is_active.is_(True))
            row = session.execute(stmt).first()
            return row[0] if row else None


class PlantMasterRepository:
    """Read-only plant master lookups used by app repositories and UI."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def list_active_hoppers(self) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            stmt = (
                select(Hopper)
                .where(Hopper.is_active.is_(True))
                .order_by(Hopper.sort_order.asc(), Hopper.hopper_code.asc())
            )
            return [
                {
                    "hopper_code": row.hopper_code,
                    "display_name": row.display_name or row.hopper_code,
                    "sort_order": row.sort_order,
                }
                for row in session.execute(stmt).scalars().all()
            ]

    def list_active_materials(self) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            stmt = (
                select(Material)
                .where(Material.is_active.is_(True))
                .order_by(Material.category_code.asc(), Material.material_code.asc())
            )
            return [
                {
                    "material_code": row.material_code,
                    "material_name": row.material_name,
                    "category_code": row.category_code,
                    "unit_code": row.unit_code,
                }
                for row in session.execute(stmt).scalars().all()
            ]

    def material_code_by_name(self) -> dict[str, str]:
        return {
            row["material_name"]: row["material_code"]
            for row in self.list_active_materials()
        }

    def material_name_by_code(self) -> dict[str, str]:
        return {
            row["material_code"]: row["material_name"]
            for row in self.list_active_materials()
        }

class HopperHistoryRepository:
    """Repository for wide hopper-material snapshot history."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _row_to_codes(row: HopperRawMaterialHistory | None) -> dict[str, str | None]:
        if row is None:
            return {column: None for column in HOPPER_COLUMNS}
        return {column: getattr(row, column) for column in HOPPER_COLUMNS}

    def _latest_row(
        self, session: Session, ts: datetime | None = None
    ) -> HopperRawMaterialHistory | None:
        stmt = select(HopperRawMaterialHistory)
        if ts is not None:
            stmt = stmt.where(HopperRawMaterialHistory.date_time <= _as_aware_utc(ts))
        stmt = stmt.order_by(
            HopperRawMaterialHistory.date_time.desc(),
            HopperRawMaterialHistory.id.desc(),
        ).limit(1)
        return session.execute(stmt).scalar_one_or_none()

    def update_hopper_snapshot(
        self,
        *,
        hopper_material_codes: dict[str, str | None],
        from_time: datetime,
        user_id: UUID | None,
        ip_address: str | None,
        source_type: str = "webapp",
    ) -> None:
        """Insert one full hopper-material snapshot."""
        from_time = _as_aware_utc(from_time)
        with self._session_factory() as session:
            snapshot = self._row_to_codes(self._latest_row(session, from_time))
            for hopper, material_code in hopper_material_codes.items():
                if hopper not in HOPPER_COLUMNS:
                    raise ValueError(f"Invalid hopper: {hopper}")
                snapshot[hopper] = material_code
            session.add(
                HopperRawMaterialHistory(
                    date_time=from_time,
                    ip_address=ip_address,
                    user_modified=user_id,
                    source_type=source_type,
                    **snapshot,
                )
            )
            session.commit()

    def get_current_hopper_material_codes(self) -> dict[str, str | None]:
        """Return current hopper to material-code map."""
        with self._session_factory() as session:
            return self._row_to_codes(self._latest_row(session))

    def get_hopper_material_code_at(
        self, hopper: str, ts: datetime
    ) -> str | None:
        """Return assigned material code for hopper at timestamp."""
        if hopper not in HOPPER_COLUMNS:
            raise ValueError(f"Invalid hopper: {hopper}")
        with self._session_factory() as session:
            row = self._latest_row(session, ts)
            return getattr(row, hopper) if row else None

    def get_hopper_material_history(self) -> list[dict[str, Any]]:
        """Return complete hopper snapshot history rows."""
        with self._session_factory() as session:
            stmt = select(HopperRawMaterialHistory).order_by(
                HopperRawMaterialHistory.date_time.desc(),
                HopperRawMaterialHistory.id.desc(),
            )
            rows = session.execute(stmt).scalars().all()
            out = []
            for row in rows:
                payload = {
                    "id": row.id,
                    "date_time": row.date_time,
                    "source_type": row.source_type,
                    "ip_address": row.ip_address,
                    "user_modified": str(row.user_modified) if row.user_modified else None,
                }
                payload.update({column: getattr(row, column) for column in HOPPER_COLUMNS})
                out.append(payload)
            return out

    def delete_hopper_material_history(self, record_ids: list[int]) -> None:
        """Delete hopper snapshot rows by IDs."""
        if not record_ids:
            return
        with self._session_factory() as session:
            session.execute(
                delete(HopperRawMaterialHistory).where(
                    HopperRawMaterialHistory.id.in_(record_ids)
                )
            )
            session.commit()


class BurdenHistoryRepository:
    """Repository for wide burden-distribution snapshot history."""

    TEXT_FIELDS = frozenset(
        {
            "coke_charge_pattern",
            "non_coke_charge_pattern",
            "burden_changing_purpose",
        }
    )

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def burden_fields() -> list[str]:
        return list(BURDEN_VALUE_COLUMNS)

    def _latest_row(self, session: Session, ts: datetime | None = None) -> BurdenHistory | None:
        stmt = select(BurdenHistory)
        if ts is not None:
            stmt = stmt.where(BurdenHistory.date_time <= _as_aware_utc(ts))
        stmt = stmt.order_by(BurdenHistory.date_time.desc(), BurdenHistory.id.desc()).limit(1)
        return session.execute(stmt).scalar_one_or_none()

    @staticmethod
    def _row_to_values(row: BurdenHistory | None) -> dict[str, Any]:
        if row is None:
            return {column: None for column in BURDEN_VALUE_COLUMNS}
        return {column: getattr(row, column) for column in BURDEN_VALUE_COLUMNS}

    def update_burden_field(
        self,
        *,
        field_name: str,
        value: Any,
        valid_from: datetime,
        user_id: UUID | None = None,
        ip: str = "",
    ) -> None:
        """Insert one snapshot with a single changed burden value."""
        self.update_burden_row(
            row_values={field_name: value},
            timestamp=valid_from,
            user_id=user_id,
            ip=ip,
        )

    def update_burden_row(
        self,
        *,
        row_values: dict[str, Any],
        timestamp: datetime,
        user_id: UUID | None = None,
        ip: str = "",
        source_type: str = "webapp",
    ) -> None:
        """Insert one full burden snapshot copied from latest prior row plus edits."""
        timestamp = _as_aware_utc(timestamp)
        unknown = sorted(set(row_values) - set(BURDEN_VALUE_COLUMNS))
        if unknown:
            raise ValueError(f"Invalid burden field(s): {unknown}")

        with self._session_factory() as session:
            snapshot = self._row_to_values(self._latest_row(session, timestamp))
            for field, value in row_values.items():
                if value == "":
                    snapshot[field] = None
                elif field in self.TEXT_FIELDS or value is None:
                    snapshot[field] = value
                else:
                    snapshot[field] = float(value)
            session.add(
                BurdenHistory(
                    date_time=timestamp,
                    source_type=source_type,
                    ip_address=ip,
                    user_modified=user_id,
                    **snapshot,
                )
            )
            session.commit()

    def get_burden_history(self) -> list[dict[str, Any]]:
        """Return full burden snapshot history."""
        with self._session_factory() as session:
            stmt = select(BurdenHistory).order_by(
                BurdenHistory.date_time.desc(),
                BurdenHistory.id.desc(),
            )
            rows = session.execute(stmt).scalars().all()
            out = []
            for row in rows:
                payload = {
                    "id": row.id,
                    "date_time": row.date_time,
                    "source_type": row.source_type,
                    "ip_address": row.ip_address,
                    "user_modified": str(row.user_modified) if row.user_modified else None,
                }
                payload.update(self._row_to_values(row))
                out.append(payload)
            return out

    def get_all_current_burden_values(self, ts: datetime) -> dict[str, Any]:
        """Return active burden values at timestamp."""
        with self._session_factory() as session:
            return self._row_to_values(self._latest_row(session, ts))

    def delete_burden_history(self, record_ids: list[int]) -> None:
        """Delete burden snapshot rows by IDs."""
        if not record_ids:
            return
        with self._session_factory() as session:
            session.execute(delete(BurdenHistory).where(BurdenHistory.id.in_(record_ids)))
            session.commit()

    def fetch_distribution_frame(self, *, start_date: date, end_date: date) -> pd.DataFrame:
        """Return latest burden snapshots overlapping the date window."""
        window_start = datetime.combine(start_date, time.min).replace(tzinfo=timezone.utc)
        window_end = datetime.combine(end_date, time.max).replace(tzinfo=timezone.utc)
        with self._session_factory() as session:
            prior = self._latest_row(session, window_start)
            stmt = (
                select(BurdenHistory)
                .where(BurdenHistory.date_time >= window_start)
                .where(BurdenHistory.date_time <= window_end)
                .order_by(BurdenHistory.date_time.asc(), BurdenHistory.id.asc())
            )
            rows = session.execute(stmt).scalars().all()
            if prior and (not rows or prior.id != rows[0].id):
                rows.insert(0, prior)

        if not rows:
            return pd.DataFrame()
        records = []
        for row in rows:
            payload = {"time": pd.to_datetime(row.date_time)}
            payload.update(self._row_to_values(row))
            records.append(payload)
        return pd.DataFrame(records).set_index("time").sort_index()
