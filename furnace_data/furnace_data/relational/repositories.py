"""Repository classes for BF2 relational persistence."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from sqlalchemy import String, and_, cast, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .models import BurdenDistributionHistory, HopperMaterialHistory, User, UserRole


class UserRepository:
    """User/auth repository operations."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def seed_admin_user(self, *, password_hash: str) -> None:
        """Seed default admin user if missing."""
        with self._session_factory() as session:
            exists_stmt = select(User.username).where(User.username == "admin").limit(1)
            if session.execute(exists_stmt).first():
                return

            session.add(
                User(
                    username="admin",
                    password_hash=password_hash,
                    role=UserRole.ADMIN,
                )
            )
            session.commit()

    def add_user(self, username: str, password_hash: str, role: str) -> None:
        """Create a user row."""
        with self._session_factory() as session:
            session.add(
                User(
                    username=username,
                    password_hash=password_hash,
                    role=UserRole(role),
                )
            )
            try:
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
                and_(
                    User.username == username,
                    User.password_hash == password_hash,
                )
            )
            row = session.execute(stmt).first()
            if row is None:
                return None
            return row[0], row[1].value if isinstance(row[1], UserRole) else str(row[1])


class HopperHistoryRepository:
    """SCD Type-2 repository for hopper-material mapping."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def seed_hoppers_if_missing(self, hoppers: list[str], now: datetime) -> None:
        """Seed missing hoppers with ``UNASSIGNED`` records."""
        with self._session_factory() as session:
            existing_stmt = select(HopperMaterialHistory.hopper).distinct()
            existing = {row[0] for row in session.execute(existing_stmt).all()}
            for hopper in hoppers:
                if hopper in existing:
                    continue
                session.add(
                    HopperMaterialHistory(
                        hopper=hopper,
                        material="UNASSIGNED",
                        valid_from=now,
                    )
                )
            session.commit()

    def update_hopper_material_with_time(
        self,
        *,
        hopper: str,
        material: str,
        from_time: datetime,
        modifier: str,
        ip_address: str,
    ) -> None:
        """Close current active row and insert a new hopper-material record."""
        with self._session_factory() as session:
            close_stmt = (
                update(HopperMaterialHistory)
                .where(
                    and_(
                        HopperMaterialHistory.hopper == hopper,
                        HopperMaterialHistory.valid_upto.is_(None),
                    )
                )
                .values(valid_upto=from_time - timedelta(seconds=1))
            )
            session.execute(close_stmt)

            session.add(
                HopperMaterialHistory(
                    hopper=hopper,
                    material=material,
                    valid_from=from_time,
                    modifier=modifier,
                    ip_address=ip_address,
                )
            )
            session.commit()

    def get_current_hopper_materials(self) -> dict[str, str]:
        """Return current hopper to material map."""
        with self._session_factory() as session:
            stmt = (
                select(HopperMaterialHistory.hopper, HopperMaterialHistory.material)
                .where(HopperMaterialHistory.valid_upto.is_(None))
                .order_by(HopperMaterialHistory.hopper.asc())
            )
            return {row[0]: row[1] for row in session.execute(stmt).all()}

    def get_hopper_material_at(self, hopper: str, ts: datetime) -> str | None:
        """Return assigned material for hopper at timestamp."""
        with self._session_factory() as session:
            stmt = (
                select(HopperMaterialHistory.material)
                .where(
                    and_(
                        HopperMaterialHistory.hopper == hopper,
                        HopperMaterialHistory.valid_from <= ts,
                        or_(
                            HopperMaterialHistory.valid_upto.is_(None),
                            HopperMaterialHistory.valid_upto >= ts,
                        ),
                    )
                )
                .order_by(HopperMaterialHistory.valid_from.desc())
                .limit(1)
            )
            row = session.execute(stmt).first()
            return row[0] if row else None

    def get_hopper_material_history(self) -> list[dict[str, Any]]:
        """Return complete hopper material history rows."""
        with self._session_factory() as session:
            stmt = select(HopperMaterialHistory).order_by(
                HopperMaterialHistory.hopper.asc(),
                HopperMaterialHistory.valid_from.desc(),
                HopperMaterialHistory.id.desc(),
            )
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": row.id,
                    "hopper": row.hopper,
                    "material": row.material,
                    "valid_from": row.valid_from,
                    "valid_upto": row.valid_upto,
                    "modifier": row.modifier,
                    "ip_address": row.ip_address,
                }
                for row in rows
            ]

    def delete_hopper_material_history(self, record_ids: list[int]) -> None:
        """Delete hopper history rows by IDs."""
        if not record_ids:
            return
        with self._session_factory() as session:
            stmt = delete(HopperMaterialHistory).where(
                HopperMaterialHistory.id.in_(record_ids)
            )
            session.execute(stmt)
            session.commit()


class BurdenHistoryRepository:
    """SCD Type-2 repository for burden-distribution fields."""

    TEXT_FIELDS = frozenset(
        {
            "COKE_CHARGE_PATTERN",
            "NON_COKE_CHARGE_PATTERN",
            "BURDEN_CHANGING_PURPOSE",
        }
    )

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def update_burden_field(
        self,
        *,
        field_name: str,
        value: Any,
        valid_from: datetime,
        modifier: str = "system",
        ip: str = "",
    ) -> None:
        """Close active value and append new value row for one burden field."""
        end_time = valid_from - timedelta(seconds=1)
        is_text_field = field_name in self.TEXT_FIELDS

        with self._session_factory() as session:
            close_stmt = (
                update(BurdenDistributionHistory)
                .where(
                    and_(
                        BurdenDistributionHistory.field_name == field_name,
                        BurdenDistributionHistory.valid_upto.is_(None),
                    )
                )
                .values(valid_upto=end_time)
            )
            session.execute(close_stmt)

            payload = {
                "field_name": field_name,
                "valid_from": valid_from,
                "valid_upto": None,
                "modifier": modifier,
                "ip_address": ip,
                "field_value_text": str(value) if is_text_field else None,
                "field_value_float": None if is_text_field else float(value),
            }
            session.add(BurdenDistributionHistory(**payload))
            session.commit()

    def update_burden_row(
        self,
        *,
        row_values: dict[str, Any],
        timestamp: datetime,
        burden_fields: list[str],
        modifier: str = "system",
        ip: str = "",
    ) -> None:
        """Bulk-update all applicable burden fields from one row-like mapping."""
        for field, value in row_values.items():
            if field in burden_fields and value is not None:
                self.update_burden_field(
                    field_name=field,
                    value=value,
                    valid_from=timestamp,
                    modifier=modifier,
                    ip=ip,
                )

    def get_burden_history(self) -> list[dict[str, Any]]:
        """Return full burden history as display-ready dictionaries."""
        with self._session_factory() as session:
            stmt = select(BurdenDistributionHistory).order_by(
                BurdenDistributionHistory.field_name.asc(),
                BurdenDistributionHistory.valid_from.desc(),
            )
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": row.id,
                    "field_name": row.field_name,
                    "value": (
                        row.field_value_text
                        if row.field_value_text is not None
                        else row.field_value_float
                    ),
                    "valid_from": row.valid_from,
                    "valid_upto": row.valid_upto,
                    "modifier": row.modifier,
                    "ip_address": row.ip_address,
                }
                for row in rows
            ]

    def get_all_current_burden_values(self, ts: datetime) -> dict[str, Any]:
        """Return active burden field values at timestamp."""
        with self._session_factory() as session:
            ranked_subquery = (
                select(
                    BurdenDistributionHistory.field_name.label("field_name"),
                    BurdenDistributionHistory.field_value_float.label(
                        "field_value_float"
                    ),
                    BurdenDistributionHistory.field_value_text.label(
                        "field_value_text"
                    ),
                    func.row_number()
                    .over(
                        partition_by=BurdenDistributionHistory.field_name,
                        order_by=BurdenDistributionHistory.valid_from.desc(),
                    )
                    .label("row_num"),
                )
                .where(
                    and_(
                        BurdenDistributionHistory.valid_from <= ts,
                        or_(
                            BurdenDistributionHistory.valid_upto.is_(None),
                            BurdenDistributionHistory.valid_upto >= ts,
                        ),
                    )
                )
                .subquery()
            )

            stmt = select(
                ranked_subquery.c.field_name,
                ranked_subquery.c.field_value_float,
                ranked_subquery.c.field_value_text,
            ).where(ranked_subquery.c.row_num == 1)

            rows = session.execute(stmt).all()
            return {row[0]: (row[2] if row[2] is not None else row[1]) for row in rows}

    def delete_burden_history(self, record_ids: list[int]) -> None:
        """Delete burden-history rows by IDs."""
        if not record_ids:
            return
        with self._session_factory() as session:
            stmt = delete(BurdenDistributionHistory).where(
                BurdenDistributionHistory.id.in_(record_ids)
            )
            session.execute(stmt)
            session.commit()

    def list_distribution_rows_for_window(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[tuple[str, str | None, datetime, datetime | None]]:
        """Return burden rows overlapping [start_date, end_date] for analytics joins."""
        window_start = datetime.combine(start_date, time.min)
        window_end = datetime.combine(end_date, time.max)

        with self._session_factory() as session:
            field_value = func.coalesce(
                cast(BurdenDistributionHistory.field_value_float, String),
                BurdenDistributionHistory.field_value_text,
            ).label("field_value")

            stmt = (
                select(
                    BurdenDistributionHistory.field_name,
                    field_value,
                    BurdenDistributionHistory.valid_from,
                    BurdenDistributionHistory.valid_upto,
                )
                .where(
                    and_(
                        BurdenDistributionHistory.valid_from <= window_end,
                        or_(
                            BurdenDistributionHistory.valid_upto.is_(None),
                            BurdenDistributionHistory.valid_upto >= window_start,
                        ),
                    )
                )
                .order_by(BurdenDistributionHistory.valid_from.asc())
            )
            return list(session.execute(stmt).all())
