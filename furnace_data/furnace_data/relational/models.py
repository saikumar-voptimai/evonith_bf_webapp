"""SQLAlchemy 2.0 ORM models for BF2 operational relational tables."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import DateTime, Enum as SqlEnum, Float, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base declarative class for relational ORM models."""


class UserRole(str, Enum):
    """Supported user roles for dashboard authorization."""

    ADMIN = "admin"
    SUPERVISOR = "supervisor"
    USER = "user"


USER_ROLE_ENUM = SqlEnum(
    UserRole,
    name="user_role",
    native_enum=False,
    values_callable=lambda enum: [e.value for e in enum],
)


class User(Base):
    """User credentials and role mapping."""

    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(128), primary_key=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[UserRole] = mapped_column(USER_ROLE_ENUM, nullable=False)


class HopperMaterialHistory(Base):
    """SCD Type-2 history for hopper to material assignments."""

    __tablename__ = "hopper_material_history"
    __table_args__ = (
        Index("ix_hopper_history_hopper_valid_from", "hopper", "valid_from"),
        Index("ix_hopper_history_hopper_valid_upto", "hopper", "valid_upto"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    hopper: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    material: Mapped[str] = mapped_column(String(256), nullable=False)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    valid_upto: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    modifier: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    ip_address: Mapped[str | None] = mapped_column(String(256), nullable=True)


class BurdenDistributionHistory(Base):
    """SCD Type-2 history for burden distribution fields."""

    __tablename__ = "burden_distribution_history"
    __table_args__ = (
        UniqueConstraint(
            "field_name",
            "valid_upto",
            name="uq_burden_active_record",
        ),
        Index("ix_burden_history_field_valid_from", "field_name", "valid_from"),
        Index("ix_burden_history_field_valid_upto", "field_name", "valid_upto"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    field_name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    field_value_float: Mapped[float | None] = mapped_column(Float, nullable=True)
    field_value_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    valid_upto: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    modifier: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    ip_address: Mapped[str | None] = mapped_column(String(256), nullable=True)