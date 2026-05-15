"""SQLAlchemy 2.0 ORM models for Neon-backed furnace app tables."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
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


class User(Base):
    """Application user stored in the identity schema."""

    __tablename__ = "users"
    __table_args__ = {"schema": "identity"}

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False, default=UserRole.USER.value)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class UserRoleAssignment(Base):
    """User-role mapping stored separately for future multi-role support."""

    __tablename__ = "user_roles"
    __table_args__ = {"schema": "identity"}

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("identity.users.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(Text, primary_key=True)
    assigned_by_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("identity.users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class MaterialCategory(Base):
    """Canonical material category."""

    __tablename__ = "material_categories"
    __table_args__ = {"schema": "plant_master"}

    category_code: Mapped[str] = mapped_column(Text, primary_key=True)
    category_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class Unit(Base):
    """Canonical unit of measure."""

    __tablename__ = "units"
    __table_args__ = {"schema": "plant_master"}

    unit_code: Mapped[str] = mapped_column(Text, primary_key=True)
    unit_name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class Material(Base):
    """Canonical material master row."""

    __tablename__ = "materials"
    __table_args__ = {"schema": "plant_master"}

    material_code: Mapped[str] = mapped_column(Text, primary_key=True)
    material_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    category_code: Mapped[str] = mapped_column(
        ForeignKey("plant_master.material_categories.category_code"), nullable=False
    )
    material_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    unit_code: Mapped[str] = mapped_column(
        ForeignKey("plant_master.units.unit_code"), nullable=False, default="MT"
    )


class Hopper(Base):
    """Canonical hopper master row."""

    __tablename__ = "hoppers"
    __table_args__ = {"schema": "plant_master"}

    hopper_code: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int | None] = mapped_column(Integer, nullable=True, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class HopperRawMaterialHistory(Base):
    """Wide hopper-to-material snapshot history."""

    __tablename__ = "hopper_raw_material_history"
    __table_args__ = (
        Index("ix_hopper_raw_material_history_date_time", "date_time"),
        {"schema": "ops_config"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    hopper_01: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_02: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_03: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_04: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_05: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_06: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_07: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_08: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_09: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_10: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_11: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_12: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_13: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_14: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_15: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_16: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_17: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_18: Mapped[str | None] = mapped_column(Text, nullable=True)
    hopper_19: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_modified: Mapped[UUID | None] = mapped_column(
        ForeignKey("identity.users.id"), nullable=True
    )
    source_type: Mapped[str] = mapped_column(Text, nullable=False, default="webapp")
    import_batch_id: Mapped[UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class BurdenHistory(Base):
    """Wide burden distribution snapshot history."""

    __tablename__ = "burden_history"
    __table_args__ = (
        Index("ix_burden_history_date_time", "date_time"),
        {"schema": "ops_config"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    coke_p01_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p02_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p03_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p04_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p05_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p06_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p07_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p08_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p09_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p10_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p11_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p01_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p02_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p03_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p04_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p05_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p06_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p07_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p08_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p09_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p10_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_p11_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_discharge_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    coke_charge_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    noncoke_p01_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p02_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p03_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p04_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p05_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p06_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p07_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p08_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p09_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p10_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p11_rings: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p01_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p02_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p03_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p04_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p05_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p06_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p07_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p08_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p09_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p10_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    noncoke_p11_angles: Mapped[float | None] = mapped_column(Float, nullable=True)
    non_coke_discharge_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    non_coke_charge_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    burden_changing_purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(Text, nullable=False, default="webapp")
    ip_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_modified: Mapped[UUID | None] = mapped_column(
        ForeignKey("identity.users.id"), nullable=True
    )
    import_batch_id: Mapped[UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


HOPPER_COLUMNS = tuple(f"hopper_{i:02d}" for i in range(1, 20))
BURDEN_VALUE_COLUMNS = tuple(
    col.name
    for col in BurdenHistory.__table__.columns
    if col.name
    not in {
        "id",
        "date_time",
        "source_type",
        "ip_address",
        "user_modified",
        "import_batch_id",
        "created_at",
    }
)

# Backward-compatible import names while callers migrate.
HopperMaterialHistory = HopperRawMaterialHistory
BurdenDistributionHistory = BurdenHistory
