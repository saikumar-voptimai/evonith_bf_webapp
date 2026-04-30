"""Relational database facade for the BF2 blast furnace web application.

This module keeps the legacy :class:`Database` public API stable while routing
all persistence through SQLAlchemy 2.0 ORM repositories.
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from sqlalchemy.exc import IntegrityError

try:
    from furnace_data import relational as _relational
except ModuleNotFoundError:
    # Prefer in-repo furnace_data source when an older site-packages build is present.
    furnace_data_src = Path(__file__).resolve().parents[2] / "furnace_data"
    if str(furnace_data_src) not in sys.path:
        sys.path.insert(0, str(furnace_data_src))
    from furnace_data import relational as _relational

Base = _relational.Base
BurdenHistoryRepository = _relational.BurdenHistoryRepository
HopperHistoryRepository = _relational.HopperHistoryRepository
UserRepository = _relational.UserRepository
build_relational_engine = _relational.build_relational_engine
build_relational_session_factory = _relational.build_relational_session_factory

# ------------------------------------------------------------
# Load Environment Variables
# ------------------------------------------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MATERIALS_FILE = os.path.join(PROJECT_ROOT, "config", "materials.yml")


class Database:
    """Compatibility facade over SQLAlchemy 2.0 repositories.

    Public method names and return shapes are kept stable for existing UI/auth
    call-sites while implementation moves away from raw SQL text.
    """

    def __init__(self, db_url: str | None = None) -> None:
        self.engine = build_relational_engine(db_url=db_url)
        self._session_factory = build_relational_session_factory(self.engine)

        self.hoppers, self.materials = self._safe_load_materials()
        self.burden_fields = self._safe_load_burden_fields()

        self._user_repository = UserRepository(self._session_factory)
        self._hopper_repository = HopperHistoryRepository(self._session_factory)
        self._burden_repository = BurdenHistoryRepository(self._session_factory)

        # Managed by Alembic going forward; kept for compatibility bootstrap.
        Base.metadata.create_all(self.engine)
        self._seed_defaults()

    # ============================================================
    # USERS
    # ============================================================
    def _seed_defaults(self) -> None:
        """Seed initial admin user and hopper rows where missing."""
        self._user_repository.seed_admin_user(
            password_hash=hashlib.sha256("admin123".encode()).hexdigest()
        )
        self._hopper_repository.seed_hoppers_if_missing(
            hoppers=self.hoppers,
            now=datetime.now(timezone.utc).replace(tzinfo=None),
        )

    def add_user(self, username: str, password: str, role: str = "user") -> None:
        """Add a new user to the database.

        Args:
            username: Unique username string.
            password: Plain-text password (stored as SHA-256 digest).
            role: One of ``"admin"``, ``"supervisor"``, or ``"user"``.

        Raises:
            ValueError: If username already exists or role is invalid.
        """
        try:
            self._user_repository.add_user(
                username=username,
                password_hash=hashlib.sha256(password.encode()).hexdigest(),
                role=role,
            )
        except IntegrityError as exc:
            raise ValueError("User already exists") from exc
        except ValueError as exc:
            raise ValueError("Invalid role") from exc

    def validate_user(self, username: str, password: str) -> tuple[str, str] | None:
        """Validate credentials and return ``(username, role)`` or ``None``."""
        return self._user_repository.validate_user(
            username=username,
            password_hash=hashlib.sha256(password.encode()).hexdigest(),
        )

    # ============================================================
    # YAML
    # ============================================================
    def _safe_load_materials(self) -> tuple[list[str], list[str]]:
        """Load hoppers/materials from YAML, returning empty lists on error."""
        try:
            return self._load_materials_from_yaml()
        except Exception:
            return [], []

    def _load_materials_from_yaml(self) -> tuple[list[str], list[str]]:
        """Parse ``materials.yml`` and return ``(hoppers, materials)`` lists."""
        with open(MATERIALS_FILE, "r", encoding="utf-8-sig") as file:
            data = yaml.safe_load(file) or {}
        return data.get("hoppers", []), data.get("materials", [])

    def _safe_load_burden_fields(self) -> list[str]:
        """Safely load burden distribution field names from YAML."""
        try:
            return self._load_burden_fields_from_yaml()
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"Warning: Failed to load burden_fields from YAML: {exc}")
            return []

    def _load_burden_fields_from_yaml(self) -> list[str]:
        """Load burden field definitions from ``materials.yml``."""
        if not os.path.exists(MATERIALS_FILE):
            raise FileNotFoundError(f"Missing configuration file: {MATERIALS_FILE}")

        with open(MATERIALS_FILE, "r", encoding="utf-8-sig") as file:
            data = yaml.safe_load(file) or {}

        fields = data.get("burden_fields", [])
        if not isinstance(fields, list):
            raise ValueError("'burden_fields' must be a list in materials.yml")
        return fields

    # ============================================================
    # HOPPER TO MATERIAL HISTORY
    # ============================================================
    def update_hopper_material_with_time(
        self,
        hopper: str,
        material: str,
        from_time: datetime,
        modifier: str,
        ip_address: str,
    ) -> None:
        """Record a new hopper-to-material assignment using SCD Type-2."""
        if hopper not in self.hoppers:
            raise ValueError("Invalid hopper")
        if material not in self.materials and material != "UNASSIGNED":
            raise ValueError("Invalid material")

        self._hopper_repository.update_hopper_material_with_time(
            hopper=hopper,
            material=material,
            from_time=from_time,
            modifier=modifier,
            ip_address=ip_address,
        )

    def get_current_hopper_materials(self) -> dict[str, str]:
        """Return current hopper-to-material mapping."""
        return self._hopper_repository.get_current_hopper_materials()

    def get_hopper_material_at(self, hopper: str, ts: datetime) -> str | None:
        """Return material assigned to hopper at timestamp."""
        return self._hopper_repository.get_hopper_material_at(hopper=hopper, ts=ts)

    def get_hopper_material_history(self) -> list[dict[str, Any]]:
        """Return full hopper-material history rows."""
        return self._hopper_repository.get_hopper_material_history()

    def delete_hopper_material_history(self, record_ids: list[int]) -> None:
        """Delete hopper history rows by IDs."""
        self._hopper_repository.delete_hopper_material_history(record_ids)

    # ============================================================
    # BURDEN DISTRIBUTION HISTORY
    # ============================================================
    def update_burden_field(
        self,
        field_name: str,
        value: Any,
        valid_from: datetime,
        modifier: str = "system",
        ip: str = "",
    ) -> None:
        """Record a new burden field value using SCD Type-2."""
        self._burden_repository.update_burden_field(
            field_name=field_name,
            value=value,
            valid_from=valid_from,
            modifier=modifier,
            ip=ip,
        )

    def update_burden_row(
        self,
        df_row: dict[str, Any],
        timestamp: datetime,
        modifier: str = "system",
        ip: str = "",
    ) -> None:
        """Bulk-update burden fields from one row mapping."""
        self._burden_repository.update_burden_row(
            row_values=dict(df_row.items()),
            timestamp=timestamp,
            burden_fields=self.burden_fields,
            modifier=modifier,
            ip=ip,
        )

    def get_burden_history(self) -> list[dict[str, Any]]:
        """Return full burden-history rows."""
        return self._burden_repository.get_burden_history()

    def get_all_current_burden_values(self, ts: datetime) -> dict[str, Any]:
        """Return current burden field values at timestamp."""
        return self._burden_repository.get_all_current_burden_values(ts)

    def delete_burden_history(self, record_ids: list[int]) -> None:
        """Delete burden-history records by IDs."""
        self._burden_repository.delete_burden_history(record_ids)

    def dispose(self) -> None:
        """Dispose underlying SQLAlchemy engine and release pooled connections."""
        self.engine.dispose()
