"""Streamlit-facing relational services for the BF2 web application.

The persistence implementation lives in :mod:`furnace_data.relational`.
This module only adapts repository methods to the webapp's session/auth and
client material-name workflows.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from sqlalchemy.exc import IntegrityError

from data.material_mapping import MaterialNameMapper
from furnace_data.relational import (
    BurdenHistoryRepository,
    HopperHistoryRepository,
    PlantMasterRepository,
    UserRepository,
    build_relational_engine,
    build_relational_session_factory,
)

load_dotenv()


class _RelationalService:
    """Base class that owns a PostgreSQL engine and session factory."""

    def __init__(self, db_url: str | None = None) -> None:
        self.engine = build_relational_engine(db_url=db_url)
        self._session_factory = build_relational_session_factory(self.engine)

    def dispose(self) -> None:
        """Dispose the underlying SQLAlchemy engine."""
        self.engine.dispose()


class UserDataService(_RelationalService):
    """User registration and credential validation service."""

    def __init__(self, db_url: str | None = None) -> None:
        super().__init__(db_url=db_url)
        self._user_repository = UserRepository(self._session_factory)
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        self._user_repository.seed_admin_user(
            password_hash=hashlib.sha256("admin123".encode()).hexdigest()
        )

    def add_user(self, username: str, password: str, role: str = "user") -> None:
        """Add a new user to the identity schema."""
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

    def get_user_id(self, username: str | None) -> str | None:
        """
        Return the identity UUID string for an active username.

        Args:
             - username: str | None - Login username stored in Streamlit session.

        Returns:
             - return: str | None - Matching ``identity.users.id`` UUID as text,
               or None when the username is empty or not active.
        """
        user_id = self._user_repository.get_user_id(username)
        return str(user_id) if user_id else None


class PlantMasterService(_RelationalService):
    """Plant master lookup service used by hopper/material workflows."""

    def __init__(self, db_url: str | None = None) -> None:
        super().__init__(db_url=db_url)
        self._plant_master_repository = PlantMasterRepository(self._session_factory)
        self._material_mapper = MaterialNameMapper.from_file()
        self.refresh()

    def refresh(self) -> None:
        """Refresh active hopper/material lookup caches from plant master."""
        hoppers = self._plant_master_repository.list_active_hoppers()
        materials = self._plant_master_repository.list_active_materials()
        active_material_names = {row["material_name"] for row in materials}
        self._material_mapper.validate_material_names(active_material_names)

        self.hopper_rows = hoppers
        self.hoppers = [row["hopper_code"] for row in hoppers]
        self.hopper_display_names = {
            row["hopper_code"]: row["display_name"] for row in hoppers
        }
        self.material_rows = materials
        self.materials = [
            entry.client_name
            for entry in self._material_mapper.entries
            if entry.material_name in active_material_names
        ]
        self._material_code_by_name = {
            row["material_name"]: row["material_code"] for row in materials
        }
        self._material_name_by_code = {
            row["material_code"]: row["material_name"] for row in materials
        }

    def client_material_to_code(self, material: str | None) -> str | None:
        """Map a webapp material display name to a plant master material code."""
        if material == "UNASSIGNED" or material is None:
            return None
        material_name = self._material_mapper.material_name_for_client(material)
        material_code = self._material_code_by_name.get(material_name)
        if material_code is None:
            raise ValueError(f"Material not active in plant_master: {material_name}")
        return material_code

    def material_code_to_client(self, material_code: str | None) -> str:
        """Map a stored material code back to the primary webapp display name."""
        if material_code is None:
            return "UNASSIGNED"
        material_name = self._material_name_by_code.get(material_code, material_code)
        return self._material_mapper.primary_client_name_for_material(material_name)


class HopperConfigService(PlantMasterService):
    """Hopper-material snapshot service."""

    def __init__(self, db_url: str | None = None) -> None:
        super().__init__(db_url=db_url)
        self._user_repository = UserRepository(self._session_factory)
        self._hopper_repository = HopperHistoryRepository(self._session_factory)

    def update_hopper_material_with_time(
        self,
        hopper: str,
        material: str,
        from_time: datetime,
        modifier: str,
        ip_address: str,
    ) -> None:
        """Record a new single-hopper assignment as a wide snapshot."""
        if hopper not in self.hoppers:
            raise ValueError("Invalid hopper")
        self._hopper_repository.update_hopper_snapshot(
            hopper_material_codes={hopper: self.client_material_to_code(material)},
            from_time=from_time,
            user_id=self._user_repository.get_user_id(modifier),
            ip_address=ip_address,
        )

    def update_hopper_materials_snapshot(
        self,
        hopper_materials: dict[str, str],
        from_time: datetime,
        modifier: str,
        ip_address: str,
    ) -> None:
        """Record one full hopper-material snapshot from client material names."""
        unknown_hoppers = sorted(set(hopper_materials) - set(self.hoppers))
        if unknown_hoppers:
            raise ValueError(f"Invalid hopper(s): {unknown_hoppers}")
        codes = {
            hopper: self.client_material_to_code(material)
            for hopper, material in hopper_materials.items()
        }
        self._hopper_repository.update_hopper_snapshot(
            hopper_material_codes=codes,
            from_time=from_time,
            user_id=self._user_repository.get_user_id(modifier),
            ip_address=ip_address,
        )

    def get_current_hopper_materials(self) -> dict[str, str]:
        """Return current hopper-to-client-material mapping."""
        codes = self._hopper_repository.get_current_hopper_material_codes()
        return {
            hopper: self.material_code_to_client(codes.get(hopper))
            for hopper in self.hoppers
        }

    def get_hopper_material_at(self, hopper: str, ts: datetime) -> str:
        """Return material assigned to hopper at timestamp."""
        code = self._hopper_repository.get_hopper_material_code_at(
            hopper=hopper, ts=ts
        )
        return self.material_code_to_client(code)

    def get_hopper_material_history(self) -> list[dict[str, Any]]:
        """Return full hopper-material history rows with client material labels."""
        rows = self._hopper_repository.get_hopper_material_history()
        for row in rows:
            for hopper in self.hoppers:
                row[hopper] = self.material_code_to_client(row.get(hopper))
        return rows

    def delete_hopper_material_history(self, record_ids: list[int]) -> None:
        """Delete hopper history rows by IDs."""
        self._hopper_repository.delete_hopper_material_history(record_ids)


class BurdenConfigService(_RelationalService):
    """Burden distribution snapshot service."""

    def __init__(self, db_url: str | None = None) -> None:
        super().__init__(db_url=db_url)
        self._user_repository = UserRepository(self._session_factory)
        self._burden_repository = BurdenHistoryRepository(self._session_factory)
        self.burden_fields = self._burden_repository.burden_fields()

    def update_burden_field(
        self,
        field_name: str,
        value: Any,
        valid_from: datetime,
        modifier: str = "system",
        ip: str = "",
    ) -> None:
        """Record a new burden field value as a wide snapshot."""
        self._burden_repository.update_burden_field(
            field_name=field_name,
            value=value,
            valid_from=valid_from,
            user_id=self._user_repository.get_user_id(modifier),
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
            user_id=self._user_repository.get_user_id(modifier),
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
