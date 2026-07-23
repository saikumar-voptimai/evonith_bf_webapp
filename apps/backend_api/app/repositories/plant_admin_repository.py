"""Backend repository facade for plant-admin configuration APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from furnace_data.relational import (
    BurdenHistoryRepository,
    HopperHistoryRepository,
    PlantMasterRepository,
    build_relational_engine,
    build_relational_session_factory,
)


class PlantAdminRepository:
    """Small facade over shared relational repositories."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory
        self._plant_master_repository: PlantMasterRepository | None = None
        self._hopper_history_repository: HopperHistoryRepository | None = None
        self._burden_history_repository: BurdenHistoryRepository | None = None

    @property
    def session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            engine = build_relational_engine()
            self._session_factory = build_relational_session_factory(engine)
        return self._session_factory

    @property
    def plant_master(self) -> PlantMasterRepository:
        if self._plant_master_repository is None:
            self._plant_master_repository = PlantMasterRepository(self.session_factory)
        return self._plant_master_repository

    @property
    def hopper_history(self) -> HopperHistoryRepository:
        if self._hopper_history_repository is None:
            self._hopper_history_repository = HopperHistoryRepository(self.session_factory)
        return self._hopper_history_repository

    @property
    def burden_history(self) -> BurdenHistoryRepository:
        if self._burden_history_repository is None:
            self._burden_history_repository = BurdenHistoryRepository(self.session_factory)
        return self._burden_history_repository

    @staticmethod
    def _uuid(value: str | None) -> UUID | None:
        if not value:
            return None
        return UUID(str(value))

    def list_active_hoppers(self) -> list[dict[str, Any]]:
        return self.plant_master.list_active_hoppers()

    def list_active_materials(self) -> list[dict[str, Any]]:
        return self.plant_master.list_active_materials()

    def get_hopper_snapshot_at(self, at: datetime) -> dict[str, Any]:
        return self.hopper_history.get_hopper_snapshot_at(at)

    def insert_hopper_snapshot(
        self,
        *,
        assignments: dict[str, str | None],
        effective_at: datetime,
        user_id: str | None,
        ip_address: str | None,
    ) -> int:
        return self.hopper_history.update_hopper_snapshot(
            hopper_material_codes=assignments,
            from_time=effective_at,
            user_id=self._uuid(user_id),
            ip_address=ip_address,
            source_type="api",
        )

    def list_hopper_history(self, *, limit: int, offset: int) -> list[dict[str, Any]]:
        return self.hopper_history.get_hopper_material_history(limit=limit, offset=offset)

    def count_hopper_history(self) -> int:
        return self.hopper_history.count_hopper_material_history()

    def delete_hopper_history(self, record_ids: list[int]) -> int:
        return self.hopper_history.delete_hopper_material_history(record_ids)

    def get_burden_snapshot_at(self, at: datetime) -> dict[str, Any]:
        return self.burden_history.get_burden_snapshot_at(at)

    def burden_fields(self) -> list[str]:
        return self.burden_history.burden_fields()

    def insert_burden_snapshot(
        self,
        *,
        values: dict[str, Any],
        effective_at: datetime,
        user_id: str | None,
        ip_address: str | None,
    ) -> int:
        return self.burden_history.update_burden_row(
            row_values=values,
            timestamp=effective_at,
            user_id=self._uuid(user_id),
            ip=ip_address or "",
            source_type="api",
        )

    def list_burden_history(self, *, limit: int, offset: int) -> list[dict[str, Any]]:
        return self.burden_history.get_burden_history(limit=limit, offset=offset)

    def count_burden_history(self) -> int:
        return self.burden_history.count_burden_history()

    def delete_burden_history(self, record_ids: list[int]) -> int:
        return self.burden_history.delete_burden_history(record_ids)
