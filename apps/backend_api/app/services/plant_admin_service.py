"""Plant administration business service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.plant_admin_repository import PlantAdminRepository
from furnace_data.material_mapping import MaterialNameMapper
from furnace_data.relational import BurdenHistoryRepository


class PlantAdminService:
    """Backend-owned plant configuration workflows."""

    def __init__(
        self,
        *,
        repository: PlantAdminRepository | None = None,
        material_mapper: MaterialNameMapper | None = None,
    ) -> None:
        self.repository = repository or PlantAdminRepository()
        self.material_mapper = material_mapper or MaterialNameMapper.from_file()

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc).replace(microsecond=0)

    @staticmethod
    def _aware_utc(value: datetime | None, *, field_name: str) -> datetime:
        if value is None:
            return PlantAdminService._now_utc()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ApiError(
                "VALIDATION_ERROR",
                f"{field_name} must include a timezone offset.",
                status_code=422,
            )
        return value.astimezone(timezone.utc).replace(microsecond=0)

    @staticmethod
    def _label_for_key(key: str) -> str:
        return str(key).replace("_", " ").title()

    @staticmethod
    def _actor(current_user: dict[str, Any]) -> dict[str, str | None]:
        return {
            "user_id": str(current_user.get("id") or "") or None,
            "username": str(current_user.get("username") or "") or None,
        }

    def _active_hoppers(self) -> list[dict[str, Any]]:
        return self.repository.list_active_hoppers()

    def _active_materials(self) -> list[dict[str, Any]]:
        materials = self.repository.list_active_materials()
        active_names = {str(row["material_name"]) for row in materials}
        self.material_mapper.validate_material_names(active_names)
        return materials

    def _hopper_codes(self) -> set[str]:
        return {str(row["hopper_code"]) for row in self._active_hoppers()}

    def _material_codes(self) -> set[str]:
        return {str(row["material_code"]) for row in self._active_materials()}

    def hopper_context(self, *, at: datetime | None = None) -> dict[str, Any]:
        at_utc = self._aware_utc(at, field_name="at")
        hoppers = self._active_hoppers()
        materials = self._active_materials()
        row = self.repository.get_hopper_snapshot_at(at_utc)
        assignments = {
            str(hopper["hopper_code"]): row.get(str(hopper["hopper_code"]))
            for hopper in hoppers
        }
        return {
            "at": at_utc,
            "snapshot_id": row.get("id"),
            "effective_at": row.get("date_time"),
            "hoppers": [
                {
                    "code": str(hopper["hopper_code"]),
                    "display_name": str(hopper.get("display_name") or hopper["hopper_code"]),
                }
                for hopper in hoppers
            ],
            "materials": [
                {
                    "code": str(material["material_code"]),
                    "canonical_name": str(material["material_name"]),
                    "display_name": self.material_mapper.primary_client_name_for_material(
                        str(material["material_name"])
                    ),
                }
                for material in materials
            ],
            "assignments": assignments,
        }

    def update_hopper_mapping(
        self,
        *,
        effective_at: datetime,
        expected_snapshot_id: int | None,
        assignments: dict[str, str | None],
        current_user: dict[str, Any],
        ip_address: str | None,
    ) -> dict[str, Any]:
        effective_utc = self._aware_utc(effective_at, field_name="effective_at")
        active_hoppers = self._hopper_codes()
        active_materials = self._material_codes()

        invalid_hoppers = sorted(set(assignments) - active_hoppers)
        if invalid_hoppers:
            raise ApiError(
                "INVALID_HOPPER",
                "One or more hopper codes are not active.",
                status_code=422,
                details={"hoppers": invalid_hoppers},
            )
        invalid_materials = sorted(
            {
                str(material_code)
                for material_code in assignments.values()
                if material_code is not None and str(material_code) not in active_materials
            }
        )
        if invalid_materials:
            raise ApiError(
                "INVALID_MATERIAL",
                "One or more material codes are not active.",
                status_code=422,
                details={"materials": invalid_materials},
            )

        latest = self.repository.get_hopper_snapshot_at(effective_utc)
        if latest.get("id") != expected_snapshot_id:
            raise ApiError(
                "CONFIG_VERSION_CONFLICT",
                "The hopper mapping changed before this update was submitted.",
                status_code=409,
            )

        clean_assignments = {
            str(hopper): (str(material) if material is not None else None)
            for hopper, material in assignments.items()
        }
        self.repository.insert_hopper_snapshot(
            assignments=clean_assignments,
            effective_at=effective_utc,
            user_id=str(current_user.get("id") or "") or None,
            ip_address=ip_address,
        )
        return self.hopper_context(at=effective_utc)

    def hopper_history(self, *, limit: int, offset: int) -> dict[str, Any]:
        limit = max(1, min(200, int(limit)))
        offset = max(0, int(offset))
        hoppers = [str(row["hopper_code"]) for row in self._active_hoppers()]
        rows = self.repository.list_hopper_history(limit=limit, offset=offset)
        return {
            "items": [
                {
                    "snapshot_id": int(row["id"]),
                    "effective_at": row["date_time"],
                    "assignments": {hopper: row.get(hopper) for hopper in hoppers},
                    "source_type": row.get("source_type"),
                    "actor": {"user_id": row.get("user_modified"), "username": None},
                    "created_at": row.get("created_at"),
                }
                for row in rows
            ],
            "total": self.repository.count_hopper_history(),
            "limit": limit,
            "offset": offset,
        }

    def delete_hopper_history(self, *, record_ids: list[int]) -> dict[str, Any]:
        try:
            deleted = self.repository.delete_hopper_history(record_ids)
        except ValueError as exc:
            raise ApiError("VALIDATION_ERROR", str(exc), status_code=422) from exc
        return {
            "deleted_count": deleted,
            "current_context": self.hopper_context(),
        }

    def _burden_fields(self) -> list[str]:
        return self.repository.burden_fields()

    def _burden_field_definitions(self) -> list[dict[str, Any]]:
        text_fields = set(BurdenHistoryRepository.TEXT_FIELDS)
        return [
            {
                "key": field,
                "label": self._label_for_key(field),
                "value_type": "text" if field in text_fields else "number",
                "nullable": True,
                "step": None if field in text_fields else 0.1,
            }
            for field in self._burden_fields()
        ]

    def burden_context(self, *, at: datetime | None = None) -> dict[str, Any]:
        at_utc = self._aware_utc(at, field_name="at")
        row = self.repository.get_burden_snapshot_at(at_utc)
        fields = self._burden_fields()
        return {
            "at": at_utc,
            "snapshot_id": row.get("id"),
            "effective_at": row.get("date_time"),
            "fields": self._burden_field_definitions(),
            "values": {field: row.get(field) for field in fields},
        }

    def _validate_burden_values(self, values: dict[str, Any]) -> dict[str, Any]:
        fields = set(self._burden_fields())
        unknown = sorted(set(values) - fields)
        if unknown:
            raise ApiError(
                "INVALID_BURDEN_FIELD",
                "One or more burden fields are not valid.",
                status_code=422,
                details={"fields": unknown},
            )
        text_fields = set(BurdenHistoryRepository.TEXT_FIELDS)
        clean: dict[str, Any] = {}
        for field, value in values.items():
            if field in text_fields:
                if value is not None and not isinstance(value, str):
                    raise ApiError(
                        "INVALID_BURDEN_FIELD",
                        f"{field} must be text or null.",
                        status_code=422,
                    )
                clean[field] = value if value not in {"", None} else None
                continue

            if value is None:
                clean[field] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ApiError(
                    "INVALID_BURDEN_FIELD",
                    f"{field} must be numeric or null.",
                    status_code=422,
                )
            clean[field] = float(value)
        return clean

    def update_burden_distribution(
        self,
        *,
        effective_at: datetime,
        expected_snapshot_id: int | None,
        values: dict[str, Any],
        current_user: dict[str, Any],
        ip_address: str | None,
    ) -> dict[str, Any]:
        effective_utc = self._aware_utc(effective_at, field_name="effective_at")
        latest = self.repository.get_burden_snapshot_at(effective_utc)
        if latest.get("id") != expected_snapshot_id:
            raise ApiError(
                "CONFIG_VERSION_CONFLICT",
                "The burden distribution changed before this update was submitted.",
                status_code=409,
            )
        clean = self._validate_burden_values(values)
        self.repository.insert_burden_snapshot(
            values=clean,
            effective_at=effective_utc,
            user_id=str(current_user.get("id") or "") or None,
            ip_address=ip_address,
        )
        return self.burden_context(at=effective_utc)

    def burden_history(self, *, limit: int, offset: int) -> dict[str, Any]:
        limit = max(1, min(200, int(limit)))
        offset = max(0, int(offset))
        fields = self._burden_fields()
        rows = self.repository.list_burden_history(limit=limit, offset=offset)
        return {
            "items": [
                {
                    "snapshot_id": int(row["id"]),
                    "effective_at": row["date_time"],
                    "values": {field: row.get(field) for field in fields},
                    "source_type": row.get("source_type"),
                    "actor": {"user_id": row.get("user_modified"), "username": None},
                    "created_at": row.get("created_at"),
                }
                for row in rows
            ],
            "total": self.repository.count_burden_history(),
            "limit": limit,
            "offset": offset,
        }

    def delete_burden_history(self, *, record_ids: list[int]) -> dict[str, Any]:
        try:
            deleted = self.repository.delete_burden_history(record_ids)
        except ValueError as exc:
            raise ApiError("VALIDATION_ERROR", str(exc), status_code=422) from exc
        return {
            "deleted_count": deleted,
            "current_context": self.burden_context(),
        }
