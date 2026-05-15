"""Tests for the thin Streamlit-facing relational services."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest

import src.data.db as db_module
from src.data.db import BurdenConfigService, HopperConfigService, UserDataService
from src.data.material_mapping import MaterialMapEntry, MaterialNameMapper


class _FakeEngine:
    def dispose(self) -> None:
        pass


class _BaseRepo:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory


class _FakeUserRepository(_BaseRepo):
    users: dict[str, tuple[str, str]] = {}
    user_roles: set[tuple[str, str]] = set()

    def seed_admin_user(self, *, password_hash: str) -> None:
        self.users.setdefault("admin", (password_hash, "admin"))
        self.user_roles.add(("admin", "admin"))

    def add_user(self, username: str, password_hash: str, role: str) -> None:
        if username in self.users:
            raise db_module.IntegrityError("duplicate", {}, None)
        self.users[username] = (password_hash, role)
        self.user_roles.add((username, role))

    def validate_user(self, username: str, password_hash: str):
        row = self.users.get(username)
        if row and row[0] == password_hash:
            return username, row[1]
        return None

    def get_user_id(self, username: str | None):
        if username == "qa":
            return UUID("00000000-0000-0000-0000-000000000001")
        return None


class _FakePlantMasterRepository(_BaseRepo):
    def list_active_hoppers(self):
        return [{"hopper_code": "hopper_01", "display_name": "Hopper 01", "sort_order": 1}]

    def list_active_materials(self):
        return [
            {
                "material_code": "ore_1",
                "material_name": "nmdc_donimalai_ore",
                "category_code": "ORE",
                "unit_code": "MT",
            }
        ]


class _FakeHopperHistoryRepository(_BaseRepo):
    last_snapshot = None

    def update_hopper_snapshot(self, **kwargs) -> None:
        self.__class__.last_snapshot = kwargs

    def get_current_hopper_material_codes(self):
        return {"hopper_01": "ore_1"}

    def get_hopper_material_code_at(self, hopper, ts):
        return "ore_1"

    def get_hopper_material_history(self):
        return [{"id": 1, "hopper_01": "ore_1"}]

    def delete_hopper_material_history(self, record_ids) -> None:
        self.__class__.last_deleted = list(record_ids)


class _FakeBurdenHistoryRepository(_BaseRepo):
    last_row = None

    @staticmethod
    def burden_fields():
        return ["coke_p01_rings"]

    def update_burden_field(self, **kwargs) -> None:
        self.__class__.last_field = kwargs

    def update_burden_row(self, **kwargs) -> None:
        self.__class__.last_row = kwargs

    def get_burden_history(self):
        return [{"id": 1, "coke_p01_rings": 4.0}]

    def get_all_current_burden_values(self, ts):
        return {"coke_p01_rings": 4.0}

    def delete_burden_history(self, record_ids) -> None:
        self.__class__.last_deleted = list(record_ids)


@pytest.fixture(autouse=True)
def _fake_repositories(monkeypatch):
    _FakeUserRepository.users = {}
    _FakeUserRepository.user_roles = set()
    _FakeHopperHistoryRepository.last_snapshot = None
    _FakeBurdenHistoryRepository.last_row = None

    monkeypatch.setattr(db_module, "build_relational_engine", lambda db_url=None: _FakeEngine())
    monkeypatch.setattr(db_module, "build_relational_session_factory", lambda engine: object())
    monkeypatch.setattr(db_module, "UserRepository", _FakeUserRepository)
    monkeypatch.setattr(db_module, "PlantMasterRepository", _FakePlantMasterRepository)
    monkeypatch.setattr(db_module, "HopperHistoryRepository", _FakeHopperHistoryRepository)
    monkeypatch.setattr(db_module, "BurdenHistoryRepository", _FakeBurdenHistoryRepository)
    monkeypatch.setattr(
        db_module.MaterialNameMapper,
        "from_file",
        classmethod(
            lambda cls: MaterialNameMapper(
                [
                    MaterialMapEntry(
                        client_name="NMDC Limited (Donimalai)",
                        material_name="nmdc_donimalai_ore",
                        is_primary=True,
                    )
                ]
            )
        ),
    )


def test_user_service_delegates_auth_flow_to_repository() -> None:
    service = UserDataService(db_url="postgresql://example")

    assert service.validate_user("admin", "admin123") == ("admin", "admin")

    service.add_user("qa", "pass123", "user")
    assert service.validate_user("qa", "pass123") == ("qa", "user")
    assert ("qa", "user") in _FakeUserRepository.user_roles

    with pytest.raises(ValueError, match="User already exists"):
        service.add_user("qa", "pass123", "user")


def test_hopper_service_maps_client_names_to_material_codes() -> None:
    service = HopperConfigService(db_url="postgresql://example")
    timestamp = datetime(2026, 1, 1, 6, 0)

    service.update_hopper_material_with_time(
        hopper="hopper_01",
        material="NMDC Limited (Donimalai)",
        from_time=timestamp,
        modifier="qa",
        ip_address="127.0.0.1",
    )

    snapshot = _FakeHopperHistoryRepository.last_snapshot
    assert snapshot["hopper_material_codes"] == {"hopper_01": "ore_1"}
    assert snapshot["user_id"] == UUID("00000000-0000-0000-0000-000000000001")
    assert service.get_current_hopper_materials() == {
        "hopper_01": "NMDC Limited (Donimalai)"
    }


def test_burden_service_delegates_wide_snapshot_updates() -> None:
    service = BurdenConfigService(db_url="postgresql://example")
    timestamp = datetime(2026, 1, 1, 6, 0)

    service.update_burden_row(
        {"coke_p01_rings": 4.0},
        timestamp=timestamp,
        modifier="qa",
        ip="127.0.0.1",
    )

    assert _FakeBurdenHistoryRepository.last_row["row_values"] == {
        "coke_p01_rings": 4.0
    }
    assert service.get_all_current_burden_values(timestamp) == {"coke_p01_rings": 4.0}
