"""Tests for Welcome dashboard and plant-admin API contracts."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.user_repository import UserRecord
from apps.backend_api.app.services.admin_service import AdminService
from apps.backend_api.app.services.dashboard_service import DashboardService
from apps.backend_api.app.services.plant_admin_service import PlantAdminService
from furnace_data.material_mapping import MaterialMapEntry, MaterialNameMapper


class FakeAuthService:
    def __init__(self) -> None:
        self.users = {
            "admin-token": {
                "id": "00000000-0000-0000-0000-000000000001",
                "username": "admin",
                "role": "admin",
                "permissions": [
                    "hopper:write",
                    "hopper:history:delete",
                    "burden:write",
                    "burden:history:delete",
                    "users:write",
                ],
                "is_active": True,
            },
            "supervisor-token": {
                "id": "00000000-0000-0000-0000-000000000002",
                "username": "supervisor",
                "role": "supervisor",
                "permissions": ["hopper:write"],
                "is_active": True,
            },
            "user-token": {
                "id": "00000000-0000-0000-0000-000000000003",
                "username": "operator",
                "role": "user",
                "permissions": [],
                "is_active": True,
            },
        }

    def current_user_from_token(self, token: str):
        return self.users[token]


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_dashboard_service_maps_fields_and_means():
    df = pd.DataFrame(
        {
            "production_per_hour": [100.0, 120.0],
            "fuel_rate": [480.0, 500.0],
            "body_etaco": [40.0, 42.0],
            "hot_blast_vol_nm3h": [150000.0, 160000.0],
        },
        index=pd.to_datetime(
            ["2026-07-23T04:00:00Z", "2026-07-23T04:15:00Z"], utc=True
        ),
    )
    service = DashboardService(fetcher=lambda **_: df, cache_ttl_seconds=60)

    payload, warnings = service.get_kpis(window="1h", bucket="15m")

    assert warnings == []
    assert payload["sample_count"] == 2
    assert payload["metrics"]["production_rate"] == {"value": 110.0, "unit": "t/h"}
    assert payload["metrics"]["blast_volume"] == {"value": 155000.0, "unit": "Nm3/h"}


def test_dashboard_service_empty_data_warns():
    service = DashboardService(fetcher=lambda **_: pd.DataFrame(), cache_ttl_seconds=60)

    payload, warnings = service.get_kpis(window="1h", bucket="15m")

    assert payload["sample_count"] == 0
    assert payload["metrics"]["fuel_rate"]["value"] is None
    assert warnings


def test_dashboard_service_failure_uses_stable_503_code():
    def failing_fetcher(**_):
        raise RuntimeError("token=secret")

    service = DashboardService(fetcher=failing_fetcher, cache_ttl_seconds=60)

    with pytest.raises(ApiError) as exc_info:
        service.get_kpis(window="1h", bucket="15m")

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "DASHBOARD_DATA_UNAVAILABLE"
    assert "secret" not in exc_info.value.message


class FakePlantRepo:
    def __init__(self) -> None:
        self.hopper_snapshot = {
            "id": 123,
            "date_time": datetime(2026, 7, 23, 4, 0, tzinfo=timezone.utc),
            "hopper_01": "mat-1",
            "source_type": "api",
            "user_modified": "00000000-0000-0000-0000-000000000001",
            "created_at": datetime(2026, 7, 23, 4, 1, tzinfo=timezone.utc),
        }
        self.burden_snapshot = {
            "id": 456,
            "date_time": datetime(2026, 7, 23, 4, 0, tzinfo=timezone.utc),
            "coke_charge_pattern": "A",
            "coke_p01_rings": 2.0,
            "source_type": "api",
            "user_modified": "00000000-0000-0000-0000-000000000001",
            "created_at": datetime(2026, 7, 23, 4, 1, tzinfo=timezone.utc),
        }
        self.inserted_hopper = None
        self.inserted_burden = None

    def list_active_hoppers(self):
        return [{"hopper_code": "hopper_01", "display_name": "Hopper 01"}]

    def list_active_materials(self):
        return [{"material_code": "mat-1", "material_name": "sinter_sp02_online"}]

    def get_hopper_snapshot_at(self, at):
        return dict(self.hopper_snapshot)

    def insert_hopper_snapshot(self, **kwargs):
        self.inserted_hopper = kwargs
        self.hopper_snapshot = {
            **self.hopper_snapshot,
            "id": 124,
            "date_time": kwargs["effective_at"],
            **kwargs["assignments"],
        }
        return 124

    def list_hopper_history(self, *, limit: int, offset: int):
        return [dict(self.hopper_snapshot)][offset : offset + limit]

    def count_hopper_history(self):
        return 1

    def delete_hopper_history(self, record_ids):
        return len(record_ids)

    def get_burden_snapshot_at(self, at):
        return dict(self.burden_snapshot)

    def burden_fields(self):
        return ["coke_charge_pattern", "coke_p01_rings"]

    def insert_burden_snapshot(self, **kwargs):
        self.inserted_burden = kwargs
        self.burden_snapshot = {
            **self.burden_snapshot,
            "id": 457,
            "date_time": kwargs["effective_at"],
            **kwargs["values"],
        }
        return 457

    def list_burden_history(self, *, limit: int, offset: int):
        return [dict(self.burden_snapshot)][offset : offset + limit]

    def count_burden_history(self):
        return 1

    def delete_burden_history(self, record_ids):
        return len(record_ids)


def _plant_service(repo: FakePlantRepo) -> PlantAdminService:
    return PlantAdminService(
        repository=repo,
        material_mapper=MaterialNameMapper(
            [MaterialMapEntry("Sinter", "sinter_sp02_online", True)]
        ),
    )


def test_hopper_update_converts_to_utc_and_uses_backend_actor():
    repo = FakePlantRepo()
    service = _plant_service(repo)

    context = service.update_hopper_mapping(
        effective_at=datetime.fromisoformat("2026-07-23T10:00:00+05:30"),
        expected_snapshot_id=123,
        assignments={"hopper_01": "mat-1"},
        current_user=FakeAuthService().users["admin-token"],
        ip_address="127.0.0.1",
    )

    assert repo.inserted_hopper["effective_at"] == datetime(2026, 7, 23, 4, 30, tzinfo=timezone.utc)
    assert repo.inserted_hopper["user_id"] == "00000000-0000-0000-0000-000000000001"
    assert context["snapshot_id"] == 124


def test_hopper_update_rejects_invalid_hopper_material_and_stale_snapshot():
    service = _plant_service(FakePlantRepo())

    with pytest.raises(ApiError, match="hopper"):
        service.update_hopper_mapping(
            effective_at=datetime.fromisoformat("2026-07-23T10:00:00+05:30"),
            expected_snapshot_id=123,
            assignments={"hopper_99": "mat-1"},
            current_user=FakeAuthService().users["admin-token"],
            ip_address=None,
        )
    with pytest.raises(ApiError) as material_error:
        service.update_hopper_mapping(
            effective_at=datetime.fromisoformat("2026-07-23T10:00:00+05:30"),
            expected_snapshot_id=123,
            assignments={"hopper_01": "bad-mat"},
            current_user=FakeAuthService().users["admin-token"],
            ip_address=None,
        )
    with pytest.raises(ApiError) as conflict_error:
        service.update_hopper_mapping(
            effective_at=datetime.fromisoformat("2026-07-23T10:00:00+05:30"),
            expected_snapshot_id=999,
            assignments={"hopper_01": "mat-1"},
            current_user=FakeAuthService().users["admin-token"],
            ip_address=None,
        )

    assert material_error.value.code == "INVALID_MATERIAL"
    assert conflict_error.value.code == "CONFIG_VERSION_CONFLICT"


def test_burden_update_rejects_invalid_fields_and_numeric_type():
    service = _plant_service(FakePlantRepo())

    with pytest.raises(ApiError) as invalid_field:
        service.update_burden_distribution(
            effective_at=datetime.fromisoformat("2026-07-23T10:00:00+05:30"),
            expected_snapshot_id=456,
            values={"bad_field": 1.0},
            current_user=FakeAuthService().users["admin-token"],
            ip_address=None,
        )
    with pytest.raises(ApiError) as invalid_type:
        service.update_burden_distribution(
            effective_at=datetime.fromisoformat("2026-07-23T10:00:00+05:30"),
            expected_snapshot_id=456,
            values={"coke_p01_rings": "two"},
            current_user=FakeAuthService().users["admin-token"],
            ip_address=None,
        )

    assert invalid_field.value.code == "INVALID_BURDEN_FIELD"
    assert invalid_type.value.code == "INVALID_BURDEN_FIELD"


def test_dashboard_route_requires_auth_and_openapi_is_concrete(app_factory):
    app = app_factory()
    app.state.auth_service = FakeAuthService()
    app.state.dashboard_service = DashboardService(
        fetcher=lambda **_: pd.DataFrame(
            {"production_per_hour": [1.0]},
            index=pd.to_datetime(["2026-07-23T04:00:00Z"], utc=True),
        ),
        cache_ttl_seconds=60,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        unauthorized = client.get("/api/v1/dashboard/kpis")
        allowed = client.get("/api/v1/dashboard/kpis", headers=_auth_headers("user-token"))
        openapi = client.get("/openapi.json").json()

    assert unauthorized.status_code == 401
    assert allowed.status_code == 200
    path = openapi["paths"]["/api/v1/dashboard/kpis"]["get"]
    assert path["operationId"] == "getDashboardKpis"
    assert "DashboardKpisApiResponse" in str(path["responses"]["200"])


class FakePlantRouteService:
    def hopper_context(self, *, at=None):
        return {
            "at": datetime(2026, 7, 23, 4, 30, tzinfo=timezone.utc),
            "snapshot_id": 123,
            "effective_at": datetime(2026, 7, 23, 4, 0, tzinfo=timezone.utc),
            "hoppers": [{"code": "hopper_01", "display_name": "Hopper 01"}],
            "materials": [{"code": "mat-1", "canonical_name": "sinter_sp02_online", "display_name": "Sinter"}],
            "assignments": {"hopper_01": "mat-1"},
        }

    def hopper_history(self, *, limit, offset):
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    def update_hopper_mapping(self, **kwargs):
        return self.hopper_context()

    def delete_hopper_history(self, *, record_ids):
        return {"deleted_count": len(record_ids), "current_context": self.hopper_context()}

    def burden_context(self, *, at=None):
        return {
            "at": datetime(2026, 7, 23, 4, 30, tzinfo=timezone.utc),
            "snapshot_id": 456,
            "effective_at": datetime(2026, 7, 23, 4, 0, tzinfo=timezone.utc),
            "fields": [{"key": "coke_p01_rings", "label": "Coke P01 Rings", "value_type": "number", "nullable": True, "step": 0.1}],
            "values": {"coke_p01_rings": 2.0},
        }

    def burden_history(self, *, limit, offset):
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    def update_burden_distribution(self, **kwargs):
        return self.burden_context()

    def delete_burden_history(self, *, record_ids):
        return {"deleted_count": len(record_ids), "current_context": self.burden_context()}


def test_plant_admin_route_permissions(app_factory):
    app = app_factory()
    app.state.auth_service = FakeAuthService()
    app.state.plant_admin_service = FakePlantRouteService()

    with TestClient(app, raise_server_exceptions=False) as client:
        supervisor_hopper = client.get(
            "/api/v1/admin/hopper-mappings/context",
            headers=_auth_headers("supervisor-token"),
        )
        normal_user = client.get(
            "/api/v1/admin/hopper-mappings/context",
            headers=_auth_headers("user-token"),
        )
        supervisor_delete = client.request(
            "DELETE",
            "/api/v1/admin/hopper-mappings/history",
            json={"record_ids": [123]},
            headers=_auth_headers("supervisor-token"),
        )
        admin_delete = client.request(
            "DELETE",
            "/api/v1/admin/hopper-mappings/history",
            json={"record_ids": [123]},
            headers=_auth_headers("admin-token"),
        )

    assert supervisor_hopper.status_code == 200
    assert normal_user.status_code == 403
    assert supervisor_delete.status_code == 403
    assert admin_delete.status_code == 200


def test_naive_timestamp_update_is_rejected(app_factory):
    app = app_factory()
    app.state.auth_service = FakeAuthService()
    app.state.plant_admin_service = FakePlantRouteService()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.put(
            "/api/v1/admin/hopper-mappings",
            json={
                "effective_at": "2026-07-23T10:00:00",
                "expected_snapshot_id": 123,
                "assignments": {"hopper_01": "mat-1"},
            },
            headers=_auth_headers("admin-token"),
        )

    assert response.status_code == 422


class FinalAdminRepo:
    def __init__(self) -> None:
        self.updated = False
        self.admin = UserRecord(
            id="00000000-0000-0000-0000-000000000001",
            username="admin",
            password_hash="hash",
            role="admin",
            is_active=True,
        )

    def find_by_id(self, user_id: str):
        return self.admin if str(user_id) == self.admin.id else None

    def count_active_admins(self):
        return 1

    def update_user(self, user_id: str, **changes):
        self.updated = True
        return self.admin


def test_admin_service_blocks_self_deactivate_and_final_admin_demotion():
    repo = FinalAdminRepo()
    service = AdminService(repository=repo, password_service=None)

    with pytest.raises(ApiError) as self_deactivate:
        service.update_user(
            repo.admin.id,
            actor_user={"id": repo.admin.id},
            is_active=False,
        )
    with pytest.raises(ApiError) as final_demotion:
        service.update_user(
            repo.admin.id,
            actor_user={"id": "00000000-0000-0000-0000-000000000099"},
            role="user",
        )

    assert self_deactivate.value.code == "FORBIDDEN"
    assert final_demotion.value.code == "FORBIDDEN"
    assert repo.updated is False
