from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.material_balance_service import MaterialBalanceService


class _FakeAuthService:
    users = {
        "operator": {"id": "u1", "role": "user", "permissions": ["material_balance:read", "material_balance:run", "material_balance:export"]},
        "supervisor": {"id": "u2", "role": "supervisor", "permissions": ["material_balance:read", "material_balance:run", "material_balance:export", "material_balance:config:write"]},
        "none": {"id": "u3", "role": "user", "permissions": []},
    }

    def current_user_from_token(self, token: str):
        user = self.users.get(token)
        if user is None:
            raise ApiError("INVALID_TOKEN", "Invalid token.", status_code=401)
        return user


class _FakeMaterialBalanceService:
    def config(self):
        return {
            "catalog_version": "material-balance-catalog-v1",
            "effective_config_version": "mbcfg-1",
            "display_timezone": "Asia/Kolkata",
            "dataset": {"dataset_id": "static_ml_dataset", "version": "dataset-1", "status": "ready", "available_date_range": {"minimum": "2026-07-20", "maximum": "2026-07-22"}},
            "defaults": {"rm_lag_hours": 0, "blast_lag_hours": 0, "dust_catcher_t": 0.0, "algorithm_version": "legacy_v1"},
            "limits": {"rm_lag_hours_min": 0, "rm_lag_hours_max": 240, "blast_lag_hours_min": 0, "blast_lag_hours_max": 48, "dust_catcher_t_min": 0.0, "dust_catcher_t_max": 500.0},
            "closure_thresholds": {"good": {"minimum": 95, "maximum": 105}, "warning": {"minimum": 85, "maximum": 115}},
            "elements": [], "materials": [], "input_streams": [], "output_streams": [],
            "algorithm_versions": [{"id": "legacy_v1", "label": "Validated legacy-compatible balance", "tracked_element_ids": ["fe", "c"]}],
            "capabilities": {"runtime_configuration_writable": True, "ash_analysis_editable": True, "dpr_mapping_editable": True, "export_available": True, "async_jobs_required": False},
            "available_sources": ["static_dataset", "input_data"],
            "warnings": [],
        }


def _headers(token: str = "operator") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_material_balance_routes_use_permissions_and_stable_openapi(app_factory):
    app = app_factory()
    app.state.auth_service = _FakeAuthService()
    app.state.material_balance_service = _FakeMaterialBalanceService()
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/api/v1/material-balance/config").status_code == 401
        assert client.get("/api/v1/material-balance/config", headers=_headers("none")).status_code == 403
        allowed = client.get("/api/v1/material-balance/config", headers=_headers())
        schema = client.get("/openapi.json").json()

    assert allowed.status_code == 200
    assert schema["paths"]["/api/v1/material-balance/config"]["get"]["operationId"] == "get_material_balance_config"
    assert schema["paths"]["/api/v1/material-balance/run"]["post"]["operationId"] == "run_material_balance"
    assert schema["paths"]["/api/v1/material-balance/validate"]["post"]["deprecated"] is True
    assert schema["paths"]["/api/v1/material-balance/jobs"]["post"]["deprecated"] is True
    assert "MaterialBalanceConfigData" in schema["components"]["schemas"]


def test_material_balance_static_run_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(
        "apps.backend_api.app.services.material_balance_service.get_static_dataset_metadata",
        lambda: {"dataset_id": "static_ml_dataset", "version": "dataset-1", "status": "ready", "available_date_range": {"minimum": datetime(2026, 7, 20).date(), "maximum": datetime(2026, 7, 22).date()}},
    )
    service = MaterialBalanceService(settings=BackendSettings(compute_require_auth=False), clock=lambda: datetime(2026, 7, 23, 4, 30, tzinfo=timezone.utc))

    assert service.validate({"source": "static_dataset"})["errors"][0]["code"] == "MATERIAL_BALANCE_DATE_REQUIRED"
    assert service.validate({"source": "static_dataset", "day": "2026-07-23"})["errors"][0]["code"] == "MATERIAL_BALANCE_PARTIAL_DAY_NOT_ALLOWED"
    assert service.validate({"source": "static_dataset", "day": "2026-07-19"})["errors"][0]["code"] == "MATERIAL_BALANCE_DATE_OUT_OF_RANGE"
    assert service.validate({"source": "static_dataset", "day": "2026-07-22", "expected_dataset_version": "old"})["errors"][0]["code"] == "MATERIAL_BALANCE_DATASET_VERSION_CONFLICT"
    assert service.validate({"source": "static_dataset", "day": "2026-07-22", "options": {"blast_lag_hours": 49}})["errors"][0]["code"] == "MATERIAL_BALANCE_INVALID_LAG"
    assert service.validate({"source": "static_dataset", "day": "2026-07-22", "options": {"dust_catcher_t": 501}})["errors"][0]["code"] == "MATERIAL_BALANCE_INVALID_DUST_CATCHER"