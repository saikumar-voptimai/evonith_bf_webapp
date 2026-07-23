from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from apps.backend_api.app.repositories.vsense_repository import VSenseRepository
from apps.backend_api.app.services.job_service import JobService
from apps.backend_api.app.services.vsense_run_service import VSenseRunService
from apps.backend_api.app.services.vsense_service import VSenseService
from furnace_data.vsense.catalog import (
    control_parameter_by_feature,
    load_vsense_catalog,
    target_for_optimization,
)


NOW = datetime(2026, 7, 23, 4, 30, tzinfo=timezone.utc)


class _FakeAuthService:
    users = {
        "operator": {
            "id": "operator-1",
            "username": "operator",
            "role": "user",
            "permissions": ["vsense:read", "vsense:run"],
        },
        "supervisor": {
            "id": "supervisor-1",
            "username": "supervisor",
            "role": "supervisor",
            "permissions": ["vsense:read", "vsense:run", "vsense:bounds:write", "vsense:llm"],
        },
        "none": {"id": "none-1", "role": "user", "permissions": []},
    }

    def current_user_from_token(self, token: str):
        from apps.backend_api.app.core.errors import ApiError

        user = self.users.get(token)
        if user is None:
            raise ApiError("INVALID_TOKEN", "Invalid token.", status_code=401)
        return user


@pytest.fixture()
def vsense_client(app_factory, tmp_path):
    app = app_factory()
    repository = VSenseRepository(tmp_path / "vsense.sqlite")
    jobs = JobService(tmp_path / "jobs.sqlite", max_workers=1)
    settings = SimpleNamespace(
        vsense_context_ttl_seconds=1800,
        vsense_max_concurrent_runs=1,
        vsense_require_approved_bounds=True,
        vsense_default_seed=42,
        vsense_llm_enabled=False,
        vsense_advanced_diagnostics=False,
        vboard_default_timezone="Asia/Kolkata",
    )
    service = VSenseService(
        settings=settings,
        repository=repository,
        history_df=_history(),
        clock=lambda: NOW,
    )
    app.state.auth_service = _FakeAuthService()
    app.state.vsense_service = service
    app.state.vsense_run_service = VSenseRunService(
        repository=repository,
        context_service=service.context_service,
        settings=settings,
        jobs=jobs,
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
    finally:
        jobs.shutdown(wait=False)


def _headers(token: str = "operator") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _history() -> pd.DataFrame:
    catalog = load_vsense_catalog()
    rows = []
    for idx, ts in enumerate(pd.date_range("2026-07-23T03:30:00Z", periods=3, freq="30min")):
        row = {}
        for feature, definition in control_parameter_by_feature().items():
            row[feature] = float(definition["default_value"]) + (idx * 0.01)
        for optimization_type_id in ("eta_co", "production_rate", "unit_cost"):
            target = target_for_optimization(optimization_type_id)
            row[target["feature_name"]] = 100.0 + idx
        row["COKE_ASH%"] = 12.0 + idx
        row["ORE_FE(T)%"] = 58.0 + idx
        rows.append(row)
    return pd.DataFrame(rows, index=pd.DatetimeIndex([r for r in pd.date_range("2026-07-23T03:30:00Z", periods=3, freq="30min")]))


def _create_context(client: TestClient) -> dict:
    response = client.post(
        "/api/v1/vsense/contexts",
        json={"optimization_type_id": "eta_co", "data_mode": "live"},
        headers={**_headers(), "Idempotency-Key": "ctx-key"},
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]


def test_catalog_requires_permission_and_exposes_public_ids(vsense_client):
    assert vsense_client.get("/api/v1/vsense/catalog").status_code == 401
    assert vsense_client.get("/api/v1/vsense/catalog", headers=_headers("none")).status_code == 403

    response = vsense_client.get("/api/v1/vsense/catalog", headers=_headers())

    assert response.status_code == 200
    data = response.json()["data"]
    assert [item["id"] for item in data["optimization_types"]] == [
        "eta_co",
        "production_rate",
        "unit_cost",
    ]
    eta = data["optimization_types"][0]
    assert eta["target"]["direction"] == "maximize"
    assert "hot_blast_pressure_bar" in eta["control_parameter_ids"]
    assert data["advisory_only"] is True
    serialized = json.dumps(data).lower()
    assert ".pkl" not in serialized
    assert "packages/" not in serialized


def test_context_is_immutable_owned_idempotent_and_frontend_neutral(vsense_client):
    first = _create_context(vsense_client)
    replay = vsense_client.post(
        "/api/v1/vsense/contexts",
        json={"optimization_type_id": "eta_co", "data_mode": "live"},
        headers={**_headers(), "Idempotency-Key": "ctx-key"},
    ).json()["data"]

    assert replay["context_id"] == first["context_id"]
    assert replay["idempotent_replay"] is True
    assert first["catalog_version"] == "vsense-catalog-v1"
    assert first["dataset"]["dataset_id"] == "static_ml_dataset"
    assert first["dataset"]["range_end"].endswith("Z")
    assert first["controls"][0]["source"] in {"historical", "configured_default", "live"}
    assert "owner_user_id" not in first
    assert all("bundle_version" in model and "status" in model for model in first["models"])


def test_control_profile_update_uses_permissions_conflict_and_idempotency(vsense_client):
    profile = vsense_client.get(
        "/api/v1/vsense/control-profiles/eta_co",
        headers=_headers(),
    ).json()["data"]

    denied = vsense_client.put(
        "/api/v1/vsense/control-profiles/eta_co",
        json={
            "profile_id": "plant-default",
            "expected_version": profile["version"],
            "parameters": profile["parameters"],
        },
        headers={**_headers(), "Idempotency-Key": "profile-key"},
    )
    assert denied.status_code == 403

    response = vsense_client.put(
        "/api/v1/vsense/control-profiles/eta_co",
        json={
            "profile_id": "plant-default",
            "expected_version": profile["version"],
            "parameters": profile["parameters"],
        },
        headers={**_headers("supervisor"), "Idempotency-Key": "profile-key"},
    )
    assert response.status_code == 200, response.text
    updated = response.json()["data"]
    assert updated["version"] == profile["version"] + 1

    replay = vsense_client.put(
        "/api/v1/vsense/control-profiles/eta_co",
        json={
            "profile_id": "plant-default",
            "expected_version": profile["version"],
            "parameters": profile["parameters"],
        },
        headers={**_headers("supervisor"), "Idempotency-Key": "profile-key"},
    )
    assert replay.json()["data"]["idempotent_replay"] is True

    conflict = vsense_client.put(
        "/api/v1/vsense/control-profiles/eta_co",
        json={
            "profile_id": "plant-default",
            "expected_version": profile["version"],
            "parameters": profile["parameters"],
        },
        headers={**_headers("supervisor"), "Idempotency-Key": "profile-key-2"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "VSENSE_CONTROL_PROFILE_VERSION_CONFLICT"


def test_run_lifecycle_persists_result_and_events(vsense_client):
    context = _create_context(vsense_client)
    control_plan = [
        {
            "parameter_id": item["parameter_id"],
            "mode": "fixed",
            "lower_bound": item["approved_min"],
            "upper_bound": item["approved_max"],
            "fixed_value": item["value"],
        }
        for item in context["controls"]
    ]

    accepted = vsense_client.post(
        "/api/v1/vsense/runs",
        json={
            "context_id": context["context_id"],
            "optimization_type_id": "eta_co",
            "control_plan": control_plan,
            "input_overrides": [],
            "options": {"lambda_reg": 0.05, "max_iterations": 1},
        },
        headers={**_headers(), "Idempotency-Key": "run-key"},
    )
    assert accepted.status_code == 202, accepted.text
    run_id = accepted.json()["data"]["run_id"]

    status = None
    for _ in range(30):
        status = vsense_client.get(f"/api/v1/vsense/runs/{run_id}", headers=_headers())
        if status.json()["data"]["status"] == "completed":
            break
        time.sleep(0.05)
    assert status is not None
    data = status.json()["data"]
    assert data["status"] == "completed"
    assert data["result"]["advisory_only"] is True
    assert data["result"]["requires_operator_review"] is True
    assert data["result"]["target"]["parameter_id"] == "eta_co"
    assert data["result"]["diagnostics"]["optimizer"]["all_controls_fixed"] is True

    events = vsense_client.get(f"/api/v1/vsense/runs/{run_id}/events", headers=_headers()).json()["data"]["events"]
    assert [event["sequence"] for event in events] == sorted(event["sequence"] for event in events)
    assert events[-1]["stage"] == "completed"


def test_openapi_has_typed_vsense_operations_and_deprecated_recommendations(vsense_client):
    schema = vsense_client.get("/openapi.json").json()

    assert schema["paths"]["/api/v1/vsense/catalog"]["get"]["operationId"] == "get_vsense_catalog"
    assert schema["paths"]["/api/v1/vsense/contexts"]["post"]["operationId"] == "create_vsense_context"
    assert schema["paths"]["/api/v1/vsense/runs"]["post"]["responses"]["202"]["content"]["application/json"]["schema"]["$ref"].endswith("VSenseRunAcceptedResponse")
    assert schema["paths"]["/api/v1/recommendations/run"]["post"]["deprecated"] is True
