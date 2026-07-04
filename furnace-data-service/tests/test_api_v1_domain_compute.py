"""Tests for Phase 7 domain compute APIs."""

from __future__ import annotations

from pathlib import Path

import joblib
from fastapi.testclient import TestClient

from app.core.config import BackendSettings
from app.services.model_registry_service import ModelRegistryService


class DoublerModel:
    def predict(self, frame):
        return [float(row[0]) * 2.0 for row in frame.values]


def _compute_client(app_factory, monkeypatch, tmp_path, *, require_auth: bool = False) -> TestClient:
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("EVONITH_COMPUTE_REQUIRE_AUTH", "true" if require_auth else "false")
    app = app_factory()
    return TestClient(app, raise_server_exceptions=False)


def _blend_payload() -> dict:
    return {
        "materials": [
            {
                "material_id": "ore_a",
                "available": True,
                "min_percent": 0,
                "max_percent": 80,
                "properties": {"fe_t_pct": 58},
                "cost": 1000,
            },
            {
                "material_id": "ore_b",
                "available": True,
                "min_percent": 0,
                "max_percent": 80,
                "properties": {"fe_t_pct": 62},
                "cost": 1200,
            },
        ],
        "constraints": {"target_total_qty_mt": 100},
        "options": {"max_candidates": 3},
    }


def test_material_balance_config_validate_run_job_and_artifact(app_factory, monkeypatch, tmp_path):
    with _compute_client(app_factory, monkeypatch, tmp_path) as client:
        config = client.get("/api/v1/material-balance/config")
        validate = client.post(
            "/api/v1/material-balance/validate",
            json={"source": "input_data", "input_data": {"inputs": {}, "outputs": {}}},
        )
        run = client.post(
            "/api/v1/material-balance/run",
            json={
                "source": "input_data",
                "input_data": {"inputs": {"Fe": 10}, "outputs": {"Fe": 9}},
                "export": True,
            },
        )
        artifact_id = run.json()["data"]["artifacts"][0]["artifact_id"]
        download = client.get(f"/api/v1/material-balance/artifacts/{artifact_id}/download")
        job = client.post(
            "/api/v1/material-balance/jobs",
            json={
                "source": "input_data",
                "input_data": {"inputs": {"C": 5}, "outputs": {"C": 4}},
            },
        )
        job_status = client.get(f"/api/v1/material-balance/jobs/{job.json()['data']['job_id']}")

    assert config.status_code == 200
    assert "input_data" in config.json()["data"]["available_sources"]
    assert validate.status_code == 200
    assert validate.json()["data"]["valid"] is True
    assert run.status_code == 200
    assert run.json()["data"]["kpis"]["overall_closure_pct"] == 90.0
    assert download.status_code == 200
    assert "text/csv" in download.headers["content-type"]
    assert job.json()["data"]["status"] == "completed"
    assert job_status.json()["data"]["workflow"] == "material_balance"


def test_recommendations_config_run_and_job(app_factory, monkeypatch, tmp_path):
    with _compute_client(app_factory, monkeypatch, tmp_path) as client:
        config = client.get("/api/v1/recommendations/config")
        run = client.post(
            "/api/v1/recommendations/run",
            json={
                "input_data": {"signals": {"PCI_KG/THM": 8, "COKE_RATE": -5}},
                "max_items": 1,
            },
        )
        job = client.post(
            "/api/v1/recommendations/jobs",
            json={"input_data": {"signals": {"PCI_KG/THM": 8}}},
        )
        job_status = client.get(f"/api/v1/recommendations/jobs/{job.json()['data']['job_id']}")

    assert config.status_code == 200
    assert config.json()["data"]["llm_available"] is False
    assert config.json()["data"]["max_items"] >= 1
    assert run.status_code == 200
    assert len(run.json()["data"]["items"]) == 1
    assert run.json()["data"]["summary"]["llm_used"] is False
    assert job.json()["data"]["status"] == "completed"
    assert job_status.json()["data"]["workflow"] == "recommendations"


def test_blend_optimizer_context_optimize_and_artifact(app_factory, monkeypatch, tmp_path):
    with _compute_client(app_factory, monkeypatch, tmp_path) as client:
        context = client.get("/api/v1/blend-optimizer/context")
        optimize = client.post(
            "/api/v1/blend-optimizer/optimize",
            json={**_blend_payload(), "export": True},
        )
        artifact_id = optimize.json()["data"]["artifacts"][0]["artifact_id"]
        download = client.get(f"/api/v1/blend-optimizer/artifacts/{artifact_id}/download")
        job = client.post("/api/v1/blend-optimizer/jobs", json=_blend_payload())
        job_status = client.get(f"/api/v1/blend-optimizer/jobs/{job.json()['data']['job_id']}")

    assert context.status_code == 200
    assert context.json()["data"]["materials"]
    assert optimize.status_code == 200
    assert optimize.json()["data"]["best_candidate"]["materials"]
    assert len(optimize.json()["data"]["candidates"]) == 3
    assert download.status_code == 200
    assert job.json()["data"]["status"] == "completed"
    assert job_status.json()["data"]["workflow"] == "blend_optimizer"


def test_compute_routes_require_auth_when_configured(app_factory, monkeypatch, tmp_path):
    with _compute_client(app_factory, monkeypatch, tmp_path, require_auth=True) as client:
        response = client.get("/api/v1/material-balance/config")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTH_REQUIRED"


def test_model_registry_lists_without_loading_and_predicts(tmp_path):
    model_path = tmp_path / "double.joblib"
    joblib.dump(DoublerModel(), model_path)
    settings = BackendSettings(
        backend_env="test",
        model_dir=str(tmp_path),
        model_cache_max_items=1,
    )
    registry = ModelRegistryService(settings=settings)

    listed = registry.list_models()
    status_before = registry.get_model_status("double")
    prediction = registry.predict("double", {"x": 3})
    status_after = registry.get_model_status("double")

    assert listed[0]["name"] == "double"
    assert status_before["loaded"] is False
    assert prediction["predictions"] == [6.0]
    assert status_after["loaded"] is True
    assert "path" not in str(status_after).lower()


def test_blend_predict_invalid_model_is_structured_error(app_factory, monkeypatch, tmp_path):
    with _compute_client(app_factory, monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/v1/blend-optimizer/predict",
            json={"model_name": "../secret", "features": {"x": 1}},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MODEL_PATH_INVALID"


def test_openapi_includes_phase7_endpoints(client):
    schema = client.get("/openapi.json").json()

    assert "/api/v1/material-balance/config" in schema["paths"]
    assert "/api/v1/material-balance/run" in schema["paths"]
    assert "/api/v1/recommendations/run" in schema["paths"]
    assert "/api/v1/blend-optimizer/context" in schema["paths"]
    assert "/api/v1/blend-optimizer/optimize" in schema["paths"]
    assert "/api/v1/blend-optimizer/predict" in schema["paths"]
