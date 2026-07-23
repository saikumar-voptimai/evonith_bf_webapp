"""Tests for API v1 OpenAPI schema and legacy route flags."""

from fastapi.testclient import TestClient


def test_openapi_includes_api_v1_health(client):
    schema = client.get("/openapi.json").json()

    assert "/api/v1/health" in schema["paths"]
    assert schema["info"]["title"] == "Evonith BF Backend API"
    assert schema["info"]["version"] == "0.1.0"


def test_data_explorer_openapi_documents_the_common_api_error_envelope(client):
    schema = client.get("/openapi.json").json()

    error_ref = {"$ref": "#/components/schemas/ApiErrorResponse"}
    assert schema["components"]["schemas"]["ApiErrorResponse"] == {
        "properties": {
            "request_id": {"type": "string", "title": "Request Id"},
            "error": {"$ref": "#/components/schemas/ApiErrorBody"},
        },
        "type": "object",
        "required": ["request_id", "error"],
        "title": "ApiErrorResponse",
    }

    target_operations = {
        "/api/v1/data/catalog": "get",
        "/api/v1/data/sources": "get",
        "/api/v1/data/offline/report-types": "get",
        "/api/v1/data/offline/tables": "get",
        "/api/v1/data/preview": "post",
        "/api/v1/data/export": "post",
        "/api/v1/data/artifacts/{artifact_id}/download": "get",
        "/api/v1/data/hot-metal-slag/preview": "post",
        "/api/v1/data/hot-metal-slag/export": "post",
        "/api/v1/datasets/static_ml_dataset": "get",
        "/api/v1/datasets/static_ml_dataset/analyses/scatter": "post",
        "/api/v1/datasets/static_ml_dataset/timeseries": "post",
        "/api/v1/datasets/static_ml_dataset/jobs": "post",
        "/api/v1/datasets/static_ml_dataset/jobs/{job_id}": "get",
        "/api/v1/datasets/static_ml_dataset/jobs/{job_id}/events": "get",
        "/api/v1/datasets/static_ml_dataset/jobs/{job_id}/cancel": "post",
        "/api/v1/datasets/static_ml_dataset/jobs/{job_id}/download": "get",
        "/api/v1/datasets/static_ml_dataset/download": "get",
        "/api/v1/datasets/static_ml_dataset/validation": "get",
    }
    expected_statuses = {"400", "401", "403", "404", "409", "410", "413", "422", "500", "503"}

    for path, method in target_operations.items():
        responses = schema["paths"][path][method]["responses"]
        assert expected_statuses <= responses.keys()
        for status in expected_statuses:
            content = responses[status]["content"]["application/json"]
            assert content["schema"] == error_ref


def test_legacy_health_route_can_be_enabled(app_factory):
    app = app_factory(legacy_routes=True)
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_raw_legacy_data_routes_remain_retired_when_compatibility_routes_are_enabled(app_factory):
    """Raw physical-table endpoints must never bypass the v1 auth boundary."""

    app = app_factory(legacy_routes=True)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/data/online/measurements")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_legacy_routes_can_be_disabled(app_factory):
    app = app_factory(legacy_routes=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        legacy_response = client.get("/health")
        versioned_response = client.get("/api/v1/health")

    assert legacy_response.status_code == 404
    assert legacy_response.json()["error"]["code"] == "NOT_FOUND"
    assert versioned_response.status_code == 200
