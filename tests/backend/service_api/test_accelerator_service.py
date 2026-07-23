"""Tests for safe CPU/CUDA accelerator selection."""

from __future__ import annotations

import pytest

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.services.accelerator_service import AcceleratorService


def _service(**settings) -> AcceleratorService:
    return AcceleratorService(BackendSettings(auth_secret_key="test-secret", **settings))


def test_auto_selects_cuda_only_when_device_and_runtime_are_available(monkeypatch):
    service = _service(ml_device="auto", xgboost_device="auto")
    monkeypatch.setattr(service, "cuda_device_accessible", lambda: True)
    monkeypatch.setattr(service, "xgboost_cuda_built", lambda: True)

    assert service.resolve_xgboost_device() == "cuda:0"


def test_auto_falls_back_to_cpu(monkeypatch):
    service = _service(ml_device="auto", xgboost_device="auto")
    monkeypatch.setattr(service, "cuda_device_accessible", lambda: True)
    monkeypatch.setattr(service, "xgboost_cuda_built", lambda: False)

    assert service.resolve_xgboost_device() == "cpu"


def test_required_cuda_fails_instead_of_silently_using_cpu(monkeypatch):
    service = _service(xgboost_device="cuda", cuda_required=True)
    monkeypatch.setattr(service, "cuda_device_accessible", lambda: False)
    monkeypatch.setattr(service, "xgboost_cuda_built", lambda: True)

    with pytest.raises(RuntimeError, match="CUDA was required"):
        service.resolve_xgboost_device()

    status = service.status()
    assert status["status"] == "degraded"
    assert status["selected_xgboost_device"] == "unavailable"


@pytest.mark.parametrize("value", ["gpu", "cuda:-1", "cuda:abc"])
def test_invalid_accelerator_device_is_rejected(value):
    with pytest.raises(ValueError, match="accelerator device"):
        BackendSettings(auth_secret_key="test-secret", ml_device=value)


def test_acceleration_settings_are_in_safe_profile_summary():
    summary = BackendSettings(
        auth_secret_key="test-secret",
        ml_device="cuda",
        xgboost_device="cuda:0",
        cuda_required=True,
    ).safe_runtime_profile_summary()

    assert summary["acceleration"] == {
        "ml_device": "cuda",
        "xgboost_device": "cuda:0",
        "cuda_required": True,
    }


def test_openapi_includes_admin_accelerator_status(app_factory):
    schema = app_factory().openapi()

    assert "/api/v1/status/accelerator" in schema["paths"]
