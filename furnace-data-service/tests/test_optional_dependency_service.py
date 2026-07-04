"""Tests for Phase 11 optional dependency helpers."""

from __future__ import annotations

import importlib.machinery

import pytest

from app.core.errors import ApiError
from app.services import optional_dependency_service as service


def test_optional_dependency_service_available_missing_and_errors():
    service.clear_optional_dependency_cache()

    assert service.is_module_available("json") is True
    assert service.is_module_available("evonith_missing_optional_module") is False
    assert service.require_optional_module("json", "backend-base").loads("{}") == {}

    with pytest.raises(ApiError) as exc_info:
        service.require_optional_module("evonith_missing_optional_module", "backend-ai")

    error = exc_info.value
    assert error.code == "DEPENDENCY_OPTIONAL_NOT_INSTALLED"
    assert error.details["feature_group"] == "backend-ai"
    assert "backend-ai" in error.details["recommendation"]
    assert "secret" not in str(error.details).lower()


def test_optional_dependency_availability_checks_are_cached(monkeypatch):
    service.clear_optional_dependency_cache()
    calls: list[str] = []

    def fake_find_spec(module_name: str):
        calls.append(module_name)
        return importlib.machinery.ModuleSpec(module_name, loader=None)

    monkeypatch.setattr(service.importlib.util, "find_spec", fake_find_spec)

    assert service.is_module_available("cached_module") is True
    assert service.is_module_available("cached_module") is True
    assert calls == ["cached_module"]


def test_optional_dependency_status_is_path_and_secret_safe():
    service.clear_optional_dependency_cache()

    status = service.get_optional_dependency_status()

    assert any(item["feature_group"] == "backend-ai" for item in status)
    assert "site-packages" not in str(status)
    assert "api_key" not in str(status).lower()
