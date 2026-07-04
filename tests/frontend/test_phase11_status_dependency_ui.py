"""Frontend tests for Phase 11 status dependency adapters."""

from __future__ import annotations

from pathlib import Path

from config.frontend_settings import load_frontend_settings
from services import status_api


class FakeClient:
    base_url = "http://localhost:8080/api/v1"

    def __init__(self):
        self.calls = []

    def get(self, path, params=None, headers=None):
        self.calls.append((path, params, headers))
        data = {
            "runtime_profile": "edge",
            "edge_mode": True,
            "optional_dependencies": [{"module": "openai", "status": "unavailable"}],
            "profile": {"runtime_profile": "edge"},
        }
        return {"request_id": "id", "data": data, "meta": {}}


def test_status_api_parses_runtime_profile_and_optional_dependencies():
    client = FakeClient()

    config = status_api.get_status_config("token", client=client)
    dependencies = status_api.get_dependency_status("token", client=client)

    assert config["runtime_profile"] == "edge"
    assert dependencies["optional_dependencies"][0]["module"] == "openai"
    assert client.calls[0] == ("/status/config", None, {"Authorization": "Bearer token"})
    assert client.calls[1][0] == "/status/dependencies"


def test_advanced_dependency_details_are_hidden_by_default(monkeypatch):
    monkeypatch.delenv("SHOW_ADVANCED_BACKEND_STATUS", raising=False)
    monkeypatch.delenv("USE_BACKEND_API_OPS", raising=False)
    settings = load_frontend_settings()

    assert settings.show_advanced_backend_status is False
    assert settings.page_api_flags["ops"] is False


def test_frontend_api_adapters_do_not_load_backend_or_heavy_modules():
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = {
        "from app",
        "import app",
        "furnace_data",
        "qdrant_client",
        "openai",
        "anthropic",
        "sklearn",
        "joblib",
        "torch",
        "sentence_transformers",
    }

    for path in (repo_root / "src" / "services").glob("*_api.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(pattern in text for pattern in forbidden)
