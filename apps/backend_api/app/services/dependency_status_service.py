"""Fast, cached dependency status checks for edge deployments."""

from __future__ import annotations

import os
import time
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.services.optional_dependency_service import get_optional_dependency_status
from apps.backend_api.app.services.runtime_status_service import RuntimeStatusService


class DependencyStatusService:
    """Report required and optional dependency status without network probes."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()
        self._cached_at = 0.0
        self._cached: dict[str, Any] | None = None

    def status(self, *, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if (
            not force
            and self._cached is not None
            and now - self._cached_at < self.settings.dependency_check_cache_seconds
        ):
            return self._cached
        dependencies = [
            self._runtime(),
            self._auth_secret(),
            self._relational_database(),
            self._influx("online", "INFLUX_ONLINE_TOKEN"),
            self._influx("offline", "INFLUX_OFFLINE_TOKEN"),
            self._qdrant(),
            self._llm("copilot", self.settings.copilot_llm_enabled, self.settings.copilot_provider, self.settings.copilot_api_key_env),
            self._llm(
                "furnacemind",
                self.settings.furnacemind_llm_enabled,
                self.settings.furnacemind_provider,
                self.settings.furnacemind_api_key_env,
            ),
        ]
        required_failed = any(item["required"] and item["status"] not in {"ok", "configured"} for item in dependencies)
        optional_degraded = any(
            not item["required"] and item["status"] not in {"ok", "configured", "unconfigured", "disabled"}
            for item in dependencies
        )
        payload = {
            "status": "degraded" if required_failed or optional_degraded else "ok",
            "timeout_seconds": self.settings.dependency_check_timeout_seconds,
            "cache_seconds": self.settings.dependency_check_cache_seconds,
            "runtime_profile": self.settings.runtime_profile,
            "edge_mode": self.settings.edge_mode,
            "profile": self.settings.safe_runtime_profile_summary(),
            "dependency_groups": [
                "backend-base",
                "backend-data",
                "backend-ml",
                "backend-ai",
                "backend-vector",
                "backend-documents",
                "frontend",
                "dev",
                "edge",
            ],
            "optional_dependencies": get_optional_dependency_status(),
            "backend_base_import": {
                "status": "ok",
                "message": "Backend app import is covered by scripts/check_backend_minimal_startup.py.",
            },
            "frontend_api_imports": {
                "status": "not_checked",
                "message": "Frontend API import safety is covered by scripts/check_frontend_api_imports.py.",
            },
            "dependencies": dependencies,
        }
        self._cached = payload
        self._cached_at = now
        return payload

    def _runtime(self) -> dict[str, Any]:
        status = RuntimeStatusService(self.settings).status(create_missing=True, include_sizes=False)
        return {
            "name": "runtime",
            "required": True,
            "status": "ok" if status["status"] in {"ok", "warning"} else "degraded",
            "message": "Runtime directory is available.",
        }

    def _auth_secret(self) -> dict[str, Any]:
        configured = bool(self.settings.auth_secret_key)
        required = self.settings.auth_enabled and self.settings.auth_require_secret_in_production
        ok = configured or self.settings.backend_env.lower() not in {"prod", "production"}
        return {
            "name": "auth_secret",
            "required": required,
            "status": "configured" if configured else "unconfigured" if ok else "degraded",
            "message": "Auth secret configured." if configured else "Auth secret not configured in this environment.",
        }

    @staticmethod
    def _relational_database() -> dict[str, Any]:
        configured = bool(os.getenv("DATABASE_URL", "").strip())
        return {
            "name": "relational_database",
            "required": False,
            "status": "configured" if configured else "unconfigured",
            "message": "Database URL configured." if configured else "Database URL not configured.",
        }

    @staticmethod
    def _influx(name: str, token_env: str) -> dict[str, Any]:
        configured = bool(os.getenv(token_env, "").strip())
        return {
            "name": f"influx_{name}",
            "required": False,
            "status": "configured" if configured else "unconfigured",
            "message": f"Influx {name} token configured." if configured else f"Influx {name} token not configured.",
        }

    def _qdrant(self) -> dict[str, Any]:
        if not self.settings.enable_optional_vector:
            return {
                "name": "qdrant",
                "required": False,
                "status": "disabled",
                "message": "Optional vector features are disabled.",
            }
        if not self.settings.furnacemind_memory_enabled:
            return {"name": "qdrant", "required": False, "status": "disabled", "message": "FurnaceMind memory is disabled."}
        configured = bool(self.settings.furnacemind_qdrant_url)
        return {
            "name": "qdrant",
            "required": False,
            "status": "configured" if configured else "unconfigured",
            "message": "Qdrant URL configured." if configured else "Qdrant URL not configured.",
        }

    @staticmethod
    def _llm(feature: str, enabled: bool, provider: str, key_env: str) -> dict[str, Any]:
        if not enabled:
            return {"name": f"{feature}_llm", "required": False, "status": "disabled", "message": f"{feature} LLM is disabled."}
        configured = bool(provider and os.getenv(key_env, "").strip())
        return {
            "name": f"{feature}_llm",
            "required": False,
            "status": "configured" if configured else "unconfigured",
            "message": f"{feature} LLM provider configured." if configured else f"{feature} LLM provider is not fully configured.",
        }
