"""V-Sense gateways for API-first mode and temporary direct rollback mode."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.services.api_client import ApiClient
from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError
from apps.frontend_streamlit.services.vsense_api import JsonDict, VSenseApi


@runtime_checkable
class VSenseGateway(Protocol):
    def get_catalog(self) -> JsonDict: ...

    def create_context(self, request: JsonDict, *, idempotency_key: str) -> JsonDict: ...

    def get_control_profile(self, optimization_type_id: str) -> JsonDict: ...

    def update_control_profile(
        self,
        optimization_type_id: str,
        request: JsonDict,
        *,
        idempotency_key: str,
    ) -> JsonDict: ...

    def create_run(self, request: JsonDict, *, idempotency_key: str) -> JsonDict: ...

    def get_run(self, run_id: str) -> JsonDict: ...

    def get_run_events(self, run_id: str, *, after: int = 0) -> JsonDict: ...

    def cancel_run(self, run_id: str) -> JsonDict: ...


class ApiVSenseGateway:
    """V-Sense gateway backed exclusively by API v1."""

    def __init__(self, access_token: str, client: ApiClient | None = None) -> None:
        self.api = VSenseApi(access_token, client)

    def get_catalog(self) -> JsonDict:
        return self.api.get_catalog()

    def create_context(self, request: JsonDict, *, idempotency_key: str) -> JsonDict:
        return self.api.create_context(request, idempotency_key=idempotency_key)

    def get_control_profile(self, optimization_type_id: str) -> JsonDict:
        return self.api.get_control_profile(optimization_type_id)

    def update_control_profile(
        self,
        optimization_type_id: str,
        request: JsonDict,
        *,
        idempotency_key: str,
    ) -> JsonDict:
        return self.api.update_control_profile(
            optimization_type_id,
            request,
            idempotency_key=idempotency_key,
        )

    def create_run(self, request: JsonDict, *, idempotency_key: str) -> JsonDict:
        return self.api.create_run(request, idempotency_key=idempotency_key)

    def get_run(self, run_id: str) -> JsonDict:
        return self.api.get_run(run_id)

    def get_run_events(self, run_id: str, *, after: int = 0) -> JsonDict:
        return self.api.get_run_events(run_id, after=after)

    def cancel_run(self, run_id: str) -> JsonDict:
        return self.api.cancel_run(run_id)


class DirectVSenseGateway:
    """Deprecated direct V-Sense gateway kept only as a rollback path."""

    def __init__(self) -> None:
        self._profile_cache: dict[str, dict[str, Any]] = {}
        self._context_cache: dict[str, dict[str, Any]] = {}
        self._run_cache: dict[str, dict[str, Any]] = {}

    def get_catalog(self) -> JsonDict:
        from furnace_data.vsense import load_vsense_catalog

        data = load_vsense_catalog()
        data["request_id"] = None
        data["warnings"] = [
            "Direct V-Sense mode is deprecated; enable USE_BACKEND_API_VSENSE for API mode."
        ]
        return data

    def create_context(self, request: JsonDict, *, idempotency_key: str) -> JsonDict:
        from furnace_data.vsense.context import build_context_snapshot

        context = build_context_snapshot(
            optimization_type_id=str(request["optimization_type_id"]),
            data_mode=str(request.get("data_mode") or "live"),
            now=datetime.now(timezone.utc),
            control_profile=self.get_control_profile(str(request["optimization_type_id"])),
        )
        context["request_id"] = None
        context.setdefault("warnings", []).append(
            "Direct V-Sense mode is deprecated; enable USE_BACKEND_API_VSENSE for API mode."
        )
        self._context_cache[context["context_id"]] = context
        return context

    def get_control_profile(self, optimization_type_id: str) -> JsonDict:
        from furnace_data.vsense.bounds import default_control_profile
        from furnace_data.vsense.catalog import CATALOG_VERSION

        if optimization_type_id not in self._profile_cache:
            self._profile_cache[optimization_type_id] = {
                "profile_id": "plant-default",
                "optimization_type_id": optimization_type_id,
                "version": 1,
                "catalog_version": CATALOG_VERSION,
                "parameters": default_control_profile(optimization_type_id),
                "updated_by_user_id": None,
                "updated_by_username": None,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "request_id": None,
            }
        return dict(self._profile_cache[optimization_type_id])

    def update_control_profile(
        self,
        optimization_type_id: str,
        request: JsonDict,
        *,
        idempotency_key: str,
    ) -> JsonDict:
        from furnace_data.vsense.bounds import validate_control_profile
        from furnace_data.vsense.catalog import CATALOG_VERSION

        current = self.get_control_profile(optimization_type_id)
        if int(request.get("expected_version", -1)) != int(current["version"]):
            raise BackendApiHTTPError(
                "Control profile version conflict.",
                status_code=409,
                error_code="VSENSE_CONTROL_PROFILE_VERSION_CONFLICT",
            )
        profile = {
            **current,
            "version": int(current["version"]) + 1,
            "catalog_version": CATALOG_VERSION,
            "parameters": validate_control_profile(
                optimization_type_id,
                list(request.get("parameters") or []),
            ),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        self._profile_cache[optimization_type_id] = profile
        return dict(profile)

    def create_run(self, request: JsonDict, *, idempotency_key: str) -> JsonDict:
        from uuid import uuid4

        from furnace_data.vsense.optimizer import run_legacy_optimization

        context = self._context_cache.get(str(request["context_id"]))
        if context is None:
            raise BackendApiHTTPError(
                "V-Sense context not found.",
                status_code=404,
                error_code="VSENSE_CONTEXT_NOT_FOUND",
            )
        run_id = f"direct_{uuid4().hex}"
        result = run_legacy_optimization(
            context=context,
            control_plan=list(request.get("control_plan") or []),
            input_overrides=list(request.get("input_overrides") or []),
            lambda_reg=float((request.get("options") or {}).get("lambda_reg", 0.05)),
        )
        self._run_cache[run_id] = {
            "run_id": run_id,
            "context_id": context["context_id"],
            "optimization_type_id": context["optimization_type_id"],
            "status": "completed",
            "progress": 100.0,
            "message": "Direct V-Sense run completed",
            "error_code": None,
            "error_message": None,
            "cancellable": False,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "result": result,
        }
        return {
            "run_id": run_id,
            "status": "completed",
            "created_at": self._run_cache[run_id]["created_at"],
            "status_url": f"/vsense/runs/{run_id}",
            "events_url": f"/vsense/runs/{run_id}/events",
            "cancellable": False,
            "idempotent_replay": False,
            "request_id": None,
        }

    def get_run(self, run_id: str) -> JsonDict:
        return dict(self._run_cache[str(run_id)])

    def get_run_events(self, run_id: str, *, after: int = 0) -> JsonDict:
        return {
            "run_id": run_id,
            "events": [
                {
                    "sequence": 1,
                    "stage": "completed",
                    "progress": 100.0,
                    "message": "Direct V-Sense run completed",
                    "created_at": self._run_cache[str(run_id)]["completed_at"],
                }
            ],
            "request_id": None,
        }

    def cancel_run(self, run_id: str) -> JsonDict:
        raise BackendApiHTTPError(
            "Direct V-Sense run is already complete.",
            status_code=409,
            error_code="VSENSE_RUN_NOT_CANCELLABLE",
        )


def get_vsense_gateway(
    *,
    access_token: str | None = None,
    client: ApiClient | None = None,
) -> VSenseGateway:
    """Return the V-Sense gateway selected by configuration."""

    if is_backend_api_enabled("vsense"):
        if not is_backend_api_enabled("auth"):
            raise BackendApiHTTPError(
                "V-Sense API mode requires USE_BACKEND_API_AUTH=true.",
                status_code=401,
                error_code="AUTH_REQUIRED",
            )
        token = str(access_token or "").strip()
        if not token:
            raise BackendApiHTTPError(
                "V-Sense API mode requires a backend access token.",
                status_code=401,
                error_code="AUTH_REQUIRED",
            )
        return ApiVSenseGateway(token, client)
    return DirectVSenseGateway()


__all__ = [
    "ApiVSenseGateway",
    "DirectVSenseGateway",
    "VSenseGateway",
    "get_vsense_gateway",
]
