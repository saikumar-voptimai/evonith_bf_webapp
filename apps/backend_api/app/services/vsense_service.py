"""Backend V-Sense application service."""

from __future__ import annotations

from typing import Any

from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.vsense_repository import VSenseRepository, fingerprint
from apps.backend_api.app.services.vsense_context_service import VSenseContextService
from furnace_data.vsense.bounds import (
    VSenseValidationError,
    default_control_profile,
    validate_control_profile,
)
from furnace_data.vsense.catalog import load_vsense_catalog, optimization_by_id


class VSenseService:
    """Coordinate V-Sense catalog, profiles, contexts, and run services."""

    def __init__(
        self,
        *,
        settings: Any | None = None,
        repository: VSenseRepository | None = None,
        audit_service: Any | None = None,
        history_df: Any | None = None,
        clock: Any | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository or VSenseRepository()
        self.audit_service = audit_service
        self.context_service = VSenseContextService(
            repository=self.repository,
            settings=settings,
            history_df=history_df,
            clock=clock,
        )

    def get_catalog(self) -> dict[str, Any]:
        return load_vsense_catalog(
            context_ttl_seconds=self._setting("vsense_context_ttl_seconds", 1800),
            max_concurrent_runs=self._setting("vsense_max_concurrent_runs", 1),
            llm_review_available=bool(self._setting("vsense_llm_enabled", False)),
            advanced_diagnostics_available=bool(
                self._setting("vsense_advanced_diagnostics", False)
            ),
            display_timezone=self._setting("vboard_default_timezone", "Asia/Kolkata"),
        )

    def create_context(
        self,
        request: dict[str, Any],
        *,
        current_user: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        return self.context_service.create_context(
            request,
            current_user=current_user,
            idempotency_key=idempotency_key,
        )

    def get_control_profile(self, optimization_type_id: str) -> dict[str, Any]:
        catalog = self.get_catalog()
        if optimization_type_id not in optimization_by_id(catalog):
            raise ApiError(
                "VSENSE_INVALID_OPTIMIZATION_TYPE",
                "Unknown V-Sense optimization type.",
                status_code=400,
            )
        try:
            params = default_control_profile(
                optimization_type_id,
                require_approved_bounds=self._require_approved_bounds(),
                catalog=catalog,
            )
        except VSenseValidationError as exc:
            raise ApiError(exc.code, str(exc), status_code=exc.status_code) from exc
        return self.repository.get_or_create_profile(
            optimization_type_id=optimization_type_id,
            default_parameters=params,
            catalog_version=catalog["catalog_version"],
        )

    def update_control_profile(
        self,
        optimization_type_id: str,
        request: dict[str, Any],
        *,
        current_user: dict[str, Any],
        idempotency_key: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        owner_user_id = _user_id(current_user)
        payload_fingerprint = fingerprint(
            {
                "optimization_type_id": optimization_type_id,
                **request,
            }
        )
        replay = self.repository.get_idempotent_response(
            owner_user_id=owner_user_id,
            scope=f"profile:{optimization_type_id}",
            idempotency_key=idempotency_key,
            request_fingerprint=payload_fingerprint,
        )
        if replay is not None:
            replay["idempotent_replay"] = True
            return replay

        catalog = self.get_catalog()
        try:
            params = validate_control_profile(
                optimization_type_id,
                list(request.get("parameters") or []),
                require_approved_bounds=self._require_approved_bounds(),
                catalog=catalog,
            )
        except VSenseValidationError as exc:
            raise ApiError(exc.code, str(exc), status_code=exc.status_code) from exc
        profile = self.repository.update_profile(
            optimization_type_id=optimization_type_id,
            profile_id=str(request.get("profile_id") or "plant-default"),
            expected_version=int(request.get("expected_version")),
            parameters=params,
            catalog_version=catalog["catalog_version"],
            actor_user_id=owner_user_id,
            actor_username=current_user.get("username"),
        )
        profile["idempotent_replay"] = False
        self.repository.store_idempotent_response(
            owner_user_id=owner_user_id,
            scope=f"profile:{optimization_type_id}",
            idempotency_key=idempotency_key,
            request_fingerprint=payload_fingerprint,
            response=profile,
        )
        self._audit(
            {
                "request_id": request_id,
                "actor_user_id": owner_user_id,
                "actor_username": current_user.get("username"),
                "event_type": "vsense.control_profile.updated",
                "resource_type": "vsense.control_profile",
                "resource_id": f"plant-default:{optimization_type_id}",
                "action": "update",
                "result": "success",
                "status_code": 200,
                "metadata": {
                    "optimization_type_id": optimization_type_id,
                    "control_profile_version": profile["version"],
                    "parameter_ids": [item["parameter_id"] for item in params],
                },
            }
        )
        return profile

    def _audit(self, payload: dict[str, Any]) -> None:
        if self.audit_service is not None:
            self.audit_service.record_event(payload)

    def _setting(self, name: str, default: Any) -> Any:
        return getattr(self.settings, name, default)

    def _require_approved_bounds(self) -> bool:
        return bool(self._setting("vsense_require_approved_bounds", True))


def _user_id(current_user: dict[str, Any]) -> str:
    return str(current_user.get("id") or current_user.get("username") or "unknown")
