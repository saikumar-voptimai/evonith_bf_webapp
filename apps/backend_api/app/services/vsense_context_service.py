"""V-Sense immutable context service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.vsense_repository import VSenseRepository, fingerprint
from furnace_data.vsense.bounds import VSenseValidationError, default_control_profile
from furnace_data.vsense.context import VSenseContextError, build_context_snapshot


class VSenseContextService:
    """Create and retrieve immutable V-Sense contexts."""

    def __init__(
        self,
        *,
        repository: VSenseRepository,
        settings: Any | None = None,
        history_df: Any | None = None,
        clock: Any | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.history_df = history_df
        self.clock = clock

    def create_context(
        self,
        request: dict[str, Any],
        *,
        current_user: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        owner_user_id = _user_id(current_user)
        payload_fingerprint = fingerprint(request)
        replay = self.repository.get_idempotent_response(
            owner_user_id=owner_user_id,
            scope="context",
            idempotency_key=idempotency_key,
            request_fingerprint=payload_fingerprint,
        )
        if replay is not None:
            replay["idempotent_replay"] = True
            return replay

        data_mode = str(request.get("data_mode") or "live")
        if data_mode == "historical_only" and "vsense:diagnostics" not in set(
            current_user.get("permissions") or []
        ):
            raise ApiError(
                "FORBIDDEN",
                "Historical-only V-Sense context requires diagnostics permission.",
                status_code=403,
            )
        optimization_type_id = str(request.get("optimization_type_id") or "")
        catalog = self._catalog()
        try:
            profile_params = default_control_profile(
                optimization_type_id,
                require_approved_bounds=self._require_approved_bounds(),
                catalog=catalog,
            )
            profile = self.repository.get_or_create_profile(
                optimization_type_id=optimization_type_id,
                default_parameters=profile_params,
                catalog_version=catalog["catalog_version"],
            )
            context = build_context_snapshot(
                optimization_type_id=optimization_type_id,
                data_mode=data_mode,
                owner_user_id=owner_user_id,
                now=self._now(),
                ttl_seconds=self._setting("vsense_context_ttl_seconds", 1800),
                catalog=catalog,
                control_profile={
                    "profile_id": profile["profile_id"],
                    "version": profile["version"],
                    "parameters": profile["parameters"],
                },
                history_df=self.history_df,
            )
        except VSenseValidationError as exc:
            raise ApiError(exc.code, str(exc), status_code=exc.status_code) from exc
        except VSenseContextError as exc:
            raise ApiError(exc.code, str(exc), status_code=exc.status_code) from exc

        stored = self.repository.store_context(context)
        response = dict(stored)
        response.pop("owner_user_id", None)
        response["idempotent_replay"] = False
        self.repository.store_idempotent_response(
            owner_user_id=owner_user_id,
            scope="context",
            idempotency_key=idempotency_key,
            request_fingerprint=payload_fingerprint,
            response=response,
        )
        return response

    def get_context_for_run(
        self,
        context_id: str,
        *,
        current_user: dict[str, Any],
    ) -> dict[str, Any]:
        context = self.repository.get_context(context_id)
        if context is None:
            raise ApiError("VSENSE_CONTEXT_NOT_FOUND", "V-Sense context not found.", 404)
        owner_user_id = str(context.get("owner_user_id") or "")
        if owner_user_id != _user_id(current_user) and "vsense:runs:read:any" not in set(
            current_user.get("permissions") or []
        ):
            raise ApiError("FORBIDDEN", "Insufficient permissions.", 403)
        expires_at = _parse_dt(context.get("expires_at"))
        if expires_at <= self._now():
            raise ApiError(
                "VSENSE_CONTEXT_EXPIRED",
                "V-Sense context has expired. Refresh current context.",
                410,
            )
        return context

    def _catalog(self) -> dict[str, Any]:
        from furnace_data.vsense.catalog import load_vsense_catalog

        return load_vsense_catalog(
            context_ttl_seconds=self._setting("vsense_context_ttl_seconds", 1800),
            max_concurrent_runs=self._setting("vsense_max_concurrent_runs", 1),
            llm_review_available=bool(self._setting("vsense_llm_enabled", False)),
            advanced_diagnostics_available=bool(
                self._setting("vsense_advanced_diagnostics", False)
            ),
        )

    def _setting(self, name: str, default: Any) -> Any:
        return getattr(self.settings, name, default)

    def _require_approved_bounds(self) -> bool:
        return bool(self._setting("vsense_require_approved_bounds", True))

    def _now(self) -> datetime:
        if self.clock is not None:
            return self.clock()
        return datetime.now(timezone.utc)


def _user_id(current_user: dict[str, Any]) -> str:
    return str(current_user.get("id") or current_user.get("username") or "unknown")


def _parse_dt(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
