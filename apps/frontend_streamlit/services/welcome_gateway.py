"""Gateway abstraction for the Welcome page and its admin panels."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID

import streamlit as st

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.services.api_errors import (
    BackendApiHTTPError,
    BackendUnavailableError,
)
from apps.frontend_streamlit.services import dashboard_api, plant_admin_api
from apps.frontend_streamlit.utils.session import current_user_id


class WelcomeGateway(Protocol):
    def get_kpis(self, *, window: str, bucket: str): ...
    def get_hopper_context(self, *, at: datetime | None = None): ...
    def list_hopper_history(self, *, limit: int, offset: int): ...
    def update_hopper_mapping(self, request: dict[str, Any]): ...
    def delete_hopper_history(self, record_ids: list[int]): ...
    def get_burden_context(self, *, at: datetime | None = None): ...
    def list_burden_history(self, *, limit: int, offset: int): ...
    def update_burden_distribution(self, request: dict[str, Any]): ...
    def delete_burden_history(self, record_ids: list[int]): ...


def _aware_utc(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone offset.")
    return value.astimezone(timezone.utc).replace(microsecond=0)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


class ApiWelcomeGateway:
    """Welcome gateway backed by authenticated API v1 calls."""

    def __init__(self, access_token: str) -> None:
        self.access_token = access_token

    def get_kpis(self, *, window: str, bucket: str) -> dict[str, Any]:
        raw = dashboard_api.get_kpis(
            self.access_token,
            window=window,
            bucket=bucket,
        )
        return {
            "data": raw.get("data", {}),
            "warnings": list((raw.get("meta") or {}).get("warnings") or []),
            "request_id": raw.get("request_id"),
        }

    def get_hopper_context(self, *, at: datetime | None = None):
        return plant_admin_api.get_hopper_context(
            self.access_token,
            at=_iso(at),
        )

    def list_hopper_history(self, *, limit: int, offset: int):
        return plant_admin_api.list_hopper_history(
            self.access_token,
            limit=limit,
            offset=offset,
        )

    def update_hopper_mapping(self, request: dict[str, Any]):
        return plant_admin_api.update_hopper_mapping(self.access_token, request)

    def delete_hopper_history(self, record_ids: list[int]):
        return plant_admin_api.delete_hopper_history(self.access_token, record_ids)

    def get_burden_context(self, *, at: datetime | None = None):
        return plant_admin_api.get_burden_context(
            self.access_token,
            at=_iso(at),
        )

    def list_burden_history(self, *, limit: int, offset: int):
        return plant_admin_api.list_burden_history(
            self.access_token,
            limit=limit,
            offset=offset,
        )

    def update_burden_distribution(self, request: dict[str, Any]):
        return plant_admin_api.update_burden_distribution(self.access_token, request)

    def delete_burden_history(self, record_ids: list[int]):
        return plant_admin_api.delete_burden_history(self.access_token, record_ids)


class DirectWelcomeGateway:
    """Temporary direct-mode gateway for rollback."""

    def get_kpis(self, *, window: str, bucket: str) -> dict[str, Any]:
        if window != "1h" or bucket != "15m":
            raise ValueError("Unsupported KPI window or bucket.")
        from furnace_data.influx.online import fetch_online_df

        df = fetch_online_df(
            selected_measurements=["process_params"],
            time_range="last 1 hour",
            window_by="15 minutes",
            column_naming="field",
        )
        units = {
            "production_rate": ("production_per_hour", "t/h"),
            "fuel_rate": ("fuel_rate", "kg/tHM"),
            "eta_co": ("body_etaco", "%"),
            "blast_volume": ("hot_blast_vol_nm3h", "Nm3/h"),
        }
        if df is None or getattr(df, "empty", True):
            return {
                "data": {
                    "as_of": datetime.now(timezone.utc).replace(microsecond=0),
                    "window": window,
                    "bucket": bucket,
                    "sample_count": 0,
                    "metrics": {
                        key: {"value": None, "unit": unit}
                        for key, (_column, unit) in units.items()
                    },
                },
                "warnings": ["No online KPI samples were returned for the requested window."],
                "request_id": None,
            }
        metrics: dict[str, dict[str, float | str | None]] = {}
        for key, (column, unit) in units.items():
            value = None
            if column in df.columns:
                series = df[column].dropna()
                if not series.empty:
                    value = round(float(series.mean()), 1)
            metrics[key] = {"value": value, "unit": unit}
        try:
            as_of = df.index.max().to_pydatetime().astimezone(timezone.utc).replace(microsecond=0)
        except Exception:
            as_of = datetime.now(timezone.utc).replace(microsecond=0)
        return {
            "data": {
                "as_of": as_of,
                "window": window,
                "bucket": bucket,
                "sample_count": int(len(df.index)),
                "metrics": metrics,
            },
            "warnings": [],
            "request_id": None,
        }

    @staticmethod
    def _client_ip() -> str:
        try:
            forwarded_for = st.context.headers.get("X-Forwarded-For")
            if forwarded_for:
                return forwarded_for.split(",", 1)[0].strip()
            return st.context.headers.get("REMOTE_ADDR", "unknown")
        except Exception:
            return "unknown"

    @staticmethod
    def _actor_user_id() -> UUID | None:
        user_id = current_user_id()
        if not user_id:
            return None
        try:
            return UUID(str(user_id))
        except ValueError:
            return None

    def _hopper_service(self):
        from apps.frontend_streamlit.data.db import HopperConfigService

        return HopperConfigService()

    def _burden_service(self):
        from apps.frontend_streamlit.data.db import BurdenConfigService

        return BurdenConfigService()

    def get_hopper_context(self, *, at: datetime | None = None):
        svc = self._hopper_service()
        at_utc = _aware_utc(at)
        row = svc._hopper_repository.get_hopper_snapshot_at(at_utc)
        return {
            "at": at_utc,
            "snapshot_id": row.get("id"),
            "effective_at": row.get("date_time"),
            "hoppers": [
                {
                    "code": hopper["hopper_code"],
                    "display_name": hopper.get("display_name") or hopper["hopper_code"],
                }
                for hopper in svc.hopper_rows
            ],
            "materials": [
                {
                    "code": material["material_code"],
                    "canonical_name": material["material_name"],
                    "display_name": svc._material_mapper.primary_client_name_for_material(
                        material["material_name"]
                    ),
                }
                for material in svc.material_rows
            ],
            "assignments": {hopper: row.get(hopper) for hopper in svc.hoppers},
        }

    def list_hopper_history(self, *, limit: int, offset: int):
        svc = self._hopper_service()
        rows = svc._hopper_repository.get_hopper_material_history(
            limit=limit,
            offset=offset,
        )
        return {
            "items": [
                {
                    "snapshot_id": int(row["id"]),
                    "effective_at": row["date_time"],
                    "assignments": {hopper: row.get(hopper) for hopper in svc.hoppers},
                    "source_type": row.get("source_type"),
                    "actor": {"user_id": row.get("user_modified"), "username": None},
                    "created_at": row.get("created_at"),
                }
                for row in rows
            ],
            "total": svc._hopper_repository.count_hopper_material_history(),
            "limit": limit,
            "offset": offset,
        }

    def update_hopper_mapping(self, request: dict[str, Any]):
        svc = self._hopper_service()
        effective_at = _aware_utc(datetime.fromisoformat(request["effective_at"]))
        latest = svc._hopper_repository.get_hopper_snapshot_at(effective_at)
        if latest.get("id") != request.get("expected_snapshot_id"):
            raise BackendApiHTTPError(
                "The hopper mapping changed before this update was submitted.",
                status_code=409,
                error_code="CONFIG_VERSION_CONFLICT",
            )
        svc._hopper_repository.update_hopper_snapshot(
            hopper_material_codes=request["assignments"],
            from_time=effective_at,
            user_id=self._actor_user_id(),
            ip_address=self._client_ip(),
            source_type="webapp",
        )
        return self.get_hopper_context(at=effective_at)

    def delete_hopper_history(self, record_ids: list[int]):
        svc = self._hopper_service()
        deleted = svc._hopper_repository.delete_hopper_material_history(record_ids)
        return {
            "deleted_count": deleted,
            "current_context": self.get_hopper_context(),
        }

    def get_burden_context(self, *, at: datetime | None = None):
        svc = self._burden_service()
        at_utc = _aware_utc(at)
        row = svc._burden_repository.get_burden_snapshot_at(at_utc)
        fields = svc.burden_fields
        text_fields = set(svc._burden_repository.TEXT_FIELDS)
        return {
            "at": at_utc,
            "snapshot_id": row.get("id"),
            "effective_at": row.get("date_time"),
            "fields": [
                {
                    "key": field,
                    "label": field.replace("_", " ").title(),
                    "value_type": "text" if field in text_fields else "number",
                    "nullable": True,
                    "step": None if field in text_fields else 0.1,
                }
                for field in fields
            ],
            "values": {field: row.get(field) for field in fields},
        }

    def list_burden_history(self, *, limit: int, offset: int):
        svc = self._burden_service()
        rows = svc._burden_repository.get_burden_history(limit=limit, offset=offset)
        return {
            "items": [
                {
                    "snapshot_id": int(row["id"]),
                    "effective_at": row["date_time"],
                    "values": {field: row.get(field) for field in svc.burden_fields},
                    "source_type": row.get("source_type"),
                    "actor": {"user_id": row.get("user_modified"), "username": None},
                    "created_at": row.get("created_at"),
                }
                for row in rows
            ],
            "total": svc._burden_repository.count_burden_history(),
            "limit": limit,
            "offset": offset,
        }

    def update_burden_distribution(self, request: dict[str, Any]):
        svc = self._burden_service()
        effective_at = _aware_utc(datetime.fromisoformat(request["effective_at"]))
        latest = svc._burden_repository.get_burden_snapshot_at(effective_at)
        if latest.get("id") != request.get("expected_snapshot_id"):
            raise BackendApiHTTPError(
                "The burden distribution changed before this update was submitted.",
                status_code=409,
                error_code="CONFIG_VERSION_CONFLICT",
            )
        svc._burden_repository.update_burden_row(
            row_values=request["values"],
            timestamp=effective_at,
            user_id=self._actor_user_id(),
            ip=self._client_ip(),
            source_type="webapp",
        )
        return self.get_burden_context(at=effective_at)

    def delete_burden_history(self, record_ids: list[int]):
        svc = self._burden_service()
        deleted = svc._burden_repository.delete_burden_history(record_ids)
        return {
            "deleted_count": deleted,
            "current_context": self.get_burden_context(),
        }


def get_welcome_gateway() -> WelcomeGateway:
    """Return the configured Welcome gateway."""
    if is_backend_api_enabled("welcome"):
        if not is_backend_api_enabled("auth"):
            raise BackendUnavailableError(
                "Welcome API mode requires USE_BACKEND_API_AUTH=true."
            )
        token = str(st.session_state.get("auth_access_token") or "").strip()
        if not token:
            raise BackendUnavailableError(
                "Welcome API mode requires a backend auth access token."
            )
        return ApiWelcomeGateway(token)
    return DirectWelcomeGateway()
