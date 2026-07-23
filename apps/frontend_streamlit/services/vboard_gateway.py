"""V-Board gateways for API-first mode and temporary direct rollback mode."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.services.api_client import ApiClient
from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError
from apps.frontend_streamlit.services.vboard_api import JsonDict, VBoardApi


@runtime_checkable
class VBoardGateway(Protocol):
    def get_catalog(self) -> JsonDict: ...

    def get_contours(self, request: JsonDict) -> JsonDict: ...

    def get_heatload_timeseries(self, request: JsonDict) -> JsonDict: ...


class ApiVBoardGateway:
    """V-Board gateway backed exclusively by API v1."""

    def __init__(self, access_token: str, client: ApiClient | None = None) -> None:
        self.api = VBoardApi(access_token, client)

    def get_catalog(self) -> JsonDict:
        return self.api.get_catalog()

    def get_contours(self, request: JsonDict) -> JsonDict:
        return self.api.get_contours(request)

    def get_heatload_timeseries(self, request: JsonDict) -> JsonDict:
        return self.api.get_heatload_timeseries(request)


class DirectVBoardGateway:
    """Deprecated direct V-Board gateway kept only as a rollback path."""

    def get_catalog(self) -> JsonDict:
        from furnace_data.vboard import load_vboard_catalog

        catalog = load_vboard_catalog()
        catalog["request_id"] = None
        return catalog

    def get_contours(self, request: JsonDict) -> JsonDict:
        from furnace_data.vboard import (
            VBoardRepository,
            load_vboard_catalog,
            resolve_time_range,
            transform_heatload_contour,
            transform_temperature_contour,
        )

        catalog = load_vboard_catalog()
        resolved = resolve_time_range(request["time_range"], now_utc=datetime.now(timezone.utc))
        repository = VBoardRepository(source="historical")
        temperature = transform_temperature_contour(
            repository.fetch_temperature_contour(resolved.start, resolved.end),
            catalog=catalog,
        )
        heatload = transform_heatload_contour(
            repository.fetch_heatload_contour(resolved.start, resolved.end),
            catalog=catalog,
        )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "resolved_range": resolved.to_dict(),
            "catalog_version": catalog["catalog_version"],
            "processing_policy_id": catalog["processing_policy"]["id"],
            "temperature": temperature,
            "heatload": heatload,
            "request_id": None,
            "warnings": [
                "Direct V-Board mode is deprecated; enable USE_BACKEND_API_VBOARD for API mode."
            ],
        }

    def get_heatload_timeseries(self, request: JsonDict) -> JsonDict:
        from furnace_data.vboard import (
            VBoardRepository,
            load_vboard_catalog,
            resolve_time_range,
            transform_heatload_timeseries,
        )
        from furnace_data.vboard.catalog import (
            auto_resolution_window_id,
            query_window_for_window_id,
            resolution_windows_by_id,
            rows_by_id,
        )

        catalog = load_vboard_catalog()
        row_id = str(request["row_id"]).upper()
        row = rows_by_id().get(row_id)
        if row is None:
            raise BackendApiHTTPError(
                "Unknown V-Board row.",
                status_code=400,
                error_code="INVALID_VBOARD_ROW",
            )
        resolved = resolve_time_range(request["time_range"], now_utc=datetime.now(timezone.utc))
        resolution = request.get("resolution") or {"mode": "auto"}
        if resolution.get("mode") == "fixed":
            window_id = str(resolution.get("window_id") or "")
        else:
            duration_seconds = int((resolved.end - resolved.start).total_seconds())
            window_id = auto_resolution_window_id(duration_seconds)
        windows = resolution_windows_by_id()
        if window_id not in windows:
            raise BackendApiHTTPError(
                "Unknown V-Board resolution window.",
                status_code=400,
                error_code="INVALID_VBOARD_RESOLUTION",
            )
        repository = VBoardRepository(source="historical")
        frame = repository.fetch_heatload_timeseries(
            resolved.start,
            resolved.end,
            row_id=row_id,
            window_by=query_window_for_window_id(window_id),
        )
        transformed = transform_heatload_timeseries(
            frame,
            row_id=row_id,
            resolved_window_seconds=windows[window_id].seconds,
            max_points_per_quadrant=catalog["limits"]["max_timeseries_points_per_quadrant"],
            catalog=catalog,
            processing_policy_id=catalog["processing_policy"]["id"],
        )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "resolved_range": resolved.to_dict(),
            "row": row,
            **transformed,
            "request_id": None,
            "warnings": list(
                dict.fromkeys(
                    [
                        *transformed.get("warnings", []),
                        "Direct V-Board mode is deprecated; enable USE_BACKEND_API_VBOARD for API mode.",
                    ]
                )
            ),
        }


def get_vboard_gateway(
    *,
    access_token: str | None = None,
    client: ApiClient | None = None,
) -> VBoardGateway:
    """Return the V-Board gateway selected by configuration."""

    if is_backend_api_enabled("vboard"):
        if not is_backend_api_enabled("auth"):
            raise BackendApiHTTPError(
                "V-Board API mode requires USE_BACKEND_API_AUTH=true.",
                status_code=401,
                error_code="AUTH_REQUIRED",
            )
        token = str(access_token or "").strip()
        if not token:
            raise BackendApiHTTPError(
                "V-Board API mode requires a backend access token.",
                status_code=401,
                error_code="AUTH_REQUIRED",
            )
        return ApiVBoardGateway(token, client)
    return DirectVBoardGateway()


__all__ = [
    "ApiVBoardGateway",
    "DirectVBoardGateway",
    "VBoardGateway",
    "get_vboard_gateway",
]
