"""Response helpers shared by API routes and exception handlers."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


def get_request_id(request: Request) -> str:
    """Return the request id stored by middleware, or a stable fallback."""
    return str(getattr(request.state, "request_id", "unknown"))


def api_error_response(
    *,
    request_id: str,
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
    legacy_detail: Any = None,
) -> JSONResponse:
    """Build the standard API error response."""
    content: dict[str, Any] = {
        "request_id": request_id,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }
    if legacy_detail is not None:
        content["detail"] = legacy_detail

    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={"X-Request-ID": request_id},
    )
