"""Structured backend API errors and exception handlers."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from apps.backend_api.app.core.responses import api_error_response, get_request_id
from apps.backend_api.app.services.redaction_service import redact_dict, redact_text

log = logging.getLogger(__name__)


class ApiError(Exception):
    """Application-level error returned in a stable JSON shape."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


async def api_error_handler(request: Request, exc: ApiError):
    request_id = get_request_id(request)
    request.state.error_code = exc.code
    return api_error_response(
        request_id=request_id,
        code=exc.code,
        message=redact_text(exc.message),
        status_code=exc.status_code,
        details=redact_dict(exc.details),
    )


async def http_exception_handler(request: Request, exc: HTTPException | StarletteHTTPException):
    if exc.status_code == 204:
        return Response(status_code=204, headers={"X-Request-ID": get_request_id(request)})

    request_id = get_request_id(request)
    code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
    request.state.error_code = code
    detail = exc.detail
    message = detail if isinstance(detail, str) else "HTTP error"
    details = detail if isinstance(detail, dict) else {"detail": detail} if detail else {}
    return api_error_response(
        request_id=request_id,
        code=code,
        message=redact_text(message),
        status_code=exc.status_code,
        details=redact_dict(details),
        legacy_detail=redact_dict(detail) if isinstance(detail, dict) else redact_text(str(detail)) if detail else detail,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = get_request_id(request)
    request.state.error_code = "VALIDATION_ERROR"
    return api_error_response(
        request_id=request_id,
        code="VALIDATION_ERROR",
        message="Request validation failed",
        status_code=422,
        details=redact_dict({"errors": exc.errors()}),
        legacy_detail=redact_dict({"errors": exc.errors()})["errors"],
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = get_request_id(request)
    request.state.error_code = "INTERNAL_SERVER_ERROR"
    log.exception("Unhandled backend error request_id=%s", request_id)
    return api_error_response(
        request_id=request_id,
        code="INTERNAL_SERVER_ERROR",
        message="Internal server error",
        status_code=500,
    )


def register_exception_handlers(app) -> None:
    """Register standard exception handlers on a FastAPI app."""
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
