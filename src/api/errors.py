"""RFC 7807 problem responses for every error the API returns.

Body (``application/problem+json``): ``type``, ``title``, ``status``, ``detail``, ``instance`` (the request path),
``request_id`` and, for validation errors, ``errors``. A 500 never carries the exception text; the log has it under the
same request id.
"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.request_context import get_request_id

log = logging.getLogger("api")

PROBLEM_JSON = "application/problem+json"


class FieldError(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ProblemDetail(BaseModel):
    """Error body of every non-2xx response (RFC 7807)."""

    type: str = Field("about:blank", description="URI reference identifying the problem type")
    title: str = Field(..., description="Short summary of the problem type")
    status: int = Field(..., description="HTTP status code")
    detail: str | None = Field(None, description="Explanation of this occurrence")
    instance: str | None = Field(None, description="Request path")
    request_id: str | None = Field(None, description="Request id, also in the X-Request-ID header and the server log")
    errors: list[FieldError] | None = Field(None, description="Validation errors (422 only)")


class ApiError(Exception):
    """Raise from a route for a problem response with a specific status, title and detail."""

    def __init__(self, status: int, detail: str | None = None, *, title: str | None = None, type: str = "about:blank"):
        super().__init__(detail or title or HTTPStatus(status).phrase)
        self.status = status
        self.detail = detail
        self.title = title or HTTPStatus(status).phrase
        self.type = type


def problem(
    status: int,
    request: Request,
    *,
    title: str | None = None,
    detail: str | None = None,
    type: str = "about:blank",
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ProblemDetail(
        type=type,
        title=title or HTTPStatus(status).phrase,
        status=status,
        detail=detail,
        instance=request.url.path,
        request_id=get_request_id(),
        errors=[FieldError(**e) for e in errors] if errors else None,
    )
    return JSONResponse(
        body.model_dump(exclude_none=True), status_code=status, media_type=PROBLEM_JSON, headers=headers
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return problem(exc.status, request, title=exc.title, detail=exc.detail, type=exc.type)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = (
            exc.detail if isinstance(exc.detail, str) and exc.detail != HTTPStatus(exc.status_code).phrase else None
        )
        return problem(exc.status_code, request, detail=detail, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "loc": [str(p) if not isinstance(p, int) else p for p in e.get("loc", ())],
                "msg": e["msg"],
                "type": e["type"],
            }
            for e in exc.errors()
        ]
        return problem(422, request, detail="The request does not match the schema.", errors=errors)

    # Unhandled exceptions are converted in RequestContextMiddleware (inside the request context), see
    # internal_error_body below.


INTERNAL_DETAIL = "The server could not complete the request. Quote the request id."


def internal_error_body(path: str) -> bytes:
    """Body of the 500 problem: never the exception text (the log has it under the same request id)."""
    body = ProblemDetail(
        title="Internal Server Error", status=500, detail=INTERNAL_DETAIL, instance=path, request_id=get_request_id()
    )
    return body.model_dump_json(exclude_none=True).encode()


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    "4XX": {"model": ProblemDetail, "description": "Client error (problem+json)"},
    "5XX": {"model": ProblemDetail, "description": "Server error (problem+json)"},
}
