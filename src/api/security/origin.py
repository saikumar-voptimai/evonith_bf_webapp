"""Cross-site request refusal for state-changing API calls.

The session cookie is SameSite=Strict, so browsers do not send it with cross-site requests. As a second line, a
POST, PUT, PATCH or DELETE to ``/api`` whose ``Origin`` header names another host than the request's ``Host`` is
refused with 403 before it reaches a route. Requests without an ``Origin`` header (command-line clients, tests) pass.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

from api.errors import PROBLEM_JSON, ProblemDetail
from api.request_context import get_request_id

UNSAFE = frozenset({"POST", "PUT", "PATCH", "DELETE"})
DETAIL = "Requests that change data must come from a page of this console."


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def cross_site(scope: Scope) -> bool:
    """True when an unsafe ``/api`` request carries an ``Origin`` that differs from its ``Host``."""
    if scope["type"] != "http" or scope["method"] not in UNSAFE or not scope["path"].startswith("/api/"):
        return False
    origin = _header(scope, b"origin")
    if origin is None:
        return False
    host = _header(scope, b"host")
    if origin == "null" or host is None:
        return True
    return urlsplit(origin).netloc.lower() != host.lower()


class OriginGuardMiddleware:
    """Pure ASGI middleware (no body buffering); install inside ``RequestContextMiddleware``."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not cross_site(scope):
            await self.app(scope, receive, send)
            return
        body = ProblemDetail(
            title="Forbidden", status=403, detail=DETAIL, instance=scope["path"], request_id=get_request_id()
        ).model_dump_json(exclude_none=True)
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-type", PROBLEM_JSON.encode()), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body.encode()})
