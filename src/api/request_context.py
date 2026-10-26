"""Request id and access log: every request gets an id (``X-Request-ID`` from the client, else a new one), echoed in
the response header and attached to every log record written while the request is served."""

from __future__ import annotations

import logging
import re
import time
import uuid
from contextvars import ContextVar

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
access_log = logging.getLogger("api.access")


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


class RequestIdFilter(logging.Filter):
    """Adds ``request_id`` to log records (``-`` outside a request)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _REQUEST_ID.get() or "-"
        return True


class RequestContextMiddleware:
    """Pure ASGI middleware (no response buffering, so server-sent events stream through unchanged)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex
        token = _REQUEST_ID.set(request_id)
        start = time.perf_counter()
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [h for h in message.get("headers", []) if h[0].lower() != b"x-request-id"]
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        started = False

        async def send_tracked(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send_with_id(message)

        try:
            await self.app(scope, receive, send_tracked)
        except Exception:
            # Unhandled errors become a problem response here, inside the request context, so the body and the
            # X-Request-ID header carry the id that the log line carries (Starlette's own 500 handler runs outside).
            access_log.getChild("error").exception("unhandled error on %s %s", scope.get("method"), scope.get("path"))
            if started:
                raise
            from api.errors import internal_error_body

            body = internal_error_body(str(scope.get("path")))
            await send_with_id(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [
                        (b"content-type", b"application/problem+json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send_with_id({"type": "http.response.body", "body": body})
        finally:
            access_log.info(
                "%s %s %s %.1f ms", scope.get("method"), scope.get("path"), status, (time.perf_counter() - start) * 1000
            )
            _REQUEST_ID.reset(token)
