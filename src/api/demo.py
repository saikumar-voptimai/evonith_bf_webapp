"""Demo mode on the wire (``BF_DEMO=1``; user decision 16 Nov 2026, demo branch): every ``/api`` response leaves the
twin-api with the plant, site, supplier and mine names replaced (``utils.demo``, rules in ``config/demo.yml``).

A pure ASGI middleware, so it sees every body the routes produce, also the server-sent events of the live stream and
of FurnaceMind's answers:

* JSON (``application/json``, ``application/problem+json``): the whole body is read, its strings redacted, re-encoded;
  the ``content-length`` follows.
* server-sent events (``text/event-stream``): passed on event by event (an event ends with a blank line), each
  ``data:`` payload redacted; the stream is never held back longer than one event.
* CSV and plain text: the whole body as text.

Read-only views that show ids as text (``id_paths`` of ``config/demo.yml``) get the names inside their ids replaced
as well; every other response keeps its ids, which the SPA sends back on a save.

Anything else (images, downloads of other types) passes unchanged. With demo mode off the middleware is a no-op.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from utils import demo

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

_BUFFERED = ("application/json", "application/problem+json", "text/csv", "text/plain")


def _content_type(headers: list[tuple[bytes, bytes]]) -> str:
    for key, value in headers:
        if key.lower() == b"content-type":
            return value.decode("latin-1").split(";")[0].strip().lower()
    return ""


class DemoRedaction:
    """Redacts the bodies of ``/api`` responses while demo mode is on."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or not demo.enabled() or not str(scope.get("path", "")).startswith("/api"):
            await self.app(scope, receive, send)
            return

        ids = demo.ids_on(str(scope.get("path", "")))
        mode = "pass"
        start: Message | None = None
        content_type = ""
        chunks: list[bytes] = []
        pending = b""

        async def wrapped(message: Message) -> None:
            nonlocal mode, start, content_type, pending
            kind = message["type"]
            if kind == "http.response.start":
                content_type = _content_type(list(message.get("headers", [])))
                if content_type == "text/event-stream":
                    mode = "sse"
                    await send(message)
                elif content_type in _BUFFERED:
                    mode = "buffer"
                    start = message
                else:
                    await send(message)
                return
            if kind != "http.response.body":
                await send(message)
                return
            body = message.get("body", b"") or b""
            more = bool(message.get("more_body", False))
            if mode == "pass":
                await send(message)
            elif mode == "buffer":
                chunks.append(body)
                if more:
                    return
                raw = b"".join(chunks)
                out = (
                    demo.redact_text_bytes(raw)
                    if content_type.startswith("text/")
                    else demo.redact_json_bytes(raw, ids=ids)
                )
                assert start is not None
                headers = [(k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"]
                headers.append((b"content-length", str(len(out)).encode("latin-1")))
                await send({**start, "headers": headers})
                await send({"type": "http.response.body", "body": out, "more_body": False})
            else:  # server-sent events: pass on every complete event at once
                pending += body.replace(b"\r\n", b"\n")
                events = pending.split(b"\n\n")
                pending = events.pop()  # the unfinished tail waits for the next chunk
                out = b"".join(demo.redact_sse_event(e) + b"\n\n" for e in events)
                if not more and pending:
                    out += demo.redact_sse_event(pending)
                    pending = b""
                if out or not more:
                    await send({"type": "http.response.body", "body": out, "more_body": more})

        await self.app(scope, receive, wrapped)
