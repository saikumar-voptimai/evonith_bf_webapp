"""Server-sent events: live status, twin snapshots, forecasts, job progress and alarms (SV-0.7, #276)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse

from api.auth import read_session, require_role
from api.live import live_state
from services.events import (
    REPLAY_LIMIT,
    HeartbeatData,
    HeartbeatEvent,
    StreamEvent,
    id_key,
    is_event_id,
    sse_frame,
)

router = APIRouter(prefix="/api/stream", tags=["stream"], dependencies=[Depends(require_role("user"))])

RETRY_MS = 3000
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}
_IDLE = object()  # no event within the wait


def _heartbeat() -> str:
    now = datetime.now(UTC)
    event = HeartbeatEvent(id="0-0", type="heartbeat", at=now, data=HeartbeatData(at=now))
    return sse_frame(event, with_id=False)


@router.get(
    "",
    summary="Live event stream (text/event-stream)",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "Server-sent events, see /api/stream/recent"}
    },
)
async def stream(
    request: Request,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    resume_after: Annotated[
        str | None,
        Query(
            alias="last_event_id",
            description="Resume after this event id when the client opens a new connection (an EventSource cannot "
            "set headers); the Last-Event-ID header of the browser's own reconnect takes precedence",
        ),
    ] = None,
) -> StreamingResponse:
    """Events as they are published, a heartbeat every 15 s, and after a reconnect with ``Last-Event-ID`` (header, or
    the ``last_event_id`` parameter) the events that were missed (at most the last 100). The stream ends when the
    session ends; the browser then reconnects and receives 401."""
    live = live_state(request)
    sub = live.hub.subscribe()  # before the replay query, so nothing falls between replay and live events
    resume = next((i for i in (last_event_id, resume_after) if is_event_id(i)), None)
    replay = await live.bus.since(resume, REPLAY_LIMIT) if resume else []
    heartbeat_s = live.settings.stream_heartbeat_s

    async def body() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        sent = resume
        # The session is checked every heartbeat interval, also while events arrive more often than that (the
        # worker status every 10 s), so a revoked or expired session ends the stream within one interval.
        check_at = loop.time() + heartbeat_s
        idle_since = loop.time()
        try:
            yield f"retry: {RETRY_MS}\n\n"
            for event in replay:
                yield sse_frame(event)
                sent = event.id
            while not sub.closed:
                wake_at = min(check_at, idle_since + heartbeat_s)
                try:
                    event = await asyncio.wait_for(sub.queue.get(), timeout=max(0.0, wake_at - loop.time()))
                except TimeoutError:
                    event = _IDLE
                if loop.time() >= check_at:
                    if read_session(request) is None:
                        break
                    check_at = loop.time() + heartbeat_s
                if event is _IDLE:
                    if loop.time() - idle_since >= heartbeat_s:
                        yield _heartbeat()
                        idle_since = loop.time()
                    continue
                if event is None:
                    break
                if sent is not None and id_key(event.id) <= id_key(sent):
                    continue
                yield sse_frame(event)
                sent = event.id
                idle_since = loop.time()
        finally:
            live.hub.unsubscribe(sub)

    return StreamingResponse(body(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/recent", response_model=list[StreamEvent], summary="Recent events")
async def recent(request: Request, limit: Annotated[int, Query(ge=1, le=REPLAY_LIMIT)] = 50) -> list[StreamEvent]:
    """The last ``limit`` events of the stream, oldest first (heartbeats are not stored)."""
    return await live_state(request).bus.recent(limit)
