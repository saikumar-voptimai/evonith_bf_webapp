"""FurnaceMind conversations: create, list, read, and ask with the answer streamed as server-sent events (SV-5.1, #312).

The agent is page 7's (``agents.furnacemind.turn``) run by ``services.furnacemind``; conversations and messages are
stored in the twin store (``twin.fm_conversation``, ``twin.fm_message``), so they survive a restart of the API. A
conversation belongs to the user who created it; an administrator may read any. Times are UTC.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.auth import require_role
from api.errors import ApiError
from api.store import twin_store
from data.twin.repositories import TwinStore
from services import furnacemind as fm
from services.auth import Principal

router = APIRouter(prefix="/api/fm", tags=["furnacemind"], dependencies=[Depends(require_role("user"))])

SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}
MAX_QUESTION_CHARS = 8_000


class EventStreamResponse(StreamingResponse):
    """A server-sent-event stream; documents its event model under ``text/event-stream``."""

    media_type = "text/event-stream"


class ConversationCreate(BaseModel):
    model_config = {"extra": "forbid"}

    title: str = Field("", max_length=200, description="Title; empty: the first question becomes the title")
    reasoning_level: fm.ReasoningLevel | None = Field(
        None, description="OpenRouter reasoning profile (default: the configured default, Medium)"
    )


class MessageCreate(BaseModel):
    model_config = {"extra": "forbid"}

    content: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS, description="The question (Markdown)")
    reasoning_level: fm.ReasoningLevel | None = Field(
        None, description="Reasoning profile of this turn (default: the conversation's)"
    )


def _principal(request: Request) -> Principal:
    return request.state.principal


def _conversation_id(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ApiError(404, "No FurnaceMind conversation with this id.") from exc


def _visible(row: Any, principal: Principal, *, write: bool = False) -> Any:
    """The conversation when ``principal`` may read it (owner or admin) or write to it (owner only); else 404."""
    if row is None or (row.owner != principal.username and (write or principal.role != "admin")):
        raise ApiError(404, "No FurnaceMind conversation with this id.")
    return row


@router.post("/conversations", response_model=fm.FmConversation, status_code=201, summary="Start a conversation")
def create_conversation(
    body: ConversationCreate, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> fm.FmConversation:
    """An empty conversation owned by the signed-in user; ask with POST /api/fm/conversations/{id}/messages."""
    principal = _principal(request)
    return fm.create_conversation(
        store,
        owner=principal.username,
        owner_id=principal.user_id,
        reasoning_level=body.reasoning_level,
        title=body.title,
    )


@router.get("/conversations", response_model=list[fm.FmConversationSummary], summary="My conversations")
def list_conversations(
    request: Request,
    store: Annotated[TwinStore, Depends(twin_store)],
    limit: Annotated[int, Query(ge=1, le=100, description="Conversations, count")] = 30,
) -> list[fm.FmConversationSummary]:
    """The signed-in user's conversations, most recently updated first."""
    return fm.recent_conversations(store, _principal(request).username, limit=limit)


@router.get("/conversations/{conversation_id}", response_model=fm.FmConversation, summary="A conversation")
def get_conversation(
    conversation_id: str, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> fm.FmConversation:
    """Every message with its parts (tool calls and results, Plotly figure JSON, dataset previews), oldest first."""
    key = _conversation_id(conversation_id)
    _visible(store.fm.get(key), _principal(request))
    view = fm.load_conversation(store, str(key))
    if view is None:
        raise ApiError(404, "No FurnaceMind conversation with this id.")
    return view


@router.post(
    "/conversations/{conversation_id}/messages",
    summary="Ask (answer streamed as text/event-stream)",
    response_class=EventStreamResponse,
    responses={
        200: {
            "model": fm.FmStreamEvent,
            "description": "Server-sent events, one FmStreamEvent per event (``event:`` is its type): message.user "
            "(the stored question), token, tool.call, tool.result, artifact (Plotly figure or dataset preview), then "
            "message.assistant (the stored answer) or error.",
        },
        409: {"description": "A turn of this conversation is running"},
        503: {"description": "The twin store or the agent (LLM, Qdrant settings) is not available"},
    },
)
async def post_message(
    conversation_id: str, body: MessageCreate, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> EventStreamResponse:
    """Store the question, run page 7's agent on the conversation (the model call streamed; tools in a worker
    thread) and stream its progress. The answer is stored when the turn completes, also if the client disconnects.
    One turn per conversation at a time."""
    principal = _principal(request)
    key = _conversation_id(conversation_id)
    row = _visible(await asyncio.to_thread(store.fm.get, key), principal, write=True)
    try:
        turn = await asyncio.to_thread(
            fm.begin_turn,
            store,
            row,
            content=body.content,
            reasoning_level=body.reasoning_level,
            user_id=principal.user_id,
        )
    except fm.Busy as exc:
        raise ApiError(409, "A turn of this conversation is running; wait for its answer.") from exc
    except fm.NotConfigured as exc:
        raise ApiError(503, f"FurnaceMind is not available: {exc}") from exc
    return EventStreamResponse(fm.start_turn(store, turn), headers=SSE_HEADERS)
