"""FurnaceMind conversations: create, list, read, and ask with the answer streamed as server-sent events (SV-5.1, #312).

The agent is page 7's (``agents.furnacemind.turn``) run by ``services.furnacemind``; conversations and messages are
stored in the twin store (``twin.fm_conversation``, ``twin.fm_message``), so they survive a restart of the API. A
conversation belongs to the user who created it; an administrator may read any. Times are UTC.

SV-5.2 (#313), for the SPA page: page 7's quick skills (``GET /api/fm/skills``; ask with ``skill`` instead of
``content``), the full DataFrame of a dataset artifact as CSV while this process holds it, and the owner's feedback on
an answer (rating and comment, ``twin.fm_feedback``, audited as ``fm.feedback``).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from api.auth import auth_state, client_address, require_role
from api.errors import ApiError
from api.store import twin_store
from data.twin.repositories import Actor, NotFound, TwinStore
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

    content: str = Field("", max_length=MAX_QUESTION_CHARS, description="The question (Markdown); empty with a skill")
    skill: str | None = Field(
        None,
        max_length=64,
        description="Slug of a quick skill (GET /api/fm/skills) to run instead of a typed question; its prompt "
        "becomes the question",
    )
    reasoning_level: fm.ReasoningLevel | None = Field(
        None, description="Reasoning profile of this turn (default: the conversation's)"
    )

    @model_validator(mode="after")
    def _question_or_skill(self) -> MessageCreate:
        if bool(self.content.strip()) == bool(self.skill):
            raise ValueError("Give either a question (content) or a quick skill (skill).")
        return self


class FeedbackSave(BaseModel):
    model_config = {"extra": "forbid"}

    rating: Literal["up", "down"] = Field(..., description="up: the answer was useful; down: it needs work")
    comment: str = Field("", max_length=fm.FEEDBACK_COMMENT_CHARS, description="What was right or wrong (optional)")


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


@router.get("/skills", response_model=fm.FmSkillList, summary="Quick skills")
def list_skills() -> fm.FmSkillList:
    """Page 7's quick skills: the active rows of the skills table, else the built-in ones (Unit Cost, Shift to Best,
    Heatloads), with the text a click shows at page 7's default shift (IST)."""
    try:
        return fm.list_skills()
    except Exception as exc:  # noqa: BLE001 - the skill stack (engine, its parameters) cannot load
        raise ApiError(503, f"FurnaceMind skills are not available: {type(exc).__name__}: {exc}") from exc


@router.put(
    "/conversations/{conversation_id}/messages/{seq}/feedback",
    response_model=fm.FmFeedback,
    summary="Rate an answer",
    responses={404: {"description": "No such conversation of the user, or message seq is not an answer"}},
)
def save_feedback(
    conversation_id: str,
    body: FeedbackSave,
    request: Request,
    store: Annotated[TwinStore, Depends(twin_store)],
    seq: Annotated[int, Path(ge=1, description="Position of the answer in the conversation, 1-based")],
) -> fm.FmFeedback:
    """The owner's rating (up / down, page 7's thumbs) and comment on one answer; a later save replaces it. Each save
    writes its audit row (``fm.feedback``, the values before and after) in the same transaction."""
    principal = _principal(request)
    key = _conversation_id(conversation_id)
    _visible(store.fm.get(key), principal, write=True)
    actor = Actor(
        name=principal.username,
        role=principal.role,
        client_ip=client_address(request, auth_state(request).settings.trusted_proxy_set),
    )
    try:
        return fm.save_feedback(
            store, str(key), seq, actor, rating=body.rating, comment=body.comment, author_id=principal.user_id
        )
    except NotFound as exc:
        raise ApiError(404, f"Message {seq} of this conversation is not an answer.") from exc


@router.get(
    "/conversations/{conversation_id}/datasets/{artifact_id}/csv",
    summary="Download a dataset (CSV)",
    response_class=Response,
    responses={
        200: {"content": {"text/csv": {}}, "description": "The full DataFrame with its index (page 7's download)"},
        404: {"description": "Not held by this API process (restart, another process, or replaced since)"},
    },
)
async def dataset_csv(
    conversation_id: str,
    request: Request,
    store: Annotated[TwinStore, Depends(twin_store)],
    artifact_id: Annotated[str, Path(pattern=r"^[0-9a-f]{1,32}$", description="artifact_id of the dataset part")],
) -> Response:
    """The full DataFrame of a dataset artifact (the part shows only its first rows), while this process holds it."""
    key = _conversation_id(conversation_id)
    _visible(await asyncio.to_thread(store.fm.get, key), _principal(request))
    held = await asyncio.to_thread(fm.dataset_csv, str(key), artifact_id)
    if held is None:
        raise ApiError(404, "The API no longer holds this dataset; download the preview or ask again.")
    name, text = held
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name) or "dataset.csv"
    return Response(
        text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe}"', "Cache-Control": "no-store"},
    )


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
        422: {"description": "Neither or both of content and skill, or a skill that is not active"},
        503: {"description": "The twin store or the agent (LLM, Qdrant settings) is not available"},
    },
)
async def post_message(
    conversation_id: str, body: MessageCreate, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> EventStreamResponse:
    """Store the question, run page 7's agent on the conversation (the model call streamed; tools in a worker
    thread) and stream its progress. The answer is stored when the turn completes, also if the client disconnects.
    One turn per conversation at a time. With ``skill`` the quick skill runs first in the turn thread and its prompt is
    stored as the question (``message.user`` follows once it is stored)."""
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
            skill=body.skill,
        )
    except fm.UnknownSkill as exc:
        raise ApiError(422, f"No active quick skill '{body.skill}'; list them with GET /api/fm/skills.") from exc
    except fm.Busy as exc:
        raise ApiError(409, "A turn of this conversation is running; wait for its answer.") from exc
    except fm.NotConfigured as exc:
        raise ApiError(503, f"FurnaceMind is not available: {exc}") from exc
    return EventStreamResponse(fm.start_turn(store, turn), headers=SSE_HEADERS)
