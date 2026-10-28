"""FurnaceMind agent service (SV-5.1, #312): conversations in the twin store, turns streamed as server-sent events.

The service holds no conversation state of its own: conversations and messages are rows of ``twin.fm_conversation``
and ``twin.fm_message`` (``data.twin``), so any API process can continue any conversation, also after a restart.
A turn is page 7's turn (``agents.furnacemind.turn.run_turn``):

1. the user message is stored (``seq`` n);
2. the history replayed to the model is page 7's: the stored messages expanded into page 7's chat entries (an
   answer's figure and dataset count as entries) and the last ``memory_summary_message_window`` entries kept
   (``turn.history_to_messages``);
3. the LangGraph tool loop runs in a thread of this module's pool (the model call is streamed; the tools fetch data
   and execute the plot sandbox there, never on the event loop), with the conversation's tool state bound;
4. tool calls, tool results, Plotly figures (JSON) and dataset previews are streamed as they happen and stored, in
   the same order, as the ``parts`` of the assistant message (``seq`` n + 1).

The tools' working state (fetched DataFrames by dataset id, the active DataFrame and figure) is kept per
conversation in this process (the last ``STATE_CONVERSATIONS`` conversations); after a restart a follow-up question
that names an earlier dataset id fetches it again (the model is told the id is unknown). One turn per conversation
at a time (409 otherwise).

Imports of the agent stack (LangGraph, Plotly, pandas, the OpenRouter client, the tools) are deferred to the first
turn: the API starts without them, and ``utils.settings`` (which validates the LLM and Qdrant settings when imported)
is only read when a turn runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from data.twin.repositories import TwinStore
from services.twin_views import utc

log = logging.getLogger(__name__)

ReasoningLevel = Literal["Low", "Medium", "High"]
REASONING_LEVELS: tuple[str, ...] = ("Low", "Medium", "High")

#: Conversations whose tool state (DataFrames, figure) this process keeps.
STATE_CONVERSATIONS = 32
#: Turns running at once in this process (threads of the turn pool).
MAX_TURNS = 4
#: Characters of a tool result kept in its part (the model receives the whole result).
TOOL_RESULT_CHARS = 8_000
#: A figure larger than this (JSON, bytes) is stored without its data (``omitted``).
FIGURE_MAX_BYTES = 5_000_000
#: Rows of a dataset preview, and the cells (rows x columns) a preview may hold.
PREVIEW_ROWS = 50
PREVIEW_MAX_CELLS = 20_000
#: Page 7's history window when ``utils.settings`` cannot be read (``MEMORY_SUMMARY_MESSAGE_WINDOW``).
DEFAULT_HISTORY_WINDOW = 8


# ---------------------------------------------------------------------------------------------------------- views


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(..., description="Final answer of the turn (Markdown)")


class ToolCallPart(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str | None = Field(None, description="Id of the call, pairs the call with its result")
    name: str = Field(..., description="Tool name, e.g. fetch_ml_data")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments the model passed")
    label: str = Field("", description="Progress text shown while the tool runs")


class ToolResultPart(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str | None = Field(None, description="Id of the call")
    name: str = Field(..., description="Tool name")
    ok: bool = Field(..., description="False when the tool raised (content is then the error)")
    content: str = Field(..., description=f"Tool output returned to the model, at most {TOOL_RESULT_CHARS} characters")
    chars: int = Field(..., description="Length of the full tool output, characters")
    truncated: bool = Field(..., description="True when content is shorter than the full output")


class PlotlyPart(BaseModel):
    type: Literal["plotly"] = "plotly"
    figure: dict[str, Any] | None = Field(None, description="Plotly figure JSON (data, layout); null when omitted")
    bytes: int = Field(..., description="Size of the figure JSON, bytes")
    omitted: bool = Field(False, description=f"True when the figure exceeds {FIGURE_MAX_BYTES} bytes and is not kept")


class DatasetPreview(BaseModel):
    columns: list[str] = Field(..., description="Column names")
    index: list[Any] = Field(..., description="Row labels (ISO 8601 times for a time index, IST as the tool gave)")
    data: list[list[Any]] = Field(..., description="Rows of values in the columns' own units; null for missing")


class DatasetPart(BaseModel):
    type: Literal["dataset"] = "dataset"
    dataset_id: str = Field("", description="Dataset id the tools return to the model")
    rows: int = Field(..., description="Rows of the dataset, count")
    columns: int = Field(..., description="Columns of the dataset, count")
    meta: dict[str, Any] = Field(default_factory=dict, description="Source, window and resampling of the dataset")
    preview_rows: int = Field(..., description="Rows in the preview, count (first rows of the dataset)")
    preview: DatasetPreview | None = Field(None, description="First rows; null when they cannot be serialised")


MessagePart = Annotated[
    TextPart | ToolCallPart | ToolResultPart | PlotlyPart | DatasetPart,
    Field(discriminator="type"),
]


class FmMessage(BaseModel):
    seq: int = Field(..., description="Position in the conversation, 1-based")
    role: Literal["user", "assistant"]
    content: str = Field(..., description="Text replayed to the model (user question or final answer)")
    parts: list[MessagePart] = Field(default_factory=list, description="What the answer showed, in order")
    status: Literal["complete", "failed"] = Field(..., description="failed: the model was unavailable")
    model: str | None = Field(None, description="Model that answered")
    created_at: datetime = Field(..., description="UTC")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Reasoning level, model status and error, duration_s (s), tool_calls (count)"
    )


class FmConversationSummary(BaseModel):
    id: str = Field(..., description="Conversation id (UUID)")
    title: str = Field(..., description="First user message, shortened; empty before it")
    reasoning_level: ReasoningLevel = Field(..., description="OpenRouter reasoning profile of the conversation")
    owner: str = Field(..., description="Username of the creator")
    created_at: datetime = Field(..., description="UTC")
    updated_at: datetime = Field(..., description="UTC, time of the last message")
    message_count: int = Field(..., description="Messages in the conversation, count")


class FmConversation(FmConversationSummary):
    messages: list[FmMessage] = Field(default_factory=list, description="Every message, oldest first")
    running: bool = Field(False, description="True while a turn runs in this API process")


class FmUserMessageEvent(BaseModel):
    type: Literal["message.user"] = "message.user"
    conversation_id: str
    message: FmMessage


class FmTokenEvent(BaseModel):
    type: Literal["token"] = "token"
    step: int = Field(..., description="Model call of the turn, 0-based (tool rounds in between)")
    text: str = Field(..., description="Visible content as it streams (reasoning blocks removed)")


class FmToolCallEvent(BaseModel):
    type: Literal["tool.call"] = "tool.call"
    part: ToolCallPart


class FmToolResultEvent(BaseModel):
    type: Literal["tool.result"] = "tool.result"
    part: ToolResultPart


class FmArtifactEvent(BaseModel):
    type: Literal["artifact"] = "artifact"
    part: PlotlyPart | DatasetPart = Field(..., discriminator="type")


class FmAssistantMessageEvent(BaseModel):
    type: Literal["message.assistant"] = "message.assistant"
    conversation_id: str
    message: FmMessage


class FmErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    status: int = Field(..., description="HTTP status the error would have had")
    title: str
    detail: str | None = None


FmStreamEvent = Annotated[
    FmUserMessageEvent
    | FmTokenEvent
    | FmToolCallEvent
    | FmToolResultEvent
    | FmArtifactEvent
    | FmAssistantMessageEvent
    | FmErrorEvent,
    Field(discriminator="type"),
]


def stream_frame(event: BaseModel) -> str:
    """One server-sent event: ``event`` (the type) and the event as JSON ``data``."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"  # type: ignore[attr-defined]


def message_view(row: Any) -> FmMessage:
    return FmMessage(
        seq=row.seq,
        role=row.role,
        content=row.content or "",
        parts=list(row.parts or []),
        status=row.status,
        model=row.model,
        created_at=utc(row.created_at),
        meta=dict(row.meta or {}),
    )


def conversation_summary(row: Any) -> FmConversationSummary:
    return FmConversationSummary(
        id=str(row.id),
        title=row.title or "",
        reasoning_level=row.reasoning_level,
        owner=row.owner,
        created_at=utc(row.created_at),
        updated_at=utc(row.updated_at),
        message_count=row.message_count,
    )


def conversation_view(row: Any, messages: Iterable[Any]) -> FmConversation:
    return FmConversation(
        **conversation_summary(row).model_dump(),
        messages=[message_view(m) for m in messages],
        running=is_running(str(row.id)),
    )


# ------------------------------------------------------------------------------------------------------- history


def chat_entries(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Stored messages as page 7's chat-history entries.

    Page 7 appends, for an answer, the turn's figure and DataFrame entries before the answer text; they count toward
    the history window but are not replayed. The same entries are rebuilt here so the window is page 7's.
    """
    entries: list[dict[str, Any]] = []
    for row in messages:
        if row.role == "assistant":
            kinds = {p.get("type") for p in row.parts or [] if isinstance(p, Mapping)}
            if "plotly" in kinds:
                entries.append({"role": "assistant", "type": "plotly", "content": ""})
            if "dataset" in kinds:
                entries.append({"role": "assistant", "type": "dataframe", "content": ""})
        entries.append({"role": row.role, "type": "text", "content": row.content or "", "metadata": row.meta or {}})
    return entries


# --------------------------------------------------------------------------------------------------------- parts


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def tool_call_part(call_id: str | None, name: str, arguments: Mapping[str, Any], label: str) -> ToolCallPart:
    return ToolCallPart(call_id=call_id, name=name, arguments=_jsonable(dict(arguments)), label=label)


def tool_result_part(call_id: str | None, name: str, result: str, ok: bool) -> ToolResultPart:
    text = str(result)
    return ToolResultPart(
        call_id=call_id,
        name=name,
        ok=ok,
        content=text[:TOOL_RESULT_CHARS],
        chars=len(text),
        truncated=len(text) > TOOL_RESULT_CHARS,
    )


def figure_part(figure: Any) -> PlotlyPart:
    """A Plotly figure as JSON (``plotly.io.to_json``, the figure page 7 renders)."""
    import plotly.io as pio

    text = pio.to_json(figure, validate=False)
    size = len(text.encode("utf-8"))
    if size > FIGURE_MAX_BYTES:
        return PlotlyPart(figure=None, bytes=size, omitted=True)
    return PlotlyPart(figure=json.loads(text), bytes=size)


def dataset_part(frame: Any, meta: Mapping[str, Any]) -> DatasetPart:
    """Shape, metadata and the first rows of a DataFrame (page 7 shows its first 50 rows)."""
    n_columns = int(frame.shape[1])
    n_rows = max(1, min(PREVIEW_ROWS, PREVIEW_MAX_CELLS // max(1, n_columns)))
    head = frame.head(n_rows)
    preview: DatasetPreview | None
    try:
        split = json.loads(head.to_json(orient="split", date_format="iso", default_handler=str))
        preview = DatasetPreview(
            columns=[str(c) for c in split.get("columns", [])],
            index=list(split.get("index", [])),
            data=list(split.get("data", [])),
        )
    except Exception:  # noqa: BLE001 - exotic column types: shape and metadata only
        preview = None
    clean_meta = _jsonable(dict(meta))
    return DatasetPart(
        dataset_id=str(clean_meta.get("dataset_id") or ""),
        rows=int(frame.shape[0]),
        columns=n_columns,
        meta=clean_meta,
        preview_rows=len(head),
        preview=preview,
    )


# -------------------------------------------------------------------------------------------------- dependencies


@dataclass
class Dependencies:
    """What a turn needs from outside the service; tests replace it (``use``)."""

    #: Chat client for a reasoning level (``OpenRouterClient(reasoning_level=...)``).
    llm: Callable[[str], Any]
    #: Tool schemas (``agents.furnace_tools.get_openai_tool_schemas``).
    tools: Callable[[], list[dict[str, Any]]]
    #: System prompt of a turn.
    system_prompt: Callable[[], str]
    #: Shared objects seeded into every conversation's tool state (vector stores, repositories).
    resources: Callable[[], Mapping[str, Any]]
    #: History window of page 7, entries.
    history_window: Callable[[], int]


def _default_llm(level: str) -> Any:
    from agents.llm.llm_client import OpenRouterClient

    return OpenRouterClient(reasoning_level=level)


def _default_tools() -> list[dict[str, Any]]:
    from agents.furnace_tools import get_openai_tool_schemas

    return get_openai_tool_schemas()


def _default_system_prompt() -> str:
    """Page 7's prompt without the per-user layers (memory summary, feedback lessons, semantic memory, skills)."""
    from agents.furnacemind import prompts
    from agents.furnacemind.context import SystemPromptContext
    from agents.furnacemind.turn import system_prompt

    return system_prompt(SystemPromptContext(), tool_policy=prompts.TOOL_POLICY)


#: A shared store that could not be built is built again on the first turn after this many seconds.
RESOURCE_RETRY_S = 60.0


class ResourceCache:
    """Shared stores built once per process. A store whose build fails is left out (the tool that needs it answers
    that it is not initialised) and built again on a later call, at most once per ``retry_s``; page 7's
    ``st.cache_resource`` does not cache a failure either."""

    def __init__(
        self,
        builders: Mapping[str, Callable[[], Any]],
        *,
        retry_s: float = RESOURCE_RETRY_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._builders = dict(builders)
        self._retry_s = retry_s
        self._clock = clock
        self._built: dict[str, Any] = {}
        self._retry_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def __call__(self) -> dict[str, Any]:
        """The stores built so far (a copy), after building each missing store that is due."""
        with self._lock:
            for key, build in self._builders.items():
                if key in self._built:
                    continue
                now = self._clock()
                if now < self._retry_at.get(key, now):
                    continue
                try:
                    self._built[key] = build()
                except Exception as exc:  # noqa: BLE001 - optional store: the turn runs without it
                    self._retry_at[key] = now + self._retry_s
                    log.warning("FurnaceMind %s unavailable (next try in %.0f s): %s", key, self._retry_s, exc)
                else:
                    self._retry_at.pop(key, None)
            return dict(self._built)


def _shift_store() -> Any:
    from agents.memory.vector_store import QdrantVectorStore

    return QdrantVectorStore()


def _knowledge_store() -> Any:
    from agents.embeddings.cloud_embedding import CloudEmbeddingClient
    from agents.memory.knowledge_vector_store import KnowledgeVectorStore

    return KnowledgeVectorStore(CloudEmbeddingClient())


def _relational(name: str) -> Callable[[], Any]:
    def build() -> Any:
        from furnace_data import relational as rel

        engine = rel.build_relational_engine()
        return getattr(rel, name)(rel.build_relational_session_factory(engine))

    return build


#: Page 7's shared stores of this process (vector stores, knowledge repositories).
_default_resources = ResourceCache(
    {
        "shift_store": _shift_store,
        "knowledge_store": _knowledge_store,
        "knowledge_document_repository": _relational("MemoryDocumentRepository"),
        "knowledge_retrieval_trace_repository": _relational("RetrievalTraceRepository"),
    }
)


def _default_history_window() -> int:
    try:
        from utils.settings import settings

        return int(settings.memory_summary_message_window)
    except Exception:  # noqa: BLE001
        return DEFAULT_HISTORY_WINDOW


def default_dependencies() -> Dependencies:
    return Dependencies(
        llm=_default_llm,
        tools=_default_tools,
        system_prompt=_default_system_prompt,
        resources=_default_resources,
        history_window=_default_history_window,
    )


_DEPS: Dependencies | None = None
_DEPS_LOCK = threading.Lock()


def dependencies() -> Dependencies:
    global _DEPS
    with _DEPS_LOCK:
        if _DEPS is None:
            _DEPS = default_dependencies()
        return _DEPS


def use(deps: Dependencies | None) -> None:
    """Replace the dependencies (tests); ``None`` restores the defaults on next use. Also forgets the tool states."""
    global _DEPS
    with _DEPS_LOCK:
        _DEPS = deps
    with _STATES_LOCK:
        _STATES.clear()


def default_reasoning_level() -> str:
    try:
        from utils.settings import normalize_openrouter_reasoning_level, settings

        return normalize_openrouter_reasoning_level(settings.llm.openrouter.default_reasoning_level)
    except Exception:  # noqa: BLE001 - settings not configured: page 7's default profile
        return "Medium"


# ---------------------------------------------------------------------------------------------- tool states, turns


_STATES: OrderedDict[str, dict[str, Any]] = OrderedDict()
_STATES_LOCK = threading.Lock()
_RUNNING: set[str] = set()
_RUNNING_LOCK = threading.Lock()
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


class Busy(RuntimeError):
    """A turn of this conversation is running."""


def is_running(conversation_id: str) -> bool:
    with _RUNNING_LOCK:
        return conversation_id in _RUNNING


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=MAX_TURNS, thread_name_prefix="furnacemind-turn")
        return _POOL


def tool_state(conversation_id: str, *, user_id: str | None, resources: Mapping[str, Any]) -> dict[str, Any]:
    """The tool state of a conversation in this process (created when missing). The turn's shared resources are
    seeded on every turn, so a store built after the conversation started reaches it."""
    with _STATES_LOCK:
        state = _STATES.get(conversation_id)
        if state is None:
            state = {"fm_conversation_id": conversation_id}
            _STATES[conversation_id] = state
            while len(_STATES) > STATE_CONVERSATIONS:
                _STATES.popitem(last=False)
        else:
            _STATES.move_to_end(conversation_id)
        state.update(resources)
        state["fm_user_id"] = user_id
        return state


def forget_states() -> None:
    """Drop every conversation's tool state (an API restart does the same)."""
    with _STATES_LOCK:
        _STATES.clear()


def create_conversation(
    store: TwinStore,
    *,
    owner: str,
    owner_id: str | None,
    reasoning_level: str | None = None,
    title: str = "",
) -> FmConversation:
    row = store.fm.create(
        owner=owner,
        owner_id=owner_id,
        reasoning_level=reasoning_level or default_reasoning_level(),
        title=" ".join(title.split()),
    )
    return conversation_view(row, [])


def recent_conversations(store: TwinStore, owner: str, *, limit: int = 30) -> list[FmConversationSummary]:
    return [conversation_summary(r) for r in store.fm.recent(owner, limit=limit)]


def load_conversation(store: TwinStore, conversation_id: str) -> FmConversation | None:
    row = store.fm.get(conversation_id)
    if row is None:
        return None
    return conversation_view(row, store.fm.messages(row.id))


class _StreamObserver:
    """Agent observer of the service: converts every event to a stream event and a stored part, in order."""

    wants_tokens = True

    def __init__(self, emit: Callable[[BaseModel], None]) -> None:
        self.emit = emit
        self.parts: list[BaseModel] = []

    def on_token(self, step: int, text: str) -> None:
        self.emit(FmTokenEvent(step=step, text=text))

    def on_tool_call(self, call_id: str | None, name: str, arguments: dict[str, Any], label: str) -> None:
        part = tool_call_part(call_id, name, arguments, label)
        self.parts.append(part)
        self.emit(FmToolCallEvent(part=part))

    def on_tool_result(self, call_id: str | None, name: str, result: str, ok: bool) -> None:
        part = tool_result_part(call_id, name, result, ok)
        self.parts.append(part)
        self.emit(FmToolResultEvent(part=part))

    def on_artifact(self, artifact: Any) -> None:
        try:
            part: PlotlyPart | DatasetPart = (
                figure_part(artifact.value)
                if artifact.kind == "figure"
                else dataset_part(artifact.value, artifact.meta)
            )
        except Exception as exc:  # noqa: BLE001 - an artifact that cannot be serialised is left out, not the answer
            log.warning("FurnaceMind artifact %s not serialised: %s", artifact.kind, exc)
            return
        self.parts.append(part)
        self.emit(FmArtifactEvent(part=part))


class NotConfigured(RuntimeError):
    """The agent stack cannot start (LLM key, Qdrant settings, missing package)."""


@dataclass
class TurnRequest:
    """A turn ready to run: the conversation (reserved), its stored user message, the client and the prompt."""

    conversation_id: str
    user_message: Any
    reasoning_level: str
    user_id: str | None
    llm: Any
    tools: list[dict[str, Any]]
    system_prompt: str
    history_window: int
    resources: Mapping[str, Any]


@dataclass
class _Prepared:
    llm: Any
    tools: list[dict[str, Any]]
    system_prompt: str
    history_window: int
    resources: Mapping[str, Any]


def prepare(reasoning_level: str) -> _Prepared:
    """Build the client, the tool schemas and the system prompt of a turn (loads the agent stack on first use).

    Synchronous; raises ``NotConfigured`` when the stack cannot start, before anything is stored.
    """
    deps = dependencies()
    try:
        return _Prepared(
            llm=deps.llm(reasoning_level),
            tools=deps.tools(),
            system_prompt=deps.system_prompt(),
            history_window=deps.history_window(),
            resources=deps.resources(),
        )
    except Exception as exc:  # noqa: BLE001 - reported as 503 by the route
        raise NotConfigured(f"{type(exc).__name__}: {exc}") from exc


def begin_turn(
    store: TwinStore,
    conversation: Any,
    *,
    content: str,
    reasoning_level: str | None,
    user_id: str | None,
    prepared: Callable[[str], _Prepared] = prepare,
) -> TurnRequest:
    """Reserve the conversation for one turn, prepare the agent and store the user message.

    Synchronous (agent start, database write): call it in a thread from async code. Raises ``Busy`` while a turn of
    the conversation runs and ``NotConfigured`` when the agent cannot start (nothing is stored then). The reservation
    ends with the turn (``run_turn_sync``), or here when the turn cannot begin.
    """
    key = str(conversation.id)
    with _RUNNING_LOCK:
        if key in _RUNNING:
            raise Busy(key)
        _RUNNING.add(key)
    level = reasoning_level or conversation.reasoning_level
    try:
        ready = prepared(level)
        row = store.fm.append(conversation.id, role="user", content=content, meta={"reasoning_level": level})
    except BaseException:
        _release(key)
        raise
    return TurnRequest(
        conversation_id=key,
        user_message=row,
        reasoning_level=level,
        user_id=user_id,
        llm=ready.llm,
        tools=ready.tools,
        system_prompt=ready.system_prompt,
        history_window=ready.history_window,
        resources=ready.resources,
    )


def _release(conversation_id: str) -> None:
    with _RUNNING_LOCK:
        _RUNNING.discard(conversation_id)


def turn_messages(store: TwinStore, conversation_id: str, *, system: str, window: int) -> list[dict[str, Any]]:
    """System prompt and page 7's history window over the stored messages (the new user message is the last)."""
    from agents.furnacemind.turn import history_to_messages

    # A stored message expands to at most 3 chat entries: ``window`` messages cover a window of ``window`` entries.
    history = chat_entries(store.fm.messages(conversation_id, last=max(window, 1)))
    return [{"role": "system", "content": system}, *history_to_messages(history, max_messages=window)]


def run_turn_sync(store: TwinStore, request: TurnRequest, emit: Callable[[BaseModel], None]) -> FmMessage | None:
    """Run the turn and store the answer; every event goes to ``emit`` (called from this thread). Releases the
    conversation. Returns the stored answer, or None after an ``error`` event (the user message stays stored)."""
    started = time.monotonic()
    try:
        from agents.furnacemind.turn import run_turn

        messages = turn_messages(
            store, request.conversation_id, system=request.system_prompt, window=request.history_window
        )
        state = tool_state(request.conversation_id, user_id=request.user_id, resources=request.resources)
        observer = _StreamObserver(emit)
        result = run_turn(
            llm=request.llm,
            messages=messages,
            tools=request.tools,
            state=state,
            observer=observer,
            turn_id=uuid.uuid4().hex,
        )
        usage_fn = getattr(request.llm, "usage_metadata", None)
        usage = _jsonable(usage_fn()) if callable(usage_fn) else {}
        meta = {
            "reasoning_level": request.reasoning_level,
            **usage,
            "tool_calls": len(result.tool_calls),
            "duration_s": round(time.monotonic() - started, 3),
        }
        if result.model_error:
            meta["model_error"] = meta.get("model_error") or result.model_error
        parts = [p.model_dump(mode="json") for p in observer.parts]
        parts.append(TextPart(text=result.final_text).model_dump(mode="json"))
        row = store.fm.append(
            request.conversation_id,
            role="assistant",
            content=result.final_text,
            parts=parts,
            status="failed" if result.model_error else "complete",
            model=usage.get("actual_model") or getattr(request.llm, "model", None),
            turn_id=result.turn_id,
            meta=meta,
        )
        message = message_view(row)
        emit(FmAssistantMessageEvent(conversation_id=request.conversation_id, message=message))
        return message
    except Exception as exc:  # noqa: BLE001 - reported on the stream; the stored user message stays
        log.exception("FurnaceMind turn of %s failed", request.conversation_id)
        from sqlalchemy.exc import OperationalError

        # As for a 500 problem response, the exception text stays in the log.
        if isinstance(exc, OperationalError):
            event = FmErrorEvent(
                status=503, title="The twin store is not reachable", detail="The answer was not stored."
            )
        else:
            event = FmErrorEvent(status=500, title="The turn failed", detail="See the API log for this conversation.")
        emit(event)
        return None
    finally:
        _release(request.conversation_id)


def start_turn(store: TwinStore, request: TurnRequest) -> AsyncIterator[str]:
    """Start the turn in the turn pool now and return its server-sent events.

    Call it from the event loop (an async route). The turn runs, is stored and releases the conversation whether or
    not the events are read: a client that goes away, even before the response starts, does not leave the
    conversation reserved. The events are the stored user message, then the turn's events as they happen, ending
    with the stored answer (``message.assistant``) or an ``error``.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[BaseModel | None] = asyncio.Queue()

    def emit(event: BaseModel | None) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:  # the event loop has closed (shutdown): the turn still completes and is stored
            pass

    def work() -> None:
        try:
            run_turn_sync(store, request, emit)
        finally:
            emit(None)

    loop.run_in_executor(_pool(), work)

    async def frames() -> AsyncIterator[str]:
        yield stream_frame(
            FmUserMessageEvent(conversation_id=request.conversation_id, message=message_view(request.user_message))
        )
        while True:
            event = await queue.get()
            if event is None:
                return
            yield stream_frame(event)

    return frames()


def parse_events(body: str) -> list[dict[str, Any]]:
    """The events of a server-sent-event body as ``{"type": ..., **data}`` (tests and scripts)."""
    events: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        data = [line[len("data: ") :] for line in block.splitlines() if line.startswith("data: ")]
        if data:
            events.append(json.loads("\n".join(data)))
    return events


__all__ = [
    "Busy",
    "DatasetPart",
    "Dependencies",
    "FmConversation",
    "FmConversationSummary",
    "FmMessage",
    "FmStreamEvent",
    "MessagePart",
    "NotConfigured",
    "PlotlyPart",
    "REASONING_LEVELS",
    "RESOURCE_RETRY_S",
    "ReasoningLevel",
    "ResourceCache",
    "TextPart",
    "ToolCallPart",
    "ToolResultPart",
    "TurnRequest",
    "begin_turn",
    "chat_entries",
    "create_conversation",
    "dataset_part",
    "default_dependencies",
    "figure_part",
    "forget_states",
    "is_running",
    "load_conversation",
    "parse_events",
    "prepare",
    "recent_conversations",
    "run_turn_sync",
    "start_turn",
    "stream_frame",
    "tool_state",
    "turn_messages",
    "use",
]
