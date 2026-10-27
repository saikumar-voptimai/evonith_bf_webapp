"""One FurnaceMind chat turn, shared by page 7 and the agent service (SV-5.1, #312).

Page 7 (``agents.furnacemind.ai_cooperate_page``) and ``services.furnacemind`` (the twin-api) compose a turn from the
same parts:

* :func:`history_to_messages` - the recent user and assistant texts replayed to the model (artifact entries and
  entries the caller rejects are skipped);
* :func:`system_prompt` - persona, static context, skill context, memory and the tool policy;
* :func:`run_turn` - the LangGraph tool loop (``agents.furnacemind.graph``) over the tools' working state
  (``agents.furnacemind.tool_state``), the artifacts the turn produced (figures and DataFrames owned by the turn id,
  in the order the tools produced them) and the operator message when the model is unavailable.

The page binds ``st.session_state`` as the tools' state and reports progress in its status placeholder; the service
binds a mapping per conversation and forwards the progress to its server-sent-event stream. No Streamlit import.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from agents.furnacemind import tool_state
from agents.furnacemind.graph import AgentObserver, NullObserver, run_furnacemind_graph_loop

#: Chat-history entry types that hold an artifact, not text (never replayed to the model).
ARTIFACT_TYPES = frozenset({"plotly", "dataframe"})
#: Fallback text when the client has no ``unavailable_message`` (test doubles).
UNAVAILABLE = "The selected model is busy. Please try another mode."


def history_to_messages(
    history: Iterable[dict[str, Any]],
    *,
    max_messages: int,
    skip: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, str]]:
    """OpenAI messages for the last ``max_messages`` history entries: user and assistant text only.

    Args:
        history: Chat entries ``{"role", "content", "type", "metadata", ...}``, oldest first.
        max_messages: Number of entries taken from the end of ``history`` (artifact entries count, as on page 7).
        skip: Entries for which it returns True are left out (page 7: revoked knowledge, inactive skills).

    Returns:
        ``[{"role": "user" | "assistant", "content": str}, ...]`` in history order.
    """
    items = list(history)
    messages: list[dict[str, str]] = []
    for item in items[-max_messages:]:
        if item.get("type") in ARTIFACT_TYPES:
            continue
        if skip is not None and skip(item):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    return messages


def system_prompt(
    context: Any,
    *,
    tool_policy: str,
    skill_id: str | None = None,
    skill_context: str | None = None,
    feedback_context: str = "",
    semantic_context: str = "",
) -> str:
    """The system prompt of a turn: ``context.build`` with the tool policy, feedback lessons and semantic memory.

    Args:
        context: ``agents.furnacemind.context.SystemPromptContext`` (refreshed for the conversation by the caller).
        tool_policy: ``agents.furnacemind.prompts.TOOL_POLICY``.
        skill_id: Quick skill of the turn, if any.
        skill_context: Selected and semantically retrieved skill context, if any.
        feedback_context: Lessons from operator feedback relevant to the question.
        semantic_context: Long-term memory facts relevant to the question.
    """
    extra = tool_policy
    if feedback_context:
        extra = f"{extra}\n\n{feedback_context}"
    if semantic_context:
        extra = f"{extra}\n\n{semantic_context}"
    return context.build(extra=extra, skill_id=skill_id, skill_context=skill_context)


@dataclass
class ToolCallRecord:
    """One tool call of a turn and its result (``result`` None while it runs)."""

    call_id: str | None
    name: str
    arguments: dict[str, Any]
    label: str
    result: str | None = None
    ok: bool | None = None


@dataclass
class Artifact:
    """A figure or DataFrame a tool produced during the turn.

    ``kind`` is ``"figure"`` (``value`` a Plotly figure) or ``"dataframe"`` (``value`` a DataFrame, ``meta`` the
    dataset metadata of the tool). ``after_call`` is the index in :attr:`TurnResult.tool_calls` of the call that
    produced it.
    """

    kind: str
    value: Any
    meta: dict[str, Any] = field(default_factory=dict)
    after_call: int | None = None


@dataclass
class TurnResult:
    """What a turn produced: the final text, the tool calls in order, the artifacts and the model status."""

    turn_id: str
    final_text: str
    tool_calls: list[ToolCallRecord]
    artifacts: list[Artifact]
    messages: list[dict[str, Any]]
    #: ``"Type: message"`` of the exception when the model call failed (``final_text`` is then the operator message).
    model_error: str | None = None


class _TurnRecorder:
    """Observer that records tool calls and artifacts, then forwards every event to the caller's observer."""

    def __init__(self, inner: AgentObserver, state: MutableMapping[str, Any], turn_id: str) -> None:
        self.inner = inner
        self.state = state
        self.turn_id = turn_id
        self.wants_tokens = bool(getattr(inner, "wants_tokens", False))
        self.calls: list[ToolCallRecord] = []
        self.artifacts: list[Artifact] = []
        self._seen_figure = state.get(tool_state.FIGURE)
        self._seen_frame = state.get(tool_state.DATAFRAME)

    def on_token(self, step: int, text: str) -> None:
        self.inner.on_token(step, text)

    def on_tool_call(self, call_id: str | None, name: str, arguments: dict[str, Any], label: str) -> None:
        self.calls.append(ToolCallRecord(call_id=call_id, name=name, arguments=dict(arguments), label=label))
        self.inner.on_tool_call(call_id, name, arguments, label)

    def on_tool_result(self, call_id: str | None, name: str, result: str, ok: bool) -> None:
        for record in reversed(self.calls):
            if record.call_id == call_id and record.name == name and record.result is None:
                record.result, record.ok = result, ok
                break
        self.inner.on_tool_result(call_id, name, result, ok)
        self.collect()

    def collect(self) -> None:
        """Record the figure and DataFrame the last tool produced (new objects owned by this turn)."""
        new: list[Artifact] = []
        after = len(self.calls) - 1 if self.calls else None
        figure = self.state.get(tool_state.FIGURE)
        if (
            figure is not None
            and figure is not self._seen_figure
            and tool_state.owned_by_turn(self.state, tool_state.FIGURE_TURN, self.turn_id)
        ):
            self._seen_figure = figure
            new.append(Artifact("figure", figure, after_call=after))
        frame = self.state.get(tool_state.DATAFRAME)
        if (
            frame is not None
            and frame is not self._seen_frame
            and not getattr(frame, "empty", False)
            and tool_state.owned_by_turn(self.state, tool_state.DATAFRAME_TURN, self.turn_id)
        ):
            self._seen_frame = frame
            meta = dict(self.state.get(tool_state.DATAFRAME_META) or {})
            if not meta.get("dataset_id"):  # not every tool writes its id into the metadata
                datasets = self.state.get("fm_datasets") or {}
                meta["dataset_id"] = next((k for k, v in datasets.items() if v.get("df") is frame), "")
            new.append(Artifact("dataframe", frame, meta=meta, after_call=after))
        for artifact in new:
            self.artifacts.append(artifact)
            on_artifact = getattr(self.inner, "on_artifact", None)
            if callable(on_artifact):
                on_artifact(artifact)


def run_turn(
    *,
    llm: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    state: MutableMapping[str, Any],
    observer: AgentObserver | None = None,
    turn_id: str | None = None,
) -> TurnResult:
    """Run one agent turn over ``state`` (bound as the tools' state for the duration of the turn).

    Args:
        llm: ``OpenRouterClient`` (or a double with ``chat_completions`` and optionally ``chat_completions_stream``).
        messages: System prompt and history (:func:`history_to_messages`); the graph appends the assistant and tool
            turns to it.
        tools: Tool schemas (``agents.furnace_tools.get_openai_tool_schemas()``).
        state: The tools' working state: page 7 ``st.session_state``, the service the conversation's mapping.
        observer: Progress receiver (page 7: ``StatusBoxObserver``; service: its stream). An observer with an
            ``on_artifact(artifact)`` method also receives each artifact as it is recorded.
        turn_id: Owner id of this turn's artifacts (a new one when omitted).

    Returns:
        :class:`TurnResult`. A failed model call does not raise: the client records the failure and the result
        carries its operator message (``llm.unavailable_message()``) and ``model_error``.
    """
    turn_id = turn_id or uuid.uuid4().hex
    recorder = _TurnRecorder(observer or NullObserver(), state, turn_id)
    model_error: str | None = None
    with tool_state.bind(state):
        state[tool_state.CURRENT_TURN] = turn_id
        try:
            final_text = run_furnacemind_graph_loop(llm=llm, messages=messages, tools=tools, observer=recorder)
        except Exception as exc:  # the model (or the network to it) failed: an operator message, not an error page
            model_error = f"{type(exc).__name__}: {exc}"
            record_failure = getattr(llm, "record_failure", None)
            if callable(record_failure):
                record_failure(exc)
            unavailable = getattr(llm, "unavailable_message", None)
            final_text = unavailable() if callable(unavailable) else UNAVAILABLE
        finally:
            state.pop(tool_state.CURRENT_TURN, None)
    recorder.collect()
    return TurnResult(
        turn_id=turn_id,
        final_text=final_text,
        tool_calls=recorder.calls,
        artifacts=recorder.artifacts,
        messages=messages,
        model_error=model_error,
    )


__all__ = [
    "ARTIFACT_TYPES",
    "Artifact",
    "ToolCallRecord",
    "TurnResult",
    "history_to_messages",
    "run_turn",
    "system_prompt",
]
