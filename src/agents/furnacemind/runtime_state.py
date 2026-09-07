"""Provide isolated FurnaceMind tool state for UI and headless executions.

FurnaceMind tools exchange transient datasets, plots, and retrieval context
through a mutable mapping. Interactive calls continue to use Streamlit's
session state, while a scheduled or otherwise headless caller can bind a plain
mapping for the duration of one agent run. A :class:`contextvars.ContextVar`
keeps those explicit bindings isolated across concurrent asynchronous contexts
and restores the previous binding after nested runs.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

FurnaceMindRuntimeState = MutableMapping[str, Any]

_BOUND_RUNTIME_STATE: ContextVar[FurnaceMindRuntimeState | None] = ContextVar(
    "furnacemind_runtime_state",
    default=None,
)


def _streamlit_runtime_state() -> FurnaceMindRuntimeState:
    """Return Streamlit session state for an interactive FurnaceMind call.

    Streamlit is imported lazily so importing FurnaceMind tools in a headless
    process does not initialize or require the UI runtime. Callers outside
    Streamlit should bind an explicit mapping with :func:`bind_runtime_state`.
    """

    import streamlit as st

    return cast(FurnaceMindRuntimeState, st.session_state)


def get_runtime_state() -> FurnaceMindRuntimeState:
    """Return the explicitly bound run state or the current UI session state."""

    state = _BOUND_RUNTIME_STATE.get()
    return state if state is not None else _streamlit_runtime_state()


@contextmanager
def bind_runtime_state(
    state: FurnaceMindRuntimeState | None = None,
) -> Iterator[FurnaceMindRuntimeState]:
    """Bind one mutable state mapping for the current FurnaceMind run.

    Args:
        state: Optional caller-owned mapping. A new dictionary is created when
            omitted, and the same mapping is yielded so the caller can collect
            generated datasets and artifacts after execution.

    Yields:
        The mapping used by FurnaceMind tools within this context.

    Raises:
        TypeError: If ``state`` is not a mutable mapping.
    """

    run_state: FurnaceMindRuntimeState = {} if state is None else state
    if not isinstance(run_state, MutableMapping):
        raise TypeError("FurnaceMind runtime state must be a mutable mapping.")

    token = _BOUND_RUNTIME_STATE.set(run_state)
    try:
        yield run_state
    finally:
        _BOUND_RUNTIME_STATE.reset(token)


__all__ = [
    "FurnaceMindRuntimeState",
    "bind_runtime_state",
    "get_runtime_state",
]
